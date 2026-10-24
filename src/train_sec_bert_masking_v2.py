"""Train and evaluate the paired SEC-BERT lexical-masking experiment.

The original model is trained on ``text_context_v2`` and evaluated on the same
test findings with original and masked inputs.  A second model is trained and
evaluated on ``text_context_masked_v2``.  Both runs retain the frozen report-
grouped partitions and select an epoch only on the validation split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sec_bert_masking_v2.json"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.texts[index], "label": self.labels[index]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    mask = frame["split"].eq(split) & frame["clean_eval"].fillna(False).astype(bool)
    return frame.loc[mask].copy().reset_index(drop=True)


def make_collator(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        tokens = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return tokens

    return collate


def make_loader(
    frame: pd.DataFrame,
    text_column: str,
    tokenizer,
    config: dict,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    batch_size = int(config["batch_size"]) if shuffle else int(config["batch_size"]) * 2
    return DataLoader(
        TextDataset(frame[text_column].tolist(), frame["label_id"].tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collator(tokenizer, int(config["max_length"])),
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


@torch.inference_mode()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits.append(model(**batch).logits.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    text_column: str,
    run_name: str,
    tokenizer,
    labels: list[str],
    label_to_id: dict[str, int],
    config: dict,
    cache_dir: str,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict], int]:
    seed = int(config["seed"])
    set_seed(seed)
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=cache_dir,
    ).to(device)
    train_loader = make_loader(train, text_column, tokenizer, config, True, seed)
    validation_loader = make_loader(validation, text_column, tokenizer, config, False, seed)

    counts = train["label_id"].value_counts().reindex(range(len(labels)), fill_value=1).to_numpy(float)
    weights = torch.tensor(len(train) / (len(labels) * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation"])
    updates_per_epoch = int(np.ceil(len(train_loader) / accumulation))
    total_updates = updates_per_epoch * int(config["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        max(1, int(total_updates * float(config["warmup_ratio"]))),
        total_updates,
    )

    best_epoch, best_f1, best_state = 0, -1.0, None
    history: list[dict] = []
    started = time.perf_counter()
    update = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            y = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(**batch).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y, weight=weights) / accumulation
            loss.backward()
            running += float(loss.item()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 100 == 0:
                    print(
                        f"run={run_name} epoch={epoch} update={update}/{total_updates} "
                        f"loss={running/step:.4f} elapsed_s={time.perf_counter()-started:.1f}",
                        flush=True,
                    )
        validation_logits, validation_y = predict(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            **metrics(validation_y, validation_logits.argmax(axis=1)),
        }
        history.append(row)
        print(json.dumps({run_name: {"validation": row}}, ensure_ascii=False), flush=True)
        if row["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, row["macro_f1"]
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )

    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for {run_name}")
    model.load_state_dict(best_state)
    model.to(device)
    checkpoint_dir = ROOT / config["checkpoint_root"] / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    return model, history, best_epoch


def evaluate_condition(
    model,
    frame: pd.DataFrame,
    text_column: str,
    condition: str,
    tokenizer,
    labels: list[str],
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    loader = make_loader(frame, text_column, tokenizer, config, False, int(config["seed"]))
    logits, y = predict(model, loader, device)
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    prediction_id = probabilities.argmax(axis=1)
    prediction = np.asarray([labels[int(index)] for index in prediction_id])
    ordered = np.sort(probabilities, axis=1)
    clipped = np.clip(probabilities, 1e-15, 1.0)

    keep = ["sample_id", "report_id", config["target"], "context_quality"]
    output = frame[keep].copy()
    output["condition"] = condition
    output["prediction"] = prediction
    output["correct"] = prediction_id == y
    output["confidence"] = probabilities.max(axis=1)
    output["margin"] = ordered[:, -1] - ordered[:, -2]
    output["entropy"] = -np.sum(clipped * np.log(clipped), axis=1)
    output["logits_json"] = [json.dumps(list(map(float, row)), separators=(",", ":")) for row in logits]
    return output, {"samples": int(len(output)), **metrics(y, prediction_id)}


def paired_report_bootstrap(
    target: np.ndarray,
    original_prediction: np.ndarray,
    masked_prediction: np.ndarray,
    report_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    codes, reports = pd.factorize(pd.Series(report_id).astype(str), sort=False)
    groups = [np.flatnonzero(codes == code) for code in range(len(reports))]
    values = {"accuracy_delta_masked_minus_original": [], "macro_f1_delta_masked_minus_original": []}
    for _ in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        y = target[indices]
        original = original_prediction[indices]
        masked = masked_prediction[indices]
        values["accuracy_delta_masked_minus_original"].append(
            float(accuracy_score(y, masked) - accuracy_score(y, original))
        )
        values["macro_f1_delta_masked_minus_original"].append(
            float(
                f1_score(y, masked, average="macro", zero_division=0)
                - f1_score(y, original, average="macro", zero_division=0)
            )
        )
    return {
        key: [float(value) for value in np.quantile(series, [0.025, 0.5, 0.975])]
        for key, series in values.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reproducibility_payload() -> dict:
    packages = {}
    for name in ("torch", "transformers", "pandas", "numpy", "scikit-learn", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
        ).splitlines()
    except subprocess.SubprocessError as error:
        freeze = [f"pip freeze failed: {error}"]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
        "pip_freeze": freeze,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts/hf_cache"))
    parser.add_argument("--skip-masked-training", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Enable a Kaggle GPU accelerator before running.")
    device = torch.device("cuda")
    print(f"device=cuda gpu={torch.cuda.get_device_name(0)}", flush=True)

    frame = pd.read_csv(ROOT / config["dataset"])
    required_columns = {
        "sample_id", "report_id", "split", "clean_eval", "context_quality",
        config["target"], config["original_text_column"], config["masked_text_column"],
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    splits = {name: clean_split(frame, name) for name in ("train", "validation", "calibration", "test")}
    observed_sizes = {name: int(len(value)) for name, value in splits.items()}
    if observed_sizes != config["expected_clean_split_sizes"]:
        raise ValueError(f"Unexpected clean split sizes: {observed_sizes}")

    labels = sorted(frame[config["target"]].dropna().astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    for split_frame in splits.values():
        split_frame["label_id"] = split_frame[config["target"]].map(label_to_id).astype(int)
        for column in (config["original_text_column"], config["masked_text_column"]):
            split_frame[column] = split_frame[column].fillna("").astype(str)

    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=args.cache_dir)
    table_dir = ROOT / "results/masking_v2/tables"
    metrics_dir = ROOT / "results/masking_v2/metrics"
    report_dir = ROOT / "results/masking_v2/reports"
    for directory in (table_dir, metrics_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    original_model, original_history, original_best_epoch = train_model(
        splits["train"], splits["validation"], config["original_text_column"],
        "original", tokenizer, labels, label_to_id, config, args.cache_dir, device,
    )
    original_calibration, original_calibration_metrics = evaluate_condition(
        original_model, splits["calibration"], config["original_text_column"],
        "original_model_original_calibration", tokenizer, labels, config, device,
    )
    original_test, original_test_metrics = evaluate_condition(
        original_model, splits["test"], config["original_text_column"],
        "original_model_original_test", tokenizer, labels, config, device,
    )
    masked_test, masked_test_metrics = evaluate_condition(
        original_model, splits["test"], config["masked_text_column"],
        "original_model_masked_test", tokenizer, labels, config, device,
    )
    original_calibration.to_csv(
        table_dir / "original_model_calibration_predictions.csv", index=False, encoding="utf-8-sig"
    )
    original_test.to_csv(
        table_dir / "original_model_original_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    masked_test.to_csv(
        table_dir / "original_model_masked_test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    target = splits["test"]["label_id"].to_numpy(int)
    original_prediction = original_test["prediction"].map(label_to_id).to_numpy(int)
    masked_prediction = masked_test["prediction"].map(label_to_id).to_numpy(int)
    paired = {
        "prediction_change_rate": float(np.mean(original_prediction != masked_prediction)),
        "accuracy_delta_masked_minus_original": float(
            masked_test_metrics["accuracy"] - original_test_metrics["accuracy"]
        ),
        "macro_f1_delta_masked_minus_original": float(
            masked_test_metrics["macro_f1"] - original_test_metrics["macro_f1"]
        ),
        "report_cluster_bootstrap_95": paired_report_bootstrap(
            target, original_prediction, masked_prediction,
            splits["test"]["report_id"].to_numpy(),
            int(config["bootstrap_replicates"]), int(config["seed"]) + 91,
        ),
    }

    masked_training = None
    masked_history = None
    masked_best_epoch = None
    if not args.skip_masked_training:
        del original_model
        torch.cuda.empty_cache()
        masked_model, masked_history, masked_best_epoch = train_model(
            splits["train"], splits["validation"], config["masked_text_column"],
            "masked", tokenizer, labels, label_to_id, config, args.cache_dir, device,
        )
        masked_trained_test, masked_training = evaluate_condition(
            masked_model, splits["test"], config["masked_text_column"],
            "masked_model_masked_test", tokenizer, labels, config, device,
        )
        masked_trained_test.to_csv(
            table_dir / "masked_model_masked_test_predictions.csv", index=False, encoding="utf-8-sig"
        )

    manifest = []
    checkpoint_root = ROOT / config["checkpoint_root"]
    for path in sorted(checkpoint_root.rglob("*")):
        if path.is_file():
            manifest.append({
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    environment = reproducibility_payload()
    (metrics_dir / "checkpoint_sha256_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metrics_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "version": config["version"],
        "model": config["model_name"],
        "split_sizes": observed_sizes,
        "labels": labels,
        "original_model": {
            "best_epoch": original_best_epoch,
            "validation_history": original_history,
            "calibration_original_input": original_calibration_metrics,
            "test_original_input": original_test_metrics,
            "test_masked_input": masked_test_metrics,
        },
        "paired_original_model_test_comparison": paired,
        "masked_model": None if args.skip_masked_training else {
            "best_epoch": masked_best_epoch,
            "validation_history": masked_history,
            "test_masked_input": masked_training,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "guardrails": [
            "Frozen report-grouped splits were reused without reassignment.",
            "Only validation macro-F1 selected an epoch.",
            "Calibration and retrospective test labels did not select weights or hyperparameters.",
            "The same pre-existing masked text column used by earlier baselines was reused.",
        ],
    }
    (metrics_dir / "sec_bert_masking_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# SEC-BERT lexical masking experiment v2", "",
        "## Original model", "",
        "| Evaluation input | Accuracy | Macro-F1 | Balanced accuracy |", "|---|---:|---:|---:|",
        f"| Original test | {original_test_metrics['accuracy']:.4f} | {original_test_metrics['macro_f1']:.4f} | {original_test_metrics['balanced_accuracy']:.4f} |",
        f"| Masked test | {masked_test_metrics['accuracy']:.4f} | {masked_test_metrics['macro_f1']:.4f} | {masked_test_metrics['balanced_accuracy']:.4f} |",
        "", f"Prediction change rate: {paired['prediction_change_rate']:.4f}.", "",
    ]
    if masked_training is not None:
        report_lines.extend([
            "## Masked-input training", "",
            "| Evaluation input | Accuracy | Macro-F1 | Balanced accuracy |", "|---|---:|---:|---:|",
            f"| Masked test | {masked_training['accuracy']:.4f} | {masked_training['macro_f1']:.4f} | {masked_training['balanced_accuracy']:.4f} |", "",
        ])
    (report_dir / "sec_bert_masking_v2_zh.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
