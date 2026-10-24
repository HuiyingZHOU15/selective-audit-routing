"""Fine-tune pre-specified Transformer encoder baselines on PCAOB taxonomy v2.

The benchmark preserves the existing report-grouped split.  Epoch selection is
performed only on clean validation data.  The already-inspected test split is
reported as retrospective evidence and never used for tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/encoder_baseline_v1.json"


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
    torch.use_deterministic_algorithms(False)


def clean_mask(frame: pd.DataFrame, split: str) -> pd.Series:
    mask = frame["split"].eq(split)
    if split in {"validation", "test"}:
        mask &= frame["clean_eval"].fillna(False).astype(bool)
    return mask


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


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


@torch.inference_mode()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits = []
    labels = []
    for batch in loader:
        y = batch.pop("labels")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = model(**batch)
        logits.append(output.logits.cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels)


def report_bootstrap(
    frame: pd.DataFrame,
    new_prediction: np.ndarray,
    reference_prediction: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    truth = frame["target_level1"].to_numpy()
    reports = frame["report_id"].astype(str).unique()
    positions = {
        report: np.flatnonzero(frame["report_id"].astype(str).to_numpy() == report) for report in reports
    }
    differences = []
    for _ in range(iterations):
        sampled_reports = rng.choice(reports, size=len(reports), replace=True)
        index = np.concatenate([positions[report] for report in sampled_reports])
        differences.append(
            accuracy_score(truth[index], new_prediction[index])
            - accuracy_score(truth[index], reference_prediction[index])
        )
    return {
        "accuracy_difference_encoder_minus_tfidf": float(
            accuracy_score(truth, new_prediction) - accuracy_score(truth, reference_prediction)
        ),
        "lower_95": float(np.quantile(differences, 0.025)),
        "median": float(np.quantile(differences, 0.5)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "probability_encoder_better": float(np.mean(np.asarray(differences) > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=["general_xsmall", "sec_domain"])
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts/hf_cache"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["models"][args.model_key]
    seed = int(config["seed"])
    set_seed(seed)
    torch.set_num_threads(min(10, torch.get_num_threads()))

    frame = pd.read_csv(ROOT / config["dataset"])
    train_frame = frame.loc[clean_mask(frame, "train")].copy().reset_index(drop=True)
    validation_frame = frame.loc[clean_mask(frame, "validation")].copy().reset_index(drop=True)
    test_frame = frame.loc[clean_mask(frame, "test")].copy().reset_index(drop=True)
    labels = sorted(frame[config["target"]].dropna().astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    text_column = config["text_column"]
    for split_frame in [train_frame, validation_frame, test_frame]:
        split_frame[text_column] = split_frame[text_column].fillna("").astype(str)
        split_frame["label_id"] = split_frame[config["target"]].map(label_to_id).astype(int)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["model_name"],
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=cache_dir,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    resolved_device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(resolved_device)
    model.to(device)
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)

    batch_size = int(model_config["batch_size"])
    collator = make_collator(tokenizer, int(config["max_length"]))
    train_loader = DataLoader(
        TextDataset(train_frame[text_column].tolist(), train_frame["label_id"].tolist()),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TextDataset(validation_frame[text_column].tolist(), validation_frame["label_id"].tolist()),
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TextDataset(test_frame[text_column].tolist(), test_frame["label_id"].tolist()),
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    counts = train_frame["label_id"].value_counts().reindex(range(len(labels)), fill_value=1).to_numpy(float)
    class_weights = torch.tensor(
        len(train_frame) / (len(labels) * counts), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(model_config["gradient_accumulation"])
    updates_per_epoch = int(np.ceil(len(train_loader) / accumulation))
    total_updates = updates_per_epoch * int(config["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * float(config["warmup_ratio"]))),
        num_training_steps=total_updates,
    )

    best = None
    history = []
    started = time.perf_counter()
    global_update = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            y = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            # Some Hub checkpoints load their encoder weights as float16 even
            # on CPU.  Compute the weighted classification loss in float32 so
            # PyTorch receives matching dtypes and retains numerical stability.
            logits = model(**batch).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights) / accumulation
            loss.backward()
            running_loss += float(loss.item()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update == 1 or global_update % 50 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"model={args.model_key} epoch={epoch} update={global_update}/{total_updates} "
                        f"loss={running_loss / step:.4f} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
        validation_logits, validation_y = predict(model, validation_loader, device)
        validation_prediction = validation_logits.argmax(axis=1)
        scores = metrics(validation_y, validation_prediction)
        row = {"epoch": epoch, "train_loss": running_loss / len(train_loader), **scores}
        history.append(row)
        print(json.dumps({"validation": row}, ensure_ascii=False), flush=True)
        key = (scores["macro_f1"], scores["accuracy"], -epoch)
        if best is None or key > best[0]:
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            best = (key, epoch, best_state)

    assert best is not None
    model.load_state_dict(best[2])
    test_logits, test_y = predict(model, test_loader, device)
    test_prediction_id = test_logits.argmax(axis=1)
    test_prediction = np.asarray([id_to_label[int(index)] for index in test_prediction_id])
    test_scores = metrics(test_y, test_prediction_id)

    reference_path = ROOT / "results/tables/model_comparison_test_predictions_v2.csv"
    reference = pd.read_csv(reference_path)[["sample_id", "tfidf_context"]]
    aligned = test_frame[["sample_id", "report_id", config["target"]]].merge(
        reference, on="sample_id", how="left", validate="one_to_one"
    )
    if aligned["tfidf_context"].isna().any():
        raise ValueError("Could not align TF-IDF reference predictions")
    paired = report_bootstrap(
        aligned,
        test_prediction,
        aligned["tfidf_context"].to_numpy(),
        iterations=1000,
        seed=seed,
    )

    probabilities = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    prediction_table = test_frame[["sample_id", "report_id", config["target"], "context_quality"]].copy()
    prediction_table["prediction"] = test_prediction
    prediction_table["confidence"] = probabilities.max(axis=1)
    prediction_table["correct"] = prediction_table["prediction"].eq(prediction_table[config["target"]])

    output_stem = f"encoder_{args.model_key}_v1"
    table_path = ROOT / f"results/tables/{output_stem}_test_predictions.csv"
    metrics_path = ROOT / f"results/metrics/{output_stem}.json"
    report_path = ROOT / f"results/reports/{output_stem}_zh.md"
    for path in [table_path, metrics_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(table_path, index=False, encoding="utf-8-sig")
    payload = {
        "version": config["version"],
        "model_key": args.model_key,
        "model": model_config,
        "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "splits": {"train": len(train_frame), "validation_clean": len(validation_frame), "test_clean": len(test_frame)},
        "best_epoch": int(best[1]),
        "validation_history": history,
        "test": test_scores,
        "paired_vs_tfidf_context": paired,
        "elapsed_seconds": float(time.perf_counter() - started),
        "test_status": "retrospective_already_inspected_no_test_tuning",
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# PCAOB encoder baseline：{args.model_key}

- Model: `{model_config['model_name']}`
- Role: `{model_config['role']}`
- Training: full encoder fine-tuning on {device}; best epoch selected only by clean-validation macro-F1
- Best epoch: {best[1]}
- Retrospective clean test accuracy: {test_scores['accuracy']:.4f}
- Retrospective clean test macro-F1: {test_scores['macro_f1']:.4f}
- Retrospective clean test balanced accuracy: {test_scores['balanced_accuracy']:.4f}
- Accuracy difference versus TF-IDF/context: {paired['accuracy_difference_encoder_minus_tfidf']:+.4f}
- Report-cluster bootstrap 95% CI: [{paired['lower_95']:+.4f}, {paired['upper_95']:+.4f}]
- Probability encoder better in report bootstrap: {paired['probability_encoder_better']:.3f}

既有 test 已被查看，因此本結果只用於補齊 baseline 與決定是否值得建立昂貴的 encoder OOF predictions，不能重新包裝成確認性結果。Calibration split 未參與訓練、epoch selection 或本次比較。
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
