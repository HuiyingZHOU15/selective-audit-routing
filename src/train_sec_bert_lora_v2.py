"""Train and export a recoverable SEC-BERT LoRA sequence classifier.

The base model remains the public ``nlpaueb/sec-bert-base`` snapshot.  The
portable artifact contains only the LoRA adapter, classification head,
tokenizer metadata, resolved base-model revision, and evaluation outputs.
Validation macro-F1 selects the epoch; calibration and test labels never
affect training or epoch selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import peft
import sklearn
import torch
import transformers
from huggingface_hub import model_info
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/sec_bert_lora_v2.json"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.texts[index], "label": self.labels[index]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TextDataset(frame[text_column].tolist(), frame["label_id"].tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collator(tokenizer, max_length),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.inference_mode()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            batch_logits = model(**batch).logits
        logits.append(batch_logits.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def prediction_frame(
    frame: pd.DataFrame,
    logits: np.ndarray,
    labels_y: np.ndarray,
    labels: list[str],
    target: str,
) -> pd.DataFrame:
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    prediction_id = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    clipped = np.clip(probabilities, 1e-15, 1.0)
    output = frame[["sample_id", "report_id", "Firm Names", "Inspection Year", target, "context_quality"]].copy()
    output["prediction"] = np.asarray([labels[int(index)] for index in prediction_id])
    output["confidence"] = probabilities.max(axis=1)
    output["margin"] = ordered[:, -1] - ordered[:, -2]
    output["entropy"] = -np.sum(clipped * np.log(clipped), axis=1)
    output["correct"] = prediction_id == labels_y
    output["logits_json"] = [json.dumps(list(map(float, row)), separators=(",", ":")) for row in logits]
    output["probability_json"] = [
        json.dumps(dict(zip(labels, map(float, row))), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in probabilities
    ]
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_hash_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "sha256_manifest.json":
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    (root / "sha256_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed = int(config["seed"])
    set_seed(seed)

    frame = pd.read_csv(ROOT / config["dataset"], low_memory=False)
    splits = {name: clean_split(frame, name) for name in ("train", "validation", "calibration", "test")}
    observed_sizes = {name: int(len(value)) for name, value in splits.items()}
    if observed_sizes != config["expected_clean_split_sizes"]:
        raise ValueError(f"Unexpected clean split sizes: {observed_sizes}")
    labels = sorted(frame[config["target"]].dropna().astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    for split_frame in splits.values():
        split_frame[config["text_column"]] = split_frame[config["text_column"]].fillna("").astype(str)
        split_frame["label_id"] = split_frame[config["target"]].map(label_to_id).astype(int)

    epochs = 1 if args.smoke_test else int(config["epochs"])
    if args.smoke_test:
        splits["train"] = splits["train"].groupby("label_id", group_keys=False).head(2).reset_index(drop=True)
        splits["validation"] = splits["validation"].groupby("label_id", group_keys=False).head(1).reset_index(drop=True)
        splits["calibration"] = splits["calibration"].groupby("label_id", group_keys=False).head(1).reset_index(drop=True)
        splits["test"] = splits["test"].groupby("label_id", group_keys=False).head(1).reset_index(drop=True)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resolved_revision = model_info(config["model_name"], revision=config["model_revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], revision=resolved_revision, cache_dir=args.cache_dir
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        revision=resolved_revision,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=args.cache_dir,
    )
    if config["gradient_checkpointing"]:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        base_model.config.use_cache = False
    lora = config["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        modules_to_save=list(lora["modules_to_save"]),
        bias="none",
    )
    model = get_peft_model(base_model, lora_config).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"device": str(device), "resolved_revision": resolved_revision, "trainable": trainable, "total": total}), flush=True)

    loaders = {
        name: make_loader(
            split_frame,
            config["text_column"],
            tokenizer,
            int(config["max_length"]),
            int(config["batch_size"] if name == "train" else config["eval_batch_size"]),
            name == "train",
            seed,
        )
        for name, split_frame in splits.items()
    }
    counts = splits["train"]["label_id"].value_counts().reindex(range(len(labels)), fill_value=1).to_numpy(float)
    weights = torch.tensor(len(splits["train"]) / (len(labels) * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation"])
    updates_per_epoch = int(np.ceil(len(loaders["train"]) / accumulation))
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        max(1, int(total_updates * float(config["warmup_ratio"]))),
        total_updates,
    )

    best_epoch, best_f1, best_state = 0, -1.0, None
    history = []
    started = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["fp16"]))
    update = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            y = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and bool(config["fp16"])):
                logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y, weight=weights) / accumulation
            scaler.scale(loss).backward()
            running += float(loss.item()) * accumulation
            if step % accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 50 == 0 or update == total_updates:
                    peak_gb = (
                        torch.cuda.max_memory_allocated() / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    )
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "update": update,
                                "total_updates": total_updates,
                                "running_loss": running / step,
                                "peak_cuda_gb": peak_gb,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
        validation_logits, validation_y = predict(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": running / len(loaders["train"]), **metrics_from_logits(validation_logits, validation_y)}
        history.append(row)
        print(json.dumps({"validation": row}), flush=True)
        if row["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, row["macro_f1"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in get_peft_model_state_dict(model).items()})

    if best_state is None:
        raise RuntimeError("No best adapter state selected")
    set_peft_model_state_dict(model, best_state)
    model.to(device)
    adapter_dir = args.output_root / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "base_model_revision.json").write_text(
        json.dumps({"model_name": config["model_name"], "requested_revision": config["model_revision"], "resolved_revision": resolved_revision}, indent=2),
        encoding="utf-8",
    )

    tables_dir = args.output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    split_metrics = {}
    reference_logits = {}
    for name in ("validation", "calibration", "test"):
        logits, y = predict(model, loaders[name], device)
        reference_logits[name] = logits
        split_metrics[name] = metrics_from_logits(logits, y)
        prediction_frame(splits[name], logits, y, labels, config["target"]).to_csv(
            tables_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )

    del model, base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reload_base = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        revision=resolved_revision,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=args.cache_dir,
    )
    reloaded = peft.PeftModel.from_pretrained(reload_base, adapter_dir).to(device)
    reload_logits, _ = predict(reloaded, loaders["calibration"], device)
    max_reload_difference = float(np.max(np.abs(reference_logits["calibration"] - reload_logits)))
    if max_reload_difference > 1e-4:
        raise RuntimeError(f"Reloaded adapter logits differ: max_abs={max_reload_difference}")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    payload = {
        "version": config["version"],
        "smoke_test": args.smoke_test,
        "base_model": config["model_name"],
        "resolved_revision": resolved_revision,
        "split_sizes": observed_sizes,
        "labels": labels,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "best_epoch": best_epoch,
        "validation_history": history,
        "metrics": split_metrics,
        "reload_max_abs_logit_difference": max_reload_difference,
        "elapsed_seconds": float(time.perf_counter() - started),
        "guardrail": config["selection"],
    }
    (args.output_root / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_hash_manifest(args.output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
