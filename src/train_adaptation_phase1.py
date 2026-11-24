"""Unified train/validation-only trainer for Phase 3--5 adaptation runs.

This module deliberately has no calibration or test loader. Retrospective test
scoring is isolated in ``evaluate_adaptation_phase1.py``.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from adaptation_phase1_common import (
    EarlyStopping,
    balanced_class_weights,
    canonical_json_sha256,
    checkpoint_size_bytes,
    classification_metrics,
    environment_payload,
    label_order,
    load_development_table,
    report_cluster_bootstrap,
    set_reproducibility,
    sha256_file,
    split_development,
    write_hash_manifest,
)
from audit_lora_modules_phase1 import candidate_specs, module_resolution


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"text": self.texts[index], "label": self.labels[index]}


def collator(tokenizer, max_length: int):
    def apply(batch):
        tokens = tokenizer(
            [row["text"] for row in batch], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        tokens["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        return tokens
    return apply


def loader(frame, tokenizer, text_column, max_length, batch_size, shuffle, seed):
    return DataLoader(
        TextDataset(frame[text_column].tolist(), frame["label_id"].tolist()),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collator(tokenizer, max_length),
        num_workers=0, pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


@torch.inference_mode()
def predict(model, data_loader, device):
    model.eval()
    logits, labels = [], []
    for batch in data_loader:
        y = batch.pop("labels")
        inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(**inputs).logits
        logits.append(output.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def selected_candidate(candidate_id: str) -> dict:
    matches = [item for item in candidate_specs() if item["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or unresolved LoRA candidate: {candidate_id}")
    return matches[0]


def attach_masked_text(frame: pd.DataFrame, mask_table: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    if expected_sha256 and sha256_file(mask_table) != expected_sha256:
        raise ValueError("Masked text table hash mismatch")
    masked = pd.read_csv(mask_table)
    required = {"sample_id", "text_audit_area_masked_phase1", "audit_area_cue_present_phase1"}
    if not required <= set(masked):
        raise ValueError(f"Masked text table missing: {sorted(required - set(masked))}")
    result = frame.merge(masked[list(required)], on="sample_id", how="left", validate="one_to_one")
    if result["text_audit_area_masked_phase1"].isna().any():
        raise ValueError("Masked text is missing for development samples")
    return result


def save_model(model, tokenizer, checkpoint: Path, is_lora: bool) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    (checkpoint / "checkpoint_type.json").write_text(
        json.dumps({"type": "peft_adapter" if is_lora else "full_model"}, indent=2), encoding="utf-8"
    )


def load_checkpoint(config: dict, checkpoint: Path, device: torch.device, num_labels: int):
    method = config["method"]
    if method == "sec_lora":
        base = AutoModelForSequenceClassification.from_pretrained(
            config["model_id"], revision=config["model_revision"], num_labels=num_labels,
            cache_dir=config["cache_dir"], local_files_only=bool(config.get("local_files_only", False)),
        )
        return PeftModel.from_pretrained(base, checkpoint).to(device)
    return AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).to(device)


def train(config: dict, output_root: Path) -> dict:
    if config["stage"] not in {"smoke", "adapter_pilot", "lr_pilot", "official"}:
        raise ValueError("Unsupported training stage")
    if config.get("evaluate_test") or config.get("load_calibration"):
        raise RuntimeError("Test embargo violation in training config")
    seed_info = set_reproducibility(int(config["seed"]), deterministic=True)
    development_path = Path(config["development_table"])
    frame = load_development_table(
        development_path,
        expected_sha256=config.get("development_sha256"),
        expected_counts=config.get("expected_development_counts"),
    )
    if config["text_condition"] == "masked":
        frame = attach_masked_text(frame, Path(config["masked_text_table"]), config.get("masked_text_sha256"))
        text_column = "text_audit_area_masked_phase1"
    elif config["text_condition"] == "original":
        text_column = "text_original"
    else:
        raise ValueError("Unknown text condition")
    train_frame, validation_frame = split_development(frame)
    labels = label_order(train_frame, config.get("target", "target_level1"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    for part in (train_frame, validation_frame):
        part[text_column] = part[text_column].fillna("").astype(str)
        part["label_id"] = part[config.get("target", "target_level1")].map(label_to_id).astype(int)

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["tokenizer_revision"], cache_dir=config["cache_dir"],
        local_files_only=bool(config.get("local_files_only", False)),
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"], revision=config["model_revision"], num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)}, label2id=label_to_id,
        cache_dir=config["cache_dir"], local_files_only=bool(config.get("local_files_only", False)),
    )
    method = config["method"]
    is_lora = method == "sec_lora"
    matched_modules = None
    if is_lora:
        spec = selected_candidate(config["adapter_candidate_id"])
        resolved = module_resolution(base)["candidates"][spec["candidate_id"]]
        matched_modules = resolved["matched_modules"]
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.SEQ_CLS, r=spec["rank"], lora_alpha=spec["alpha"],
                lora_dropout=0.05, target_modules=matched_modules,
                modules_to_save=["classifier"], bias="none",
            ),
        )
    elif method in {"sec_full", "bert_full"}:
        model = base
        if not all(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Full fine-tuning model contains frozen parameters")
    else:
        raise ValueError(f"Unknown method: {method}")
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if is_lora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    model.to(device)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if is_lora:
        if not any("lora_" in name for name in trainable_names):
            raise RuntimeError("No trainable LoRA parameters")
        if not any("classifier" in name for name in trainable_names):
            raise RuntimeError("Classification head is not trainable")

    batch_size = int(config.get("physical_batch_size", 2))
    eval_batch_size = int(config.get("eval_batch_size", 8))
    accumulation = int(config.get("gradient_accumulation", 8))
    if batch_size * accumulation != int(config.get("effective_batch_size", 16)):
        raise ValueError("Physical batch and accumulation do not match effective batch size")
    train_loader = loader(train_frame, tokenizer, text_column, int(config.get("max_length", 256)), batch_size, True, int(config["seed"]))
    validation_loader = loader(validation_frame, tokenizer, text_column, int(config.get("max_length", 256)), eval_batch_size, False, int(config["seed"]))
    if config["loss"] == "weighted":
        weight_values = balanced_class_weights(train_frame["label_id"], len(labels))
    elif config["loss"] == "unweighted":
        weight_values = np.ones(len(labels), dtype=float)
    else:
        raise ValueError("Unknown loss mode")
    weights = torch.tensor(weight_values, dtype=torch.float32, device=device)
    optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    if {id(p) for p in optimizer_parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("Optimizer does not exactly cover trainable parameters")
    max_epochs = int(config.get("max_epochs", 6))
    updates_per_epoch = int(np.ceil(len(train_loader) / accumulation))
    total_updates = max_epochs * updates_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_updates * float(config.get("warmup_ratio", 0.1)))), total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stopper = EarlyStopping(
        patience=int(config.get("early_stopping_patience_epochs", 2)),
        min_delta=float(config.get("early_stopping_min_delta", 1e-4)),
    )
    checkpoint = output_root / "checkpoint"
    history, first_gradient_asserted = [], False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            y = batch.pop("labels").to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            group_start = ((step - 1) // accumulation) * accumulation + 1
            group_end = min(group_start + accumulation - 1, len(train_loader))
            group_size = group_end - group_start + 1
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**inputs).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, y, weight=weights) / group_size
            scaler.scale(loss).backward(); running += float(loss.item()) * group_size
            if not first_gradient_asserted:
                if not any(parameter.grad is not None for parameter in optimizer_parameters):
                    raise RuntimeError("No gradients observed")
                first_gradient_asserted = True
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(optimizer_parameters, float(config.get("max_gradient_norm", 1.0)))
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        validation_logits, validation_y = predict(model, validation_loader, device)
        metrics = classification_metrics(validation_y, validation_logits.argmax(axis=1), list(range(len(labels))))
        row = {"epoch": epoch, "train_loss": running / len(train_loader), **{k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")}}
        history.append(row)
        new_best, should_stop = stopper.update(epoch, row["macro_f1"])
        if new_best:
            save_model(model, tokenizer, checkpoint, is_lora)
        if should_stop:
            stop_reason = "early_stopping"
            break
    training_seconds = time.perf_counter() - training_started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    del model, base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = load_checkpoint(config, checkpoint, device, len(labels))
    logits, y_true = predict(reloaded, validation_loader, device)
    y_pred = logits.argmax(axis=1)
    validation_metrics = classification_metrics(y_true, y_pred, list(range(len(labels))))
    validation_frame = validation_frame.copy()
    validation_frame["label_id"] = y_true
    validation_frame["prediction_id"] = y_pred
    validation_frame["logits_json"] = [json.dumps(list(map(float, row)), separators=(",", ":")) for row in logits]
    output_root.mkdir(parents=True, exist_ok=True)
    validation_frame[["sample_id", "report_id", "label_id", "prediction_id", "logits_json"]].to_csv(
        output_root / "validation_predictions.csv", index=False, encoding="utf-8"
    )
    payload = {
        "run_id": config["run_id"], "stage": config["stage"], "method": method,
        "test_scored": False, "calibration_loaded": False,
        "config_sha256": canonical_json_sha256(config),
        "development_table": development_path.as_posix(),
        "development_sha256": sha256_file(development_path),
        "source_sha256": sha256_file(Path(__file__)),
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "text_condition": config["text_condition"], "loss": config["loss"],
        "seed": config["seed"], "seed_state": seed_info,
        "adapter_candidate_id": config.get("adapter_candidate_id"),
        "matched_lora_modules": matched_modules,
        "class_weights": weight_values.tolist(),
        "total_parameters": total_parameter_count,
        "trainable_parameters": trainable_parameter_count,
        "trainable_parameter_names": trainable_names if is_lora else None,
        "best_epoch": stopper.best_epoch, "stop_reason": stop_reason,
        "validation_history": history, "validation_metrics": validation_metrics,
        "validation_report_bootstrap": report_cluster_bootstrap(
            validation_frame, y_pred, list(range(len(labels))),
            int(config.get("smoke_bootstrap_iterations", 50)), int(config["seed"]),
        ),
        "training_time_seconds": training_seconds,
        "peak_cuda_memory_bytes": int(peak_memory),
        "checkpoint_size_bytes": checkpoint_size_bytes(checkpoint),
        "physical_batch_size": batch_size, "effective_batch_size": batch_size * accumulation,
        "environment": environment_payload(),
    }
    (output_root / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_hash_manifest(output_root)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(train(config, args.output_root), ensure_ascii=False, indent=2))
    except Exception as exc:
        failure = {"run_id": config.get("run_id"), "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (args.output_root / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()


