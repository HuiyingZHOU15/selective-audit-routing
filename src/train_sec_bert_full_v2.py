"""Memory-efficient full SEC-BERT fine-tuning with a recoverable checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from huggingface_hub import model_info
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from train_sec_bert_lora_v2 import (
    ROOT,
    clean_split,
    make_loader,
    metrics_from_logits,
    predict,
    prediction_frame,
    set_seed,
    write_hash_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sec_bert_full_v2.json")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
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
        for name, rows_per_class in (("train", 2), ("validation", 1), ("calibration", 1), ("test", 1)):
            splits[name] = splits[name].groupby("label_id", group_keys=False).head(rows_per_class).reset_index(drop=True)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resolved_revision = model_info(config["model_name"], revision=config["model_revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=resolved_revision, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], revision=resolved_revision, num_labels=len(labels),
        id2label=id_to_label, label2id=label_to_id, cache_dir=args.cache_dir,
    )
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"device": str(device), "resolved_revision": resolved_revision, "trainable": total_parameters}), flush=True)

    loaders = {
        name: make_loader(
            split_frame, config["text_column"], tokenizer, int(config["max_length"]),
            int(config["batch_size"] if name == "train" else config["eval_batch_size"]),
            name == "train", seed,
        )
        for name, split_frame in splits.items()
    }
    counts = splits["train"]["label_id"].value_counts().reindex(range(len(labels)), fill_value=1).to_numpy(float)
    weights = torch.tensor(len(splits["train"]) / (len(labels) * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    accumulation = int(config["gradient_accumulation"])
    updates_per_epoch = int(np.ceil(len(loaders["train"]) / accumulation))
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_updates * float(config["warmup_ratio"]))), total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["fp16"]))
    started = time.perf_counter()
    history, best_epoch, best_f1, update = [], 0, -1.0, 0
    best_dir = args.output_root / "model"

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
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 50 == 0 or update == total_updates:
                    print(json.dumps({
                        "epoch": epoch, "update": update, "total_updates": total_updates,
                        "running_loss": running / step,
                        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
                        "elapsed_seconds": time.perf_counter() - started,
                    }), flush=True)
        validation_logits, validation_y = predict(model, loaders["validation"], device)
        row = {"epoch": epoch, "train_loss": running / len(loaders["train"]), **metrics_from_logits(validation_logits, validation_y)}
        history.append(row)
        print(json.dumps({"validation": row}), flush=True)
        if row["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, row["macro_f1"]
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "base_model_revision.json").write_text(json.dumps({
                "model_name": config["model_name"], "resolved_revision": resolved_revision,
            }, indent=2), encoding="utf-8")

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    tables_dir = args.output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    split_metrics, reference_calibration = {}, None
    for name in ("validation", "calibration", "test"):
        logits, y = predict(model, loaders[name], device)
        if name == "calibration":
            reference_calibration = logits
        split_metrics[name] = metrics_from_logits(logits, y)
        prediction_frame(splits[name], logits, y, labels, config["target"]).to_csv(
            tables_dir / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    reload_logits, _ = predict(reloaded, loaders["calibration"], device)
    max_reload_difference = float(np.max(np.abs(reference_calibration - reload_logits)))
    if max_reload_difference > 1e-4:
        raise RuntimeError(f"Reloaded full model logits differ: max_abs={max_reload_difference}")

    payload = {
        "version": config["version"], "smoke_test": args.smoke_test,
        "base_model": config["model_name"], "resolved_revision": resolved_revision,
        "split_sizes": observed_sizes, "labels": labels, "trainable_parameters": total_parameters,
        "best_epoch": best_epoch, "validation_history": history, "metrics": split_metrics,
        "reload_max_abs_logit_difference": max_reload_difference,
        "elapsed_seconds": float(time.perf_counter() - started), "guardrail": config["selection"],
    }
    environment = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (args.output_root / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_hash_manifest(args.output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
