"""Retrain the frozen SEC-BERT recipe and export untouched calibration logits.

This is a small Kaggle GPU follow-up to the completed encoder experiment.  It
uses train for fitting, validation for epoch selection, and calibration only
for post-hoc probability calibration in the next stage.  Test is not loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sec_bert_calibration_export_v1.json"


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


def clean_mask(frame: pd.DataFrame, split: str) -> pd.Series:
    return frame["split"].eq(split) & frame["clean_eval"].fillna(False).astype(bool)


def make_collator(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        tokens = tokenizer(
            [item["text"] for item in batch], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        tokens["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return tokens
    return collate


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts/hf_cache"))
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this frozen SEC-BERT export")
    device = torch.device("cuda")
    set_seed(int(config["seed"]))
    print(f"device=cuda gpu={torch.cuda.get_device_name(0)}", flush=True)

    frame = pd.read_csv(ROOT / config["dataset"])
    train = frame.loc[clean_mask(frame, "train")].copy().reset_index(drop=True)
    validation = frame.loc[clean_mask(frame, "validation")].copy().reset_index(drop=True)
    calibration = frame.loc[clean_mask(frame, "calibration")].copy().reset_index(drop=True)
    if (len(train), len(validation), len(calibration)) != (3982, 927, 608):
        raise ValueError(f"Unexpected split sizes: {(len(train), len(validation), len(calibration))}")

    labels = sorted(frame[config["target"]].dropna().astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    for split_frame in (train, validation, calibration):
        split_frame[config["text_column"]] = split_frame[config["text_column"]].fillna("").astype(str)
        split_frame["label_id"] = split_frame[config["target"]].map(label_to_id).astype(int)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=len(labels), id2label=id_to_label,
        label2id=label_to_id, cache_dir=args.cache_dir,
    ).to(device)
    collator = make_collator(tokenizer, int(config["max_length"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        TextDataset(train[config["text_column"]].tolist(), train["label_id"].tolist()),
        batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collator,
        num_workers=0, pin_memory=True, generator=generator,
    )
    validation_loader = DataLoader(
        TextDataset(validation[config["text_column"]].tolist(), validation["label_id"].tolist()),
        batch_size=int(config["batch_size"]) * 2, shuffle=False, collate_fn=collator,
        num_workers=0, pin_memory=True,
    )
    calibration_loader = DataLoader(
        TextDataset(calibration[config["text_column"]].tolist(), calibration["label_id"].tolist()),
        batch_size=int(config["batch_size"]) * 2, shuffle=False, collate_fn=collator,
        num_workers=0, pin_memory=True,
    )

    counts = train["label_id"].value_counts().reindex(range(len(labels)), fill_value=1).to_numpy(float)
    weights = torch.tensor(len(train) / (len(labels) * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation"])
    updates_per_epoch = int(np.ceil(len(train_loader) / accumulation))
    total_updates = updates_per_epoch * int(config["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(total_updates * float(config["warmup_ratio"]))), total_updates,
    )

    best_epoch, best_f1, best_state = 0, -1.0, None
    history = []
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
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                update += 1
                if update == 1 or update % 100 == 0:
                    print(
                        f"epoch={epoch} update={update}/{total_updates} "
                        f"loss={running/step:.4f} elapsed_s={time.perf_counter()-started:.1f}", flush=True,
                    )
        validation_logits, validation_y = predict(model, validation_loader, device)
        validation_prediction = validation_logits.argmax(axis=1)
        row = {
            "epoch": epoch, "train_loss": running / len(train_loader),
            "accuracy": float(accuracy_score(validation_y, validation_prediction)),
            "macro_f1": float(f1_score(validation_y, validation_prediction, average="macro", zero_division=0)),
        }
        history.append(row)
        print(json.dumps({"validation": row}, ensure_ascii=False), flush=True)
        if row["macro_f1"] > best_f1:
            best_epoch, best_f1 = epoch, row["macro_f1"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("No best checkpoint selected")
    model.load_state_dict(best_state)
    model.to(device)
    calibration_logits, calibration_y = predict(model, calibration_loader, device)
    probabilities = torch.softmax(torch.tensor(calibration_logits), dim=1).numpy()
    prediction_id = probabilities.argmax(axis=1)
    prediction = np.asarray([id_to_label[int(index)] for index in prediction_id])
    sorted_probabilities = np.sort(probabilities, axis=1)
    clipped = np.clip(probabilities, 1e-15, 1.0)

    output = calibration[["sample_id", "report_id", config["target"], "context_quality"]].copy()
    output["prediction"] = prediction
    output["confidence"] = probabilities.max(axis=1)
    output["margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    output["entropy"] = -np.sum(clipped * np.log(clipped), axis=1)
    output["correct"] = prediction_id == calibration_y
    output["logits_json"] = [json.dumps(list(map(float, row)), separators=(",", ":")) for row in calibration_logits]
    output["probability_json"] = [
        json.dumps(dict(zip(labels, map(float, row))), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in probabilities
    ]

    table_dir = ROOT / "results/tables"
    metrics_dir = ROOT / "results/metrics"
    checkpoint_dir = ROOT / config["checkpoint_directory"]
    for directory in (table_dir, metrics_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    output.to_csv(table_dir / "sec_bert_calibration_logits_v1.csv", index=False, encoding="utf-8-sig")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    payload = {
        "version": config["version"], "model": config["model_name"],
        "best_epoch": best_epoch, "validation_history": history,
        "calibration": {
            "samples": int(len(output)),
            "accuracy_before_calibration": float(accuracy_score(calibration_y, prediction_id)),
            "macro_f1_before_calibration": float(f1_score(
                calibration_y, prediction_id, average="macro", zero_division=0
            )),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "guardrail": "Calibration labels were not used for fitting or epoch selection.",
    }
    (metrics_dir / "sec_bert_calibration_export_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
