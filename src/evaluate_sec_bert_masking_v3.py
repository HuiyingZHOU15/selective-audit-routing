"""Evaluate the canonical and masked-trained SEC-BERT checkpoints on paired inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_sec_bert_lora_v2 import clean_split, make_loader, predict, set_seed


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-model", type=Path, required=True)
    parser.add_argument("--masked-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def evaluate(model, loader, frame: pd.DataFrame, labels: list[str], condition: str, device):
    logits, y = predict(model, loader, device)
    probability = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probability.argmax(axis=1)
    ordered = np.sort(probability, axis=1)
    output = frame[["sample_id", "report_id", "target_level1", "context_quality"]].copy()
    output["condition"] = condition
    output["prediction"] = [labels[int(index)] for index in pred]
    output["correct"] = pred == y
    output["confidence"] = probability.max(axis=1)
    output["margin"] = ordered[:, -1] - ordered[:, -2]
    clipped = np.clip(probability, 1e-15, 1.0)
    output["entropy"] = -np.sum(clipped * np.log(clipped), axis=1)
    output["logits_json"] = [json.dumps(row.tolist(), separators=(",", ":")) for row in logits]
    return output, y, pred, {"samples": int(len(y)), **scores(y, pred)}


def paired_bootstrap(y, original_pred, masked_pred, report_ids, replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    codes, reports = pd.factorize(pd.Series(report_ids).astype(str), sort=False)
    groups = [np.flatnonzero(codes == code) for code in range(len(reports))]
    accuracy_delta, macro_delta = [], []
    for _ in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[value] for value in sampled])
        accuracy_delta.append(accuracy_score(y[index], masked_pred[index]) - accuracy_score(y[index], original_pred[index]))
        macro_delta.append(
            f1_score(y[index], masked_pred[index], average="macro", zero_division=0)
            - f1_score(y[index], original_pred[index], average="macro", zero_division=0)
        )
    return {
        "accuracy_delta_masked_minus_original": [float(x) for x in np.quantile(accuracy_delta, [0.025, 0.5, 0.975])],
        "macro_f1_delta_masked_minus_original": [float(x) for x in np.quantile(macro_delta, [0.025, 0.5, 0.975])],
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed = 20261002
    set_seed(seed)
    frame = pd.read_csv(ROOT / "data/processed/model_dataset_v2.csv", low_memory=False)
    test = clean_split(frame, "test")
    labels = sorted(frame["target_level1"].dropna().astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    test["label_id"] = test["target_level1"].map(label_to_id).astype(int)
    for column in ("text_context_v2", "text_context_masked_v2"):
        test[column] = test[column].fillna("").astype(str)

    tokenizer = AutoTokenizer.from_pretrained(args.canonical_model)
    original_loader = make_loader(test, "text_context_v2", tokenizer, 256, 8, False, seed)
    masked_loader = make_loader(test, "text_context_masked_v2", tokenizer, 256, 8, False, seed)
    canonical = AutoModelForSequenceClassification.from_pretrained(args.canonical_model).to(device)
    original_table, y, original_pred, original_metrics = evaluate(
        canonical, original_loader, test, labels, "canonical_original_input", device
    )
    masked_table, masked_y, masked_pred, masked_metrics = evaluate(
        canonical, masked_loader, test, labels, "canonical_masked_input", device
    )
    if not np.array_equal(y, masked_y):
        raise RuntimeError("Paired test labels are misaligned")
    del canonical
    if device.type == "cuda":
        torch.cuda.empty_cache()
    masked_model = AutoModelForSequenceClassification.from_pretrained(args.masked_model).to(device)
    masked_trained_table, trained_y, trained_pred, trained_metrics = evaluate(
        masked_model, masked_loader, test, labels, "masked_trained_masked_input", device
    )
    if not np.array_equal(y, trained_y):
        raise RuntimeError("Masked-trained test labels are misaligned")

    args.output_root.mkdir(parents=True, exist_ok=True)
    original_table.to_csv(args.output_root / "canonical_original_test_predictions.csv", index=False, encoding="utf-8-sig")
    masked_table.to_csv(args.output_root / "canonical_masked_test_predictions.csv", index=False, encoding="utf-8-sig")
    masked_trained_table.to_csv(args.output_root / "masked_trained_masked_test_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "version": "sec-bert-masking-v3-canonical-20261022",
        "canonical_checkpoint": str(args.canonical_model.resolve()),
        "masked_checkpoint": str(args.masked_model.resolve()),
        "canonical_original_input": original_metrics,
        "canonical_masked_input": masked_metrics,
        "masked_trained_masked_input": trained_metrics,
        "paired_canonical_comparison": {
            "prediction_change_rate": float(np.mean(original_pred != masked_pred)),
            "accuracy_delta_masked_minus_original": float(masked_metrics["accuracy"] - original_metrics["accuracy"]),
            "macro_f1_delta_masked_minus_original": float(masked_metrics["macro_f1"] - original_metrics["macro_f1"]),
            "report_cluster_bootstrap_95": paired_bootstrap(
                y, original_pred, masked_pred, test["report_id"].to_numpy(), args.bootstrap_replicates, seed + 91
            ),
        },
        "guardrail": "Canonical weights were frozen before masked evaluation; masked-model epoch was selected only on validation macro-F1."
    }
    (args.output_root / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
