"""Calibrate a saved canonical SEC-BERT run and evaluate frozen policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from calibrate_sec_bert_temperature_v1 import (
    expected_calibration_error,
    multiclass_brier,
    negative_log_likelihood,
    probability_features,
    report_cluster_bootstrap as calibration_cluster_bootstrap,
    selective_metrics as calibration_selective_metrics,
    softmax,
    threshold_for_budget,
)
from evaluate_sec_bert_frozen_test_v1 import (
    calibration_metrics as test_calibration_metrics,
    cluster_bootstrap as test_cluster_bootstrap,
    selective_metrics as test_selective_metrics,
)
from train_sec_bert_lora_v2 import write_hash_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def load_predictions(path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    labels = sorted(frame["target_level1"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    y = frame["target_level1"].map(label_to_id).to_numpy(int)
    logits = np.asarray([json.loads(value) for value in frame["logits_json"]], dtype=float)
    if logits.shape != (len(frame), len(labels)):
        raise ValueError(f"Unexpected logits shape in {path}: {logits.shape}")
    if not np.array_equal(logits.argmax(axis=1), frame["prediction"].map(label_to_id).to_numpy(int)):
        raise ValueError(f"Logits do not reproduce predictions in {path}")
    return frame, labels, y, logits


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    calibration, labels, calibration_y, calibration_logits = load_predictions(
        run_root / "tables/calibration_predictions.csv"
    )
    test, test_labels, test_y, test_logits = load_predictions(run_root / "tables/test_predictions.csv")
    if labels != test_labels or len(calibration) != 608 or len(test) != 890:
        raise ValueError("Unexpected labels or split sizes")

    optimization = minimize_scalar(
        lambda log_t: negative_log_likelihood(calibration_logits, calibration_y, float(np.exp(log_t))),
        bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-10},
    )
    temperature = float(np.exp(optimization.x))
    raw_calibration_probabilities = softmax(calibration_logits)
    scaled_calibration_probabilities = softmax(calibration_logits / temperature)
    raw_confidence, raw_margin, raw_entropy = probability_features(raw_calibration_probabilities)
    scaled_confidence, scaled_margin, scaled_entropy = probability_features(scaled_calibration_probabilities)
    calibration["raw_confidence_recomputed"] = raw_confidence
    calibration["raw_margin_recomputed"] = raw_margin
    calibration["raw_entropy_recomputed"] = raw_entropy
    calibration["calibrated_confidence"] = scaled_confidence
    calibration["calibrated_margin"] = scaled_margin
    calibration["calibrated_entropy"] = scaled_entropy

    calibration_metrics = {}
    for name, probabilities, current_temperature in (
        ("raw", raw_calibration_probabilities, 1.0),
        ("temperature_scaled", scaled_calibration_probabilities, temperature),
    ):
        current_ece, _ = expected_calibration_error(probabilities, calibration_y, 10)
        prediction = probabilities.argmax(axis=1)
        calibration_metrics[name] = {
            "temperature": current_temperature,
            "accuracy": float(accuracy_score(calibration_y, prediction)),
            "macro_f1": float(f1_score(calibration_y, prediction, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(calibration_y, prediction)),
            "nll": negative_log_likelihood(calibration_logits, calibration_y, current_temperature),
            "multiclass_brier": multiclass_brier(probabilities, calibration_y),
            "ece_10_bins": current_ece,
        }

    freeze = {
        "version": "sec-bert-full-v2-calibration-freeze-20261022",
        "model_run": run_root.name,
        "temperature": temperature,
        "score": "calibrated_top1_minus_top2_probability",
        "operator": "refer_if_score_below",
        "operating_points": {},
    }
    calibration_operating_points = {}
    for name, budget in (("high_coverage", 0.05), ("conservative", 0.10)):
        threshold, intended = threshold_for_budget(scaled_margin, budget)
        freeze["operating_points"][name] = {
            "threshold": threshold,
            "requested_referral_rate": budget,
            "calibration_referrals": intended,
        }
        metrics = calibration_selective_metrics(calibration, scaled_margin, threshold, labels)
        metrics["report_cluster_bootstrap_95"] = calibration_cluster_bootstrap(
            calibration, scaled_margin, threshold,
            {"random_seed": args.seed, "bootstrap_replicates": args.bootstrap_replicates},
        )
        calibration_operating_points[name] = metrics

    raw_test_probabilities = softmax(test_logits)
    scaled_test_probabilities = softmax(test_logits / temperature)
    ordered_test = np.sort(scaled_test_probabilities, axis=1)
    test_margin = ordered_test[:, -1] - ordered_test[:, -2]
    test["calibrated_confidence"] = ordered_test[:, -1]
    test["calibrated_margin"] = test_margin
    test_probability_metrics = {
        "raw": test_calibration_metrics(test_logits, raw_test_probabilities, test_y, 1.0),
        "temperature_scaled": test_calibration_metrics(test_logits, scaled_test_probabilities, test_y, temperature),
    }
    test_operating_points = {}
    for index, (name, specification) in enumerate(freeze["operating_points"].items(), start=1):
        threshold = float(specification["threshold"])
        referred = test_margin < threshold
        metrics = test_selective_metrics(test, referred, labels)
        metrics["threshold"] = threshold
        metrics["report_cluster_bootstrap_95"] = test_cluster_bootstrap(
            test, referred, replicates=args.bootstrap_replicates, seed=args.seed + index,
        )
        test_operating_points[name] = metrics

    downstream = run_root / "downstream"
    downstream.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(downstream / "calibration_scaled.csv", index=False, encoding="utf-8-sig")
    test.to_csv(downstream / "test_scored.csv", index=False, encoding="utf-8-sig")
    (downstream / "operating_points.freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = {
        "version": "sec-bert-full-v2-downstream-20261022",
        "temperature_optimization_success": bool(optimization.success),
        "temperature": temperature,
        "calibration_probability_metrics": calibration_metrics,
        "calibration_operating_points": calibration_operating_points,
        "retrospective_test_probability_metrics": test_probability_metrics,
        "retrospective_test_operating_points": test_operating_points,
        "guardrail": (
            "Temperature and thresholds use only calibration labels. The retrospective test remains "
            "previously inspected and is not a never-observed confirmatory holdout."
        ),
    }
    (downstream / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_hash_manifest(run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
