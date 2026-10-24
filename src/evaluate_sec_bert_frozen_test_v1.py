"""Evaluate frozen SEC-BERT selective-classification operating points on test.

This script reads the immutable operating-point file selected on development
and calibration.  It never chooses a temperature, score, threshold, or target
coverage from retrospective test labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import beta
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/sec_bert_operating_points_v1.freeze.json"
TEST_PATH = ROOT / "results/tables/sec_bert_retrospective_test_logits_v1.csv"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def ece(probabilities: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return result


def calibration_metrics(logits: np.ndarray, probabilities: np.ndarray, y: np.ndarray, temperature: float) -> dict:
    prediction = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[y]
    scaled = logits / temperature
    return {
        "temperature": temperature,
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "nll": float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bins": ece(probabilities, y),
    }


def upper_binomial_risk(errors: int, accepted: int, confidence: float = 0.95) -> float:
    if accepted <= 0:
        return float("nan")
    if errors >= accepted:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, accepted - errors))


def selective_metrics(frame: pd.DataFrame, referred: np.ndarray, labels: list[str]) -> dict:
    accepted = ~referred
    errors = ~frame["prediction"].eq(frame["target_level1"]).to_numpy()
    accepted_errors = int(errors[accepted].sum())
    referred_errors = int(errors[referred].sum())
    return {
        "samples": int(len(frame)),
        "referred": int(referred.sum()),
        "referral_rate": float(referred.mean()),
        "coverage": float(accepted.mean()),
        "selective_accuracy": float(1.0 - accepted_errors / accepted.sum()),
        "selective_risk": float(accepted_errors / accepted.sum()),
        "sample_level_binomial_risk_upper_95": upper_binomial_risk(accepted_errors, int(accepted.sum())),
        "accepted_macro_f1": float(f1_score(
            frame.loc[accepted, "target_level1"], frame.loc[accepted, "prediction"],
            labels=labels, average="macro", zero_division=0,
        )),
        "errors_remaining": accepted_errors,
        "errors_captured": referred_errors,
        "error_capture_rate": float(referred_errors / errors.sum()),
        "referral_precision": float(referred_errors / referred.sum()) if referred.sum() else float("nan"),
        "oracle_assisted_accuracy_upper_bound": float(1.0 - accepted_errors / len(frame)),
    }


def cluster_bootstrap(
    frame: pd.DataFrame, referred: np.ndarray, replicates: int = 5000, seed: int = 20261013
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    report_codes, reports = pd.factorize(frame["report_id"].astype(str), sort=False)
    groups = [np.flatnonzero(report_codes == code) for code in range(len(reports))]
    errors = ~frame["prediction"].eq(frame["target_level1"]).to_numpy()
    values = {"coverage": [], "selective_accuracy": [], "error_capture_rate": [], "referral_precision": []}
    for _ in range(replicates):
        sampled_codes = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[code] for code in sampled_codes])
        selected = referred[indices]
        accepted = ~selected
        sampled_errors = errors[indices]
        referred_errors = int(sampled_errors[selected].sum())
        total_errors = int(sampled_errors.sum())
        values["coverage"].append(float(accepted.mean()))
        values["selective_accuracy"].append(float(1.0 - sampled_errors[accepted].mean()))
        values["error_capture_rate"].append(float(referred_errors / total_errors) if total_errors else np.nan)
        values["referral_precision"].append(float(referred_errors / selected.sum()) if selected.sum() else np.nan)
    return {
        key: [float(x) for x in np.nanquantile(series, [0.025, 0.5, 0.975])]
        for key, series in values.items()
    }


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(TEST_PATH)
    if len(frame) != 890 or frame["sample_id"].nunique() != 890:
        raise ValueError("Expected exactly 890 unique retrospective test samples")
    labels = sorted(frame["target_level1"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    y = frame["target_level1"].map(label_to_id).to_numpy(int)
    logits = np.asarray([json.loads(value) for value in frame["logits_json"]], dtype=float)
    if logits.shape != (len(frame), len(labels)):
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    prediction_id = frame["prediction"].map(label_to_id).to_numpy(int)
    if not np.array_equal(logits.argmax(axis=1), prediction_id):
        raise ValueError("Logits do not reproduce exported predictions")

    temperature = float(freeze["temperature"])
    raw_probabilities = softmax(logits)
    calibrated_probabilities = softmax(logits / temperature)
    ordered = np.sort(calibrated_probabilities, axis=1)
    calibrated_margin = ordered[:, -1] - ordered[:, -2]
    calibrated_confidence = ordered[:, -1]
    frame["calibrated_margin"] = calibrated_margin
    frame["calibrated_confidence"] = calibrated_confidence

    probability_evaluation = {
        "raw": calibration_metrics(logits, raw_probabilities, y, 1.0),
        "temperature_scaled": calibration_metrics(logits, calibrated_probabilities, y, temperature),
    }
    operating_rows = []
    bootstraps = {}
    class_rows = []
    for name, specification in freeze["operating_points"].items():
        threshold = float(specification["threshold"])
        referred = calibrated_margin < threshold
        metrics = selective_metrics(frame, referred, labels)
        metrics.update({"operating_point": name, "threshold": threshold})
        operating_rows.append(metrics)
        bootstraps[name] = cluster_bootstrap(frame, referred, seed=20261013 + len(operating_rows))
        temporary = frame.assign(referred=referred)
        for label, group in temporary.groupby("prediction", sort=True):
            class_rows.append({
                "operating_point": name, "predicted_class": label, "samples": int(len(group)),
                "referral_rate": float(group["referred"].mean()),
                "error_rate": float((~group["correct"].astype(bool)).mean()),
            })

    results = pd.DataFrame(operating_rows)
    table_dir = ROOT / "results/tables"
    metrics_dir = ROOT / "results/metrics"
    report_dir = ROOT / "results/reports"
    figure_dir = ROOT / "results/figures"
    for directory in (table_dir, metrics_dir, report_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_dir / "sec_bert_frozen_test_scored_v1.csv", index=False, encoding="utf-8-sig")
    results.to_csv(table_dir / "sec_bert_frozen_test_operating_points_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(class_rows).to_csv(
        table_dir / "sec_bert_frozen_test_class_referral_v1.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(6.4, 4.5))
    x = np.arange(len(results))
    plt.bar(x - 0.18, results["coverage"], width=0.36, label="coverage")
    plt.bar(x + 0.18, results["selective_accuracy"], width=0.36, label="selective accuracy")
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, results["operating_point"].str.replace("_", " "))
    plt.ylim(0.80, 1.00)
    plt.ylabel("Proportion")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "sec_bert_frozen_test_operating_points_v1.png", dpi=220)
    plt.close()

    payload = {
        "version": "sec-bert-frozen-retrospective-test-v1-20261013",
        "status": "retrospective_already_inspected_no_test_tuning",
        "samples": int(len(frame)), "reports": int(frame["report_id"].nunique()),
        "freeze_file": str(FREEZE_PATH.relative_to(ROOT)),
        "temperature": temperature,
        "probability_metrics": probability_evaluation,
        "operating_points": operating_rows,
        "report_cluster_bootstrap_95": bootstraps,
        "protocol_guardrail": (
            "Temperature, score, and thresholds were frozen before this file was read. "
            "No retrospective test label changed any decision rule."
        ),
        "claim_guardrail": (
            "This test has been inspected in prior experiments and remains retrospective. "
            "Confirmatory claims require a new temporal holdout."
        ),
    }
    (metrics_dir / "sec_bert_frozen_retrospective_test_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# SEC-BERT凍結路由：retrospective test v1", "",
        f"在讀取890筆test logits前，temperature、margin score與兩個threshold均已凍結。"
        f"Base accuracy={probability_evaluation['raw']['accuracy']:.4f}、macro-F1={probability_evaluation['raw']['macro_f1']:.4f}。", "",
        "## 凍結操作點", "",
        "| 工作點 | Threshold | Referral | Coverage | Selective accuracy | Error capture | Referral precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.operating_point} | {row.threshold:.6f} | {row.referral_rate:.4f} | {row.coverage:.4f} | "
            f"{row.selective_accuracy:.4f} | {row.error_capture_rate:.4f} | {row.referral_precision:.4f} |"
        )
    lines.extend(["", "## 解讀", "", (
        "這是鎖定規則的回顧性轉移檢查，而非全新確認性測試。不得依據本結果修改temperature或threshold；"
        "正式risk-controlled主張仍需未接觸的PCAOB temporal holdout。"
    ), ""])
    (report_dir / "sec_bert_frozen_retrospective_test_v1_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
