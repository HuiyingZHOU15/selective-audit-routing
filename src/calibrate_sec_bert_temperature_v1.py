"""Fit SEC-BERT temperature scaling and freeze referral thresholds.

The primary risk score (top-1/top-2 margin) was selected on development OOF.
This script uses only the locked calibration split to fit one temperature and
to freeze 5% and 10% referral thresholds.  It never reads retrospective test.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sec_bert_temperature_calibration_v1.json"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))


def multiclass_brier(probabilities: np.ndarray, y: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[y]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray, y: np.ndarray, bins: int
) -> tuple[float, pd.DataFrame]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = abs(mean_confidence - accuracy)
            ece += count / len(y) * gap
        else:
            mean_confidence = accuracy = gap = float("nan")
        rows.append({
            "bin": index + 1, "lower": lower, "upper": upper, "samples": count,
            "mean_confidence": mean_confidence, "accuracy": accuracy, "absolute_gap": gap,
        })
    return float(ece), pd.DataFrame(rows)


def probability_features(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    clipped = np.clip(probabilities, 1e-15, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    return confidence, margin, entropy


def threshold_for_budget(values: np.ndarray, budget: float) -> tuple[float, int]:
    """Return a strict-less-than threshold producing the requested low-score count."""
    count = int(round(len(values) * budget))
    ordered = np.sort(values)
    if count <= 0:
        return float("-inf"), 0
    if count >= len(values):
        return float("inf"), len(values)
    threshold = float((ordered[count - 1] + ordered[count]) / 2.0)
    actual = int((values < threshold).sum())
    if actual != count:
        raise RuntimeError(f"Margin tie prevents exact budget: requested={count}, actual={actual}")
    return threshold, count


def upper_binomial_risk(errors: int, accepted: int, confidence: float = 0.95) -> float:
    if accepted <= 0:
        return float("nan")
    if errors >= accepted:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, accepted - errors))


def selective_metrics(
    frame: pd.DataFrame, score: np.ndarray, threshold: float, labels: list[str]
) -> dict:
    referred = score < threshold
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
    }


def report_cluster_bootstrap(
    frame: pd.DataFrame, score: np.ndarray, threshold: float, config: dict
) -> dict[str, list[float]]:
    rng = np.random.default_rng(int(config["random_seed"]))
    report_codes, reports = pd.factorize(frame["report_id"].astype(str), sort=False)
    groups = [np.flatnonzero(report_codes == code) for code in range(len(reports))]
    errors = ~frame["prediction"].eq(frame["target_level1"]).to_numpy()
    values = {"coverage": [], "selective_accuracy": [], "error_capture_rate": [], "referral_precision": []}
    for _ in range(int(config["bootstrap_replicates"])):
        sampled_codes = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[code] for code in sampled_codes])
        referred = score[indices] < threshold
        accepted = ~referred
        sampled_errors = errors[indices]
        referred_errors = int(sampled_errors[referred].sum())
        total_errors = int(sampled_errors.sum())
        values["coverage"].append(float(accepted.mean()))
        values["selective_accuracy"].append(float(1.0 - sampled_errors[accepted].mean()))
        values["error_capture_rate"].append(float(referred_errors / total_errors) if total_errors else np.nan)
        values["referral_precision"].append(float(referred_errors / referred.sum()) if referred.sum() else np.nan)
    return {
        key: [float(x) for x in np.nanquantile(series, [0.025, 0.5, 0.975])]
        for key, series in values.items()
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / config["calibration_logits"])
    if len(frame) != 608 or frame["sample_id"].nunique() != 608:
        raise ValueError("Expected exactly 608 unique calibration samples")

    labels = sorted(frame["target_level1"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    y = frame["target_level1"].map(label_to_id).to_numpy(int)
    logits = np.asarray([json.loads(value) for value in frame["logits_json"]], dtype=float)
    if logits.shape != (len(frame), len(labels)):
        raise ValueError(f"Unexpected logits shape {logits.shape}; labels={len(labels)}")
    if not np.array_equal(logits.argmax(axis=1), frame["prediction"].map(label_to_id).to_numpy(int)):
        raise ValueError("Logit order does not reproduce exported predictions")

    optimization = minimize_scalar(
        lambda log_t: negative_log_likelihood(logits, y, float(np.exp(log_t))),
        bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-10},
    )
    temperature = float(np.exp(optimization.x))
    raw_probabilities = softmax(logits)
    calibrated_probabilities = softmax(logits / temperature)
    raw_confidence, raw_margin, raw_entropy = probability_features(raw_probabilities)
    calibrated_confidence, calibrated_margin, calibrated_entropy = probability_features(calibrated_probabilities)

    rows = []
    reliability_tables = {}
    for name, probabilities in {"raw": raw_probabilities, "temperature_scaled": calibrated_probabilities}.items():
        ece, reliability = expected_calibration_error(probabilities, y, int(config["ece_bins"]))
        prediction = probabilities.argmax(axis=1)
        rows.append({
            "method": name,
            "temperature": 1.0 if name == "raw" else temperature,
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "nll": negative_log_likelihood(logits, y, 1.0 if name == "raw" else temperature),
            "multiclass_brier": multiclass_brier(probabilities, y),
            "ece": ece,
        })
        reliability["method"] = name
        reliability_tables[name] = reliability
    comparison = pd.DataFrame(rows)

    output = frame.copy()
    output["raw_confidence_recomputed"] = raw_confidence
    output["raw_margin_recomputed"] = raw_margin
    output["raw_entropy_recomputed"] = raw_entropy
    output["calibrated_confidence"] = calibrated_confidence
    output["calibrated_margin"] = calibrated_margin
    output["calibrated_entropy"] = calibrated_entropy

    operating_points = []
    thresholds = {}
    bootstrap = {}
    for score_name, score in {
        "calibrated_margin": calibrated_margin,
        "calibrated_confidence": calibrated_confidence,
    }.items():
        for budget in [float(value) for value in config["referral_budgets"]]:
            threshold, intended_count = threshold_for_budget(score, budget)
            key = f"{score_name}_{int(round(100 * budget))}pct"
            thresholds[key] = {
                "score": score_name, "operator": "refer_if_score_below",
                "threshold": threshold, "requested_referral_rate": budget,
                "intended_referral_count": intended_count,
            }
            metrics = selective_metrics(output, score, threshold, labels)
            metrics.update({"score": score_name, "requested_referral_rate": budget, "threshold": threshold})
            operating_points.append(metrics)
            bootstrap[key] = report_cluster_bootstrap(output, score, threshold, config)

    table_dir = ROOT / "results/tables"
    metrics_dir = ROOT / "results/metrics"
    report_dir = ROOT / "results/reports"
    figure_dir = ROOT / "results/figures"
    for directory in (table_dir, metrics_dir, report_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    output.to_csv(table_dir / "sec_bert_calibration_scaled_v1.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(table_dir / "sec_bert_calibration_metrics_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(operating_points).to_csv(
        table_dir / "sec_bert_calibration_operating_points_v1.csv", index=False, encoding="utf-8-sig"
    )
    reliability = pd.concat(reliability_tables.values(), ignore_index=True)
    reliability.to_csv(table_dir / "sec_bert_reliability_bins_v1.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(5.8, 5.2))
    for method, group in reliability.groupby("method", sort=False):
        present = group["samples"].gt(0)
        plt.plot(group.loc[present, "mean_confidence"], group.loc[present, "accuracy"], marker="o", label=method)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figure_dir / "sec_bert_reliability_diagram_v1.png", dpi=220)
    plt.close()

    payload = {
        "version": config["version"],
        "samples": int(len(frame)), "reports": int(frame["report_id"].nunique()),
        "temperature": temperature, "optimization_success": bool(optimization.success),
        "calibration_metrics": comparison.to_dict(orient="records"),
        "frozen_thresholds": thresholds,
        "operating_points": operating_points,
        "report_cluster_bootstrap_95": bootstrap,
        "protocol_guardrail": config["protocol"],
        "claim_guardrail": (
            "Calibration estimates and thresholds are not confirmatory test results. "
            "Risk-controlled wording requires unchanged evaluation on temporal holdout."
        ),
    }
    (metrics_dir / "sec_bert_temperature_calibration_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary = pd.DataFrame(operating_points)
    lines = [
        "# SEC-BERT temperature calibration與路由閾值v1", "",
        f"使用608筆、{frame['report_id'].nunique()}份report的鎖定calibration split；temperature={temperature:.6f}。", "",
        "## 概率校準", "",
        "| 方法 | Accuracy | Macro-F1 | NLL | Multiclass Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.method} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.nll:.4f} | "
            f"{row.multiclass_brier:.4f} | {row.ece:.4f} |"
        )
    lines.extend(["", "## 凍結工作點", "",
                  "| Score | Referral | Threshold | Coverage | Selective accuracy | Error capture | Referral precision |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in primary.itertuples():
        lines.append(
            f"| {row.score} | {row.referral_rate:.4f} | {row.threshold:.6f} | {row.coverage:.4f} | "
            f"{row.selective_accuracy:.4f} | {row.error_capture_rate:.4f} | {row.referral_precision:.4f} |"
        )
    lines.extend(["", "## 限制", "", (
        "Margin已在development OOF預先選定；本步只擬合temperature及凍結閾值。Calibration結果不能當成測試結果，"
        "也不能單獨支持risk-controlled主張。下一步必須用保存的checkpoint產生retrospective test logits，"
        "套用完全相同的temperature與threshold，不得再次調整。"
    ), ""])
    (report_dir / "sec_bert_temperature_calibration_v1_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
