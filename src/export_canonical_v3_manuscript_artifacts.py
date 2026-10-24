"""Export manuscript-facing tables from the frozen canonical v3 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-run", type=Path, required=True)
    parser.add_argument("--masking-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def interval(values: list[float]) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.5, 0.975])]


def reliability_rows(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for method, column in (("raw", "raw_confidence_recomputed"),
                           ("temperature_scaled", "calibrated_confidence")):
        confidence = frame[column].to_numpy(float)
        correct = frame["correct"].astype(str).str.lower().eq("true").to_numpy(bool)
        bin_id = np.minimum((confidence * 10).astype(int), 9)
        for index in range(10):
            keep = bin_id == index
            rows.append({
                "method": method,
                "bin": index,
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "samples": int(keep.sum()),
                "mean_confidence": float(confidence[keep].mean()) if keep.any() else np.nan,
                "accuracy": float(correct[keep].mean()) if keep.any() else np.nan,
            })
    return rows


def main() -> None:
    args = parse_args()
    tables = args.output_root / "tables"
    metrics_dir = args.output_root / "metrics"
    tables.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_metrics = json.loads((args.canonical_run / "metrics.json").read_text(encoding="utf-8"))
    downstream = json.loads((args.canonical_run / "downstream" / "metrics.json").read_text(encoding="utf-8"))
    scored = pd.read_csv(args.canonical_run / "downstream" / "test_scored.csv")
    calibration = pd.read_csv(args.canonical_run / "downstream" / "calibration_scaled.csv")
    baseline = pd.read_csv(ROOT / "results" / "tables" / "model_comparison_test_predictions_v2.csv")
    masking = json.loads((args.masking_run / "metrics.json").read_text(encoding="utf-8"))
    masked_predictions = pd.read_csv(args.masking_run / "masked_trained_masked_test_predictions.csv")

    paired = scored[["sample_id", "report_id", "target_level1", "prediction"]].merge(
        baseline[["sample_id", "tfidf_context"]], on="sample_id", how="inner", validate="one_to_one"
    )
    if len(paired) != len(scored):
        raise RuntimeError("Canonical and TF-IDF test predictions are not aligned")
    true = paired["target_level1"].to_numpy(str)
    sec = paired["prediction"].to_numpy(str)
    tfidf = paired["tfidf_context"].to_numpy(str)
    codes, reports = pd.factorize(paired["report_id"].astype(str), sort=False)
    groups = [np.flatnonzero(codes == code) for code in range(len(reports))]
    rng = np.random.default_rng(20261002 + 301)
    acc_delta: list[float] = []
    macro_delta: list[float] = []
    for _ in range(args.bootstrap_replicates):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[value] for value in chosen])
        acc_delta.append(accuracy_score(true[idx], sec[idx]) - accuracy_score(true[idx], tfidf[idx]))
        macro_delta.append(
            f1_score(true[idx], sec[idx], average="macro", zero_division=0)
            - f1_score(true[idx], tfidf[idx], average="macro", zero_division=0)
        )

    policies = downstream["retrospective_test_operating_points"]
    thresholds = {name: float(row["threshold"]) for name, row in policies.items()}
    scored["correct"] = scored["correct"].astype(str).str.lower().eq("true")
    scored["high_coverage"] = scored["calibrated_margin"] < thresholds["high_coverage"]
    scored["conservative"] = scored["calibrated_margin"] < thresholds["conservative"]
    scored.to_csv(tables / "sec_bert_canonical_v3_test_scored.csv", index=False, encoding="utf-8-sig")

    context_map = {
        "same_report_issuer_audit_area_preamble": "context_added",
        "reference_without_available_preamble": "context_unresolved",
        "original_sufficient": "original_sufficient",
    }
    context = scored.assign(context_group=scored["context_quality"].map(context_map))
    context_rows = []
    for name, group in context.groupby("context_group"):
        errors = ~group["correct"]
        context_rows.append({
            "context_group": name,
            "samples": int(len(group)),
            "errors": int(errors.sum()),
            "accuracy": float(group["correct"].mean()),
            "high_coverage_referral_rate": float(group["high_coverage"].mean()),
            "conservative_referral_rate": float(group["conservative"].mean()),
        })
    pd.DataFrame(context_rows).to_csv(
        tables / "sec_bert_context_quality_test_v3.csv", index=False, encoding="utf-8-sig"
    )

    confusion = (
        scored.loc[~scored["correct"]]
        .groupby(["target_level1", "prediction"], as_index=False)
        .agg(errors=("sample_id", "size"), mean_calibrated_margin=("calibrated_margin", "mean"), reports=("report_id", "nunique"))
        .sort_values(["errors", "reports", "target_level1"], ascending=[False, False, True])
    )
    confusion.to_csv(tables / "sec_bert_confusion_pairs_test_v3.csv", index=False, encoding="utf-8-sig")

    labels = sorted(scored["target_level1"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        scored["target_level1"], scored["prediction"], labels=labels, zero_division=0
    )
    pd.DataFrame({"label": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}).to_csv(
        tables / "sec_bert_class_metrics_test_v3.csv", index=False, encoding="utf-8-sig"
    )

    class_referral_rows = []
    for prediction, group in scored.groupby("prediction"):
        errors = ~group["correct"]
        class_referral_rows.append({
            "prediction": prediction,
            "samples": int(len(group)),
            "errors": int(errors.sum()),
            "error_rate": float(errors.mean()),
            "high_coverage_referral_rate": float(group["high_coverage"].mean()),
            "conservative_referral_rate": float(group["conservative"].mean()),
            "high_coverage_captured": int((errors & group["high_coverage"]).sum()),
            "conservative_captured": int((errors & group["conservative"]).sum()),
        })
    pd.DataFrame(class_referral_rows).to_csv(
        tables / "sec_bert_class_referral_v3.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(reliability_rows(calibration)).to_csv(
        tables / "sec_bert_reliability_bins_v3.csv", index=False, encoding="utf-8-sig"
    )
    probability_rows = []
    for method, row in downstream["calibration_probability_metrics"].items():
        probability_rows.append({"method": method, **row})
    pd.DataFrame(probability_rows).to_csv(
        tables / "sec_bert_calibration_metrics_v3.csv", index=False, encoding="utf-8-sig"
    )

    mask_true = masked_predictions["target_level1"].to_numpy(str)
    mask_pred = masked_predictions["prediction"].to_numpy(str)
    m_precision, m_recall, m_f1, m_support = precision_recall_fscore_support(
        mask_true, mask_pred, labels=labels, zero_division=0
    )
    pd.DataFrame({"label": labels, "precision": m_precision, "recall": m_recall, "f1": m_f1, "support": m_support}).to_csv(
        tables / "sec_bert_masked_class_metrics_test_v3.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "version": "sec-bert-canonical-v3-manuscript-20261024",
        "canonical": train_metrics,
        "downstream": downstream,
        "masking": masking,
        "sec_bert_minus_tfidf": {
            "accuracy_delta": float(accuracy_score(true, sec) - accuracy_score(true, tfidf)),
            "accuracy_delta_report_cluster_bootstrap_95": interval(acc_delta),
            "macro_f1_delta": float(
                f1_score(true, sec, average="macro", zero_division=0)
                - f1_score(true, tfidf, average="macro", zero_division=0)
            ),
            "macro_f1_delta_report_cluster_bootstrap_95": interval(macro_delta),
        },
        "artifact_paths": {
            "canonical_run": str(args.canonical_run.resolve()),
            "masking_run": str(args.masking_run.resolve()),
        },
    }
    (metrics_dir / "sec_bert_canonical_v3_manuscript.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["sec_bert_minus_tfidf"], indent=2))
    print(pd.DataFrame(context_rows).to_string(index=False))
    print(confusion.head(8).to_string(index=False))
    print(pd.DataFrame({"label": labels, "f1": m_f1, "support": m_support}).to_string(index=False))


if __name__ == "__main__":
    main()
