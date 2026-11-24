"""Aggregate Phase 6 predictions with report-clustered uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


CONTRASTS = (
    ("primary_lora_vs_full_original_weighted", "C01_SEC_LORA_ORIGINAL_WEIGHTED", "C04_SEC_FULL_ORIGINAL_WEIGHTED"),
    ("lora_weighted_vs_unweighted_original", "C01_SEC_LORA_ORIGINAL_WEIGHTED", "C02_SEC_LORA_ORIGINAL_UNWEIGHTED"),
    ("sec_full_weighted_vs_unweighted_original", "C04_SEC_FULL_ORIGINAL_WEIGHTED", "C05_SEC_FULL_ORIGINAL_UNWEIGHTED"),
    ("sec_full_vs_bert_full_original_weighted", "C04_SEC_FULL_ORIGINAL_WEIGHTED", "C07_BERT_FULL_ORIGINAL_WEIGHTED"),
    ("lora_vs_full_matched_masked", "C03_SEC_LORA_MASKED_WEIGHTED", "C06_SEC_FULL_MASKED_WEIGHTED"),
)
SEEDS = (20261114, 20261115, 20261116)
LABEL_IDS = list(range(12))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def macro_f1(frame: pd.DataFrame, column: str = "prediction_id") -> float:
    return float(f1_score(frame["label_id"], frame[column], labels=LABEL_IDS, average="macro", zero_division=0))


def report_confusions(frame: pd.DataFrame, prediction: str = "prediction_id", reports=None):
    reports = np.asarray(sorted(frame["report_id"].astype(str).unique()) if reports is None else reports)
    matrices = np.zeros((len(reports), 12, 12), dtype=np.int64)
    report_to_index = {report: index for index, report in enumerate(reports)}
    for row in frame.itertuples(index=False):
        matrices[report_to_index[str(row.report_id)], int(row.label_id), int(getattr(row, prediction))] += 1
    return reports, matrices


def macro_f1_from_confusions(matrices: np.ndarray) -> np.ndarray:
    true_sum = matrices.sum(axis=2)
    pred_sum = matrices.sum(axis=1)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    denominator = true_sum + pred_sum
    per_class = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return per_class.mean(axis=1)


def bootstrap_scores(matrices: np.ndarray, replicates: int, rng: np.random.Generator, counts=None) -> np.ndarray:
    report_count = len(matrices)
    if counts is None:
        counts = rng.multinomial(report_count, np.full(report_count, 1.0 / report_count), size=replicates)
    sampled = (counts @ matrices.reshape(report_count, -1)).reshape(replicates, 12, 12)
    return macro_f1_from_confusions(sampled)


def interval(values: list[float]) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-freeze", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], required=True)
    parser.add_argument("--bootstrap-replicates", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261121)
    args = parser.parse_args()

    freeze = read_json(args.phase6_freeze)
    if args.partition == "test":
        expected = int(freeze["uncertainty"]["report_cluster_bootstrap_replicates"])
        if args.bootstrap_replicates != expected:
            raise RuntimeError(f"Test aggregation requires exactly {expected} bootstrap replicates")
    run_rows, predictions = [], {}
    expected_ids = None
    for spec in freeze["runs"]:
        run_dir = args.evaluation_root / spec["run_id"]
        metrics = read_json(run_dir / "metrics.json")
        frame = pd.read_csv(run_dir / f"{args.partition}_predictions.csv")
        if metrics["partition"] != args.partition:
            raise RuntimeError(f"Partition mismatch for {spec['run_id']}")
        ids = set(frame["sample_id"].astype(str))
        if expected_ids is None: expected_ids = ids
        elif ids != expected_ids: raise RuntimeError(f"Membership mismatch for {spec['run_id']}")
        predictions[(spec["condition_id"], int(spec["seed"]))] = frame
        m = metrics["metrics"]
        run_rows.append({
            "order": spec["order"], "run_id": spec["run_id"], "condition_id": spec["condition_id"],
            "seed": spec["seed"], "method": spec["method"], "text_condition": spec["text_condition"],
            "loss": spec["loss"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "weighted_f1": m["weighted_f1"], "balanced_accuracy": m["balanced_accuracy"],
            "throughput_examples_per_second": metrics["median_examples_per_second"],
        })
    if len(run_rows) != 24 or len(predictions) != 24:
        raise RuntimeError("All 24 frozen runs are required")

    rng = np.random.default_rng(args.bootstrap_seed)
    run_bootstrap = []
    for row in run_rows:
        frame = predictions[(row["condition_id"], int(row["seed"]))]
        _, matrices = report_confusions(frame)
        values = bootstrap_scores(matrices, args.bootstrap_replicates, rng)
        low, high = interval(values.tolist())
        run_bootstrap.append({"run_id": row["run_id"], "macro_f1": row["macro_f1"], "ci95_low": low, "ci95_high": high})

    frame_rows = pd.DataFrame(run_rows)
    condition_rows = []
    for condition, group in frame_rows.groupby("condition_id", sort=True):
        condition_rows.append({
            "condition_id": condition, "method": group.iloc[0]["method"],
            "text_condition": group.iloc[0]["text_condition"], "loss": group.iloc[0]["loss"],
            "n_seeds": len(group), "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_sample_sd": float(group["macro_f1"].std(ddof=1)),
            "macro_f1_worst": float(group["macro_f1"].min()), "macro_f1_best": float(group["macro_f1"].max()),
        })

    contrast_rows = []
    for name, left, right in CONTRASTS:
        observed = [macro_f1(predictions[(left, seed)]) - macro_f1(predictions[(right, seed)]) for seed in SEEDS]
        paired_matrices = {}
        for seed in SEEDS:
            left_frame = predictions[(left, seed)]
            right_frame = predictions[(right, seed)][["sample_id", "prediction_id"]]
            paired = left_frame.merge(right_frame, on="sample_id", suffixes=("_left", "_right"), validate="one_to_one")
            reports, left_matrices = report_confusions(paired, "prediction_id_left")
            _, right_matrices = report_confusions(paired, "prediction_id_right", reports)
            paired_matrices[seed] = (left_matrices, right_matrices)
        slot_values = []
        for _slot in range(len(SEEDS)):
            chosen = rng.choice(SEEDS, size=args.bootstrap_replicates, replace=True)
            values = np.zeros(args.bootstrap_replicates, dtype=float)
            for seed in SEEDS:
                left_matrices, right_matrices = paired_matrices[seed]
                report_count = len(left_matrices)
                counts = rng.multinomial(
                    report_count, np.full(report_count, 1.0 / report_count), size=args.bootstrap_replicates
                )
                left_draws = bootstrap_scores(left_matrices, args.bootstrap_replicates, rng, counts)
                right_draws = bootstrap_scores(right_matrices, args.bootstrap_replicates, rng, counts)
                mask = chosen == seed
                values[mask] = (left_draws - right_draws)[mask]
            slot_values.append(values)
        draws = np.mean(np.vstack(slot_values), axis=0)
        low, high = interval(draws.tolist())
        contrast_rows.append({
            "contrast": name, "left_condition": left, "right_condition": right,
            "delta_seed_20261114": observed[0], "delta_seed_20261115": observed[1],
            "delta_seed_20261116": observed[2], "delta_mean": float(np.mean(observed)),
            "delta_sample_sd": float(np.std(observed, ddof=1)), "hierarchical_ci95_low": low,
            "hierarchical_ci95_high": high, "bootstrap_replicates": args.bootstrap_replicates,
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / f"phase6_{args.partition}_seed_results_v1_1.csv", run_rows)
    write_csv(args.output_root / f"phase6_{args.partition}_condition_summary_v1_1.csv", condition_rows)
    write_csv(args.output_root / f"phase6_{args.partition}_run_bootstrap_v1_1.csv", run_bootstrap)
    write_csv(args.output_root / f"phase6_{args.partition}_contrasts_v1_1.csv", contrast_rows)
    summary = {"status": "phase6_aggregation_complete", "partition": args.partition, "runs": 24,
               "conditions": 8, "contrasts": 5, "bootstrap_replicates": args.bootstrap_replicates,
               "statistical_unit": "inspection_report", "test_previously_inspected": args.partition == "test"}
    (args.output_root / "aggregation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

