"""Verify and finalize the frozen Phase 6B retrospective test evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_BASELINES = (
    ("tfidf_balanced_lr", "original"),
    ("tfidf_balanced_lr", "masked"),
    ("frozen_minilm_balanced_lr", "original"),
    ("frozen_minilm_balanced_lr", "masked"),
    ("frozen_sec_bert_balanced_lr", "original"),
    ("frozen_sec_bert_balanced_lr", "masked"),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = read_json(root / "sha256_manifest.json")
    for item in manifest:
        path = root / item["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing manifest file: {path}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Manifest mismatch: {path}")
    return len(manifest)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-freeze", type=Path, required=True)
    parser.add_argument("--phase6a-acceptance", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--aggregation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failed-attempt-root", type=Path)
    args = parser.parse_args()

    freeze = read_json(args.phase6_freeze)
    phase6a = read_json(args.phase6a_acceptance)
    if phase6a.get("status") != "phase6a_accepted_test_gate_ready":
        raise RuntimeError("Phase 6A was not accepted")
    expected_rows = int(freeze["test_gate"]["clean_test_rows"])
    tiers = freeze["frequency_tiers_from_train_only"]
    label_order = sorted(tiers["high"] + tiers["medium"] + tiers["low"])
    if len(label_order) != 12 or len(set(label_order)) != 12:
        raise RuntimeError("Frozen frequency tiers do not define exactly 12 labels")
    expected_evaluator = phase6a["evaluator_sha256"]
    expected_baseline_evaluator = phase6a["baseline_evaluator_sha256"]
    expected_freeze_hash = phase6a["phase6_freeze_sha256"]
    if sha256_file(args.phase6_freeze) != expected_freeze_hash:
        raise RuntimeError("Phase 6 freeze differs from Phase 6A acceptance")

    official_summary = read_json(args.official_root / "execution_summary.json")
    required_official = {
        "status": "evaluation_complete",
        "partition": "test",
        "runs_completed": 24,
        "runs_requested": 24,
        "test_scored": True,
        "calibration_loaded": False,
        "evaluator_sha256": expected_evaluator,
        "phase6_freeze_sha256": expected_freeze_hash,
    }
    for key, value in required_official.items():
        if official_summary.get(key) != value:
            raise RuntimeError(f"Official summary mismatch for {key}")

    expected_ids = None
    expected_reports = None
    official_manifest_files = 0
    tier_seed_rows: list[dict] = []
    per_class_rows: list[dict] = []
    throughput_rows: list[dict] = []
    for spec in freeze["runs"]:
        run_root = args.official_root / spec["run_id"]
        official_manifest_files += verify_manifest(run_root)
        metrics = read_json(run_root / "metrics.json")
        predictions = pd.read_csv(run_root / "test_predictions.csv", low_memory=False)
        if metrics.get("partition") != "test" or not metrics.get("test_previously_inspected"):
            raise RuntimeError(f"Test positioning mismatch: {spec['run_id']}")
        if metrics.get("calibration_loaded") is not False or metrics.get("rows") != expected_rows:
            raise RuntimeError(f"Official row/calibration mismatch: {spec['run_id']}")
        if metrics.get("evaluator_sha256") != expected_evaluator:
            raise RuntimeError(f"Official evaluator mismatch: {spec['run_id']}")
        if len(predictions) != expected_rows or predictions["sample_id"].duplicated().any():
            raise RuntimeError(f"Official prediction membership invalid: {spec['run_id']}")
        sample_ids = frozenset(predictions["sample_id"].astype(str))
        report_ids = frozenset(predictions["report_id"].astype(str))
        if expected_ids is None:
            expected_ids, expected_reports = sample_ids, report_ids
        elif sample_ids != expected_ids or report_ids != expected_reports:
            raise RuntimeError(f"Official membership mismatch: {spec['run_id']}")
        for tier, values in metrics["metrics"]["frequency_tiers"].items():
            tier_seed_rows.append({
                "run_id": spec["run_id"], "condition_id": spec["condition_id"],
                "seed": spec["seed"], "method": spec["method"],
                "text_condition": spec["text_condition"], "loss": spec["loss"],
                "tier": tier, "tier_macro_f1": values["macro_f1"],
                "tier_macro_recall": values["macro_recall"],
            })
        for item in metrics["metrics"]["per_class"]:
            label_id = int(item["label_id"])
            per_class_rows.append({
                "run_id": spec["run_id"], "condition_id": spec["condition_id"],
                "seed": spec["seed"], "method": spec["method"],
                "text_condition": spec["text_condition"], "loss": spec["loss"],
                "label_id": label_id, "label": label_order[label_id],
                "precision": item["precision"], "recall": item["recall"],
                "f1": item["f1"], "support": item["support"],
            })
        throughput_rows.append({
            "run_id": spec["run_id"], "condition_id": spec["condition_id"],
            "seed": spec["seed"], "method": spec["method"],
            "text_condition": spec["text_condition"], "loss": spec["loss"],
            "device_class": "GPU", "warmup_passes": metrics["warmup_passes"],
            "timed_passes": metrics["timed_passes"],
            "median_examples_per_second": metrics["median_examples_per_second"],
        })

    tier_frame = pd.DataFrame(tier_seed_rows)
    tier_summary_rows: list[dict] = []
    for (condition_id, tier), group in tier_frame.groupby(["condition_id", "tier"], sort=True):
        tier_summary_rows.append({
            "condition_id": condition_id,
            "method": group.iloc[0]["method"],
            "text_condition": group.iloc[0]["text_condition"],
            "loss": group.iloc[0]["loss"], "tier": tier, "n_seeds": len(group),
            "tier_macro_f1_mean": float(group["tier_macro_f1"].mean()),
            "tier_macro_f1_sample_sd": float(group["tier_macro_f1"].std(ddof=1)),
            "tier_macro_f1_worst": float(group["tier_macro_f1"].min()),
            "tier_macro_recall_mean": float(group["tier_macro_recall"].mean()),
        })

    baseline_summary = read_json(args.baseline_root / "execution_summary.json")
    if baseline_summary.get("runs_completed") != 6 or baseline_summary.get("partition") != "test":
        raise RuntimeError("Baseline execution summary is incomplete")
    if baseline_summary.get("test_scored") is not True or baseline_summary.get("calibration_loaded") is not False:
        raise RuntimeError("Baseline test/calibration status mismatch")
    if baseline_summary.get("baseline_evaluator_sha256") != expected_baseline_evaluator:
        raise RuntimeError("Baseline evaluator mismatch")
    baseline_rows: list[dict] = []
    baseline_manifest_files = 0
    for method, text_condition in EXPECTED_BASELINES:
        run_id = f"baseline__{method}__{text_condition}__phase1v1_1"
        run_root = args.baseline_root / run_id
        baseline_manifest_files += verify_manifest(run_root)
        metrics = read_json(run_root / "metrics.json")
        predictions = pd.read_csv(run_root / "test_predictions.csv", low_memory=False)
        if metrics.get("rows") != expected_rows or metrics.get("test_scored") is not True:
            raise RuntimeError(f"Baseline row/test mismatch: {run_id}")
        if metrics.get("calibration_loaded") is not False:
            raise RuntimeError(f"Baseline calibration mismatch: {run_id}")
        if metrics.get("baseline_evaluator_sha256") != expected_baseline_evaluator:
            raise RuntimeError(f"Baseline evaluator mismatch: {run_id}")
        if len(predictions) != expected_rows or frozenset(predictions["sample_id"].astype(str)) != expected_ids:
            raise RuntimeError(f"Baseline membership mismatch: {run_id}")
        values = metrics["metrics"]
        baseline_rows.append({
            "run_id": run_id, "method": method, "text_condition": text_condition,
            "macro_f1": values["macro_f1"], "accuracy": values["accuracy"],
            "balanced_accuracy": values["balanced_accuracy"],
            "weighted_f1": values["weighted_f1"],
            "device_class": "CPU",
            "median_examples_per_second": metrics["median_examples_per_second"],
        })

    aggregation = read_json(args.aggregation_root / "aggregation_summary.json")
    required_aggregation = {
        "status": "phase6_aggregation_complete", "partition": "test", "runs": 24,
        "conditions": 8, "contrasts": 5, "bootstrap_replicates": 5000,
        "statistical_unit": "inspection_report", "test_previously_inspected": True,
    }
    for key, value in required_aggregation.items():
        if aggregation.get(key) != value:
            raise RuntimeError(f"Aggregation summary mismatch for {key}")
    condition_table = pd.read_csv(args.aggregation_root / "phase6_test_condition_summary_v1_1.csv")
    contrast_table = pd.read_csv(args.aggregation_root / "phase6_test_contrasts_v1_1.csv")
    seed_table = pd.read_csv(args.aggregation_root / "phase6_test_seed_results_v1_1.csv")
    if len(condition_table) != 8 or len(contrast_table) != 5 or len(seed_table) != 24:
        raise RuntimeError("Aggregation table row count mismatch")

    failed_attempt = None
    if args.failed_attempt_root is not None:
        partial = list(args.failed_attempt_root.glob("*/metrics.json"))
        failed_attempt = {
            "preserved": True,
            "completed_run_outputs_before_environment_failure": len(partial),
            "execution_summary_present": (args.failed_attempt_root / "execution_summary.json").is_file(),
            "failure_class": "offline_model_loading_environment_retry",
            "result_selection_or_tuning_performed": False,
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "phase6_test_frequency_tier_seed_results_v1_1.csv", tier_seed_rows)
    write_csv(args.output_root / "phase6_test_frequency_tier_summary_v1_1.csv", tier_summary_rows)
    write_csv(args.output_root / "phase6_test_per_class_seed_results_v1_1.csv", per_class_rows)
    write_csv(args.output_root / "phase6_test_official_throughput_v1_1.csv", throughput_rows)
    write_csv(args.output_root / "phase6_test_baseline_summary_v1_1.csv", baseline_rows)

    acceptance = {
        "version": "phase6b-acceptance-v1.1",
        "status": "phase6b_retrospective_test_accepted",
        "evaluation_positioning": freeze["evaluation_positioning"],
        "test_previously_inspected": True, "test_scored": True,
        "calibration_loaded": False, "clean_test_rows": expected_rows,
        "unique_test_reports": len(expected_reports),
        "official_runs_completed": 24, "baseline_runs_completed": 6,
        "conditions_reported": 8, "prespecified_contrasts_reported": 5,
        "bootstrap_replicates": 5000, "statistical_unit": "inspection_report",
        "official_manifest_files_verified": official_manifest_files,
        "baseline_manifest_files_verified": baseline_manifest_files,
        "phase6_freeze_sha256": expected_freeze_hash,
        "evaluator_sha256": expected_evaluator,
        "baseline_evaluator_sha256": expected_baseline_evaluator,
        "aggregator_sha256": phase6a["aggregator_sha256"],
        "failed_attempt_audit": failed_attempt,
        "selection_prohibitions_preserved": freeze["selection_prohibitions"],
    }
    acceptance_path = args.output_root / "phase6b_acceptance_v1_1.json"
    acceptance_path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

