"""Post-hoc near-duplicate sensitivity analysis using saved predictions only.

The analysis is explicitly diagnostic after retrospective test access. It does not
retrain, select a new checkpoint, or modify any frozen artifact.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aggregate_phase6_results_v1_1 import (  # noqa: E402
    CONTRASTS,
    LABEL_IDS,
    SEEDS,
    bootstrap_scores,
    interval,
    report_confusions,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def macro_f1(frame: pd.DataFrame) -> float:
    return float(
        f1_score(
            frame["label_id"], frame["prediction_id"],
            labels=LABEL_IDS, average="macro", zero_division=0,
        )
    )


def filter_ids(frame: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    result = frame.loc[~frame["sample_id"].astype(str).isin(excluded)].copy()
    if len(frame) - len(result) != len(excluded):
        missing = excluded - set(frame["sample_id"].astype(str))
        raise RuntimeError(f"Prediction file did not contain all excluded IDs: {sorted(missing)[:5]}")
    return result.reset_index(drop=True)


def zip_run_table(archive: Path, excluded: set[str]) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(archive) as handle:
        prediction_members = sorted(
            name for name in handle.namelist()
            if name.startswith("runs/") and name.endswith("/validation_predictions.csv")
        )
        for member in prediction_members:
            run_root = member.rsplit("/", 1)[0]
            config = json.loads(handle.read(f"{run_root}/resolved_config.json"))
            frame = pd.read_csv(io.BytesIO(handle.read(member)))
            filtered = filter_ids(frame, excluded)
            rows.append({
                "run_id": config["run_id"], "method": config["method"],
                "adapter_candidate_id": config.get("adapter_candidate_id"),
                "learning_rate": float(config["learning_rate"]),
                "original_rows": len(frame), "filtered_rows": len(filtered),
                "original_macro_f1": macro_f1(frame),
                "filtered_macro_f1": macro_f1(filtered),
                "delta_filtered_minus_original": macro_f1(filtered) - macro_f1(frame),
            })
    return pd.DataFrame(rows)


def condition_summary(rows: pd.DataFrame, score_prefix: str) -> pd.DataFrame:
    records = []
    for condition, group in rows.groupby("condition_id", sort=True):
        base = {
            "condition_id": condition, "method": group.iloc[0]["method"],
            "text_condition": group.iloc[0]["text_condition"],
            "loss": group.iloc[0]["loss"], "n_seeds": len(group),
        }
        for metric in ("original_macro_f1", "filtered_macro_f1"):
            base[f"{metric}_mean"] = float(group[metric].mean())
            base[f"{metric}_sample_sd"] = float(group[metric].std(ddof=1))
            base[f"{metric}_worst"] = float(group[metric].min())
        base["mean_delta_filtered_minus_original"] = (
            base["filtered_macro_f1_mean"] - base["original_macro_f1_mean"]
        )
        records.append(base)
    return pd.DataFrame(records)


def contrast_rows(
    predictions: dict[tuple[str, int], pd.DataFrame], replicates: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, left, right in CONTRASTS:
        observed = [
            macro_f1(predictions[(left, current_seed)])
            - macro_f1(predictions[(right, current_seed)])
            for current_seed in SEEDS
        ]
        paired_matrices = {}
        for current_seed in SEEDS:
            left_frame = predictions[(left, current_seed)]
            right_frame = predictions[(right, current_seed)][["sample_id", "prediction_id"]]
            paired = left_frame.merge(
                right_frame, on="sample_id", suffixes=("_left", "_right"), validate="one_to_one"
            )
            reports, left_matrices = report_confusions(paired, "prediction_id_left")
            _, right_matrices = report_confusions(paired, "prediction_id_right", reports)
            paired_matrices[current_seed] = (left_matrices, right_matrices)
        slots = []
        for _ in range(len(SEEDS)):
            chosen = rng.choice(SEEDS, size=replicates, replace=True)
            values = np.zeros(replicates, dtype=float)
            for current_seed in SEEDS:
                left_matrices, right_matrices = paired_matrices[current_seed]
                report_count = len(left_matrices)
                counts = rng.multinomial(
                    report_count, np.full(report_count, 1.0 / report_count), size=replicates
                )
                left_draws = bootstrap_scores(left_matrices, replicates, rng, counts)
                right_draws = bootstrap_scores(right_matrices, replicates, rng, counts)
                mask = chosen == current_seed
                values[mask] = (left_draws - right_draws)[mask]
            slots.append(values)
        draws = np.mean(np.vstack(slots), axis=0)
        low, high = interval(draws.tolist())
        rows.append({
            "contrast": name, "left_condition": left, "right_condition": right,
            "delta_seed_20261114": observed[0], "delta_seed_20261115": observed[1],
            "delta_seed_20261116": observed[2], "delta_mean": float(np.mean(observed)),
            "delta_sample_sd": float(np.std(observed, ddof=1)),
            "hierarchical_ci95_low": low, "hierarchical_ci95_high": high,
            "bootstrap_replicates": replicates,
            "multiplicity_adjusted": False,
            "analysis_positioning": "posthoc_near_duplicate_sensitivity",
        })
    return pd.DataFrame(rows)


def tar_csv(handle: tarfile.TarFile, suffix: str) -> pd.DataFrame:
    matches = [member for member in handle.getmembers() if member.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one tar member ending with {suffix}, found {len(matches)}")
    extracted = handle.extractfile(matches[0])
    if extracted is None:
        raise RuntimeError(f"Could not read tar member: {matches[0].name}")
    return pd.read_csv(extracted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--phase4a-zip", type=Path, required=True)
    parser.add_argument("--phase4b-zip", type=Path, required=True)
    parser.add_argument("--phase5-root", type=Path, required=True)
    parser.add_argument("--phase6-freeze", type=Path, required=True)
    parser.add_argument("--phase6-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261121)
    args = parser.parse_args()

    audit = read_json(args.audit)
    validation_excluded = {
        row["later_sample_id"] for row in audit["affected_rows"]
        if row["later_split"] == "validation"
    }
    test_excluded = {
        row["later_sample_id"] for row in audit["affected_rows"]
        if row["later_split"] == "test"
    }
    if len(validation_excluded) != 44 or len(test_excluded) != 27:
        raise RuntimeError("Audit exclusions do not match the independently reproduced counts")

    phase4a = zip_run_table(args.phase4a_zip, validation_excluded)
    phase4a["candidate_id"] = phase4a["adapter_candidate_id"]
    original_a_max = phase4a["original_macro_f1"].max()
    filtered_a_max = phase4a["filtered_macro_f1"].max()
    original_a_selected = phase4a.loc[phase4a["original_macro_f1"].idxmax(), "candidate_id"]
    filtered_a_selected = phase4a.loc[phase4a["filtered_macro_f1"].idxmax(), "candidate_id"]
    phase4a["filtered_efficiency_set"] = phase4a["filtered_macro_f1"].ge(filtered_a_max - 0.005)

    phase4b = zip_run_table(args.phase4b_zip, validation_excluded)
    selected_lr = {}
    for method, group in phase4b.groupby("method"):
        maximum = group["filtered_macro_f1"].max()
        tie = group.loc[group["filtered_macro_f1"].ge(maximum - 1e-4)]
        selected_lr[method] = float(tie["learning_rate"].min())
    phase4b["filtered_selected"] = phase4b.apply(
        lambda row: bool(np.isclose(row["learning_rate"], selected_lr[row["method"]])), axis=1
    )

    freeze = read_json(args.phase6_freeze)
    phase5_rows = []
    validation_predictions = {}
    for spec in freeze["runs"]:
        path = args.phase5_root / spec["run_relative_path"] / "validation_predictions.csv"
        frame = pd.read_csv(path)
        filtered = filter_ids(frame, validation_excluded)
        validation_predictions[(spec["condition_id"], int(spec["seed"]))] = filtered
        phase5_rows.append({
            "run_id": spec["run_id"], "condition_id": spec["condition_id"],
            "seed": spec["seed"], "method": spec["method"],
            "text_condition": spec["text_condition"], "loss": spec["loss"],
            "original_rows": len(frame), "filtered_rows": len(filtered),
            "original_macro_f1": macro_f1(frame), "filtered_macro_f1": macro_f1(filtered),
            "delta_filtered_minus_original": macro_f1(filtered) - macro_f1(frame),
        })
    phase5_runs = pd.DataFrame(phase5_rows)
    phase5_conditions = condition_summary(phase5_runs, "validation")

    test_rows = []
    test_predictions = {}
    baseline_rows = []
    with tarfile.open(args.phase6_bundle, "r:gz") as handle:
        for spec in freeze["runs"]:
            suffix = (
                f"phase6b_retrospective_test_official_retry1_v1_1/{spec['run_id']}"
                "/test_predictions.csv"
            )
            frame = tar_csv(handle, suffix)
            filtered = filter_ids(frame, test_excluded)
            test_predictions[(spec["condition_id"], int(spec["seed"]))] = filtered
            test_rows.append({
                "run_id": spec["run_id"], "condition_id": spec["condition_id"],
                "seed": spec["seed"], "method": spec["method"],
                "text_condition": spec["text_condition"], "loss": spec["loss"],
                "original_rows": len(frame), "filtered_rows": len(filtered),
                "original_macro_f1": macro_f1(frame), "filtered_macro_f1": macro_f1(filtered),
                "delta_filtered_minus_original": macro_f1(filtered) - macro_f1(frame),
            })
        baseline_suffixes = [
            ("tfidf_balanced_lr", "original"), ("tfidf_balanced_lr", "masked"),
            ("frozen_minilm_balanced_lr", "original"),
            ("frozen_minilm_balanced_lr", "masked"),
            ("frozen_sec_bert_balanced_lr", "original"),
            ("frozen_sec_bert_balanced_lr", "masked"),
        ]
        for method, text_condition in baseline_suffixes:
            run_id = f"baseline__{method}__{text_condition}__phase1v1_1"
            frame = tar_csv(
                handle,
                f"phase6b_retrospective_test_baselines_v1_1/{run_id}/test_predictions.csv",
            )
            filtered = filter_ids(frame, test_excluded)
            baseline_rows.append({
                "run_id": run_id, "method": method, "text_condition": text_condition,
                "original_rows": len(frame), "filtered_rows": len(filtered),
                "original_macro_f1": macro_f1(frame), "filtered_macro_f1": macro_f1(filtered),
                "delta_filtered_minus_original": macro_f1(filtered) - macro_f1(frame),
            })

    test_runs = pd.DataFrame(test_rows)
    test_conditions = condition_summary(test_runs, "test")
    test_contrasts = contrast_rows(
        test_predictions, args.bootstrap_replicates, args.bootstrap_seed
    )
    baselines = pd.DataFrame(baseline_rows)
    remaining_reports = int(next(iter(test_predictions.values()))["report_id"].astype(str).nunique())

    args.output_root.mkdir(parents=True, exist_ok=True)
    phase4a.to_csv(args.output_root / "phase4a_validation_near_duplicate_sensitivity_v1_1.csv", index=False)
    phase4b.to_csv(args.output_root / "phase4b_validation_near_duplicate_sensitivity_v1_1.csv", index=False)
    phase5_runs.to_csv(args.output_root / "phase5_validation_seed_near_duplicate_sensitivity_v1_1.csv", index=False)
    phase5_conditions.to_csv(args.output_root / "phase5_validation_condition_near_duplicate_sensitivity_v1_1.csv", index=False)
    test_runs.to_csv(args.output_root / "phase6_test_seed_near_duplicate_sensitivity_v1_1.csv", index=False)
    test_conditions.to_csv(args.output_root / "phase6_test_condition_near_duplicate_sensitivity_v1_1.csv", index=False)
    test_contrasts.to_csv(args.output_root / "phase6_test_contrast_near_duplicate_sensitivity_v1_1.csv", index=False)
    baselines.to_csv(args.output_root / "phase6_test_baseline_near_duplicate_sensitivity_v1_1.csv", index=False)

    summary = {
        "version": "phase6-posthoc-near-duplicate-sensitivity-v1.1",
        "status": "diagnostic_after_retrospective_test_access",
        "confirmatory_claim": False,
        "retraining_performed": False,
        "checkpoint_or_threshold_changed": False,
        "validation_rows_original": 927,
        "validation_rows_excluded": len(validation_excluded),
        "validation_rows_remaining": 927 - len(validation_excluded),
        "test_rows_original": 890,
        "test_rows_excluded": len(test_excluded),
        "test_rows_remaining": 890 - len(test_excluded),
        "test_reports_remaining": remaining_reports,
        "phase4a_original_selected": original_a_selected,
        "phase4a_filtered_selected_using_saved_best_checkpoints": filtered_a_selected,
        "phase4a_selection_changed": original_a_selected != filtered_a_selected,
        "phase4a_filtered_efficiency_set": phase4a.loc[
            phase4a["filtered_efficiency_set"], "candidate_id"
        ].tolist(),
        "phase4b_filtered_selected_learning_rates": selected_lr,
        "epoch_selection_recheck_possible_from_saved_artifacts": False,
        "epoch_selection_limitation": (
            "Only the previously selected best-checkpoint predictions were saved; per-epoch "
            "predictions are unavailable, so the best epoch cannot be reselected after filtering."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "inspection_report",
        "multiplicity_adjustment": "none; intervals remain descriptive and unadjusted",
    }
    (args.output_root / "sensitivity_summary_v1_1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

