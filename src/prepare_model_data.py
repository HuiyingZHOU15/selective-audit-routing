"""Freeze taxonomy v1 and create leakage-audited report-grouped splits."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "data" / "interim" / "findings.csv"
NEAR_DUPLICATES = ROOT / "results" / "tables" / "near_duplicate_report.csv"
CONFIG = ROOT / "configs" / "taxonomy_v1.json"
OUTPUT = ROOT / "data" / "processed" / "model_dataset_v1.csv"
SUMMARY = ROOT / "results" / "reports" / "split_summary_v1.md"
TABLE = ROOT / "results" / "tables" / "split_label_counts_v1.csv"

SPLIT_PRIORITY = {"train": 0, "validation": 1, "calibration": 2, "test": 3}


def choose_group_split(
    frame: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a grouped split with good size and class-distribution balance."""
    labels = sorted(frame["target_label"].unique())
    global_dist = frame["target_label"].value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_fraction, random_state=seed + offset)
        keep_idx, hold_idx = next(splitter.split(frame, groups=frame["report_id"]))
        hold = frame.iloc[hold_idx]
        hold_dist = hold["target_label"].value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
        missing_penalty = 10 * sum(label not in set(hold["target_label"]) for label in labels)
        size_penalty = abs(len(hold) / len(frame) - holdout_fraction) * 8
        distribution_penalty = float(np.abs(global_dist - hold_dist).sum())
        score = missing_penalty + size_penalty + distribution_penalty
        if best is None or score < best[0]:
            best = (score, keep_idx, hold_idx)
    assert best is not None
    return best[1], best[2]


def mask_text(text: str, label_pattern: re.Pattern[str]) -> str:
    result = str(text)
    result = label_pattern.sub("[AUDIT_AREA]", result)
    result = re.sub(r"\bAS\s+\d{3,4}(?:\.\d+)?\b", "[STANDARD_REF]", result, flags=re.IGNORECASE)
    result = re.sub(r"\(\s*\[STANDARD_REF\][^)]*\)", "[STANDARD_REF]", result)
    return re.sub(r"\s+", " ", result).strip()


def assign_splits(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    seed = int(config["split_seed"])
    keep, test = choose_group_split(frame, 0.15, seed)
    remaining = frame.iloc[keep].copy()
    test_frame = frame.iloc[test].copy()

    keep, calibration = choose_group_split(remaining, 0.10 / 0.85, seed + 10_000)
    calibration_frame = remaining.iloc[calibration].copy()
    remaining = remaining.iloc[keep].copy()

    train, validation = choose_group_split(remaining, 0.15 / 0.75, seed + 20_000)
    train_frame = remaining.iloc[train].copy()
    validation_frame = remaining.iloc[validation].copy()

    pieces = []
    for name, part in (
        ("train", train_frame),
        ("validation", validation_frame),
        ("calibration", calibration_frame),
        ("test", test_frame),
    ):
        part = part.copy()
        part["split"] = name
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def mark_cross_split_leakage(frame: pd.DataFrame, near_pairs: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["clean_eval"] = True
    result["leakage_reason"] = ""
    split_by_id = result.set_index("sample_id")["split"].to_dict()
    row_by_id = {sample_id: index for index, sample_id in result["sample_id"].items()}

    def mark_later(sample_ids: list[str], reason: str) -> None:
        present = [sample_id for sample_id in sample_ids if sample_id in split_by_id]
        if len({split_by_id[sample_id] for sample_id in present}) <= 1:
            return
        earliest = min(SPLIT_PRIORITY[split_by_id[sample_id]] for sample_id in present)
        for sample_id in present:
            if SPLIT_PRIORITY[split_by_id[sample_id]] > earliest:
                index = row_by_id[sample_id]
                result.at[index, "clean_eval"] = False
                old = result.at[index, "leakage_reason"]
                result.at[index, "leakage_reason"] = " | ".join(filter(None, [old, reason]))

    for _, group in result.groupby("description_norm"):
        if len(group) > 1:
            mark_later(group["sample_id"].tolist(), "exact_duplicate_across_split")
    if not near_pairs.empty:
        for row in near_pairs.itertuples():
            mark_later([row.sample_id, row.neighbor_sample_id], "near_duplicate_across_split")
    return result


def simple_markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.reset_index().fillna("").astype(str)
    header = "| " + " | ".join(shown.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()]
    return "\n".join([header, divider, *rows])


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    findings = pd.read_csv(FINDINGS)
    near_pairs = pd.read_csv(NEAR_DUPLICATES)

    model = findings.loc[findings["audit_area_count"].eq(1)].copy()
    model["audit_area_norm"] = model["audit_areas"].replace(config["alias_map"])
    fixed = set(config["target_labels"]) - {"Other"}
    model["target_label"] = model["audit_area_norm"].where(model["audit_area_norm"].isin(fixed), "Other")

    # A byte-identical normalized input with two target labels is not learnable
    # as a deterministic single-label task. Exclude it before splitting.
    conflicts = model.groupby("description_norm")["target_label"].transform("nunique").gt(1)
    model["excluded_label_conflict"] = conflicts
    conflict_count = int(conflicts.sum())
    model = model.loc[~conflicts].copy()

    all_phrases = sorted(
        {str(value) for value in findings["audit_areas"].str.split(" | ").explode().dropna()},
        key=len,
        reverse=True,
    )
    label_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(phrase) for phrase in all_phrases) + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    model["text_original"] = model["Description in the Firm's Inspection Report"].astype(str)
    model["text_masked"] = model["text_original"].map(lambda text: mask_text(text, label_pattern))
    model = assign_splits(model, config)
    model = mark_cross_split_leakage(model, near_pairs)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_COLUMNS = [
        "sample_id",
        "report_id",
        "Registration ID",
        "Firm Names",
        "Inspection Year",
        "Issuer Reference Key",
        "audit_areas",
        "audit_area_norm",
        "target_label",
        "split",
        "clean_eval",
        "leakage_reason",
        "description_norm",
        "text_original",
        "text_masked",
        "standard_refs",
    ]
    model[OUTPUT_COLUMNS].to_csv(OUTPUT, index=False, encoding="utf-8")

    counts = (
        model.groupby(["split", "target_label"])
        .agg(total=("sample_id", "size"), clean=("clean_eval", "sum"), reports=("report_id", "nunique"))
        .reset_index()
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(TABLE, index=False, encoding="utf-8")

    split_stats = model.groupby("split").agg(
        samples=("sample_id", "size"),
        clean_samples=("clean_eval", "sum"),
        reports=("report_id", "nunique"),
        firms=("Registration ID", "nunique"),
    )
    report_sets = {name: set(group["report_id"]) for name, group in model.groupby("split")}
    report_overlap = sum(
        len(report_sets[a] & report_sets[b])
        for i, a in enumerate(report_sets)
        for b in list(report_sets)[i + 1 :]
    )
    summary = f"""# Taxonomy and Split Summary v1

## Frozen taxonomy

- Input population: explicit single-label findings only.
- Alias merge: `Business Combination` → `Business Combinations`.
- Target: 12 named classes plus `Other`.
- Exact normalized inputs with conflicting target labels excluded: {conflict_count:,} samples.
- Label mapping: `configs/taxonomy_v1.json`.

## Split policy

- Report-grouped 60/15/10/15 train/validation/calibration/test design.
- Split seed: {config['split_seed']}.
- Report overlap across splits: {report_overlap}.
- Evaluation samples that duplicate an earlier split at exact or audited ≥0.90 near-duplicate level are retained for traceability but marked `clean_eval = false`.

## Counts

{simple_markdown_table(split_stats)}

## Leakage-controlled evaluation

Headline validation, calibration, and test metrics must use only `clean_eval = true`. A secondary unfiltered result may be reported to quantify the optimism caused by template leakage.
"""
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(summary, encoding="utf-8")
    print(split_stats.to_string())


if __name__ == "__main__":
    main()
