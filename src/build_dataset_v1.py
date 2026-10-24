"""Build the versioned PCAOB Selective Routing Dataset v1 release.

The release is a research dataset, not a new source-data download. It freezes the
official snapshot, cleaning/taxonomy policies, grouped and temporal partitions,
and observed small-model/Jev comparative outcomes in one auditable package.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/dataset_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_successful_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "ok":
            rows.append(item)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No successful responses in {path}")
    return frame.drop_duplicates(["sample_id", "prompt_bundle_sha256"], keep="last")


def comparative_outcomes(config: dict) -> pd.DataFrame:
    llm_config = json.loads((ROOT / config["llm_config"]).read_text(encoding="utf-8"))
    partitions = []
    development = pd.read_csv(ROOT / config["development_comparative_outcomes"])
    development["outcome_partition"] = "development_oof"
    development["outcome_usage"] = "router_training_crossfit"
    development["llm_model"] = development["response_model"]
    development["llm_probabilities_json"] = development["probabilities"].map(canonical_json)
    development["llm_usage_json"] = development["usage"].map(canonical_json)
    partitions.append(
        development[
            [
                "sample_id", "outcome_partition", "outcome_usage", "small_prediction",
                "small_confidence", "small_correct", "llm_prediction", "llm_confidence",
                "llm_correct", "routing_outcome", "llm_model", "latency_ms",
                "llm_usage_json", "llm_probabilities_json", "prompt_bundle_sha256",
            ]
        ]
    )
    specs = [
        (
            "calibration",
            ROOT / llm_config["calibration_candidate_table"],
            ROOT / llm_config["calibration_response_log"],
            "routing_development",
        ),
        (
            "test",
            ROOT / llm_config["candidate_table"],
            ROOT / llm_config["response_log"],
            "retrospective_evaluation_only",
        ),
    ]
    for partition, candidate_path, response_path, usage in specs:
        candidates = pd.read_csv(candidate_path)
        responses = load_successful_jsonl(response_path)
        joined = candidates.merge(
            responses,
            on=["sample_id", "prompt_bundle_sha256"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_response"),
        )
        if len(joined) != len(candidates):
            raise RuntimeError(
                f"{partition}: expected {len(candidates)} successful responses, found {len(joined)}"
            )
        joined["outcome_partition"] = partition
        joined["outcome_usage"] = usage
        partitions.append(joined)

    development_frame = partitions[0]
    frame = pd.concat(partitions[1:], ignore_index=True)
    frame["small_correct"] = frame["small_prediction"].eq(frame["target_level1"])
    frame["llm_prediction"] = frame["prediction"]
    frame["llm_confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["llm_correct"] = frame["llm_prediction"].eq(frame["target_level1"])
    frame["routing_outcome"] = "neutral_both_wrong"
    frame.loc[frame["small_correct"] & frame["llm_correct"], "routing_outcome"] = "neutral_both_correct"
    frame.loc[~frame["small_correct"] & frame["llm_correct"], "routing_outcome"] = "beneficial"
    frame.loc[frame["small_correct"] & ~frame["llm_correct"], "routing_outcome"] = "harmful"
    frame["llm_probabilities_json"] = frame["probabilities"].map(canonical_json)
    frame["llm_usage_json"] = frame["usage"].map(canonical_json)
    response_model = frame.get("response_model", pd.Series("", index=frame.index)).fillna("")
    requested_model = frame.get("requested_model", pd.Series("", index=frame.index)).fillna("")
    frame["llm_model"] = response_model.where(response_model.ne(""), requested_model)
    frame = frame[
        [
            "sample_id",
            "outcome_partition",
            "outcome_usage",
            "small_prediction",
            "small_confidence",
            "small_correct",
            "llm_prediction",
            "llm_confidence",
            "llm_correct",
            "routing_outcome",
            "llm_model",
            "latency_ms",
            "llm_usage_json",
            "llm_probabilities_json",
            "prompt_bundle_sha256",
        ]
    ]
    return pd.concat([development_frame, frame], ignore_index=True)


def temporal_partition(year: int, policy: dict) -> str:
    if year <= int(policy["development_through_inspection_year"]):
        return "development"
    if year == int(policy["calibration_inspection_year"]):
        return "calibration"
    return "retrospective_test"


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.fillna("").astype(str)
    return "\n".join(
        [
            "| " + " | ".join(shown.columns) + " |",
            "| " + " | ".join(["---"] * len(shown.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()],
        ]
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    release_dir = ROOT / "data/releases" / config["release_slug"]
    release_dir.mkdir(parents=True, exist_ok=True)

    model = pd.read_csv(ROOT / config["model_table"])
    findings = pd.read_csv(ROOT / config["finding_table"])
    source_extra = findings[
        [
            "sample_id",
            "Inspection Type",
            "Country",
            "Global Network",
            "Inspection Report Date",
            "Inspection Report Section",
            "Firm-Identified Risk Assessment",
            "Audits Affected by the Deficiencies Identified in Part I.A",
            "Classification of Audits with Part I.A Deficiencies",
            "source_row_count",
            "audit_area_count",
            "standard_paragraphs",
        ]
    ].drop_duplicates("sample_id")
    data = model.merge(source_extra, on="sample_id", how="left", validate="one_to_one")
    data = data.rename(columns={"split": "group_split", "clean_eval": "group_split_clean_eval"})

    if not data["audit_area_count"].eq(1).all():
        raise RuntimeError("The v1 release must contain explicit single-label findings only.")
    if data["sample_id"].duplicated().any():
        raise RuntimeError("sample_id is not unique.")
    if data.groupby("report_id")["group_split"].nunique().max() != 1:
        raise RuntimeError("A report crosses group-split partitions.")

    years = pd.to_numeric(data["Inspection Year"], errors="raise").astype(int)
    policy = config["temporal_policy"]
    data["temporal_split"] = years.map(lambda year: temporal_partition(year, policy))
    data["temporal_confirmatory_status"] = policy["current_snapshot_status"]

    duplicated = data["description_norm"].duplicated(keep=False)
    data["exact_duplicate_flag"] = duplicated
    data["exact_duplicate_group_id"] = data["description_norm"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    ).where(duplicated, "")
    near = pd.read_csv(ROOT / config["near_duplicate_table"])
    near_ids = set(near.get("sample_id", pd.Series(dtype=str)).astype(str)) | set(
        near.get("neighbor_sample_id", pd.Series(dtype=str)).astype(str)
    )
    data["near_duplicate_flag"] = data["sample_id"].isin(near_ids)

    outcomes = comparative_outcomes(config)
    data = data.merge(outcomes, on="sample_id", how="left", validate="one_to_one")
    data["routing_outcome"] = data["routing_outcome"].fillna("not_observed")
    data["outcome_usage"] = data["outcome_usage"].fillna("not_available")
    data["outcome_partition"] = data["outcome_partition"].fillna("")
    data["small_model_id"] = config["small_model_id"]
    data["llm_system_id"] = config["llm_system_id"]

    observed = data["routing_outcome"].ne("not_observed")
    expected = pd.Series("neutral_both_wrong", index=data.index)
    expected.loc[data["small_correct"].eq(True) & data["llm_correct"].eq(True)] = "neutral_both_correct"
    expected.loc[data["small_correct"].eq(False) & data["llm_correct"].eq(True)] = "beneficial"
    expected.loc[data["small_correct"].eq(True) & data["llm_correct"].eq(False)] = "harmful"
    if not data.loc[observed, "routing_outcome"].eq(expected.loc[observed]).all():
        raise RuntimeError("Comparative outcome labels are inconsistent with correctness fields.")

    ordered = [
        "sample_id", "report_id", "Registration ID", "Firm Names", "Country", "Global Network",
        "Inspection Type", "Inspection Year", "Inspection Report Date", "Inspection Report Section",
        "Issuer Reference Key", "Finding Count", "source_row_count", "audit_area_count",
        "audit_areas", "target_level2", "target_level1", "standard_refs", "standard_paragraphs",
        "Firm-Identified Risk Assessment", "Audits Affected by the Deficiencies Identified in Part I.A",
        "Classification of Audits with Part I.A Deficiencies", "text_original", "text_context_v2",
        "text_context_masked_v2", "description_norm", "context_needs_context", "context_added",
        "context_quality", "group_split", "group_split_clean_eval", "leakage_reason",
        "temporal_split", "temporal_confirmatory_status", "exact_duplicate_flag",
        "exact_duplicate_group_id", "near_duplicate_flag", "outcome_partition", "outcome_usage",
        "small_model_id", "small_prediction", "small_confidence", "small_correct", "llm_system_id",
        "llm_model", "llm_prediction", "llm_confidence", "llm_correct", "routing_outcome",
        "latency_ms", "llm_usage_json", "llm_probabilities_json", "prompt_bundle_sha256",
    ]
    data = data[ordered].sort_values("sample_id").reset_index(drop=True)
    observed = data["routing_outcome"].ne("not_observed")
    dataset_path = release_dir / "pcaob_selective_routing_v1.csv"
    data.to_csv(dataset_path, index=False, encoding="utf-8")

    taxonomy_path = release_dir / "taxonomy_v2_mapping.csv"
    shutil.copyfile(ROOT / config["taxonomy_mapping"], taxonomy_path)

    split_stats = (
        data.groupby(["group_split", "temporal_split"], dropna=False)
        .agg(samples=("sample_id", "size"), reports=("report_id", "nunique"), firms=("Registration ID", "nunique"))
        .reset_index()
    )
    split_stats.to_csv(release_dir / "split_statistics.csv", index=False, encoding="utf-8")
    outcome_stats = (
        data.loc[observed]
        .groupby(["outcome_partition", "routing_outcome"], dropna=False)
        .size().rename("samples").reset_index()
    )
    outcome_stats.to_csv(release_dir / "routing_outcome_statistics.csv", index=False, encoding="utf-8")

    schema = pd.DataFrame(
        [{"column": column, "dtype": str(data[column].dtype), "non_null": int(data[column].notna().sum()),
          "unique": int(data[column].nunique(dropna=True))} for column in data.columns]
    )
    schema.to_csv(release_dir / "schema.csv", index=False, encoding="utf-8")

    source_manifest = json.loads((ROOT / config["source_manifest"]).read_text(encoding="utf-8"))
    future_protocol = {
        "protocol_version": "1.0.0",
        "baseline_snapshot_sha256": source_manifest["sha256"],
        "baseline_downloaded_at_utc": source_manifest["downloaded_at_utc"],
        "eligibility_rule": policy["confirmatory_rule"],
        "prohibited_actions_before_unblinding": [
            "Do not tune taxonomy rules, model hyperparameters, routing features, or thresholds on future-holdout outcomes.",
            "Do not move reports already present in the baseline snapshot into the confirmatory holdout.",
            "Do not split findings from one report across partitions."
        ],
        "unblinding_condition": "Freeze code, taxonomy, small model, LLM prompt/version, and routing policy before scoring the future holdout."
    }
    (release_dir / "future_temporal_holdout_protocol.json").write_text(
        json.dumps(future_protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    raw_findings = len(findings)
    observed_counts = data.loc[observed, "routing_outcome"].value_counts()
    dataset_card = f"""# PCAOB Selective Routing Dataset v1

Version: `{config['version']}`  
Source snapshot date: `2026-09-27`  
Status: research release; the current temporal test is retrospective, not confirmatory.

## Purpose

This release supports reproducible study of selective routing between a low-cost text classifier and an LLM on PCAOB Part I.A inspection findings. It unifies source provenance, finding-level aggregation, taxonomy, leakage controls, report-grouped partitions, a retrospective temporal partition, and observed comparative outcomes.

## Population and cleaning

- Official raw rows are aggregated into {raw_findings:,} finding-level records because one finding may cite multiple standards/paragraphs.
- The modeling release contains {len(data):,} explicit single-label findings.
- Exact normalized texts with conflicting deterministic targets were excluded upstream.
- Exact and audited near-duplicate indicators are retained. Evaluation rows duplicating earlier partitions are marked by `group_split_clean_eval = false`, not silently deleted.
- Context is added only from the same report, issuer reference, and official audit area when a finding contains an unresolved reference to earlier text.

## Labels

- `target_level2`: normalized official PCAOB audit-area label.
- `target_level1`: one of {data['target_level1'].nunique()} frozen coarse classes assigned from label names by `configs/taxonomy_v2.json`.
- The mapping is a reproducible research taxonomy, not an official PCAOB taxonomy or expert-consensus standard.

## Partitions

- `group_split`: existing seeded report-grouped train/validation/calibration/test split. A report never crosses partitions.
- `temporal_split`: development through inspection year 2023, calibration in 2024, and retrospective test from 2025.
- Every row in this snapshot is marked `retrospective_only_not_confirmatory`, because models and analyses predate this release packaging.
- Confirmatory evidence must follow `future_temporal_holdout_protocol.json` and use only reports first appearing after the frozen snapshot.

## Comparative routing outcomes

Observed outcomes exist for {int(observed.sum()):,} clean samples on which both systems were run, including 4,613 cross-fitted development samples, 608 calibration samples, and 890 retrospective-test samples:

- beneficial: {int(observed_counts.get('beneficial', 0)):,}
- harmful: {int(observed_counts.get('harmful', 0)):,}
- neutral, both correct: {int(observed_counts.get('neutral_both_correct', 0)):,}
- neutral, both wrong: {int(observed_counts.get('neutral_both_wrong', 0)):,}

Cross-fitted development outcomes are marked `router_training_crossfit`; calibration outcomes may be used only for threshold/operating-point selection. Existing test outcomes are marked `retrospective_evaluation_only` and must not be used to train or tune a claimed confirmatory router.

## Leakage and independence

The independent replication unit for uncertainty estimation is the inspection report, not each finding. Multiple findings from one report are nested observations. Report-cluster bootstrap or an equivalent grouped analysis is required.

## Source and reuse

Source: {source_manifest['source_page']}  
Snapshot SHA-256: `{source_manifest['sha256']}`

The underlying material is official public PCAOB inspection data. Users remain responsible for checking PCAOB terms and for describing this derived research dataset accurately. LLM predictions are model outputs, not PCAOB annotations or professional audit judgments.
"""
    (release_dir / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")

    files = [
        dataset_path, taxonomy_path, release_dir / "split_statistics.csv",
        release_dir / "routing_outcome_statistics.csv", release_dir / "schema.csv",
        release_dir / "future_temporal_holdout_protocol.json", release_dir / "DATASET_CARD.md",
    ]
    manifest = {
        "dataset_name": config["dataset_name"],
        "version": config["version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": source_manifest,
        "builder_config_sha256": sha256_file(CONFIG_PATH),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "counts": {
            "samples": len(data),
            "reports": int(data["report_id"].nunique()),
            "firms": int(data["Registration ID"].nunique()),
            "observed_comparative_outcomes": int(observed.sum()),
        },
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files},
    }
    (release_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(markdown_table(outcome_stats))
    print(f"Release: {release_dir}")


if __name__ == "__main__":
    main()
