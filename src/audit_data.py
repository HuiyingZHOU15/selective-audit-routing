"""Create the Phase-1 data audit for the PCAOB Part I.A flat file."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_audit.json"
INTERIM_DIR = ROOT / "data" / "interim"
TABLES_DIR = ROOT / "results" / "tables"
REPORTS_DIR = ROOT / "results" / "reports"

DESCRIPTION = "Description in the Firm's Inspection Report"
LABEL = "Audit Area"
STANDARD = "Auditing Standard"
PARAGRAPH = "Paragraph of the Auditing Standard"


def normalize_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: object) -> str:
    text = normalize_space(value).lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_report_id(frame: pd.DataFrame) -> pd.Series:
    parts = ["Registration ID", "Inspection Year", "Inspection Report Date"]
    return frame[parts].fillna("").astype(str).agg("|".join, axis=1)


def aggregate_findings(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["report_id"] = make_report_id(frame)
    frame["description_norm"] = frame[DESCRIPTION].map(normalize_text)

    # A finding can occupy multiple rows because each cited standard/paragraph
    # is represented separately. These columns identify the finding while the
    # standard fields are aggregated below.
    finding_key = [
        "report_id",
        "Issuer Reference Key",
        "Finding Count",
        "Audits Affected by the Deficiencies Identified in Part I.A",
        "Classification of Audits with Part I.A Deficiencies",
        "Firm-Identified Risk Assessment",
        "description_norm",
    ]
    metadata = [
        "Inspection Type",
        "Registration ID",
        "Firm Names",
        "Inspection Year",
        "Country",
        "Global Network",
        "Inspection Report Date",
        "Inspection Report Section",
        "Firm played a role but was not the lead auditor",
        DESCRIPTION,
    ]

    def joined_unique(series: pd.Series) -> str:
        values = sorted({normalize_space(value) for value in series.dropna() if normalize_space(value)})
        return " | ".join(values)

    grouped = frame.groupby(finding_key, dropna=False, sort=False)
    findings = grouped.agg(
        **{column: (column, "first") for column in metadata},
        audit_areas=(LABEL, joined_unique),
        standard_refs=(STANDARD, joined_unique),
        standard_paragraphs=(PARAGRAPH, joined_unique),
        source_row_count=(DESCRIPTION, "size"),
    ).reset_index()
    findings["audit_area_count"] = findings["audit_areas"].map(
        lambda value: len([item for item in str(value).split(" | ") if item])
    )
    findings.insert(0, "sample_id", [f"PCAOB-PIA-{i:06d}" for i in range(1, len(findings) + 1)])
    return findings


def schema_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        non_empty = series.dropna().astype(str).map(str.strip).ne("").sum()
        examples = [normalize_space(v)[:120] for v in series.dropna().astype(str).unique()[:3]]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": len(series),
                "non_empty": int(non_empty),
                "missing_or_empty": int(len(series) - non_empty),
                "missing_pct": round(100 * (len(series) - non_empty) / max(len(series), 1), 2),
                "unique_non_null": int(series.dropna().nunique()),
                "examples": " || ".join(examples),
            }
        )
    return pd.DataFrame(rows)


def near_duplicate_pairs(findings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    texts = findings["description_norm"].fillna("").tolist()
    if len(texts) < 2:
        return pd.DataFrame(columns=["sample_id", "neighbor_sample_id", "similarity"])
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000)
    matrix = vectorizer.fit_transform(texts)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=-1)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    rows = []
    for i in range(len(findings)):
        j = int(indices[i, 1])
        similarity = float(1.0 - distances[i, 1])
        if similarity >= threshold and i < j:
            rows.append(
                {
                    "sample_id": findings.iloc[i]["sample_id"],
                    "neighbor_sample_id": findings.iloc[j]["sample_id"],
                    "similarity": round(similarity, 5),
                    "same_label_set": findings.iloc[i]["audit_areas"] == findings.iloc[j]["audit_areas"],
                    "same_report": findings.iloc[i]["report_id"] == findings.iloc[j]["report_id"],
                    "labels": findings.iloc[i]["audit_areas"],
                    "neighbor_labels": findings.iloc[j]["audit_areas"],
                    "text": findings.iloc[i][DESCRIPTION],
                    "neighbor_text": findings.iloc[j][DESCRIPTION],
                }
            )
    return pd.DataFrame(rows).sort_values("similarity", ascending=False) if rows else pd.DataFrame()


def leakage_table(findings: pd.DataFrame) -> pd.DataFrame:
    stop = {"and", "or", "the", "of", "to", "in", "for", "a", "an", "with", "related", "other"}
    rows = []
    standard_pattern = re.compile(r"\bAS\s+\d{3,4}\b", flags=re.IGNORECASE)
    for _, row in findings.iterrows():
        for label in str(row["audit_areas"]).split(" | "):
            text_norm = normalize_text(row[DESCRIPTION])
            label_norm = normalize_text(label)
            label_tokens = {token for token in label_norm.split() if token not in stop and len(token) >= 3}
            text_tokens = set(text_norm.split())
            overlap = sorted(label_tokens & text_tokens)
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "audit_area": label,
                    "finding_label_count": row["audit_area_count"],
                    "exact_label_phrase_in_text": bool(label_norm and label_norm in text_norm),
                    "label_content_token_count": len(label_tokens),
                    "overlapping_label_tokens": " | ".join(overlap),
                    "label_token_coverage": round(len(overlap) / max(len(label_tokens), 1), 4),
                    "standard_reference_in_text": bool(standard_pattern.search(str(row[DESCRIPTION]))),
                    "text": row[DESCRIPTION],
                }
            )
    return pd.DataFrame(rows)


def to_markdown_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    shown = frame.head(limit) if limit else frame
    if shown.empty:
        return "_No rows._"
    safe = shown.fillna("").astype(str).map(lambda value: value.replace("|", "\\|").replace("\n", " "))
    header = "| " + " | ".join(safe.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(safe.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in safe.to_numpy().tolist()]
    return "\n".join([header, divider, *body])


def write_reports(
    raw: pd.DataFrame,
    findings: pd.DataFrame,
    schema: pd.DataFrame,
    label_counts: pd.DataFrame,
    duplicates: pd.DataFrame,
    near_duplicates: pd.DataFrame,
    leakage: pd.DataFrame,
    config: dict,
    snapshot: Path,
) -> None:
    report_count = findings["report_id"].nunique()
    firm_count = findings["Registration ID"].nunique()
    year_min = int(pd.to_numeric(findings["Inspection Year"], errors="coerce").min())
    year_max = int(pd.to_numeric(findings["Inspection Year"], errors="coerce").max())
    multirow = int((findings["source_row_count"] > 1).sum())

    schema_report = f"""# PCAOB Part I.A Schema Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Snapshot

- File: `{snapshot.relative_to(ROOT).as_posix()}`
- SHA-256: `{sha256(snapshot)}`
- Encoding used: `Windows-1252`
- Raw rows: {len(raw):,}
- Finding-level rows after aggregating standard references: {len(findings):,}
- Reports: {report_count:,}
- Registered firms: {firm_count:,}
- Inspection years: {year_min}–{year_max}
- Findings represented by more than one raw row: {multirow:,}

## Columns

{to_markdown_table(schema)}
"""
    (REPORTS_DIR / "schema_report.md").write_text(schema_report, encoding="utf-8")

    single_label_count = int(findings["audit_area_count"].eq(1).sum())
    multi_label_count = int(findings["audit_area_count"].gt(1).sum())
    eligible = label_counts[label_counts["single_label_finding_count"] >= config["minimum_class_size"]]
    top12_share = label_counts.head(12)["single_label_finding_count"].sum() / max(single_label_count, 1)
    exact_dup_groups = int(duplicates["duplicate_count"].gt(1).sum()) if not duplicates.empty else 0
    cross_label_exact = int(duplicates["label_set_count"].gt(1).sum()) if not duplicates.empty else 0
    exact_leak = float(leakage["exact_label_phrase_in_text"].mean())
    token_leak = float(leakage["label_token_coverage"].ge(0.5).mean())
    standard_leak = float(leakage["standard_reference_in_text"].mean())

    reasons = []
    if len(eligible) >= config["target_label_count_min"]:
        reasons.append(
            f"GO: {len(eligible)} labels have at least {config['minimum_class_size']} single-label findings."
        )
    else:
        reasons.append(
            f"NO-GO for the proposed label breadth: only {len(eligible)} labels have at least "
            f"{config['minimum_class_size']} single-label findings; use fewer/coarser labels."
        )
    if cross_label_exact:
        reasons.append(f"CAUTION: {cross_label_exact} repeated normalized texts have different label sets.")
    else:
        reasons.append("GO: exact normalized-text duplicates have consistent label sets.")
    if token_leak >= 0.25 or standard_leak >= 0.25:
        reasons.append("CAUTION: leakage-masked text must be a primary evaluation condition.")
    if near_duplicates.empty:
        reasons.append("GO: no cross-sample near-duplicate pairs met the configured similarity threshold.")
    else:
        reasons.append(
            f"CAUTION: {len(near_duplicates)} near-duplicate nearest-neighbor pairs require grouped handling."
        )

    feasibility = f"""# Phase-1 Data Feasibility Decision

## Decision

**Conditional GO** for a grouped, leakage-controlled classification feasibility study.

The dataset is large enough to proceed, but raw CSV rows must not be treated as independent findings. Standard citations create repeated rows, and text/template leakage must be controlled before any headline model result is accepted.

## Evidence

- {len(raw):,} raw rows collapse to {len(findings):,} finding-level samples.
- {report_count:,} reports, {firm_count:,} firms, inspection years {year_min}–{year_max}.
- {len(label_counts):,} non-empty audit-area labels.
- Single-label findings: {single_label_count:,}; multi-label findings: {multi_label_count:,}.
- {len(eligible):,} labels contain at least {config['minimum_class_size']} single-label findings.
- Top 12 labels cover {top12_share:.1%} of single-label findings before creating `Other`.
- Exact normalized-text duplicate groups: {exact_dup_groups:,}; differing label-set groups: {cross_label_exact:,}.
- Near-duplicate pairs at cosine similarity ≥ {config['near_duplicate_threshold']}: {len(near_duplicates):,}.
- Exact full label phrase appears in {exact_leak:.1%} of descriptions.
- At least half of the label's content tokens appear in {token_leak:.1%} of descriptions.
- An explicit `AS ####` reference appears in {standard_leak:.1%} of descriptions.

## Gate assessment

{chr(10).join(f'- {reason}' for reason in reasons)}

## Required rules for Phase 2

1. Use `sample_id` rows from `data/interim/findings.csv`, not raw CSV rows.
2. Keep every `report_id` wholly within one split; additionally test grouping by firm and time.
3. First model only the explicit single-label subset; preserve multi-label findings for a later study.
4. Define a top-label policy from `label_counts.csv`; do not select labels using test performance.
5. Report both original text and a pre-specified leakage-masked version.
6. Fit preprocessing, vocabulary, classifier, and calibration only on their permitted partitions.

## Recommended immediate next step

Inspect the top labels and the sampled leakage/duplicate tables, freeze the label taxonomy, then create a report-grouped train/validation/calibration/test split. Do not call an LLM yet.
"""
    (REPORTS_DIR / "data_feasibility.md").write_text(feasibility, encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    snapshot = ROOT / config["snapshot_file"]
    for directory in (INTERIM_DIR, TABLES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(snapshot, encoding="cp1252", dtype=str, keep_default_na=True)
    raw.columns = [normalize_space(column) for column in raw.columns]
    required = {DESCRIPTION, LABEL, STANDARD, PARAGRAPH, "Registration ID", "Inspection Year"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    findings = aggregate_findings(raw)
    schema = schema_table(raw)
    exploded = findings.assign(audit_area=findings["audit_areas"].str.split(r" \| ")).explode("audit_area")
    exploded["is_single_label"] = exploded["audit_area_count"].eq(1)
    label_counts = (
        exploded.groupby("audit_area", dropna=False)
        .agg(
            finding_count=("sample_id", "size"),
            single_label_finding_count=("is_single_label", "sum"),
            report_count=("report_id", "nunique"),
            firm_count=("Registration ID", "nunique"),
            earliest_year=("Inspection Year", "min"),
            latest_year=("Inspection Year", "max"),
        )
        .reset_index()
        .sort_values("single_label_finding_count", ascending=False)
    )
    label_counts["share_pct"] = (100 * label_counts["finding_count"] / len(findings)).round(2)

    duplicate_groups = (
        findings.groupby("description_norm", dropna=False)
        .agg(
            duplicate_count=("sample_id", "size"),
            label_set_count=("audit_areas", "nunique"),
            label_sets=("audit_areas", lambda s: " || ".join(sorted(set(s.dropna().astype(str))))),
            report_count=("report_id", "nunique"),
            sample_ids=("sample_id", lambda s: " | ".join(s.astype(str))),
        )
        .reset_index()
        .query("duplicate_count > 1")
        .sort_values(["label_set_count", "duplicate_count"], ascending=False)
    )
    near_duplicates = near_duplicate_pairs(findings, float(config["near_duplicate_threshold"]))
    leakage = leakage_table(findings)

    manifest = {
        "source_page": config["source_page"],
        "source_url": config["source_url"],
        "downloaded_at_utc": config["downloaded_at_utc"],
        "snapshot_file": config["snapshot_file"],
        "byte_size": snapshot.stat().st_size,
        "sha256": sha256(snapshot),
        "encoding": "Windows-1252",
        "analysis_generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Keep raw/ immutable. The download may have been created by an elevated
    # process on Windows, so metadata is deliberately stored one level above.
    (ROOT / "data" / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    findings.to_csv(INTERIM_DIR / "findings.csv", index=False, encoding="utf-8")
    schema.to_csv(TABLES_DIR / "schema.csv", index=False, encoding="utf-8")
    label_counts.to_csv(TABLES_DIR / "label_counts.csv", index=False, encoding="utf-8")
    duplicate_groups.to_csv(TABLES_DIR / "duplicate_report.csv", index=False, encoding="utf-8")
    near_duplicates.to_csv(TABLES_DIR / "near_duplicate_report.csv", index=False, encoding="utf-8")
    leakage.to_csv(TABLES_DIR / "leakage_examples.csv", index=False, encoding="utf-8")

    write_reports(
        raw,
        findings,
        schema,
        label_counts,
        duplicate_groups,
        near_duplicates,
        leakage,
        config,
        snapshot,
    )
    print(
        json.dumps(
            {
                "raw_rows": len(raw),
                "finding_rows": len(findings),
                "labels": len(label_counts),
                "near_duplicate_pairs": len(near_duplicates),
                "output": str(REPORTS_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
