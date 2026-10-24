"""Build a leakage-controlled hierarchical taxonomy and reconstruct shared report context."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "taxonomy_v2.json"
V1_PATH = ROOT / "data" / "processed" / "model_dataset_v1.csv"
FINDINGS_PATH = ROOT / "data" / "interim" / "findings.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "model_dataset_v2.csv"
MAPPING_PATH = ROOT / "results" / "tables" / "taxonomy_v2_mapping.csv"
CONTEXT_PATH = ROOT / "results" / "tables" / "taxonomy_v2_context_audit.csv"
COUNTS_PATH = ROOT / "results" / "tables" / "taxonomy_v2_counts.csv"
REPORT_PATH = ROOT / "results" / "reports" / "taxonomy_v2_summary_zh.md"


def normalize_label(label: str, aliases: dict[str, str]) -> str:
    cleaned = re.sub(r"\s+", " ", str(label)).strip().rstrip(".")
    exact = aliases.get(cleaned)
    if exact:
        return exact
    casefold_aliases = {key.casefold(): value for key, value in aliases.items()}
    return casefold_aliases.get(cleaned.casefold(), cleaned)


def assign_level1(label: str, rules: list[dict]) -> str:
    for rule in rules:
        if re.search(rule["pattern"], label, flags=re.IGNORECASE):
            return rule["label"]
    raise RuntimeError(f"No level-1 rule matched: {label}")


def extract_preamble(text: str, marker: str) -> str | None:
    value = str(text).strip()
    position = value.casefold().find(marker.casefold())
    if position <= 0:
        return None
    preamble = value[:position].strip(" \n\r:-")
    return preamble if len(preamble) >= 40 else None


def mask_context_text(text: str, label_pattern: re.Pattern[str]) -> str:
    value = label_pattern.sub("[AUDIT_AREA]", str(text))
    value = re.sub(r"\bAS\s+\d{3,4}(?:\.\d+)?\b", "[STANDARD_REF]", value, flags=re.IGNORECASE)
    value = re.sub(r"\(\s*\[STANDARD_REF\][^)]*\)", "[STANDARD_REF]", value)
    return re.sub(r"\s+", " ", value).strip()


def choose_group_preambles(findings: pd.DataFrame, marker: str) -> dict[tuple[str, str, str], tuple[str, str]]:
    candidates: dict[tuple[str, str, str], tuple[int, str, str]] = {}
    for _, row in findings.iterrows():
        preamble = extract_preamble(row["Description in the Firm's Inspection Report"], marker)
        if not preamble:
            continue
        key = (str(row["report_id"]), str(row["Issuer Reference Key"]), str(row["audit_areas"]))
        order = int(row["Finding Count"]) if pd.notna(row["Finding Count"]) else 10**9
        existing = candidates.get(key)
        if existing is None or order < existing[0]:
            candidates[key] = (order, str(row["sample_id"]), preamble)
    return {key: (value[1], value[2]) for key, value in candidates.items()}


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include="number").columns:
        shown[column] = shown[column].map(lambda value: f"{value:,}")
    shown = shown.astype(str)
    return "\n".join(
        [
            "| " + " | ".join(shown.columns) + " |",
            "| " + " | ".join(["---"] * len(shown.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()],
        ]
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v1 = pd.read_csv(V1_PATH)
    findings = pd.read_csv(FINDINGS_PATH)
    source_columns = [
        "sample_id",
        "report_id",
        "Issuer Reference Key",
        "Finding Count",
        "audit_areas",
        "Description in the Firm's Inspection Report",
    ]
    source = findings[source_columns].drop_duplicates("sample_id")
    model = v1.drop(columns=["audit_areas"], errors="ignore").merge(
        source, on=["sample_id", "report_id", "Issuer Reference Key"], how="left", validate="one_to_one"
    )
    if model["audit_areas"].isna().any():
        raise RuntimeError("Some v1 samples could not be joined to the official finding table.")

    model["target_level2"] = model["audit_areas"].map(
        lambda label: normalize_label(label, config["explicit_aliases"])
    )
    model["target_level1"] = model["target_level2"].map(
        lambda label: assign_level1(label, config["level1_rules"])
    )

    preambles = choose_group_preambles(findings, config["shared_preamble_marker"])
    reference_pattern = re.compile(config["context_reference_pattern"], flags=re.IGNORECASE)
    context_rows = []
    context_texts = []
    for _, row in model.iterrows():
        text = str(row["text_original"])
        key = (str(row["report_id"]), str(row["Issuer Reference Key"]), str(row["audit_areas"]))
        candidate = preambles.get(key)
        needs_context = bool(reference_pattern.search(text))
        add_context = bool(candidate and candidate[0] != row["sample_id"] and needs_context)
        if add_context:
            source_id, preamble = candidate
            context_text = f"Shared report context: {preamble}\n\nCurrent finding: {text}"
            quality = "same_report_issuer_audit_area_preamble"
        else:
            source_id = None
            context_text = text
            quality = "reference_without_available_preamble" if needs_context else "original_sufficient"
        context_texts.append(context_text)
        context_rows.append(
            {
                "sample_id": row["sample_id"],
                "report_id": row["report_id"],
                "issuer_reference_key": row["Issuer Reference Key"],
                "raw_audit_area": row["audit_areas"],
                "needs_context": needs_context,
                "context_added": add_context,
                "context_source_sample_id": source_id,
                "context_quality": quality,
            }
        )
    model["text_context_v2"] = context_texts
    context_audit = pd.DataFrame(context_rows)
    model["context_needs_context"] = context_audit["needs_context"].to_numpy()
    model["context_added"] = context_audit["context_added"].to_numpy()
    model["context_quality"] = context_audit["context_quality"].to_numpy()
    all_label_phrases = sorted(
        {str(value) for value in findings["audit_areas"].str.split(" | ").explode().dropna()},
        key=len,
        reverse=True,
    )
    label_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(value) for value in all_label_phrases) + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )
    model["text_context_masked_v2"] = model["text_context_v2"].map(
        lambda text: mask_context_text(text, label_pattern)
    )

    mapping = (
        model.groupby(["audit_areas", "target_level2", "target_level1"])
        .agg(samples=("sample_id", "size"), reports=("report_id", "nunique"))
        .reset_index()
        .rename(columns={"audit_areas": "raw_audit_area"})
        .sort_values(["target_level1", "samples"], ascending=[True, False])
    )
    counts = (
        model.groupby(["split", "target_level1"])
        .agg(samples=("sample_id", "size"), clean_samples=("clean_eval", "sum"), reports=("report_id", "nunique"))
        .reset_index()
    )

    output_columns = [
        "sample_id",
        "report_id",
        "Registration ID",
        "Firm Names",
        "Inspection Year",
        "Issuer Reference Key",
        "Finding Count",
        "audit_areas",
        "target_level2",
        "target_level1",
        "split",
        "clean_eval",
        "leakage_reason",
        "description_norm",
        "text_original",
        "text_context_v2",
        "text_context_masked_v2",
        "context_needs_context",
        "context_added",
        "context_quality",
        "text_masked",
        "standard_refs",
    ]
    for path in (OUTPUT_PATH, MAPPING_PATH, CONTEXT_PATH, COUNTS_PATH, REPORT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    model[output_columns].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    mapping.to_csv(MAPPING_PATH, index=False, encoding="utf-8")
    context_audit.to_csv(CONTEXT_PATH, index=False, encoding="utf-8")
    counts.to_csv(COUNTS_PATH, index=False, encoding="utf-8")

    overall = (
        model.groupby("target_level1")
        .agg(samples=("sample_id", "size"), reports=("report_id", "nunique"), fine_labels=("target_level2", "nunique"))
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    ambiguous = int(context_audit["needs_context"].sum())
    restored = int(context_audit["context_added"].sum())
    fallback_count = int(model["target_level1"].eq("Cash and Other Assets").sum())
    report = f"""# Hierarchical Taxonomy v2 與上下文修復摘要

## 設計決策

- 保留 v1 的 {len(model):,} 筆樣本、report-level split 與 leakage 標記，確保後續結果可直接比較。
- 細粒度 `target_level2` 保留並正規化官方 audit area，不再把長尾標籤全部折疊成 `Other`。
- 粗粒度 `target_level1` 僅根據官方標籤名稱的預先聲明規則產生；沒有使用測試預測或 DeepSeek 回答調整規則。
- 明確別名合併：{sum(model['audit_areas'].ne(model['target_level2'])):,} 筆。
- 原始細類：{model['audit_areas'].nunique():,} 種；正規化細類：{model['target_level2'].nunique():,} 種；粗類：{model['target_level1'].nunique():,} 種。
- `Cash and Other Assets` 粗類：{fallback_count:,} 筆；其中通用資產名稱仍保留各自的官方細類，不再作為單一細粒度答案。

## 上下文完整性

- 偵測到指代表達、可能依賴前文：{ambiguous:,} 筆。
- 從同一 report、issuer reference 與官方 audit area 的共享前言安全補回：{restored:,} 筆。
- 仍缺少可用共享前言：{ambiguous - restored:,} 筆；後續模型應將其視為 ambiguity/OOD 訊號，而不是強制分類。

## 粗粒度分布

{markdown_table(overall)}

## 使用限制

此版本是可重現的資料工程 taxonomy，不是審計專家共識標準。正式論文應對映射表抽樣雙人覆核，報告一致率，並在凍結後才訓練 v2 模型。v1 的 DeepSeek 實驗保留為 pilot，不用 v2 回溯改寫。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(overall.to_string(index=False))
    print(f"context_needing={ambiguous} context_restored={restored} fallback={fallback_count}")


if __name__ == "__main__":
    main()
