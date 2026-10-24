"""Build a joint small-model/DeepSeek/Jev disagreement review queue."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_PATH = ROOT / "results" / "tables" / "core_v2_llm_audit.csv"
JEV_PATH = ROOT / "results" / "tables" / "jev_core_v2_audit.csv"
FULL_OUT = ROOT / "results" / "tables" / "three_model_comparison_v2.csv"
REVIEW_OUT = ROOT / "results" / "tables" / "three_model_error_review_v2.csv"
REPORT_OUT = ROOT / "results" / "reports" / "three_model_error_review_v2_zh.md"


def relation(small: bool, other: bool) -> str:
    if small and other:
        return "both_correct"
    if small and not other:
        return "small_correct_llm_wrong"
    if not small and other:
        return "small_wrong_llm_correct"
    return "both_wrong"


def priority(row: pd.Series) -> str:
    if not row["small_correct"] and not row["deepseek_correct"] and not row["jev_correct"]:
        return "P1_all_three_wrong"
    if not row["small_correct"] and (row["deepseek_correct"] or row["jev_correct"]):
        return "P2_llm_can_correct_small"
    if row["small_correct"] and (not row["deepseek_correct"] or not row["jev_correct"]):
        return "P3_llm_can_harm_small"
    return "P4_all_three_correct"


def main() -> None:
    deepseek = pd.read_csv(DEEPSEEK_PATH)
    jev = pd.read_csv(JEV_PATH)
    base_columns = [
        "sample_id", "report_id", "target_level1", "target_level2", "audit_areas",
        "context_quality", "text_context_v2", "small_prediction", "small_confidence", "small_correct",
        "route_10pct", "route_20pct", "route_30pct",
    ]
    deepseek_columns = base_columns + [
        "llm_prediction", "llm_confidence", "llm_correct", "llm_abstain",
        "evidence_exact", "auto_decision_valid",
    ]
    jev_columns = ["sample_id", "prediction", "confidence", "jev_correct"]
    table = deepseek[deepseek_columns].rename(
        columns={
            "llm_prediction": "deepseek_prediction",
            "llm_confidence": "deepseek_confidence",
            "llm_correct": "deepseek_correct",
            "llm_abstain": "deepseek_abstain",
            "evidence_exact": "deepseek_evidence_exact",
            "auto_decision_valid": "deepseek_auto_valid",
        }
    ).merge(
        jev[jev_columns].rename(
            columns={"prediction": "jev_prediction", "confidence": "jev_confidence"}
        ),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    for column in ["small_correct", "deepseek_correct", "jev_correct"]:
        table[column] = table[column].astype(bool)
    table["deepseek_error_pattern"] = [
        relation(small, other) for small, other in zip(table["small_correct"], table["deepseek_correct"])
    ]
    table["jev_error_pattern"] = [
        relation(small, other) for small, other in zip(table["small_correct"], table["jev_correct"])
    ]
    table["three_model_pattern"] = table.apply(
        lambda r: f"S{int(r.small_correct)}_D{int(r.deepseek_correct)}_J{int(r.jev_correct)}", axis=1
    )
    table["review_priority"] = table.apply(priority, axis=1)
    table["adjudicated_label"] = ""
    table["error_source"] = ""  # model / context_missing / taxonomy_noise / source_label_error / ambiguous
    table["review_notes"] = ""
    order = {
        "P1_all_three_wrong": 1,
        "P2_llm_can_correct_small": 2,
        "P3_llm_can_harm_small": 3,
        "P4_all_three_correct": 4,
    }
    table["_priority_order"] = table["review_priority"].map(order)
    table = table.sort_values(["_priority_order", "small_confidence", "sample_id"]).drop(columns="_priority_order")
    review = table.loc[table["review_priority"].ne("P4_all_three_correct")].copy()
    for path in [FULL_OUT, REVIEW_OUT, REPORT_OUT]:
        path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FULL_OUT, index=False, encoding="utf-8-sig")
    review.to_csv(REVIEW_OUT, index=False, encoding="utf-8-sig")

    priority_counts = table["review_priority"].value_counts()
    deepseek_counts = table["deepseek_error_pattern"].value_counts()
    jev_counts = table["jev_error_pattern"].value_counts()
    pattern_counts = table["three_model_pattern"].value_counts().sort_index()
    report = f"""# 三模型錯誤與分歧審查隊列（v2）

- 全測試集：{len(table)} 筆
- 至少有一個模型錯誤、需審查：{len(review)} 筆
- 三者全錯：{priority_counts.get('P1_all_three_wrong', 0)} 筆
- 小模型錯、至少一個 LLM 正確：{priority_counts.get('P2_llm_can_correct_small', 0)} 筆
- 小模型正確、至少一個 LLM 錯誤：{priority_counts.get('P3_llm_can_harm_small', 0)} 筆
- 三者全對：{priority_counts.get('P4_all_three_correct', 0)} 筆

## DeepSeek 相對小模型

{deepseek_counts.to_string()}

## Jev 相對小模型

{jev_counts.to_string()}

## 三模型正誤組合

`S/D/J` 分別代表 small model、DeepSeek、Jev；1=正確、0=錯誤。

{pattern_counts.to_string()}

人工審查時填寫 `adjudicated_label`、`error_source` 與 `review_notes`。建議 `error_source` 使用：`model`、`context_missing`、`taxonomy_noise`、`source_label_error`、`ambiguous`。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved review queue: {REVIEW_OUT}")


if __name__ == "__main__":
    main()
