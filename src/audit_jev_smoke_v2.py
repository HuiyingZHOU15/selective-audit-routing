"""Audit the authorized 24-sample stratified Jev smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_common import ROOT


def main() -> None:
    candidates = pd.read_csv(ROOT / "data/interim/deepseek_core_v2_candidates.csv")
    selection = pd.read_csv(ROOT / "data/interim/jev_stratified_smoke_v2.csv")
    response_path = ROOT / "artifacts/llm_responses/jev_core_v2.jsonl"
    responses = []
    for line in response_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "ok":
            responses.append(item)
    response_frame = pd.DataFrame(responses).drop_duplicates(
        ["sample_id", "prompt_bundle_sha256"], keep="last"
    )
    audit = (
        selection.merge(candidates, on=["sample_id", "prompt_bundle_sha256", "target_level1"], validate="one_to_one")
        .merge(response_frame, on=["sample_id", "prompt_bundle_sha256"], validate="one_to_one")
    )
    audit["jev_correct"] = audit["prediction"].eq(audit["target_level1"])
    audit["small_correct"] = audit["small_prediction"].eq(audit["target_level1"])
    audit["jev_corrected_small"] = audit["jev_correct"] & ~audit["small_correct"]
    audit["jev_introduced_error"] = ~audit["jev_correct"] & audit["small_correct"]
    audit["high_confidence_error"] = ~audit["jev_correct"] & audit["confidence"].ge(0.8)
    per_class = audit.groupby("target_level1").agg(
        n=("sample_id", "size"),
        jev_accuracy=("jev_correct", "mean"),
        small_accuracy=("small_correct", "mean"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    usage = audit["usage"].map(lambda value: value if isinstance(value, dict) else {})
    input_tokens = int(usage.map(lambda value: value.get("input_tokens", 0)).sum())
    metrics = {
        "n": len(audit),
        "class_count": int(audit["target_level1"].nunique()),
        "jev_accuracy": float(audit["jev_correct"].mean()),
        "small_accuracy": float(audit["small_correct"].mean()),
        "mean_confidence": float(audit["confidence"].mean()),
        "median_latency_ms": float(audit["latency_ms"].median()),
        "corrected_small_errors": int(audit["jev_corrected_small"].sum()),
        "introduced_errors": int(audit["jev_introduced_error"].sum()),
        "high_confidence_errors_ge_0_8": int(audit["high_confidence_error"].sum()),
        "input_tokens": input_tokens,
        "estimated_cost_usd": input_tokens * 0.042 / 1_000_000,
        "per_class": per_class.to_dict(orient="records"),
    }
    table = "\n".join(
        f"| {row.target_level1} | {int(row.n)} | {row.jev_accuracy:.0%} | {row.small_accuracy:.0%} | {row.mean_confidence:.3f} |"
        for row in per_class.itertuples()
    )
    errors = audit.loc[~audit["jev_correct"], ["sample_id", "target_level1", "prediction", "confidence", "small_prediction"]]
    error_table = "無。" if errors.empty else "\n".join(
        f"- `{row.sample_id}`：真值 `{row.target_level1}`，Jev `{row.prediction}`（{row.confidence:.3f}），小模型 `{row.small_prediction}`"
        for row in errors.itertuples()
    )
    report = f"""# Jev 分層煙霧測試（taxonomy v2）

- 樣本：{len(audit)}（12 類 × 每類 2 筆）
- Jev accuracy：{metrics['jev_accuracy']:.4f}
- 小模型 accuracy：{metrics['small_accuracy']:.4f}
- Jev 平均 confidence：{metrics['mean_confidence']:.4f}
- 中位延遲：{metrics['median_latency_ms']:.1f} ms
- Jev 修正小模型錯誤：{metrics['corrected_small_errors']}；引入錯誤：{metrics['introduced_errors']}
- Jev 高信心錯誤（confidence ≥ 0.8）：{metrics['high_confidence_errors_ge_0_8']}
- 輸入 tokens：{input_tokens}；估算成本：USD {metrics['estimated_cost_usd']:.6f}

| 類別 | n | Jev accuracy | 小模型 accuracy | Jev 平均信心 |
| --- | ---: | ---: | ---: | ---: |
{table}

## Jev 錯誤

{error_table}

這是分層 smoke test，不是總體無偏估計；每類僅兩筆，不能據此宣稱完整測試集的最終性能。
"""
    outputs = [
        ROOT / "results/tables/jev_stratified_smoke_v2.csv",
        ROOT / "results/metrics/jev_stratified_smoke_v2.json",
        ROOT / "results/reports/jev_stratified_smoke_v2_zh.md",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(outputs[0], index=False, encoding="utf-8")
    outputs[1].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs[2].write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
