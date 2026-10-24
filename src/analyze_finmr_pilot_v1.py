"""Statistical and diagnostic analysis of the FinMR paired pilot."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from finmr_structured import decimal_value
from llm_common import ROOT
from prepare_finmr_candidates_v1 import load_config


def numeric_equal(left, right) -> bool:
    return decimal_value(str(left)) == decimal_value(str(right))


def main() -> None:
    config = load_config()
    audit = pd.read_csv(ROOT / config["audit_table"])
    paired = audit.pivot(index="sample_id", columns="condition", values="joint_correct").dropna().astype(bool)
    beneficial = int((~paired["direct"] & paired["structured_rag"]).sum())
    harmful = int((paired["direct"] & ~paired["structured_rag"]).sum())
    delta = paired["structured_rag"].astype(int) - paired["direct"].astype(int)
    rng = np.random.default_rng(20261002)
    boot = rng.choice(delta.to_numpy(), size=(20_000, len(delta)), replace=True).mean(axis=1)
    discordant = beneficial + harmful
    p_value = float(binomtest(min(beneficial, harmful), discordant, 0.5).pvalue) if discordant else 1.0

    direct = audit.loc[audit.condition.eq("direct")].set_index("sample_id")
    structured = audit.loc[audit.condition.eq("structured_rag")].set_index("sample_id")
    deterministic_agreement = (
        structured.apply(lambda row: numeric_equal(row["extracted_value"], row["deterministic_extracted_value"]), axis=1)
        & structured.apply(lambda row: numeric_equal(row["calculated_value"], row["deterministic_calculated_value"]), axis=1)
    )
    suspected = structured.loc[
        ~structured["joint_correct"]
        & deterministic_agreement
        & structured.apply(
            lambda row: not numeric_equal(row["deterministic_calculated_value"], row["gold_calculated_value"]), axis=1
        )
    ]
    token_reduction = 1 - structured["input_tokens"].mean() / direct["input_tokens"].mean()
    latency_reduction = 1 - structured["latency_ms"].mean() / direct["latency_ms"].mean()

    result = {
        "version": config["version"],
        "design_note": "Exploratory development-only subset with query-complete DQC_US_0126 evidence; not representative of all FinMR cases.",
        "paired_n": len(paired),
        "direct_joint_accuracy": float(direct["joint_correct"].mean()),
        "structured_rag_joint_accuracy": float(structured["joint_correct"].mean()),
        "beneficial": beneficial,
        "harmful": harmful,
        "net": beneficial - harmful,
        "accuracy_delta": float(delta.mean()),
        "paired_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "mcnemar_exact_two_sided_p": p_value,
        "input_token_reduction": float(token_reduction),
        "latency_reduction": float(latency_reduction),
        "structured_vs_deterministic_exact_numeric_agreement": float(deterministic_agreement.mean()),
        "suspected_gold_inconsistencies": [
            {
                "sample_id": sample_id,
                "source_id": int(row["source_id"]),
                "target": row["target"],
                "gold_calculated_value": str(row["gold_calculated_value"]),
                "deterministic_and_model_value": str(row["deterministic_calculated_value"]),
            }
            for sample_id, row in suspected.iterrows()
        ],
        "holdout_status": "FinMR_Sub test was inspected during parser coverage diagnostics and must be treated as retrospective, not confirmatory.",
    }
    output_json = ROOT.parent / "rag_pilot_runtime" / "finmr_pilot_v1_statistical_analysis.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# FinMR structured-RAG feasibility pilot：統計與診斷", "",
        "> 探索性 development-only 結果，只涵蓋 query 中具完整 calculation evidence 的 DQC_US_0126；不代表全部 FinMR。", "",
        f"- Direct Joint ACC：{direct['joint_correct'].mean():.2%}（{int(direct['joint_correct'].sum())}/{len(direct)}）。",
        f"- Structured-RAG Joint ACC：{structured['joint_correct'].mean():.2%}（{int(structured['joint_correct'].sum())}/{len(structured)}）。",
        f"- 配對轉移：beneficial {beneficial}、harmful {harmful}、net {beneficial-harmful:+d}；delta {delta.mean():+.2%}。",
        f"- Paired bootstrap 95% CI：[{np.quantile(boot, .025):+.2%}, {np.quantile(boot, .975):+.2%}]；exact McNemar p={p_value:.4f}。",
        f"- Input tokens 減少 {token_reduction:.1%}；latency 減少 {latency_reduction:.1%}。",
        f"- Structured-RAG 與 deterministic checker 的數值結果一致率：{deterministic_agreement.mean():.2%}。", "",
        "## Benchmark consistency", "",
        "官方 gold 中發現 1 筆高度疑似多一個 0：children 為 577,282 + 108,087 − 653,367 = 32,002，但 gold 為 320,002。主要表格保留官方 gold 分數；另把此案例列為資料品質診斷，不擅自修改標籤。", "",
        "## 判讀", "",
        "Structured evidence 在本子集上以約 5% 的 input tokens 達到略高於 Direct 的 Joint ACC，但只有 1 個 beneficial event，統計區間跨 0，不能宣稱品質顯著提升。可支持的強結論是：結構化取證大幅降低成本與延遲，且 deterministic verification 足以重現 structured-RAG 的全部數值答案；因此 Verified Audit-RAG 應優先把可執行 checker 當核心，LLM 僅處理 parser／規則無法覆蓋的案例。", "",
        "## Holdout 限制", "",
        "FinMR_Sub test 在 parser coverage 診斷時已查看 aggregate gold 結果，必須標記為 retrospective；不能再把它寫成未接觸的確認性 test。",
    ]
    output_md = ROOT.parent / "rag_pilot_runtime" / "finmr_pilot_v1_statistical_analysis_zh.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
