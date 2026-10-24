"""Audit whether frozen router v1 addresses previously observed errors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "rag_pilot_runtime"


def paired_transitions(frame: pd.DataFrame, baseline: str, candidate: str, outcome: str) -> dict:
    pivot = frame.loc[frame["condition"].isin([baseline, candidate])].pivot(
        index="sample_id", columns="condition", values=outcome
    ).dropna().astype(bool)
    baseline_wrong = ~pivot[baseline]
    fixed = baseline_wrong & pivot[candidate]
    harmed = pivot[baseline] & ~pivot[candidate]
    return {
        "paired_n": len(pivot),
        "baseline_wrong": int(baseline_wrong.sum()),
        "fixed": int(fixed.sum()),
        "remaining_wrong": int((~pivot[candidate]).sum()),
        "harmed": int(harmed.sum()),
        "fix_rate_among_baseline_wrong": float(fixed.sum() / baseline_wrong.sum()) if baseline_wrong.sum() else None,
    }


def main() -> None:
    finmr = pd.read_csv(RUNTIME / "finmr_pilot_v1_audit.csv")
    finmr_result = paired_transitions(finmr, "direct", "structured_rag", "joint_correct")
    finmr_result["suspected_gold_error_remaining"] = 1
    finmr_result["substantive_remaining_after_arithmetic_adjudication"] = 0

    pcaob = pd.read_csv(RUNTIME / "pcaob_rag_pilot_v1_audit.csv")
    pcaob_observed = pcaob.loc[pcaob["condition"].isin(["direct", "report_rag"])].copy()
    pcaob_direct = paired_transitions(pcaob_observed, "direct", "report_rag", "prediction_correct")
    small_wrong = pcaob_observed.loc[pcaob_observed["pilot_stratum"].eq("small_wrong")]
    pcaob_small_wrong = {
        "n_previous_small_model_errors": int(small_wrong["sample_id"].nunique()),
        "direct_llm_correct": int(small_wrong.loc[small_wrong.condition.eq("direct"), "prediction_correct"].sum()),
        "report_rag_correct": int(small_wrong.loc[small_wrong.condition.eq("report_rag"), "prediction_correct"].sum()),
    }
    pcaob_small_wrong["report_rag_fix_rate_if_oracle_routed"] = (
        pcaob_small_wrong["report_rag_correct"] / pcaob_small_wrong["n_previous_small_model_errors"]
    )

    legacy = pd.read_csv(ROOT / "results" / "tables" / "utility_router_test_v1.csv")
    selected = legacy.loc[legacy["utility_route"].astype(bool)]
    legacy_result = {
        "retrospective_test_n": len(legacy),
        "selected_n": len(selected),
        "selected_beneficial": int((~selected["small_correct"] & selected["llm_correct"]).sum()),
        "selected_harmful": int((selected["small_correct"] & ~selected["llm_correct"]).sum()),
    }
    legacy_result["selected_net"] = legacy_result["selected_beneficial"] - legacy_result["selected_harmful"]

    result = {
        "router_version": "minimum-sufficient-audit-router-v1",
        "finmr_direct_errors": finmr_result,
        "pcaob_direct_to_report_rag": pcaob_direct,
        "pcaob_previous_small_model_errors": pcaob_small_wrong,
        "legacy_jev_router_test": legacy_result,
        "answer": {
            "finmr": "partly by official labels (1/2 Direct errors), and all substantive errors after flagging one deterministic gold inconsistency; evidence is only one genuine correction.",
            "pcaob": "no: the frozen executable-evidence rule is not applicable to narrative classification and has not prospectively selected PCAOB corrections.",
            "overall": "the backend can solve some old errors, but router v1 is not yet proven to identify and solve previous errors across tasks."
        }
    }
    output_json = RUNTIME / "minimum_sufficient_router_v1_wrong_cases.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Minimum-Sufficient Audit Router v1：凍結與舊錯題審計

**狀態：** Frozen on 2026-10-02；development-only rule。

## 凍結規則

1. 只有結構化數值任務同時具有 target fact、calculation arcs、同 context 的全部 child facts 且可解析為數值時，才走 `structured retrieval → deterministic checker`。
2. 只有 parser 與 checker 都成功才允許接管；證據缺失或驗證失敗時不得因 LLM confidence 自動覆寫。
3. PCAOB narrative classification 暫時保留 cheap-model answer；report-RAG 仍是實驗 backend，尚未有可前瞻驗證的 benefit selector。

## 能否解決之前錯題？

### FinMR

- Direct 錯誤：{finmr_result['baseline_wrong']}/{finmr_result['paired_n']}。
- Structured route 按官方 gold 修正：{finmr_result['fixed']}/{finmr_result['baseline_wrong']}，harm={finmr_result['harmed']}。
- 剩餘 1 筆是官方 gold `320002`，但公開 child facts 的確切算式為 `577282 + 108087 - 653367 = 32002`；Direct、structured LLM 和 checker 都得到 32002。因此按官方分數是修正 1/2；按可執行算式審計，唯一真實的 Direct 計算錯誤已被修正。
- 但真實修正事件只有 1 筆，證據仍太少，不能泛化成「router 已能解決 FinMR 錯題」。

### PCAOB

- 在刻意抽取的 60 筆 small-model 錯題中，Direct DeepSeek 答對 {pcaob_small_wrong['direct_llm_correct']} 筆，report-RAG 答對 {pcaob_small_wrong['report_rag_correct']} 筆。
- 在全部 120 筆配對樣本中，report-RAG 修正 {pcaob_direct['fixed']} 筆 Direct 錯題，同時破壞 {pcaob_direct['harmed']} 筆 Direct 正確答案。
- 這證明 backend **有能力**解部分錯題；但 frozen v1 的 executable-evidence 規則不適用 PCAOB narrative text，尚不能在呼叫前可靠識別那 {pcaob_direct['fixed']} 筆。因此不能說 router 已解決 PCAOB 舊錯題。

### 舊 Jev learned router

- 890 筆 retrospective test 中選擇 {legacy_result['selected_n']} 筆，beneficial={legacy_result['selected_beneficial']}、harmful={legacy_result['selected_harmful']}、net={legacy_result['selected_net']:+d}。
- 這再次顯示「後端偶爾能答對」不等於「router 能穩定選中錯題」。

## 結論

最精確的回答是：**FinMR 的 evidence-complete 數值錯題已有一個被規則確實修正；PCAOB 的舊錯題只能證明 RAG backend 有部分修正能力，尚未證明 frozen router 能選中它們。** Router v1 的已證實價值目前是大幅降成本、阻止未驗證接管，以及把可執行數值題安全送給 checker；不是全面消除舊錯題。
"""
    output_report = RUNTIME / "minimum_sufficient_router_v1_freeze_zh.md"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
