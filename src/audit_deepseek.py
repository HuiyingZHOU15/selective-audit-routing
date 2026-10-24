"""Audit DeepSeek outputs for accuracy, corrections, evidence grounding, cost, and latency."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from llm_common import LLMDecision, ROOT, evidence_match, load_config


def read_latest_responses(path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            key = (str(item["sample_id"]), str(item.get("prompt_bundle_sha256", "")))
            latest[key] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return latest


def safe_rate(series: pd.Series) -> float | None:
    return float(series.mean()) if len(series) else None


def binary_ece(correct: pd.Series, confidence: pd.Series, bins: int = 10) -> float | None:
    if not len(correct):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = confidence.ge(lower) & (confidence.lt(upper) if index < bins - 1 else confidence.le(upper))
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def main() -> None:
    config = load_config()
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    responses = read_latest_responses(ROOT / config["response_log"])
    taxonomy = json.loads((ROOT / config["taxonomy"]).read_text(encoding="utf-8"))
    labels = set(taxonomy["target_labels"])
    rows = []
    for _, candidate in candidates.iterrows():
        sample_id = str(candidate["sample_id"])
        response = responses.get((sample_id, str(candidate["prompt_bundle_sha256"])))
        row = candidate.to_dict()
        row.update(
            {
                "response_status": "missing" if response is None else response.get("status", "error"),
                "schema_valid": False,
                "label_valid": False,
                "evidence_exact": False,
                "evidence_normalized": False,
                "llm_prediction": None,
                "llm_confidence": np.nan,
                "llm_abstain": True,
                "llm_correct": False,
                "small_correct": candidate["small_prediction"] == candidate["target_label"],
                "corrected_small_error": False,
                "introduced_error": False,
                "auto_decision_valid": False,
                "latency_ms": np.nan,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
        if response and response.get("status") == "ok":
            try:
                decision = LLMDecision.model_validate(response["decision"])
                exact, normalized = evidence_match(str(candidate["text"]), decision.evidence_quote)
                label_valid = decision.predicted_label in labels if decision.predicted_label else decision.abstain
                llm_correct = (not decision.abstain) and decision.predicted_label == candidate["target_label"]
                auto_valid = (not decision.abstain) and label_valid and exact
                row.update(
                    {
                        "schema_valid": True,
                        "label_valid": label_valid,
                        "evidence_exact": exact,
                        "evidence_normalized": normalized,
                        "llm_prediction": decision.predicted_label,
                        "llm_confidence": decision.confidence,
                        "llm_abstain": decision.abstain,
                        "llm_correct": llm_correct,
                        "corrected_small_error": (not row["small_correct"]) and llm_correct and auto_valid,
                        "introduced_error": row["small_correct"] and (not llm_correct) and auto_valid,
                        "auto_decision_valid": auto_valid,
                        "latency_ms": response.get("latency_ms", np.nan),
                        "input_tokens": response.get("input_tokens", 0),
                        "output_tokens": response.get("output_tokens", 0),
                    }
                )
            except Exception:
                pass
        rows.append(row)

    audit = pd.DataFrame(rows)
    audit_path = ROOT / config["audit_table"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8")

    completed = audit.loc[audit["response_status"].eq("ok")]
    auto = completed.loc[completed["auto_decision_valid"]]
    decisions = completed.loc[~completed["llm_abstain"]]
    audit["gated_prediction"] = audit["small_prediction"].where(
        ~audit["auto_decision_valid"], audit["llm_prediction"]
    )
    audit["gated_correct"] = audit["gated_prediction"].eq(audit["target_label"])
    audit.to_csv(audit_path, index=False, encoding="utf-8")

    routes = pd.read_csv(ROOT / config["routing_table"])
    route_mask = routes[config["route_column"]].astype(bool)
    baseline_correct = routes[config["small_prediction_column"]].eq(routes["target_label"])
    non_routed_correct = int(baseline_correct[~route_mask].sum())
    full_count = len(routes)
    gated_full_correct = non_routed_correct + int(audit["gated_correct"].sum())
    accepted_count = int((~route_mask).sum()) + len(auto)
    accepted_correct = non_routed_correct + int(auto["llm_correct"].sum())

    class_metrics = (
        audit.groupby("target_label")
        .agg(
            n=("sample_id", "size"),
            small_accuracy=("small_correct", "mean"),
            abstention_rate=("llm_abstain", "mean"),
            auto_coverage=("auto_decision_valid", "mean"),
            llm_accuracy_all=("llm_correct", "mean"),
            gated_accuracy=("gated_correct", "mean"),
            corrections=("corrected_small_error", "sum"),
            introduced_errors=("introduced_error", "sum"),
        )
        .reset_index()
    )
    class_path = audit_path.with_name("deepseek_class_metrics_v1.csv")
    class_metrics.to_csv(class_path, index=False, encoding="utf-8")
    pricing = config["pricing_usd_per_million_tokens"]
    input_tokens = int(completed["input_tokens"].sum())
    output_tokens = int(completed["output_tokens"].sum())
    metrics = {
        "version": config["version"],
        "provider": config["provider"],
        "model": config["model"],
        "candidate_count": int(len(audit)),
        "completed_count": int(len(completed)),
        "completion_rate": safe_rate(audit["response_status"].eq("ok")),
        "schema_valid_rate_completed": safe_rate(completed["schema_valid"]),
        "exact_evidence_rate_completed": safe_rate(completed["evidence_exact"]),
        "abstention_rate_completed": safe_rate(completed["llm_abstain"]),
        "auto_decision_count": int(len(auto)),
        "auto_decision_accuracy": safe_rate(auto["llm_correct"]),
        "auto_decision_macro_f1": (
            float(
                f1_score(
                    auto["target_label"],
                    auto["llm_prediction"],
                    labels=sorted(labels),
                    average="macro",
                    zero_division=0,
                )
            )
            if len(auto)
            else None
        ),
        "llm_confidence_ece_10bin": binary_ece(
            decisions["llm_correct"].astype(float), decisions["llm_confidence"]
        ),
        "llm_confidence_brier": (
            float(np.mean((decisions["llm_confidence"] - decisions["llm_correct"].astype(float)) ** 2))
            if len(decisions)
            else None
        ),
        "high_confidence_errors_ge_0_8": int(
            ((decisions["llm_confidence"] >= 0.8) & ~decisions["llm_correct"]).sum()
        ),
        "corrected_small_errors": int(auto["corrected_small_error"].sum()),
        "introduced_errors": int(auto["introduced_error"].sum()),
        "routed_small_model_accuracy": safe_rate(audit["small_correct"]),
        "routed_gated_fallback_accuracy": safe_rate(audit["gated_correct"]),
        "full_test_small_model_accuracy": float(baseline_correct.mean()),
        "full_test_gated_fallback_accuracy": gated_full_correct / full_count,
        "full_test_gated_fallback_net_correct": gated_full_correct - int(baseline_correct.sum()),
        "full_test_human_review_coverage": accepted_count / full_count,
        "full_test_human_review_accepted_accuracy": accepted_correct / accepted_count,
        "full_test_unresolved_for_human": full_count - accepted_count,
        "median_latency_ms": float(completed["latency_ms"].median()) if len(completed) else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd_off_peak": (
            input_tokens * pricing["off_peak_input_cache_miss"] + output_tokens * pricing["off_peak_output"]
        ) / 1_000_000,
        "estimated_cost_usd_peak": (
            input_tokens * pricing["peak_input_cache_miss"] + output_tokens * pricing["peak_output"]
        ) / 1_000_000,
        "readiness": "complete" if len(completed) == len(audit) else "awaiting_api_responses",
    }
    metrics_path = ROOT / config["metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# DeepSeek 路由可靠性審計（v1）

## 狀態

- 模型：`{config['model']}`（DeepSeek Responses API）
- 20% 路由候選：{len(audit)} 筆
- 已取得有效回應：{len(completed)} 筆
- 可自動採納（結構、標籤、逐字證據均合格）：{len(auto)} 筆
- 狀態：`{metrics['readiness']}`

## 可靠性門檻

LLM 回答只有在 JSON 結構有效、標籤屬於凍結 taxonomy、未棄權，且 `evidence_quote` 是原始 finding 的逐字連續片段時，才算可自動採納。其他情形保留給人工覆核，不會以缺失回應推算模型表現。

## 目前可報告結果

- 自動採納準確率：{metrics['auto_decision_accuracy']}
- 自動採納 macro-F1：{metrics['auto_decision_macro_f1']}
- 修正小模型錯誤：{metrics['corrected_small_errors']} 筆
- 新引入錯誤：{metrics['introduced_errors']} 筆
- 路由子集小模型準確率：{metrics['routed_small_model_accuracy']:.4f}
- 路由子集以 LLM 合格答案覆蓋、其餘回退小模型後：{metrics['routed_gated_fallback_accuracy']:.4f}
- 全測試集小模型準確率：{metrics['full_test_small_model_accuracy']:.4f}
- 全測試集 gated fallback 準確率：{metrics['full_test_gated_fallback_accuracy']:.4f}（淨增加 {metrics['full_test_gated_fallback_net_correct']} 筆正確）
- 若不回退、而將不合格 LLM 回答交人工：自動覆蓋率 {metrics['full_test_human_review_coverage']:.4f}，已覆蓋樣本準確率 {metrics['full_test_human_review_accepted_accuracy']:.4f}，待人工 {metrics['full_test_unresolved_for_human']} 筆
- LLM 信心 ECE（10 bins）：{metrics['llm_confidence_ece_10bin']:.4f}
- 高信心（≥0.8）錯誤：{metrics['high_confidence_errors_ge_0_8']} 筆
- 完成樣本估計 API 成本：USD {metrics['estimated_cost_usd_off_peak']:.6f}–{metrics['estimated_cost_usd_peak']:.6f}

## 解讀限制

這裡的「正確」首先指與凍結的 top-12 + `Other` 標籤完全一致。`Other` 是多種官方原始 audit area 的折疊桶，不是單一語義類別；因此 LLM 選出文字上合理的具名類別時，仍可能被此指標判錯。完整判斷需同時查看逐筆審計表中的 `raw_audit_area`，不能把所有不一致直接解讀為幻覺。
"""
    report_path = ROOT / config["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
