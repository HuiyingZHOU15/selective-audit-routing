"""Audit all-test DeepSeek outputs and compare routing strategies fairly."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from llm_common import LLMDecision, ROOT, evidence_match
from prepare_deepseek_core_v2 import load_config_v2


def read_latest(path) -> dict[tuple[str, str], dict]:
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                latest[(str(item["sample_id"]), str(item.get("prompt_bundle_sha256", "")))] = item
            except (json.JSONDecodeError, KeyError):
                pass
    return latest


def metric_pair(y, pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
    }


def cluster_bootstrap_delta(audit, comparison_col, iterations, seed) -> dict:
    rng = np.random.default_rng(seed)
    reports = audit["report_id"].drop_duplicates().to_numpy()
    grouped = {key: group for key, group in audit.groupby("report_id", sort=False)}
    values = []
    for _ in range(iterations):
        sampled = rng.choice(reports, size=len(reports), replace=True)
        frame = pd.concat([grouped[key] for key in sampled], ignore_index=True)
        values.append(float(frame[comparison_col].mean() - frame["small_correct"].mean()))
    return {"mean": float(np.mean(values)), "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}


def main() -> None:
    config = load_config_v2()
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    responses = read_latest(ROOT / config["response_log"])
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    labels = list(definitions)
    rows = []
    for _, candidate in candidates.iterrows():
        key = (str(candidate["sample_id"]), str(candidate["prompt_bundle_sha256"]))
        response = responses.get(key)
        row = candidate.to_dict()
        row.update({"response_status": "missing", "schema_valid": False, "label_valid": False,
                    "evidence_exact": False, "llm_prediction": None, "llm_confidence": np.nan,
                    "llm_abstain": True, "llm_correct": False, "auto_decision_valid": False,
                    "latency_ms": np.nan, "input_tokens": 0, "output_tokens": 0})
        if response:
            row["response_status"] = response.get("status", "error")
        if response and response.get("status") == "ok":
            try:
                decision = LLMDecision.model_validate(response["decision"])
                exact, _ = evidence_match(str(candidate["text"]), decision.evidence_quote)
                valid_label = decision.predicted_label in definitions if decision.predicted_label else decision.abstain
                auto = bool(not decision.abstain and valid_label and exact)
                row.update({"schema_valid": True, "label_valid": valid_label, "evidence_exact": exact,
                            "llm_prediction": decision.predicted_label, "llm_confidence": decision.confidence,
                            "llm_abstain": decision.abstain,
                            "llm_correct": bool(not decision.abstain and decision.predicted_label == candidate["target_level1"]),
                            "auto_decision_valid": auto, "latency_ms": response.get("latency_ms", np.nan),
                            "input_tokens": response.get("input_tokens", 0), "output_tokens": response.get("output_tokens", 0)})
            except Exception:
                pass
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit["small_correct"] = audit["small_prediction"].eq(audit["target_level1"])
    audit["gated_prediction_all"] = audit["small_prediction"].where(~audit["auto_decision_valid"], audit["llm_prediction"])
    audit["gated_correct_all"] = audit["gated_prediction_all"].eq(audit["target_level1"])
    y = audit["target_level1"].to_numpy()
    rng = np.random.default_rng(20260927)
    strategies = []
    for budget in (10, 20, 30):
        route_col = f"route_{budget}pct"
        route = audit[route_col].astype(bool).to_numpy()
        use_llm = route & audit["auto_decision_valid"].to_numpy()
        pred = audit["small_prediction"].to_numpy(dtype=object).copy()
        pred[use_llm] = audit.loc[use_llm, "llm_prediction"].to_numpy()
        correct_col = f"hybrid_{budget}_correct"
        audit[f"hybrid_{budget}_prediction"] = pred
        audit[correct_col] = pred == y
        observed = metric_pair(y, pred, labels)
        k = int(route.sum())
        random_acc, random_f1 = [], []
        for _ in range(config["random_iterations"]):
            random_route = np.zeros(len(audit), dtype=bool)
            random_route[rng.choice(len(audit), size=k, replace=False)] = True
            random_use = random_route & audit["auto_decision_valid"].to_numpy()
            random_pred = audit["small_prediction"].to_numpy(dtype=object).copy()
            random_pred[random_use] = audit.loc[random_use, "llm_prediction"].to_numpy()
            random_acc.append(float(np.mean(random_pred == y)))
            random_f1.append(float(f1_score(y, random_pred, labels=labels, average="macro", zero_division=0)))
        accepted = (~route) | use_llm
        accepted_pred = pred[accepted]
        accepted_y = y[accepted]
        correctable = (~audit["small_correct"] & audit["auto_decision_valid"] & audit["llm_correct"]).to_numpy()
        corrected = int((route & correctable).sum())
        introduced = int((route & audit["small_correct"].to_numpy() & use_llm & ~audit["llm_correct"].to_numpy()).sum())
        strategies.append({
            "budget": budget / 100, "routed_count": k, **observed,
            "corrected_small_errors": corrected, "introduced_errors": introduced,
            "human_review_count": int((route & ~audit["auto_decision_valid"].to_numpy()).sum()),
            "accepted_coverage": float(accepted.mean()),
            "accepted_accuracy": float(accuracy_score(accepted_y, accepted_pred)),
            "random_accuracy_mean": float(np.mean(random_acc)),
            "random_accuracy_ci95": [float(np.quantile(random_acc, .025)), float(np.quantile(random_acc, .975))],
            "random_macro_f1_mean": float(np.mean(random_f1)),
            "random_macro_f1_ci95": [float(np.quantile(random_f1, .025)), float(np.quantile(random_f1, .975))],
            "probability_beats_random": float(np.mean(observed["accuracy"] > np.asarray(random_acc))),
            "oracle_accuracy_upper": float((audit["small_correct"].sum() + min(k, int(correctable.sum()))) / len(audit)),
            "report_cluster_bootstrap_accuracy_delta": cluster_bootstrap_delta(audit, correct_col, config["bootstrap_iterations"], 20260927 + budget),
        })

    completed = audit.loc[audit["response_status"].eq("ok")]
    auto = audit.loc[audit["auto_decision_valid"]]
    price = config["pricing_usd_per_million_tokens"]
    input_tokens, output_tokens = int(completed["input_tokens"].sum()), int(completed["output_tokens"].sum())
    metrics = {
        "version": config["version"], "model": config["model"], "test_samples": len(audit),
        "completed": len(completed), "completion_rate": float(len(completed) / len(audit)),
        "small_only": metric_pair(y, audit["small_prediction"], labels),
        "llm_only_accuracy_all": float(audit["llm_correct"].mean()),
        "llm_auto_coverage": float(audit["auto_decision_valid"].mean()),
        "llm_auto_accuracy": float(auto["llm_correct"].mean()) if len(auto) else None,
        "llm_auto_macro_f1": float(f1_score(auto["target_level1"], auto["llm_prediction"], labels=labels, average="macro", zero_division=0)) if len(auto) else None,
        "evidence_exact_rate_completed": float(completed["evidence_exact"].mean()) if len(completed) else None,
        "abstention_rate_completed": float(completed["llm_abstain"].mean()) if len(completed) else None,
        "strategies": strategies,
        "median_latency_ms": float(completed["latency_ms"].median()) if len(completed) else None,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "estimated_cost_usd_off_peak": (input_tokens * price["off_peak_input_cache_miss"] + output_tokens * price["off_peak_output"]) / 1e6,
        "estimated_cost_usd_peak": (input_tokens * price["peak_input_cache_miss"] + output_tokens * price["peak_output"]) / 1e6,
        "readiness": "complete" if len(completed) == len(audit) else "awaiting_api_responses",
    }
    audit_path = ROOT / config["audit_table"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    (ROOT / config["metrics"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    table = "\n".join(f"| {s['budget']:.0%} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {s['random_accuracy_mean']:.4f} [{s['random_accuracy_ci95'][0]:.4f}, {s['random_accuracy_ci95'][1]:.4f}] | {s['corrected_small_errors']} | {s['introduced_errors']} | {s['human_review_count']} |" for s in strategies)
    report = f"""# 核心實驗結果（taxonomy v2）

- 狀態：`{metrics['readiness']}`；DeepSeek 回應 {len(completed)}/{len(audit)}。
- 小模型：accuracy {metrics['small_only']['accuracy']:.4f}，macro-F1 {metrics['small_only']['macro_f1']:.4f}。
- LLM-only（棄權/不合格按錯誤）：accuracy {metrics['llm_only_accuracy_all']:.4f}。
- 證據門控自動覆蓋率 {metrics['llm_auto_coverage']:.2%}；覆蓋內 accuracy {metrics['llm_auto_accuracy']}。

| 路由預算 | 混合 accuracy | 混合 macro-F1 | 隨機路由 accuracy 均值 [95%區間] | 修正 | 引入錯誤 | 待人工 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

同一批 DeepSeek 回應被所有策略重用；置信度路由與隨機路由在相同實際路由數下比較。置信度門控的選擇不使用 test 標籤；oracle 只作不可部署的上限。bootstrap 以 report 為叢集重抽樣。

全測試集一次性 DeepSeek 成本估計：USD {metrics['estimated_cost_usd_off_peak']:.4f}–{metrics['estimated_cost_usd_peak']:.4f}；中位延遲 {metrics['median_latency_ms']} ms。
"""
    (ROOT / config["report"]).write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
