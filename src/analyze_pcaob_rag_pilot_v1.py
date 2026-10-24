"""Paired statistical analysis for the completed PCAOB RAG pilot."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from llm_common import ROOT
from prepare_pcaob_rag_pilot_v1 import load_config


def wilson(successes: int, n: int, alpha: float = 0.05) -> list[float]:
    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(centre - half), float(centre + half)]


def main() -> None:
    config = load_config()
    audit = pd.read_csv(ROOT / config["audit_table"])
    observed = audit.loc[
        audit["completed"] & audit["condition"].isin(["direct", "report_rag"])
    ].copy()
    paired = observed.pivot(index="sample_id", columns="condition", values="prediction_correct")
    paired = paired.dropna().astype(bool)
    beneficial = int((~paired["direct"] & paired["report_rag"]).sum())
    harmful = int((paired["direct"] & ~paired["report_rag"]).sum())
    discordant = beneficial + harmful
    mcnemar_p = float(binomtest(min(beneficial, harmful), discordant, 0.5).pvalue) if discordant else 1.0

    delta = paired["report_rag"].astype(int) - paired["direct"].astype(int)
    rng = np.random.default_rng(20261002)
    values = delta.to_numpy()
    boot = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)

    conditions = {}
    for condition, group in observed.groupby("condition"):
        correct = int(group["prediction_correct"].sum())
        n = len(group)
        conditions[condition] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "accuracy_wilson_95": wilson(correct, n),
            "evidence_exact_rate": float(group["evidence_exact"].mean()),
            "evidence_valid_rate": float(group["evidence_normalized"].mean()),
            "mean_input_tokens": float(group["input_tokens"].mean()),
            "mean_latency_ms": float(group["latency_ms"].mean()),
        }

    # Context counts are condition-specific; take them from the report-RAG row,
    # not from the first (usually direct) row for each sample.
    metadata = observed.loc[observed["condition"].eq("report_rag")].set_index("sample_id")
    paired_meta = paired.join(metadata[["pilot_stratum", "report_context_count", "target_level1"]])
    strata = {}
    for name, group in paired_meta.groupby("pilot_stratum"):
        b = int((~group["direct"] & group["report_rag"]).sum())
        h = int((group["direct"] & ~group["report_rag"]).sum())
        strata[name] = {
            "n": len(group),
            "direct_accuracy": float(group["direct"].mean()),
            "report_rag_accuracy": float(group["report_rag"].mean()),
            "beneficial": b,
            "harmful": h,
            "net": b - h,
        }

    context_groups = {}
    for available, group in paired_meta.groupby(paired_meta["report_context_count"].gt(0)):
        b = int((~group["direct"] & group["report_rag"]).sum())
        h = int((group["direct"] & ~group["report_rag"]).sum())
        context_groups["available" if available else "unavailable"] = {
            "n": len(group),
            "direct_accuracy": float(group["direct"].mean()),
            "report_rag_accuracy": float(group["report_rag"].mean()),
            "beneficial": b,
            "harmful": h,
            "net": b - h,
        }

    direct = conditions["direct"]
    rag = conditions["report_rag"]
    result = {
        "version": config["version"],
        "design_note": "Exploratory development-only, outcome-stratified sample; not a population prevalence estimate or confirmatory test.",
        "protocol_deviation": "The first 216 valid responses used max_output_tokens=800. The final 24 previously truncated/empty responses used 2000; model, prompts, samples, and decoding settings were unchanged. Any confirmatory run must use 2000 uniformly.",
        "paired_n": len(paired),
        "beneficial": beneficial,
        "harmful": harmful,
        "net": beneficial - harmful,
        "accuracy_delta": float(delta.mean()),
        "accuracy_delta_cluster_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "conditions": conditions,
        "cost_change": {
            "input_tokens_absolute": rag["mean_input_tokens"] - direct["mean_input_tokens"],
            "input_tokens_relative": rag["mean_input_tokens"] / direct["mean_input_tokens"] - 1,
            "latency_ms_absolute": rag["mean_latency_ms"] - direct["mean_latency_ms"],
            "latency_relative": rag["mean_latency_ms"] / direct["mean_latency_ms"] - 1,
        },
        "by_pilot_stratum": strata,
        "by_report_context_availability": context_groups,
    }

    out_json = ROOT / "../rag_pilot_runtime/pcaob_rag_pilot_v1_statistical_analysis.json"
    out_json.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# PCAOB report-RAG feasibility pilot：統計判讀", "",
        "> Development-only 探索性、依 outcome 分層的 120-report 樣本；不得當作母體盛行率或確認性檢驗。", "",
        "> Protocol deviation：前 216 個有效回傳使用 800-token 輸出上限；最後 24 個先前被截斷／空白的回傳使用 2,000-token 上限。模型、prompt、樣本與其他 decoding 設定未變；任何確認性重跑必須全程統一使用 2,000。", "",
        f"- Direct accuracy：{direct['accuracy']:.2%}（{direct['correct']}/{direct['n']}）。",
        f"- Report-RAG accuracy：{rag['accuracy']:.2%}（{rag['correct']}/{rag['n']}）。",
        f"- 配對轉移：beneficial {beneficial}、harmful {harmful}、net {beneficial-harmful}；accuracy delta {delta.mean():+.2%}。",
        f"- Report-cluster bootstrap 95% CI：[{np.quantile(boot, .025):+.2%}, {np.quantile(boot, .975):+.2%}]。",
        f"- Exact McNemar p = {mcnemar_p:.4f}（探索性；沒有達到傳統 .05 門檻）。", "",
        "## 分層結果", "", "| pilot stratum | n | direct | report-RAG | beneficial | harmful | net |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in strata.items():
        lines.append(f"| {name} | {value['n']} | {value['direct_accuracy']:.2%} | {value['report_rag_accuracy']:.2%} | {value['beneficial']} | {value['harmful']} | {value['net']:+d} |")
    lines += ["", "## 成本與證據", "",
              f"- Mean input tokens：{direct['mean_input_tokens']:.1f} → {rag['mean_input_tokens']:.1f}（{result['cost_change']['input_tokens_relative']:+.1%}）。",
              f"- Mean latency：{direct['mean_latency_ms']:.1f} ms → {rag['mean_latency_ms']:.1f} ms（{result['cost_change']['latency_relative']:+.1%}）。",
              f"- Report-RAG evidence validity：{rag['evidence_valid_rate']:.2%}；exact quote：{rag['evidence_exact_rate']:.2%}。", "",
              "## 決策", "",
              "Report context 出現正淨收益，通過最弱的 feasibility go 條件；但只有 7 個 beneficial events、信賴區間跨 0，不能據此宣稱 RAG 已被證明有效，也不足以單獨訓練穩定的 learned router。下一步應補齊 standards/combined RAG 或轉到 FinAuditing structured-RAG 任務，以確認可重複的 Route-C events。"]
    out_md = ROOT / "../rag_pilot_runtime/pcaob_rag_pilot_v1_statistical_analysis_zh.md"
    out_md.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
