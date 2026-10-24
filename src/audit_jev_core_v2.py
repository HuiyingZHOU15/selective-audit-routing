"""Full-test audit of Jev and small-model-to-Jev selective routing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

from llm_common import ROOT

SEED = 20260927
RANDOM_ITERATIONS = 2000
BOOTSTRAP_ITERATIONS = 500


def pair(y, pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
    }


def binary_ece(correct, confidence, bins=15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def cluster_bootstrap(audit, correct_column, iterations, seed) -> dict:
    rng = np.random.default_rng(seed)
    report_ids = audit["report_id"].drop_duplicates().to_numpy()
    grouped = {key: frame for key, frame in audit.groupby("report_id", sort=False)}
    deltas = []
    for _ in range(iterations):
        sample = rng.choice(report_ids, len(report_ids), replace=True)
        frame = pd.concat([grouped[key] for key in sample], ignore_index=True)
        deltas.append(float(frame[correct_column].mean() - frame["small_correct"].mean()))
    return {
        "mean": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
        "probability_positive": float(np.mean(np.asarray(deltas) > 0)),
    }


def main() -> None:
    candidates = pd.read_csv(ROOT / "data/interim/deepseek_core_v2_candidates.csv")
    definitions = json.loads((ROOT / "configs/llm_label_definitions_v2.json").read_text(encoding="utf-8"))
    labels = list(definitions)
    responses = []
    for line in (ROOT / "artifacts/llm_responses/jev_core_v2.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "ok":
            responses.append(item)
    response_frame = pd.DataFrame(responses).drop_duplicates(["sample_id", "prompt_bundle_sha256"], keep="last")
    audit = candidates.merge(response_frame, on=["sample_id", "prompt_bundle_sha256"], validate="one_to_one")
    if len(audit) != len(candidates):
        raise RuntimeError(f"Expected {len(candidates)} responses, found {len(audit)}")
    y = audit["target_level1"].to_numpy()
    small = audit["small_prediction"].to_numpy(dtype=object)
    jev = audit["prediction"].to_numpy(dtype=object)
    audit["small_correct"] = small == y
    audit["jev_correct"] = jev == y
    audit["jev_corrected_small"] = audit["jev_correct"] & ~audit["small_correct"]
    audit["jev_introduced_error"] = ~audit["jev_correct"] & audit["small_correct"]
    confidence = audit["confidence"].to_numpy(float)

    probability_labels = sorted(labels)
    probability_matrix = np.asarray([
        [row.get(label, 0.0) for label in probability_labels]
        for row in audit["probabilities"]
    ], dtype=float)
    row_sums = probability_matrix.sum(axis=1, keepdims=True)
    probability_matrix = np.divide(probability_matrix, row_sums, out=np.zeros_like(probability_matrix), where=row_sums != 0)
    rng = np.random.default_rng(SEED)
    strategies = []
    for budget in (10, 20, 30):
        route = audit[f"route_{budget}pct"].astype(bool).to_numpy()
        pred = small.copy()
        pred[route] = jev[route]
        audit[f"small_to_jev_{budget}_prediction"] = pred
        correct_column = f"small_to_jev_{budget}_correct"
        audit[correct_column] = pred == y
        observed = pair(y, pred, labels)
        k = int(route.sum())
        random_accuracy, random_macro_f1 = [], []
        for _ in range(RANDOM_ITERATIONS):
            random_route = np.zeros(len(audit), dtype=bool)
            random_route[rng.choice(len(audit), k, replace=False)] = True
            random_pred = small.copy()
            random_pred[random_route] = jev[random_route]
            random_accuracy.append(float(np.mean(random_pred == y)))
            random_macro_f1.append(float(f1_score(y, random_pred, labels=labels, average="macro", zero_division=0)))
        corrected = int((route & audit["jev_corrected_small"].to_numpy()).sum())
        introduced = int((route & audit["jev_introduced_error"].to_numpy()).sum())
        high_confidence = route & (confidence >= .8)
        gated_pred = small.copy()
        gated_pred[high_confidence] = jev[high_confidence]
        accepted = (~route) | high_confidence
        correctable = audit["jev_corrected_small"].to_numpy()
        strategies.append({
            "budget": budget / 100, "routed_count": k, **observed,
            "corrected_small_errors": corrected, "introduced_errors": introduced,
            "net_correct": corrected - introduced,
            "random_accuracy_mean": float(np.mean(random_accuracy)),
            "random_accuracy_ci95": [float(np.quantile(random_accuracy, .025)), float(np.quantile(random_accuracy, .975))],
            "random_macro_f1_mean": float(np.mean(random_macro_f1)),
            "random_macro_f1_ci95": [float(np.quantile(random_macro_f1, .025)), float(np.quantile(random_macro_f1, .975))],
            "probability_beats_random": float(np.mean(observed["accuracy"] > np.asarray(random_accuracy))),
            "oracle_accuracy_upper": float((audit["small_correct"].sum() + min(k, int(correctable.sum()))) / len(audit)),
            "report_cluster_bootstrap_accuracy_delta": cluster_bootstrap(audit, correct_column, BOOTSTRAP_ITERATIONS, SEED + budget),
            "exploratory_confidence_gate_0_8": {
                **pair(y, gated_pred, labels),
                "jev_used_count": int(high_confidence.sum()),
                "human_review_count": int((route & ~high_confidence).sum()),
                "accepted_coverage": float(accepted.mean()),
                "accepted_accuracy": float(np.mean(gated_pred[accepted] == y[accepted])),
            },
        })

    usage = audit["usage"].map(lambda value: value if isinstance(value, dict) else {})
    input_tokens = int(usage.map(lambda value: value.get("input_tokens", 0)).sum())
    output_tokens = int(usage.map(lambda value: value.get("output_tokens", 0)).sum())
    per_class = audit.groupby("target_level1").agg(
        n=("sample_id", "size"), small_accuracy=("small_correct", "mean"),
        jev_accuracy=("jev_correct", "mean"), mean_jev_confidence=("confidence", "mean"),
        jev_corrected_small=("jev_corrected_small", "sum"), jev_introduced_error=("jev_introduced_error", "sum"),
    ).reset_index()
    metrics = {
        "version": "jev-core-v2-1", "model": str(audit["response_model"].mode().iloc[0]),
        "n": len(audit), "small_only": pair(y, small, labels), "jev_only": pair(y, jev, labels),
        "jev_binary_ece_15bin": binary_ece(audit["jev_correct"].to_numpy(float), confidence),
        "jev_confidence_brier": float(np.mean((confidence - audit["jev_correct"].to_numpy(float)) ** 2)),
        "jev_multiclass_log_loss": float(log_loss(y, probability_matrix, labels=probability_labels)),
        "jev_high_confidence_errors_ge_0_8": int(((confidence >= .8) & ~audit["jev_correct"].to_numpy()).sum()),
        "jev_corrected_small_total": int(audit["jev_corrected_small"].sum()),
        "jev_introduced_error_total": int(audit["jev_introduced_error"].sum()),
        "strategies": strategies,
        "median_latency_ms": float(audit["latency_ms"].median()),
        "p95_latency_ms": float(audit["latency_ms"].quantile(.95)),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "estimated_cost_usd": input_tokens * 0.042 / 1_000_000,
        "per_class": per_class.to_dict(orient="records"),
    }
    table = "\n".join(
        f"| {s['budget']:.0%} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} | {s['net_correct']:+d} | "
        f"{s['random_accuracy_mean']:.4f} [{s['random_accuracy_ci95'][0]:.4f}, {s['random_accuracy_ci95'][1]:.4f}] | "
        f"{s['report_cluster_bootstrap_accuracy_delta']['ci95'][0]:+.4f}, {s['report_cluster_bootstrap_accuracy_delta']['ci95'][1]:+.4f} |"
        for s in strategies
    )
    class_table = "\n".join(
        f"| {r.target_level1} | {int(r.n)} | {r.small_accuracy:.3f} | {r.jev_accuracy:.3f} | {r.mean_jev_confidence:.3f} |"
        for r in per_class.itertuples()
    )
    report = f"""# Jev 全測試集核心實驗（taxonomy v2）

## 總體

- 樣本與回應：{len(audit)}/{len(candidates)}
- 小模型：accuracy {metrics['small_only']['accuracy']:.4f}，macro-F1 {metrics['small_only']['macro_f1']:.4f}
- Jev-only：accuracy {metrics['jev_only']['accuracy']:.4f}，macro-F1 {metrics['jev_only']['macro_f1']:.4f}
- Jev ECE：{metrics['jev_binary_ece_15bin']:.4f}；confidence Brier：{metrics['jev_confidence_brier']:.4f}
- Jev 高信心錯誤（≥0.8）：{metrics['jev_high_confidence_errors_ge_0_8']}
- 中位／P95 延遲：{metrics['median_latency_ms']:.1f}/{metrics['p95_latency_ms']:.1f} ms
- 估算成本：USD {metrics['estimated_cost_usd']:.4f}

## Selective routing

| 預算 | 小模型→Jev accuracy | macro-F1 | 淨正確數 | 隨機路由 accuracy 均值 [95%區間] | report-bootstrap accuracy 差 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

## 分類別

| 類別 | n | 小模型 accuracy | Jev accuracy | Jev 平均信心 |
| --- | ---: | ---: | ---: | ---: |
{class_table}

主結果直接以 Jev 覆蓋被路由樣本；0.8 信心門控只作探索性診斷，未作主結論。隨機路由使用相同路由數重複 {RANDOM_ITERATIONS} 次；不確定性以 report 為叢集重抽樣 {BOOTSTRAP_ITERATIONS} 次。
"""
    outputs = [
        ROOT / "results/tables/jev_core_v2_audit.csv",
        ROOT / "results/metrics/jev_core_v2.json",
        ROOT / "results/reports/jev_core_v2_zh.md",
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(outputs[0], index=False, encoding="utf-8")
    outputs[1].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs[2].write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
