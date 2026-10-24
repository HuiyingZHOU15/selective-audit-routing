"""Select the Jev confidence harness on calibration and evaluate once on test."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from llm_common import ROOT

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
BUDGETS = [0.10, 0.20, 0.30]
SEED = 20260927
RANDOM_ITERATIONS = 2000
BOOTSTRAP_ITERATIONS = 2000


def responses(path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "ok":
            rows.append(item)
    return pd.DataFrame(rows).drop_duplicates(["sample_id", "prompt_bundle_sha256"], keep="last")


def metrics(y, pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
    }


def apply(frame, route, threshold, labels) -> tuple[np.ndarray, dict]:
    y = frame["target_level1"].to_numpy()
    small = frame["small_prediction"].to_numpy(dtype=object)
    jev = frame["prediction"].to_numpy(dtype=object)
    use_jev = route & frame["confidence"].to_numpy(float).__ge__(threshold)
    pred = small.copy()
    pred[use_jev] = jev[use_jev]
    result = metrics(y, pred, labels)
    result.update({
        "routed_count": int(route.sum()),
        "jev_used_count": int(use_jev.sum()),
        "human_review_count": int((route & ~use_jev).sum()),
        "corrected_small_errors": int((use_jev & (small != y) & (jev == y)).sum()),
        "introduced_errors": int((use_jev & (small == y) & (jev != y)).sum()),
    })
    result["net_correct"] = result["corrected_small_errors"] - result["introduced_errors"]
    return pred, result


def exact_route(confidence, budget) -> np.ndarray:
    route = np.zeros(len(confidence), dtype=bool)
    route[np.argsort(confidence, kind="stable")[: int(round(len(confidence) * budget))]] = True
    return route


def cluster_bootstrap(frame, selected_correct, iterations, seed) -> dict:
    work = frame[["report_id"]].copy()
    work["small_correct"] = frame["small_prediction"].eq(frame["target_level1"]).to_numpy()
    work["selected_correct"] = selected_correct
    grouped = {key: value for key, value in work.groupby("report_id", sort=False)}
    reports = np.asarray(list(grouped))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sample = rng.choice(reports, len(reports), replace=True)
        boot = pd.concat([grouped[key] for key in sample], ignore_index=True)
        values.append(float(boot["selected_correct"].mean() - boot["small_correct"].mean()))
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
        "probability_positive": float(np.mean(np.asarray(values) > 0)),
    }


def main() -> None:
    definitions = json.loads((ROOT / "configs/llm_label_definitions_v2.json").read_text(encoding="utf-8"))
    labels = list(definitions)
    calibration = pd.read_csv(ROOT / "data/interim/jev_calibration_v2_candidates.csv").merge(
        responses(ROOT / "artifacts/llm_responses/jev_calibration_v2.jsonl"),
        on=["sample_id", "prompt_bundle_sha256"], validate="one_to_one",
    )
    test = pd.read_csv(ROOT / "data/interim/deepseek_core_v2_candidates.csv").merge(
        responses(ROOT / "artifacts/llm_responses/jev_core_v2.jsonl"),
        on=["sample_id", "prompt_bundle_sha256"], validate="one_to_one",
    )
    if len(calibration) != 608 or len(test) != 890:
        raise RuntimeError(f"Unexpected sample counts calibration={len(calibration)} test={len(test)}")

    grid = []
    for budget in BUDGETS:
        route = exact_route(calibration["small_confidence"].to_numpy(float), budget)
        for threshold in THRESHOLDS:
            _, result = apply(calibration, route, threshold, labels)
            grid.append({"budget": budget, "jev_confidence_threshold": threshold, **result})
    grid_frame = pd.DataFrame(grid)
    selected = max(
        grid,
        key=lambda row: (
            row["macro_f1"], row["accuracy"], -row["jev_used_count"], -row["routed_count"], row["jev_confidence_threshold"]
        ),
    )
    budget_pct = int(round(selected["budget"] * 100))
    test_route = test[f"route_{budget_pct}pct"].astype(bool).to_numpy()
    test_pred, test_result = apply(test, test_route, selected["jev_confidence_threshold"], labels)
    y_test = test["target_level1"].to_numpy()
    small_test = test["small_prediction"].to_numpy(dtype=object)

    rng = np.random.default_rng(SEED)
    random_acc, random_f1 = [], []
    for _ in range(RANDOM_ITERATIONS):
        random_route = np.zeros(len(test), dtype=bool)
        random_route[rng.choice(len(test), int(test_route.sum()), replace=False)] = True
        random_pred, random_result = apply(test, random_route, selected["jev_confidence_threshold"], labels)
        random_acc.append(random_result["accuracy"])
        random_f1.append(random_result["macro_f1"])

    selected_correct = test_pred == y_test
    accepted = (~test_route) | (test_route & (test["confidence"].to_numpy(float) >= selected["jev_confidence_threshold"]))
    payload = {
        "selection_policy": "maximize calibration macro_f1; then accuracy; then fewer Jev uses; then lower routed count; then higher threshold",
        "threshold_grid": THRESHOLDS,
        "budget_grid": BUDGETS,
        "selected": selected,
        "calibration_small_only": metrics(calibration["target_level1"], calibration["small_prediction"], labels),
        "test_small_only": metrics(y_test, small_test, labels),
        "test_locked_harness": test_result,
        "test_accuracy_delta": float(test_result["accuracy"] - np.mean(small_test == y_test)),
        "test_macro_f1_delta": float(test_result["macro_f1"] - metrics(y_test, small_test, labels)["macro_f1"]),
        "test_accepted_coverage_if_low_jev_confidence_goes_to_human": float(accepted.mean()),
        "test_accepted_accuracy_if_low_jev_confidence_goes_to_human": float(np.mean(selected_correct[accepted])),
        "random_route_same_budget": {
            "iterations": RANDOM_ITERATIONS,
            "accuracy_mean": float(np.mean(random_acc)),
            "accuracy_ci95": [float(np.quantile(random_acc, .025)), float(np.quantile(random_acc, .975))],
            "macro_f1_mean": float(np.mean(random_f1)),
            "macro_f1_ci95": [float(np.quantile(random_f1, .025)), float(np.quantile(random_f1, .975))],
            "probability_locked_accuracy_beats_random": float(np.mean(test_result["accuracy"] > np.asarray(random_acc))),
        },
        "report_cluster_bootstrap_delta": cluster_bootstrap(test, selected_correct, BOOTSTRAP_ITERATIONS, SEED),
        "estimated_deployment_cost_usd_for_test_batch": float(0.033106374 * selected["budget"]),
    }
    locked = {
        "version": "jev-harness-v2-locked",
        "selection_split": "calibration",
        "routing_budget": selected["budget"],
        "jev_confidence_threshold": selected["jev_confidence_threshold"],
        "fallback": "small_model",
        "selection_objective": payload["selection_policy"],
    }
    report = f"""# Jev harness 獨立校準與鎖定測試

## 鎖定規則

- 路由預算：{selected['budget']:.0%}
- Jev 接管門檻：confidence ≥ {selected['jev_confidence_threshold']:.2f}
- 低於門檻：回退小模型；也可作為人工覆核隊列
- 選擇資料：608 筆 clean calibration；test 標籤未參與選擇

## 一次性 test 結果

- 小模型：accuracy {payload['test_small_only']['accuracy']:.4f}，macro-F1 {payload['test_small_only']['macro_f1']:.4f}
- 鎖定 harness：accuracy {test_result['accuracy']:.4f}，macro-F1 {test_result['macro_f1']:.4f}
- 差值：accuracy {payload['test_accuracy_delta']:+.4f}，macro-F1 {payload['test_macro_f1_delta']:+.4f}
- 路由 {test_result['routed_count']}，Jev 實際接管 {test_result['jev_used_count']}，修正 {test_result['corrected_small_errors']}，引入 {test_result['introduced_errors']}，淨正確 {test_result['net_correct']:+d}
- report-cluster bootstrap accuracy 差 95% CI：[{payload['report_cluster_bootstrap_delta']['ci95'][0]:+.4f}, {payload['report_cluster_bootstrap_delta']['ci95'][1]:+.4f}]
- 同預算隨機路由 accuracy：{payload['random_route_same_budget']['accuracy_mean']:.4f}，95%區間 [{payload['random_route_same_budget']['accuracy_ci95'][0]:.4f}, {payload['random_route_same_budget']['accuracy_ci95'][1]:.4f}]
- 若低 Jev 信心樣本交人工：自動覆蓋率 {payload['test_accepted_coverage_if_low_jev_confidence_goes_to_human']:.2%}，已覆蓋 accuracy {payload['test_accepted_accuracy_if_low_jev_confidence_goes_to_human']:.4f}
- 該批次部署成本估計：USD {payload['estimated_deployment_cost_usd_for_test_batch']:.4f}

門檻與預算由固定網格在 calibration 上選擇，test 僅用於最終一次性評估。主結果不使用 test 後驗調整。
"""
    outputs = {
        ROOT / "configs/jev_harness_v2_locked.json": json.dumps(locked, ensure_ascii=False, indent=2),
        ROOT / "results/metrics/jev_harness_v2_locked.json": json.dumps(payload, ensure_ascii=False, indent=2),
        ROOT / "results/reports/jev_harness_v2_locked_zh.md": report,
    }
    grid_path = ROOT / "results/tables/jev_harness_calibration_grid_v2.csv"
    prediction_path = ROOT / "results/tables/jev_harness_locked_test_v2.csv"
    for path in [*outputs, grid_path, prediction_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    grid_frame.to_csv(grid_path, index=False, encoding="utf-8")
    test_output = test[["sample_id", "report_id", "target_level1", "small_prediction", "small_confidence", "prediction", "confidence"]].copy()
    test_output["routed"] = test_route
    test_output["locked_prediction"] = test_pred
    test_output["locked_correct"] = selected_correct
    test_output.to_csv(prediction_path, index=False, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
