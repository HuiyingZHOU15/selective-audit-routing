"""Calibration-only ablation of simple and audit-aware Jev acceptance gates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from llm_common import ROOT
from train_audit_aware_router_v2 import SEED, exact_route, load_split, system_metrics


BUDGET = 0.20
FOLDS = 5
THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.02), 2)
JEV_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
COMPACT_NUMERIC = ["small_confidence", "jev_confidence", "model_agree"]
AUDIT_NUMERIC = COMPACT_NUMERIC + [
    "context_reference", "account_topic_hits", "crosscut_hits", "multi_topic", "text_log_words"
]
AUDIT_CATEGORICAL = ["small_prediction", "jev_prediction", "context_quality"]


def model(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("features", pre),
        ("classifier", LogisticRegression(class_weight="balanced", C=0.25, max_iter=2000, random_state=SEED)),
    ])


def learned_candidate(calibration, test, route_cal, route_test, numeric, categorical, route_only):
    columns = numeric + categorical
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(calibration), np.nan)
    for train, valid in splitter.split(calibration, calibration["outcome"], calibration["report_id"]):
        train_mask = calibration.iloc[train]["outcome"].isin(["benefit", "harm"]).to_numpy()
        if route_only:
            train_mask &= route_cal[train]
        train_decisive = train[train_mask]
        gate = model(numeric, categorical)
        gate.fit(calibration.iloc[train_decisive][columns], calibration.iloc[train_decisive]["outcome"])
        benefit = list(gate.named_steps["classifier"].classes_).index("benefit")
        oof[valid] = gate.predict_proba(calibration.iloc[valid][columns])[:, benefit]
    grid = []
    for threshold in THRESHOLDS:
        use = route_cal & (oof >= threshold)
        grid.append({"threshold": float(threshold), **system_metrics(calibration, use, LABELS)})
    selected = max(grid, key=lambda x: (x["macro_f1"], x["accuracy"], -x["jev_used_count"]))
    decisive = calibration["outcome"].isin(["benefit", "harm"]).to_numpy()
    if route_only:
        decisive &= route_cal
    gate = model(numeric, categorical)
    gate.fit(calibration.loc[decisive, columns], calibration.loc[decisive, "outcome"])
    benefit = list(gate.named_steps["classifier"].classes_).index("benefit")
    test_score = gate.predict_proba(test[columns])[:, benefit]
    test_result = system_metrics(test, route_test & (test_score >= selected["threshold"]), LABELS)
    return selected, test_result


def main() -> None:
    global LABELS
    LABELS = list(json.loads((ROOT / "configs/llm_label_definitions_v2.json").read_text(encoding="utf-8")))
    calibration, _ = load_split("calibration")
    test, _ = load_split("test")
    route_cal = exact_route(calibration["small_confidence"].to_numpy(float), BUDGET)
    route_test = exact_route(test["small_confidence"].to_numpy(float), BUDGET)
    rows = []

    for threshold in JEV_THRESHOLDS:
        cal = system_metrics(calibration, route_cal & calibration["jev_confidence"].ge(threshold).to_numpy(), LABELS)
        rows.append({"candidate": "jev_confidence_only", "train_scope": "none", "threshold": threshold, "calibration": cal})

    for name, numeric, categorical in [
        ("compact_model_signals", COMPACT_NUMERIC, []),
        ("audit_aware_signals", AUDIT_NUMERIC, AUDIT_CATEGORICAL),
    ]:
        for route_only in [False, True]:
            selected, test_result = learned_candidate(
                calibration, test, route_cal, route_test, numeric, categorical, route_only
            )
            rows.append({
                "candidate": name,
                "train_scope": "routed_decisive_only" if route_only else "all_decisive",
                "threshold": selected["threshold"],
                "calibration": {k: v for k, v in selected.items() if k != "threshold"},
                "test": test_result,
            })

    confidence_rows = [row for row in rows if row["candidate"] == "jev_confidence_only"]
    best_confidence = max(confidence_rows, key=lambda x: (x["calibration"]["macro_f1"], x["calibration"]["accuracy"], -x["calibration"]["jev_used_count"]))
    best_confidence["test"] = system_metrics(
        test, route_test & test["jev_confidence"].ge(best_confidence["threshold"]).to_numpy(), LABELS
    )
    finalists = [best_confidence] + [row for row in rows if row["candidate"] != "jev_confidence_only"]
    selected = max(finalists, key=lambda x: (x["calibration"]["macro_f1"], x["calibration"]["accuracy"], -x["calibration"]["jev_used_count"]))
    flat = []
    for row in finalists:
        flat.append({
            "candidate": row["candidate"], "train_scope": row["train_scope"], "threshold": row["threshold"],
            **{f"cal_{k}": v for k, v in row["calibration"].items()},
            **{f"test_{k}": v for k, v in row["test"].items()},
            "selected_on_calibration": row is selected,
        })
    output = pd.DataFrame(flat)
    table_path = ROOT / "results/tables/audit_router_feature_ablation_v2.csv"
    metrics_path = ROOT / "results/metrics/audit_router_feature_ablation_v2.json"
    report_path = ROOT / "results/reports/audit_router_feature_ablation_v2_zh.md"
    for path in [table_path, metrics_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(table_path, index=False, encoding="utf-8-sig")
    payload = {
        "selection_split": "calibration",
        "fixed_route_budget": BUDGET,
        "selected_candidate": selected,
        "candidates": flat,
        "test_status": "exploratory_after_prior_test_inspection",
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = "\n".join(
        f"| {r['candidate']} | {r['train_scope']} | {r['threshold']:.2f} | {r['cal_macro_f1']:.4f} | {r['cal_accuracy']:.4f} | {r['test_macro_f1']:.4f} | {r['test_accuracy']:.4f} |"
        for r in flat
    )
    report = f"""# Audit-aware router 特徵消融

| candidate | training scope | threshold | calibration OOF macro-F1 | calibration accuracy | exploratory test macro-F1 | exploratory test accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{lines}

僅按 calibration 選中的方案：**{selected['candidate']} / {selected['train_scope']} / threshold {selected['threshold']:.2f}**。

新的 learned gate 使用 report-group OOF 分數選門檻。既有 test 已被查看，因此 test 欄只作探索性診斷，不作新的確認性主結果。
"""
    report_path.write_text(report, encoding="utf-8")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
