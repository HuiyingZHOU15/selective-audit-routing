"""Train an audit-aware comparative-advantage gate for small model vs Jev.

The gate is developed only on the clean calibration split.  It predicts
whether Jev is preferable to the small model after a sample has entered the
fixed 20% low-small-confidence route.  Test results are exploratory because
the test aggregate has already been inspected during earlier diagnosis.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from llm_common import ROOT


SEED = 20260927
BUDGET = 0.20
FOLDS = 5
BOOTSTRAPS = 2000
MODEL_PATH = ROOT / "artifacts/models/tfidf_lr_context_taxonomy_v2.joblib"
MODEL_OUT = ROOT / "artifacts/models/audit_aware_jev_gate_v2.joblib"
CAL_OUT = ROOT / "results/tables/audit_aware_router_calibration_oof_v2.csv"
TEST_OUT = ROOT / "results/tables/audit_aware_router_test_v2.csv"
PROFILE_OUT = ROOT / "results/tables/error_feature_profile_v2.csv"
METRICS_OUT = ROOT / "results/metrics/audit_aware_router_v2.json"
REPORT_OUT = ROOT / "results/reports/audit_aware_router_v2_zh.md"

NUMERIC = [
    "small_confidence", "small_margin", "small_entropy", "jev_confidence",
    "jev_margin", "jev_entropy", "model_agree", "text_log_chars",
    "text_log_words", "context_reference", "account_topic_hits",
    "crosscut_hits", "multi_topic",
]
CATEGORICAL = ["small_prediction", "jev_prediction", "context_quality"]

ACCOUNT_PATTERNS = [
    r"revenue|receiv", r"inventory|cost of sales", r"goodwill|intangible|property|impairment|lease",
    r"loan|credit loss|deposit|insurance", r"investment|securit|derivative|financial instrument",
    r"debt|equity|warrant|liabilit|payable", r"income tax|deferred tax",
    r"expense|compensation|pension|benefit|payroll", r"business combination|acquisition|consolidat",
    r"related part|significant transaction", r"cash",
]
CROSSCUT_PATTERNS = [r"control", r"estimate|valuation", r"audit procedure|audit evidence", r"journal entr", r"presentation|disclosure|reporting"]
CONTEXT_PATTERN = re.compile(
    r"\b(?:these|those|such|aforementioned|above-mentioned)\s+"
    r"(?:assets?|accounts?|transactions?|controls?|data|assumptions?|liabilities|procedures?)\b|"
    r"\bthis control\b|\bas discussed above\b|\bprocedures discussed above\b",
    re.I,
)


def read_responses(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "ok":
            rows.append(item)
    return pd.DataFrame(rows).drop_duplicates(["sample_id", "prompt_bundle_sha256"], keep="last")


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1) / math.log(probabilities.shape[1])


def margins(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def exact_route(confidence: np.ndarray, budget: float) -> np.ndarray:
    route = np.zeros(len(confidence), dtype=bool)
    route[np.argsort(confidence, kind="stable")[: int(round(len(confidence) * budget))]] = True
    return route


def load_split(split: str) -> tuple[pd.DataFrame, np.ndarray]:
    if split == "calibration":
        candidate_path = ROOT / "data/interim/jev_calibration_v2_candidates.csv"
        response_path = ROOT / "artifacts/llm_responses/jev_calibration_v2.jsonl"
    else:
        candidate_path = ROOT / "data/interim/deepseek_core_v2_candidates.csv"
        response_path = ROOT / "artifacts/llm_responses/jev_core_v2.jsonl"
    frame = pd.read_csv(candidate_path).merge(
        read_responses(response_path), on=["sample_id", "prompt_bundle_sha256"], validate="one_to_one"
    )
    model = joblib.load(MODEL_PATH)
    small_prob = model.predict_proba(frame["text"].fillna(""))
    classes = list(model.classes_)
    jev_prob = np.asarray(
        [[float(row.get(label, 0.0)) for label in classes] for row in frame["probabilities"]], dtype=float
    )
    jev_prob = jev_prob / np.clip(jev_prob.sum(axis=1, keepdims=True), 1e-12, None)
    frame["small_margin"] = margins(small_prob)
    frame["small_entropy"] = entropy(small_prob)
    frame["jev_prediction"] = frame["prediction"]
    frame["jev_confidence"] = frame["confidence"].astype(float)
    frame["jev_margin"] = margins(jev_prob)
    frame["jev_entropy"] = entropy(jev_prob)
    frame["model_agree"] = frame["small_prediction"].eq(frame["jev_prediction"]).astype(int)
    text = frame["text"].fillna("").astype(str)
    lower = text.str.lower()
    frame["text_log_chars"] = np.log1p(text.str.len())
    frame["text_log_words"] = np.log1p(text.str.split().str.len())
    frame["context_reference"] = text.str.contains(CONTEXT_PATTERN).astype(int)
    frame["account_topic_hits"] = sum(lower.str.contains(pattern, regex=True).astype(int) for pattern in ACCOUNT_PATTERNS)
    frame["crosscut_hits"] = sum(lower.str.contains(pattern, regex=True).astype(int) for pattern in CROSSCUT_PATTERNS)
    frame["multi_topic"] = frame["account_topic_hits"].ge(2).astype(int)
    frame["small_correct"] = frame["small_prediction"].eq(frame["target_level1"])
    frame["jev_correct"] = frame["jev_prediction"].eq(frame["target_level1"])
    frame["outcome"] = np.select(
        [~frame["small_correct"] & frame["jev_correct"], frame["small_correct"] & ~frame["jev_correct"]],
        ["benefit", "harm"], default="neutral",
    )
    return frame, small_prob


def make_gate() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("features", pre),
            ("classifier", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=SEED)),
        ]
    )


def system_metrics(frame: pd.DataFrame, use_jev: np.ndarray, labels: list[str]) -> dict:
    y = frame["target_level1"].to_numpy()
    pred = frame["small_prediction"].to_numpy(dtype=object).copy()
    pred[use_jev] = frame.loc[use_jev, "jev_prediction"].to_numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "jev_used_count": int(use_jev.sum()),
        "corrected_small_errors": int((use_jev & ~frame["small_correct"] & frame["jev_correct"]).sum()),
        "introduced_errors": int((use_jev & frame["small_correct"] & ~frame["jev_correct"]).sum()),
    }


def cluster_bootstrap(frame: pd.DataFrame, selected_correct: np.ndarray) -> dict:
    work = pd.DataFrame({
        "report_id": frame["report_id"],
        "small": frame["small_correct"].to_numpy(bool),
        "selected": selected_correct,
    })
    grouped = {key: value for key, value in work.groupby("report_id", sort=False)}
    reports = np.asarray(list(grouped))
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(BOOTSTRAPS):
        boot = pd.concat([grouped[key] for key in rng.choice(reports, len(reports), replace=True)])
        deltas.append(float(boot["selected"].mean() - boot["small"].mean()))
    return {
        "mean": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
        "probability_positive": float(np.mean(np.asarray(deltas) > 0)),
    }


def main() -> None:
    definitions = json.loads((ROOT / "configs/llm_label_definitions_v2.json").read_text(encoding="utf-8"))
    labels = list(definitions)
    calibration, _ = load_split("calibration")
    test, _ = load_split("test")

    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(calibration), np.nan)
    stratify = calibration["outcome"]
    for train_index, valid_index in splitter.split(calibration, stratify, calibration["report_id"]):
        decisive = calibration.iloc[train_index]["outcome"].isin(["benefit", "harm"]).to_numpy()
        decisive_index = train_index[decisive]
        gate = make_gate()
        gate.fit(calibration.iloc[decisive_index][NUMERIC + CATEGORICAL], calibration.iloc[decisive_index]["outcome"])
        class_position = list(gate.named_steps["classifier"].classes_).index("benefit")
        oof[valid_index] = gate.predict_proba(calibration.iloc[valid_index][NUMERIC + CATEGORICAL])[:, class_position]
    if np.isnan(oof).any():
        raise RuntimeError("Missing out-of-fold gate scores.")

    calibration_route = exact_route(calibration["small_confidence"].to_numpy(float), BUDGET)
    threshold_grid = np.round(np.arange(0.20, 0.81, 0.02), 2)
    grid = []
    for threshold in threshold_grid:
        use = calibration_route & (oof >= threshold)
        result = system_metrics(calibration, use, labels)
        grid.append({"gate_threshold": float(threshold), **result})
    selected = max(
        grid,
        key=lambda row: (row["macro_f1"], row["accuracy"], row["corrected_small_errors"] - row["introduced_errors"], -row["jev_used_count"]),
    )

    decisive_mask = calibration["outcome"].isin(["benefit", "harm"])
    gate = make_gate()
    gate.fit(calibration.loc[decisive_mask, NUMERIC + CATEGORICAL], calibration.loc[decisive_mask, "outcome"])
    class_position = list(gate.named_steps["classifier"].classes_).index("benefit")
    test_score = gate.predict_proba(test[NUMERIC + CATEGORICAL])[:, class_position]
    test_route = exact_route(test["small_confidence"].to_numpy(float), BUDGET)
    gate_use = test_route & (test_score >= selected["gate_threshold"])
    locked_use = test_route & test["jev_confidence"].ge(0.70).to_numpy()
    gate_result = system_metrics(test, gate_use, labels)
    locked_result = system_metrics(test, locked_use, labels)
    small_result = system_metrics(test, np.zeros(len(test), dtype=bool), labels)
    gate_pred = test["small_prediction"].to_numpy(dtype=object).copy()
    gate_pred[gate_use] = test.loc[gate_use, "jev_prediction"].to_numpy()
    selected_correct = gate_pred == test["target_level1"].to_numpy()

    decisive_oof = calibration["outcome"].isin(["benefit", "harm"])
    auc = roc_auc_score(calibration.loc[decisive_oof, "outcome"].eq("benefit"), oof[decisive_oof])
    payload = {
        "version": "audit-aware-router-v2-1",
        "development_split": "calibration only",
        "fixed_route_budget": BUDGET,
        "calibration_samples": int(len(calibration)),
        "calibration_outcomes": calibration["outcome"].value_counts().to_dict(),
        "grouped_oof_folds": FOLDS,
        "oof_decisive_auc": float(auc),
        "selected_gate_threshold": selected["gate_threshold"],
        "selected_calibration_oof": selected,
        "test_status": "exploratory_after_prior_test_inspection",
        "test_small_only": small_result,
        "test_locked_confidence_gate": locked_result,
        "test_audit_aware_gate": gate_result,
        "test_gate_accuracy_delta_vs_small": gate_result["accuracy"] - small_result["accuracy"],
        "test_gate_macro_f1_delta_vs_small": gate_result["macro_f1"] - small_result["macro_f1"],
        "test_report_cluster_bootstrap_delta": cluster_bootstrap(test, selected_correct),
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    }

    cal_output = calibration[["sample_id", "report_id", "target_level1", "small_prediction", "jev_prediction", "small_correct", "jev_correct", "outcome"] + NUMERIC].copy()
    cal_output["gate_oof_benefit_probability"] = oof
    cal_output["route_20pct"] = calibration_route
    cal_output["gate_would_use_jev"] = calibration_route & (oof >= selected["gate_threshold"])
    test_output = test[["sample_id", "report_id", "target_level1", "small_prediction", "jev_prediction", "small_correct", "jev_correct", "outcome"] + NUMERIC].copy()
    test_output["gate_benefit_probability"] = test_score
    test_output["route_20pct"] = test_route
    test_output["gate_use_jev"] = gate_use
    test_output["gate_prediction"] = gate_pred
    test_output["gate_correct"] = selected_correct

    profile = (
        test.groupby("outcome")[NUMERIC]
        .mean(numeric_only=True)
        .assign(n=test.groupby("outcome").size())
        .reset_index()
    )
    for path in [MODEL_OUT, CAL_OUT, TEST_OUT, PROFILE_OUT, METRICS_OUT, REPORT_OUT]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gate, MODEL_OUT)
    cal_output.to_csv(CAL_OUT, index=False, encoding="utf-8-sig")
    test_output.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")
    profile.to_csv(PROFILE_OUT, index=False, encoding="utf-8-sig")
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 審計特性感知比較優勢路由器（探索性）

- 開發資料：clean calibration {len(calibration)} 筆、{calibration['report_id'].nunique()} 份報告。
- outcome：benefit {payload['calibration_outcomes'].get('benefit', 0)}、harm {payload['calibration_outcomes'].get('harm', 0)}、neutral {payload['calibration_outcomes'].get('neutral', 0)}。
- 5-fold report-group OOF decisive AUC：{auc:.4f}。
- 固定路由預算：20%；OOF 選定 gate threshold：{selected['gate_threshold']:.2f}。

| test 方案 | accuracy | macro-F1 | Jev 接管 | 修正 | 引入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 小模型 | {small_result['accuracy']:.4f} | {small_result['macro_f1']:.4f} | 0 | 0 | 0 |
| 原 locked confidence gate | {locked_result['accuracy']:.4f} | {locked_result['macro_f1']:.4f} | {locked_result['jev_used_count']} | {locked_result['corrected_small_errors']} | {locked_result['introduced_errors']} |
| Audit-aware comparative gate | {gate_result['accuracy']:.4f} | {gate_result['macro_f1']:.4f} | {gate_result['jev_used_count']} | {gate_result['corrected_small_errors']} | {gate_result['introduced_errors']} |

相對小模型：accuracy {payload['test_gate_accuracy_delta_vs_small']:+.4f}，macro-F1 {payload['test_gate_macro_f1_delta_vs_small']:+.4f}。report-cluster bootstrap accuracy delta 95% CI：[{payload['test_report_cluster_bootstrap_delta']['ci95'][0]:+.4f}, {payload['test_report_cluster_bootstrap_delta']['ci95'][1]:+.4f}]。

## 解讀限制

Gate 的所有擬合、OOF 評估與 threshold 選擇均只用 calibration。由於既有 test 聚合結果已在方法設計前被查看，本次 test 結果只屬探索性；正式確認仍需新的 report-level 或 temporal holdout。人工 taxonomy/label adjudication 尚未完成。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
