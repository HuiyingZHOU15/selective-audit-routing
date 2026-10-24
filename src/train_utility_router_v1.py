"""Train a pre-call two-head utility router on full PCAOB comparative outcomes."""

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
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from llm_common import ROOT


CONFIG_PATH = ROOT / "configs/utility_router_v1.json"
MODEL_OUT = ROOT / "artifacts/models/pcaob_utility_router_v1.joblib"
DEV_OUT = ROOT / "results/tables/utility_router_development_oof_v1.csv"
CAL_OUT = ROOT / "results/tables/utility_router_calibration_v1.csv"
TEST_OUT = ROOT / "results/tables/utility_router_test_v1.csv"
GRID_OUT = ROOT / "results/tables/utility_router_calibration_grid_v1.csv"
METRICS_OUT = ROOT / "results/metrics/utility_router_v1.json"
REPORT_OUT = ROOT / "results/reports/utility_router_v1_zh.md"

NUMERIC = [
    "small_raw_confidence", "small_raw_margin", "small_raw_entropy",
    "text_log_chars", "text_log_words", "context_reference",
    "account_topic_hits", "crosscut_hits", "multi_topic", "context_added",
]
CATEGORICAL = ["small_prediction", "context_quality"]

ACCOUNT_PATTERNS = [
    r"revenue|receiv", r"inventory|cost of sales", r"goodwill|intangible|property|impairment|lease",
    r"loan|credit loss|deposit|insurance", r"investment|securit|derivative|financial instrument",
    r"debt|equity|warrant|liabilit|payable", r"income tax|deferred tax",
    r"expense|compensation|pension|benefit|payroll", r"business combination|acquisition|consolidat",
    r"related part|significant transaction", r"cash",
]
CROSSCUT_PATTERNS = [
    r"control", r"estimate|valuation", r"audit procedure|audit evidence",
    r"journal entr", r"presentation|disclosure|reporting",
]
CONTEXT_PATTERN = re.compile(
    r"\b(?:these|those|such|aforementioned|above-mentioned)\s+"
    r"(?:assets?|accounts?|transactions?|controls?|data|assumptions?|liabilities|procedures?)\b|"
    r"\bthis control\b|\bas discussed above\b|\bprocedures discussed above\b",
    re.I,
)


def read_responses(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame([row for row in rows if row.get("status") == "ok"]).drop_duplicates(
        ["sample_id", "prompt_bundle_sha256"], keep="last"
    )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1) / math.log(probabilities.shape[1])


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    text = result["text"].fillna("").astype(str)
    lower = text.str.lower()
    result["text_log_chars"] = np.log1p(text.str.len())
    result["text_log_words"] = np.log1p(text.str.split().str.len())
    result["context_reference"] = text.str.contains(CONTEXT_PATTERN).astype(int)
    result["account_topic_hits"] = sum(lower.str.contains(pattern, regex=True).astype(int) for pattern in ACCOUNT_PATTERNS)
    result["crosscut_hits"] = sum(lower.str.contains(pattern, regex=True).astype(int) for pattern in CROSSCUT_PATTERNS)
    result["multi_topic"] = result["account_topic_hits"].ge(2).astype(int)
    result["context_added"] = result["context_added"].fillna(False).astype(bool).astype(int)
    return result


def add_raw_small_features(frame: pd.DataFrame, model) -> pd.DataFrame:
    result = frame.copy()
    probabilities = model.predict_proba(result["text"].fillna(""))
    ordered = np.sort(probabilities, axis=1)
    result["small_raw_confidence"] = probabilities.max(axis=1)
    result["small_raw_margin"] = ordered[:, -1] - ordered[:, -2]
    result["small_raw_entropy"] = normalized_entropy(probabilities)
    return result


def load_existing_split(config: dict, split: str, raw_model) -> pd.DataFrame:
    if split == "calibration":
        candidate_path = ROOT / config["calibration_candidates"]
        response_path = ROOT / config["calibration_responses"]
    else:
        candidate_path = ROOT / config["test_candidates"]
        response_path = ROOT / config["test_responses"]
    frame = pd.read_csv(candidate_path).merge(
        read_responses(response_path), on=["sample_id", "prompt_bundle_sha256"], validate="one_to_one"
    )
    metadata = pd.read_csv(ROOT / config["model_dataset"])[
        ["sample_id", "context_needs_context", "context_added"]
    ]
    frame = frame.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    frame["llm_prediction"] = frame["prediction"]
    frame["small_correct"] = frame["small_prediction"].eq(frame["target_level1"])
    frame["llm_correct"] = frame["llm_prediction"].eq(frame["target_level1"])
    frame["routing_outcome"] = "neutral_both_wrong"
    frame.loc[frame["small_correct"] & frame["llm_correct"], "routing_outcome"] = "neutral_both_correct"
    frame.loc[~frame["small_correct"] & frame["llm_correct"], "routing_outcome"] = "beneficial"
    frame.loc[frame["small_correct"] & ~frame["llm_correct"], "routing_outcome"] = "harmful"
    return add_text_features(add_raw_small_features(frame, raw_model))


def load_development(config: dict) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / config["development_outcomes"])
    frame["text"] = frame["text"].fillna("")
    return add_text_features(frame)


def make_head(config: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("features", preprocessor),
            ("classifier", LogisticRegression(C=config["classifier_c"], max_iter=3000, random_state=config["random_seed"])),
        ]
    )


def fit_heads(frame: pd.DataFrame, config: dict) -> tuple[Pipeline, Pipeline]:
    features = frame[NUMERIC + CATEGORICAL]
    benefit = make_head(config)
    harm = make_head(config)
    benefit.fit(features, frame["routing_outcome"].eq("beneficial"))
    harm.fit(features, frame["routing_outcome"].eq("harmful"))
    return benefit, harm


def score_heads(frame: pd.DataFrame, benefit: Pipeline, harm: Pipeline, penalty: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame[NUMERIC + CATEGORICAL]
    p_benefit = benefit.predict_proba(features)[:, list(benefit.classes_).index(True)]
    p_harm = harm.predict_proba(features)[:, list(harm.classes_).index(True)]
    return p_benefit, p_harm, p_benefit - penalty * p_harm


def select_route(scores: np.ndarray, threshold: float, maximum_budget: float) -> np.ndarray:
    eligible = np.flatnonzero(scores >= threshold)
    maximum = int(round(maximum_budget * len(scores)))
    if len(eligible) > maximum:
        eligible = eligible[np.argsort(-scores[eligible], kind="stable")[:maximum]]
    route = np.zeros(len(scores), dtype=bool)
    route[eligible] = True
    return route


def exact_low_confidence(frame: pd.DataFrame, budget: float) -> np.ndarray:
    route = np.zeros(len(frame), dtype=bool)
    count = int(round(budget * len(frame)))
    confidence = frame["small_raw_confidence"].to_numpy(float)
    route[np.argsort(confidence, kind="stable")[:count]] = True
    return route


def system_metrics(frame: pd.DataFrame, use_llm: np.ndarray, labels: list[str]) -> dict:
    truth = frame["target_level1"].to_numpy()
    prediction = frame["small_prediction"].to_numpy(dtype=object).copy()
    prediction[use_llm] = frame.loc[use_llm, "llm_prediction"].to_numpy()
    beneficial = frame["routing_outcome"].eq("beneficial").to_numpy()
    harmful = frame["routing_outcome"].eq("harmful").to_numpy()
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "llm_calls": int(use_llm.sum()),
        "call_rate": float(use_llm.mean()),
        "corrected": int((use_llm & beneficial).sum()),
        "harmed": int((use_llm & harmful).sum()),
        "net_correct": int((use_llm & beneficial).sum() - (use_llm & harmful).sum()),
    }


def cluster_bootstrap(frame: pd.DataFrame, use_llm: np.ndarray, iterations: int, seed: int) -> dict:
    selected = frame["small_correct"].to_numpy(bool).copy()
    selected[use_llm] = frame.loc[use_llm, "llm_correct"].to_numpy(bool)
    work = pd.DataFrame({"report_id": frame["report_id"], "small": frame["small_correct"], "selected": selected})
    grouped = {key: value for key, value in work.groupby("report_id", sort=False)}
    reports = np.asarray(list(grouped))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        boot = pd.concat([grouped[key] for key in rng.choice(reports, len(reports), replace=True)])
        deltas.append(float(boot["selected"].mean() - boot["small"].mean()))
    return {
        "mean": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
        "probability_positive": float(np.mean(np.asarray(deltas) > 0)),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw_model = joblib.load(ROOT / config["raw_small_model"])
    development = load_development(config)
    calibration = load_existing_split(config, "calibration", raw_model)
    test = load_existing_split(config, "test", raw_model)
    labels = sorted(development["target_level1"].unique())

    development["router_p_benefit_oof"] = np.nan
    development["router_p_harm_oof"] = np.nan
    for fold in sorted(development["outer_fold"].unique()):
        train = development["outer_fold"].ne(fold)
        valid = ~train
        benefit, harm = fit_heads(development.loc[train], config)
        p_benefit, p_harm, _ = score_heads(development.loc[valid], benefit, harm, config["harm_penalty"])
        development.loc[valid, "router_p_benefit_oof"] = p_benefit
        development.loc[valid, "router_p_harm_oof"] = p_harm
    development["router_utility_oof"] = (
        development["router_p_benefit_oof"] - config["harm_penalty"] * development["router_p_harm_oof"]
    )
    if development[["router_p_benefit_oof", "router_p_harm_oof"]].isna().any().any():
        raise RuntimeError("Development router OOF scores are incomplete")

    benefit_head, harm_head = fit_heads(development, config)
    cal_pb, cal_ph, cal_score = score_heads(calibration, benefit_head, harm_head, config["harm_penalty"])
    test_pb, test_ph, test_score = score_heads(test, benefit_head, harm_head, config["harm_penalty"])
    calibration["router_p_benefit"] = cal_pb
    calibration["router_p_harm"] = cal_ph
    calibration["router_utility"] = cal_score
    test["router_p_benefit"] = test_pb
    test["router_p_harm"] = test_ph
    test["router_utility"] = test_score

    thresholds = np.unique(np.quantile(cal_score, np.linspace(0, 1, config["threshold_quantiles"])))
    thresholds = np.r_[thresholds, np.inf]
    grid = []
    for threshold in thresholds:
        route = select_route(cal_score, float(threshold), config["maximum_jev_budget"])
        grid.append({"threshold": float(threshold), **system_metrics(calibration, route, labels)})
    selected = max(
        grid,
        key=lambda row: (row["net_correct"], row["accuracy"], row["macro_f1"], -row["llm_calls"]),
    )
    selected_threshold = selected["threshold"]

    development_route = select_route(
        development["router_utility_oof"].to_numpy(float), -np.inf, config["maximum_jev_budget"]
    )
    calibration_route = select_route(cal_score, selected_threshold, config["maximum_jev_budget"])
    test_route = select_route(test_score, selected_threshold, config["maximum_jev_budget"])
    low_conf_cal = exact_low_confidence(calibration, config["maximum_jev_budget"])
    low_conf_test = exact_low_confidence(test, config["maximum_jev_budget"])
    none_test = np.zeros(len(test), dtype=bool)
    all_test = np.ones(len(test), dtype=bool)

    development["utility_route_20pct"] = development_route
    calibration["utility_route"] = calibration_route
    test["utility_route"] = test_route
    test["low_confidence_route_20pct"] = low_conf_test
    payload = {
        "version": config["version"],
        "features_available_before_llm_call": {"numeric": NUMERIC, "categorical": CATEGORICAL},
        "development_samples": len(development),
        "development_outcomes": development["routing_outcome"].value_counts().to_dict(),
        "development_benefit_roc_auc": float(roc_auc_score(development["routing_outcome"].eq("beneficial"), development["router_p_benefit_oof"])),
        "development_benefit_average_precision": float(average_precision_score(development["routing_outcome"].eq("beneficial"), development["router_p_benefit_oof"])),
        "development_harm_roc_auc": float(roc_auc_score(development["routing_outcome"].eq("harmful"), development["router_p_harm_oof"])),
        "development_harm_average_precision": float(average_precision_score(development["routing_outcome"].eq("harmful"), development["router_p_harm_oof"])),
        "development_top20_utility": system_metrics(development, development_route, labels),
        "calibration_selected_threshold": selected_threshold,
        "calibration_selected": selected,
        "calibration_low_confidence_20pct": system_metrics(calibration, low_conf_cal, labels),
        "test_status": config["test_status"],
        "test_small_only": system_metrics(test, none_test, labels),
        "test_always_jev": system_metrics(test, all_test, labels),
        "test_low_confidence_20pct": system_metrics(test, low_conf_test, labels),
        "test_utility_router": system_metrics(test, test_route, labels),
        "test_report_cluster_bootstrap_delta": cluster_bootstrap(
            test, test_route, config["bootstrap_iterations"], config["random_seed"]
        ),
    }
    for path in (MODEL_OUT, DEV_OUT, CAL_OUT, TEST_OUT, GRID_OUT, METRICS_OUT, REPORT_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "benefit_head": benefit_head,
        "harm_head": harm_head,
        "harm_penalty": config["harm_penalty"],
        "threshold": selected_threshold,
        "maximum_jev_budget": config["maximum_jev_budget"],
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "version": config["version"],
    }, MODEL_OUT)
    development.to_csv(DEV_OUT, index=False, encoding="utf-8")
    calibration.to_csv(CAL_OUT, index=False, encoding="utf-8")
    test.to_csv(TEST_OUT, index=False, encoding="utf-8")
    pd.DataFrame(grid).to_csv(GRID_OUT, index=False, encoding="utf-8")
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    small = payload["test_small_only"]
    low = payload["test_low_confidence_20pct"]
    utility = payload["test_utility_router"]
    always = payload["test_always_jev"]
    report = f"""# PCAOB pre-call utility router v1

## Development

- Clean nested-OOF comparative outcomes：{len(development):,}；beneficial {payload['development_outcomes'].get('beneficial', 0)}；harmful {payload['development_outcomes'].get('harmful', 0)}。
- Benefit head OOF ROC-AUC/AP：{payload['development_benefit_roc_auc']:.4f}/{payload['development_benefit_average_precision']:.4f}。
- Harm head OOF ROC-AUC/AP：{payload['development_harm_roc_auc']:.4f}/{payload['development_harm_average_precision']:.4f}。
- Router 只使用 Jev 呼叫前可得的 small-model probability、文本長度／指代／topic 特徵、small prediction 與 context quality。

## Calibration

- 608 筆 clean calibration 只用於選 utility threshold；最大 Jev budget 固定為 20%。
- 選定 threshold：{selected_threshold:.6f}；Jev calls {selected['llm_calls']}；net corrections {selected['net_correct']:+d}。

## Retrospective test

| 方法 | accuracy | macro-F1 | Jev calls | corrected | harmed | net |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small only | {small['accuracy']:.4f} | {small['macro_f1']:.4f} | 0 | 0 | 0 | 0 |
| Always Jev | {always['accuracy']:.4f} | {always['macro_f1']:.4f} | {always['llm_calls']} | {always['corrected']} | {always['harmed']} | {always['net_correct']:+d} |
| Lowest-confidence 20% | {low['accuracy']:.4f} | {low['macro_f1']:.4f} | {low['llm_calls']} | {low['corrected']} | {low['harmed']} | {low['net_correct']:+d} |
| Utility router | {utility['accuracy']:.4f} | {utility['macro_f1']:.4f} | {utility['llm_calls']} | {utility['corrected']} | {utility['harmed']} | {utility['net_correct']:+d} |

Utility router 相對 small-only 的 report-cluster bootstrap accuracy delta 95% CI：[{payload['test_report_cluster_bootstrap_delta']['ci95'][0]:+.4f}, {payload['test_report_cluster_bootstrap_delta']['ci95'][1]:+.4f}]。

此 test 已在先前研究中被查看，因此結果只屬 retrospective；確認性主張仍需下一批 PCAOB temporal holdout。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
