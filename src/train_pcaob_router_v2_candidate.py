"""Exploratory grouped-CV selector for cheap model -> PCAOB report-RAG.

This script does not freeze or overwrite Router v1.  The 120-case pilot was
deliberately outcome-stratified, so deployment-prevalence numbers are reported
only as sensitivity analyses.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from llm_common import ROOT


CONFIG_PATH = ROOT / "configs/pcaob_router_v2_candidate.json"

NUMERIC_FEATURES = [
    "small_confidence",
    "small_margin",
    "small_entropy",
    "small_raw_confidence",
    "small_raw_margin",
    "small_raw_entropy",
    "word_count",
    "sentence_count",
    "digit_count",
    "standard_ref_count",
    "enumeration_count",
    "context_needs_context_num",
    "context_added_num",
    "report_context_count",
    "report_context_available",
]
CATEGORICAL_FEATURES = ["small_prediction", "context_quality"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("features", pre),
            (
                "classifier",
                LogisticRegression(
                    C=0.15,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=20261002,
                ),
            ),
        ]
    )


def load_pilot(config: dict) -> pd.DataFrame:
    audit = pd.read_csv(ROOT / config["audit_table"])
    backend = audit.loc[audit["condition"].eq(config["conditions"]["backend"])].copy()
    if backend["sample_id"].duplicated().any():
        raise ValueError("Expected one report-RAG row per sample")
    backend["small_correct"] = as_bool(backend["small_correct"])
    backend["rag_correct"] = as_bool(backend["prediction_correct"])
    backend["rag_eligible"] = as_bool(backend["takeover_eligible"])
    backend["benefit"] = (~backend["small_correct"] & backend["rag_correct"] & backend["rag_eligible"]).astype(int)
    backend["harm"] = (backend["small_correct"] & (~backend["rag_correct"] | ~backend["rag_eligible"])).astype(int)
    backend["small_wrong"] = (~backend["small_correct"]).astype(int)
    backend["rag_wrong"] = (~backend["rag_correct"] | ~backend["rag_eligible"]).astype(int)
    backend["outcome"] = np.select(
        [backend["benefit"].eq(1), backend["harm"].eq(1), backend["small_correct"] & backend["rag_correct"]],
        ["benefit", "harm", "both_correct"],
        default="both_wrong",
    )

    text = backend["text_context_v2"].fillna("").astype(str)
    backend["word_count"] = text.str.findall(r"\b\w+\b").str.len().astype(float)
    backend["sentence_count"] = text.str.count(r"[.!?;]").clip(lower=1).astype(float)
    backend["digit_count"] = text.str.count(r"\d").astype(float)
    backend["standard_ref_count"] = backend["standard_refs"].fillna("").astype(str).str.count(r"(?i)\bAS\s*\d{4}").astype(float)
    backend["enumeration_count"] = text.str.count(r"(?:\(\d+\)|\b\d+[.)])").astype(float)
    backend["context_needs_context_num"] = as_bool(backend["context_needs_context"]).astype(float)
    backend["context_added_num"] = as_bool(backend["context_added"]).astype(float)
    backend["report_context_count"] = pd.to_numeric(backend["report_context_count"], errors="coerce").fillna(0.0)
    backend["report_context_available"] = backend["report_context_count"].gt(0).astype(float)
    for column in NUMERIC_FEATURES:
        backend[column] = pd.to_numeric(backend[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        backend[column] = backend[column].fillna("missing").astype(str)
    return backend.reset_index(drop=True)


def fit_probability(train: pd.DataFrame, valid: pd.DataFrame, target: str) -> np.ndarray:
    y = train[target].astype(int)
    if y.nunique() < 2:
        return np.full(len(valid), float(y.iloc[0]))
    model = logistic_pipeline()
    model.fit(train[FEATURES], y)
    positive = list(model.named_steps["classifier"].classes_).index(1)
    return model.predict_proba(valid[FEATURES])[:, positive]


def crossfit_scores(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = frame.copy()
    for name in ["p_benefit", "p_harm", "p_small_wrong", "p_rag_wrong"]:
        result[name] = np.nan
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(frame, frame["outcome"], groups=frame["report_id"]), start=1
    ):
        train = frame.iloc[train_index]
        valid = frame.iloc[valid_index]
        for target, score_name in [
            ("benefit", "p_benefit"),
            ("harm", "p_harm"),
            ("small_wrong", "p_small_wrong"),
            ("rag_wrong", "p_rag_wrong"),
        ]:
            result.loc[valid_index, score_name] = fit_probability(train, valid, target)
        result.loc[valid_index, "cv_fold"] = fold
    if result[["p_benefit", "p_harm", "p_small_wrong", "p_rag_wrong"]].isna().any().any():
        raise RuntimeError("Cross-fitting left missing scores")
    result["learned_route_score"] = result["p_benefit"] - result["p_harm"]
    result["confidence_route_score"] = 1.0 - result["small_confidence"]
    result["evidence_route_score"] = (
        2.0 * result["context_needs_context_num"] * result["report_context_available"]
        + 0.5 * result["report_context_available"]
        + 0.1 * (1.0 - result["small_confidence"])
    )
    return result


def exact_top_fraction(score: pd.Series, fraction: float) -> np.ndarray:
    count = int(math.ceil(len(score) * fraction))
    chosen = np.zeros(len(score), dtype=bool)
    if count:
        order = np.argsort(-score.to_numpy(float), kind="stable")
        chosen[order[:count]] = True
    return chosen


def scenario_weights(frame: pd.DataFrame, small_error_prevalence: float | None) -> np.ndarray:
    if small_error_prevalence is None:
        return np.ones(len(frame), dtype=float)
    wrong = frame["small_wrong"].to_numpy(bool)
    n_wrong = max(int(wrong.sum()), 1)
    n_correct = max(int((~wrong).sum()), 1)
    weights = np.where(wrong, small_error_prevalence / n_wrong, (1.0 - small_error_prevalence) / n_correct)
    return weights / weights.sum()


def weighted_rate(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values.astype(float), weights=weights))


def safe_auc(y: pd.Series, score: pd.Series) -> float:
    return float(roc_auc_score(y, score)) if y.nunique() == 2 else float("nan")


def policy_metrics(
    frame: pd.DataFrame,
    route: np.ndarray,
    weights: np.ndarray,
    target_accuracy: float,
) -> dict:
    small_correct = frame["small_correct"].to_numpy(bool)
    rag_correct = frame["rag_correct"].to_numpy(bool) & frame["rag_eligible"].to_numpy(bool)
    final_correct = np.where(route, rag_correct, small_correct)
    predicted_risk = np.where(route, frame["p_rag_wrong"], frame["p_small_wrong"])
    review_order = np.argsort(-predicted_risk, kind="stable")
    review = np.zeros(len(frame), dtype=bool)
    minimum_review = None
    curve = []
    for count in range(len(frame) + 1):
        if count:
            review[review_order[count - 1]] = True
        system_correct = final_correct | review
        accuracy = weighted_rate(system_correct, weights)
        coverage = 1.0 - weighted_rate(review, weights)
        if count == 0 or count == len(frame) or count % max(1, len(frame) // 20) == 0:
            curve.append({"review_count": count, "auto_coverage": coverage, "system_accuracy": accuracy})
        if minimum_review is None and accuracy >= target_accuracy:
            minimum_review = {
                "review_count": count,
                "human_review_rate": weighted_rate(review, weights),
                "auto_coverage": coverage,
                "system_accuracy": accuracy,
                "observed_errors_reviewed": int((~final_correct & review).sum()),
                "observed_correct_cases_reviewed": int((final_correct & review).sum()),
            }
    selected = route.astype(bool)
    fixed_review = np.zeros(len(frame), dtype=bool)
    fixed_review_weight = 0.0
    for index in review_order:
        candidate_weight = float(weights[index] / weights.sum())
        if fixed_review_weight + candidate_weight <= 0.05 + 1e-12:
            fixed_review[index] = True
            fixed_review_weight += candidate_weight
    fixed_system_correct = final_correct | fixed_review
    automatic_accuracy = weighted_rate(final_correct, weights)
    return {
        "route_rate": weighted_rate(selected, weights),
        "automatic_accuracy": automatic_accuracy,
        "beneficial_takeover_rate": weighted_rate(selected & ~small_correct & rag_correct, weights),
        "harmful_takeover_rate": weighted_rate(selected & small_correct & ~rag_correct, weights),
        "net_correction_rate": weighted_rate(selected & (rag_correct.astype(int) - small_correct.astype(int) > 0), weights)
        - weighted_rate(selected & (rag_correct.astype(int) - small_correct.astype(int) < 0), weights),
        "beneficial_count": int((selected & ~small_correct & rag_correct).sum()),
        "harmful_count": int((selected & small_correct & ~rag_correct).sum()),
        "oracle_minimum_review_rate_for_95pct": max(0.0, target_accuracy - automatic_accuracy),
        "fixed_5pct_review_rate": weighted_rate(fixed_review, weights),
        "system_accuracy_at_5pct_review": weighted_rate(fixed_system_correct, weights),
        "errors_reviewed_at_5pct": int((~final_correct & fixed_review).sum()),
        "correct_reviewed_at_5pct": int((final_correct & fixed_review).sum()),
        "minimum_review_for_target": minimum_review,
        "review_curve": curve,
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = load_pilot(config)
    scored = crossfit_scores(frame, int(config["folds"]), int(config["seed"]))
    development = pd.read_csv(ROOT / config["development_oof"])
    clean = as_bool(development["oof_clean"])
    development_accuracy = as_bool(development.loc[clean, "small_correct"]).mean()
    scenarios = {
        "pilot_unweighted": None,
        "development_oof_standardized": float(1.0 - development_accuracy),
        "retrospective_test_standardized": float(config["prevalence_scenarios"]["retrospective_test"]),
    }
    policies = {
        "small_only": ("confidence_route_score", 0.0),
        "report_rag_all": ("confidence_route_score", 1.0),
    }
    for budget in config["route_budgets"]:
        for name, score in [
            ("confidence", "confidence_route_score"),
            ("evidence_need", "evidence_route_score"),
            ("two_head_logistic", "learned_route_score"),
        ]:
            policies[f"{name}@{int(100 * budget)}pct"] = (score, float(budget))

    rows = []
    detail = []
    for policy_name, (score_name, budget) in policies.items():
        route = exact_top_fraction(scored[score_name], budget)
        scored[f"route__{policy_name}"] = route
        for scenario_name, prevalence in scenarios.items():
            metrics = policy_metrics(
                scored,
                route,
                scenario_weights(scored, prevalence),
                float(config["target_system_accuracy"]),
            )
            minimum = metrics.pop("minimum_review_for_target")
            curve = metrics.pop("review_curve")
            rows.append(
                {
                    "policy": policy_name,
                    "scenario": scenario_name,
                    "route_budget": budget,
                    **metrics,
                    "minimum_human_review_rate_for_95pct": None if minimum is None else minimum["human_review_rate"],
                    "auto_coverage_at_95pct": None if minimum is None else minimum["auto_coverage"],
                    "review_count_for_95pct": None if minimum is None else minimum["review_count"],
                    "errors_reviewed_for_95pct": None if minimum is None else minimum["observed_errors_reviewed"],
                    "correct_reviewed_for_95pct": None if minimum is None else minimum["observed_correct_cases_reviewed"],
                }
            )
            detail.append({"policy": policy_name, "scenario": scenario_name, "minimum": minimum, "curve": curve})

    results = pd.DataFrame(rows)
    learned = results.loc[
        results["policy"].str.startswith("two_head_logistic")
        & results["scenario"].eq("development_oof_standardized")
    ].copy()
    learned["feasible_5pct_human"] = learned["minimum_human_review_rate_for_95pct"].le(0.05)
    selected = learned.sort_values(
        ["feasible_5pct_human", "minimum_human_review_rate_for_95pct", "automatic_accuracy", "route_budget"],
        ascending=[False, True, False, True],
        na_position="last",
    ).iloc[0]

    diagnostics = {
        "n": len(scored),
        "reports": int(scored["report_id"].nunique()),
        "outcomes": scored["outcome"].value_counts().to_dict(),
        "development_small_error_prevalence": float(1.0 - development_accuracy),
        "oof_auc": {
            "benefit": safe_auc(scored["benefit"], scored["p_benefit"]),
            "harm": safe_auc(scored["harm"], scored["p_harm"]),
            "small_wrong": safe_auc(scored["small_wrong"], scored["p_small_wrong"]),
            "rag_wrong": safe_auc(scored["rag_wrong"], scored["p_rag_wrong"]),
        },
        "selected_exploratory_policy": selected.to_dict(),
        "confirmatory_status": "not_confirmatory_outcome_stratified_120_case_pilot",
    }

    table_path = ROOT / "results/tables/pcaob_router_v2_cv_policy.csv"
    oof_path = ROOT / "results/tables/pcaob_router_v2_oof_predictions.csv"
    metrics_path = ROOT / "results/metrics/pcaob_router_v2_cv.json"
    report_path = ROOT / "results/reports/pcaob_router_v2_cv_zh.md"
    for path in [table_path, oof_path, metrics_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(table_path, index=False, encoding="utf-8-sig")
    safe_columns = [
        "sample_id", "report_id", "pilot_stratum", "outcome", "small_correct", "rag_correct",
        "rag_eligible", "cv_fold", "p_benefit", "p_harm", "p_small_wrong", "p_rag_wrong",
        "learned_route_score", "confidence_route_score", "evidence_route_score",
    ] + [column for column in scored if column.startswith("route__")]
    scored[safe_columns].to_csv(oof_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(
        json.dumps({"config": config, "diagnostics": diagnostics, "review_details": detail}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dev_rows = results.loc[results["scenario"].eq("development_oof_standardized")].copy()
    dev_rows = dev_rows.sort_values(["route_budget", "policy"])
    lines = []
    for _, row in dev_rows.iterrows():
        review = row["minimum_human_review_rate_for_95pct"]
        lines.append(
            f"| {row['policy']} | {row['route_rate']:.1%} | {row['automatic_accuracy']:.1%} | "
            f"{row['beneficial_count']} | {row['harmful_count']} | "
            f"{row['oracle_minimum_review_rate_for_95pct']:.1%} | "
            f"{row['system_accuracy_at_5pct_review']:.1%} | "
            f"{'NA' if pd.isna(review) else f'{review:.1%}'} |"
        )
    selected_name = str(selected["policy"])
    selected_review = selected["minimum_human_review_rate_for_95pct"]
    feasibility = bool(pd.notna(selected_review) and selected_review <= 0.05)
    report = f"""# PCAOB Router V2 探索性分組交叉驗證

## 結論先行

這是 **V2 候選實驗，不是已凍結的生產路由器**。120 筆 pilot 能用來判斷是否存在可學習的排序信號，但因抽樣刻意平衡小模型正誤，不能直接宣稱部署後的準確率或人工兜底率。

在 report-level 5-fold cross-validation 中，benefit AUC 為 **{diagnostics['oof_auc']['benefit']:.3f}**，harm AUC 為 **{diagnostics['oof_auc']['harm']:.3f}**。按 development OOF 的小模型錯誤率 **{diagnostics['development_small_error_prevalence']:.1%}** 做敏感度標準化後，探索性首選為 **{selected_name}**。

在完全知道錯題位置的理想情況下，這一自動正確率至少仍需 **{selected['oracle_minimum_review_rate_for_95pct']:.1%}** 人工覆核才能達到 95%；使用目前 cross-fitted 風險排序時，估計最小人工覆核率為 **{'NA' if pd.isna(selected_review) else f'{selected_review:.1%}'}**。固定約 5% 人工預算下，系統正確率為 **{selected['system_accuracy_at_5pct_review']:.1%}**。因此目前 **{'達到' if feasibility else '尚未達到'}**「人工兜底不超過 5%」的目標。這些數字仍只是敏感度分析，不是確認性部署估計。

## Development-prevalence 標準化結果

| policy | RAG 調用率 | 自動路徑正確率 | beneficial（樣本數） | harmful（樣本數） | 理想人工下限 | 約 5% 人工後正確率 | 現有風險排序達 95% 所需人工率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

## 方法邊界

- 僅使用 LLM 呼叫前可取得的訊號：小模型信心、margin、entropy、文本形態、context 標記、同報告上下文是否存在，以及小模型預測類別。
- `small_correct`、真實標籤、Direct/LLM/RAG 輸出及其信心均未進入特徵。
- two-head logistic 分別預測 beneficial 與 harmful，再以 `P(benefit)-P(harm)` 排序。
- 人工覆核曲線假設人工可修正被覆核案例；風險排序由 cross-fitted 的 small-error/RAG-error 模型產生。
- Pilot 的 correct-risk/control 亦非隨機人口樣本，所以 prevalence 標準化只能回答「在若干錯誤率情境下可能如何」，不能替代 608 筆 calibration 或新的 temporal holdout。

## 決策

若兩個 AUC 接近 0.5、learned router 不優於 confidence/evidence rule，便不應擴大訓練複雜模型；應先增加具代表性的 paired report-RAG outcomes。若信號為正且 harmful 明顯受控，下一步才是在 608 筆 calibration 生成 report-RAG outcomes，固定門檻與人工 risk gate，最後留給全新 temporal holdout 驗證。
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"wrote {table_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
