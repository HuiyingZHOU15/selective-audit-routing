"""Compare deployable risk scores for SEC-BERT selective classification.

The learned risk model is cross-fitted by the already locked report-grouped
outer folds.  Every score for a development sample therefore comes from a
router that did not train on that sample's report.  This script evaluates the
development OOF population only; calibration/test threshold selection belongs
to the next protocol stage.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/sec_bert_risk_router_v1.json"

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

NUMERIC = [
    "sec_confidence", "sec_margin", "sec_entropy", "text_log_chars", "text_log_words",
    "context_reference", "account_topic_hits", "crosscut_hits", "multi_topic",
    "tfidf_disagree", "small_confidence", "small_margin", "small_entropy",
]
CATEGORICAL = ["sec_prediction", "context_quality"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    text = result["text_context_v2"].fillna("").astype(str)
    lower = text.str.lower()
    result["text_log_chars"] = np.log1p(text.str.len())
    result["text_log_words"] = np.log1p(text.str.split().str.len())
    result["context_reference"] = text.str.contains(CONTEXT_PATTERN).astype(int)
    result["account_topic_hits"] = sum(
        lower.str.contains(pattern, regex=True).astype(int) for pattern in ACCOUNT_PATTERNS
    )
    result["crosscut_hits"] = sum(
        lower.str.contains(pattern, regex=True).astype(int) for pattern in CROSSCUT_PATTERNS
    )
    result["multi_topic"] = result["account_topic_hits"].ge(2).astype(int)
    result["tfidf_disagree"] = result["small_prediction"].ne(result["sec_prediction"]).astype(int)
    return result


def make_router(config: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    classifier = LogisticRegression(
        C=float(config["logistic_c"]),
        class_weight="balanced",
        max_iter=3000,
        random_state=int(config["random_seed"]),
    )
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def crossfit_risk(frame: pd.DataFrame, config: dict) -> np.ndarray:
    scores = np.full(len(frame), np.nan, dtype=float)
    for fold in sorted(frame["outer_fold"].astype(int).unique()):
        train = frame["outer_fold"].astype(int).ne(fold)
        hold = ~train
        train_reports = set(frame.loc[train, "report_id"].astype(str))
        hold_reports = set(frame.loc[hold, "report_id"].astype(str))
        if train_reports & hold_reports:
            raise RuntimeError(f"Report leakage in router fold {fold}")
        model = make_router(config)
        model.fit(frame.loc[train, NUMERIC + CATEGORICAL], frame.loc[train, "sec_error"])
        classes = list(model.named_steps["classifier"].classes_)
        scores[hold] = model.predict_proba(frame.loc[hold, NUMERIC + CATEGORICAL])[:, classes.index(True)]
    if np.isnan(scores).any():
        raise RuntimeError("Cross-fitted router did not score every row")
    return scores


def upper_binomial_risk(errors: int, accepted: int, confidence: float) -> float:
    if accepted <= 0:
        return float("nan")
    if errors >= accepted:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, accepted - errors))


def referral_mask(frame: pd.DataFrame, score_column: str, budget: float) -> np.ndarray:
    n_refer = int(round(len(frame) * budget))
    ordered = frame.sort_values([score_column, "sample_id"], ascending=[False, True]).index
    mask = np.zeros(len(frame), dtype=bool)
    if n_refer:
        mask[ordered[:n_refer].to_numpy()] = True
    return mask


def evaluate_mask(frame: pd.DataFrame, referred: np.ndarray, confidence: float) -> dict:
    accepted = ~referred
    n_accept = int(accepted.sum())
    n_refer = int(referred.sum())
    accept_errors = int(frame.loc[accepted, "sec_error"].sum())
    referred_errors = int(frame.loc[referred, "sec_error"].sum())
    total_errors = int(frame["sec_error"].sum())
    labels = sorted(frame["target_level1"].astype(str).unique())
    return {
        "samples": int(len(frame)),
        "referred": n_refer,
        "referral_rate": n_refer / len(frame),
        "coverage": n_accept / len(frame),
        "selective_accuracy": 1.0 - accept_errors / n_accept if n_accept else float("nan"),
        "selective_risk": accept_errors / n_accept if n_accept else float("nan"),
        "selective_risk_upper_95": upper_binomial_risk(accept_errors, n_accept, confidence),
        "accepted_macro_f1": float(f1_score(
            frame.loc[accepted, "target_level1"], frame.loc[accepted, "sec_prediction"],
            labels=labels, average="macro", zero_division=0,
        )) if n_accept else float("nan"),
        "errors_remaining": accept_errors,
        "errors_captured": referred_errors,
        "error_capture_rate": referred_errors / total_errors if total_errors else float("nan"),
        "referral_precision": referred_errors / n_refer if n_refer else float("nan"),
        "oracle_assisted_accuracy_upper_bound": 1.0 - accept_errors / len(frame),
    }


def risk_curve(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    accepted_order = frame.sort_values([score_column, "sample_id"], ascending=[True, True])
    errors = accepted_order["sec_error"].astype(int).to_numpy()
    accepted = np.arange(1, len(frame) + 1)
    return pd.DataFrame({
        "method": score_column.removeprefix("risk_"),
        "coverage": accepted / len(frame),
        "selective_risk": np.cumsum(errors) / accepted,
        "selective_accuracy": 1.0 - np.cumsum(errors) / accepted,
    })


def grouped_bootstrap(
    frame: pd.DataFrame, score_column: str, budget: float, config: dict
) -> dict[str, list[float]]:
    rng = np.random.default_rng(int(config["random_seed"]) + int(round(budget * 10000)))
    report_codes, report_ids = pd.factorize(frame["report_id"].astype(str), sort=False)
    group_indices = [np.flatnonzero(report_codes == code) for code in range(len(report_ids))]
    scores = frame[score_column].to_numpy(float)
    errors = frame["sec_error"].to_numpy(bool)
    values: dict[str, list[float]] = {
        "coverage": [], "selective_accuracy": [], "error_capture_rate": [],
        "referral_precision": [], "selective_risk_upper_95": [],
    }
    for _ in range(int(config["bootstrap_replicates"])):
        sampled_codes = rng.integers(0, len(group_indices), size=len(group_indices))
        sampled_indices = np.concatenate([group_indices[code] for code in sampled_codes])
        boot_scores = scores[sampled_indices]
        boot_errors = errors[sampled_indices]
        n_refer = int(round(len(sampled_indices) * budget))
        order = np.argsort(-boot_scores, kind="stable")
        referred_errors = int(boot_errors[order[:n_refer]].sum()) if n_refer else 0
        total_errors = int(boot_errors.sum())
        accepted = len(sampled_indices) - n_refer
        accepted_errors = total_errors - referred_errors
        values["coverage"].append(accepted / len(sampled_indices))
        values["selective_accuracy"].append(1.0 - accepted_errors / accepted)
        values["error_capture_rate"].append(referred_errors / total_errors if total_errors else np.nan)
        values["referral_precision"].append(referred_errors / n_refer if n_refer else np.nan)
        values["selective_risk_upper_95"].append(upper_binomial_risk(
            accepted_errors, accepted, float(config["risk_confidence_level"])
        ))
    return {
        key: [float(x) for x in np.quantile(series, [0.025, 0.5, 0.975])]
        for key, series in values.items()
    }


def first_certified_point(frame: pd.DataFrame, score_column: str, config: dict) -> dict | None:
    target_risk = 1.0 - float(config["target_selective_accuracy"])
    confidence = float(config["risk_confidence_level"])
    ordered = frame.sort_values([score_column, "sample_id"], ascending=[False, True])
    error = ordered["sec_error"].astype(int).to_numpy()
    total_errors = int(error.sum())
    for referred in range(len(frame) + 1):
        accepted_errors = total_errors - int(error[:referred].sum())
        accepted = len(frame) - referred
        if accepted and upper_binomial_risk(accepted_errors, accepted, confidence) <= target_risk:
            mask = np.zeros(len(frame), dtype=bool)
            if referred:
                mask[ordered.index[:referred].to_numpy()] = True
            result = evaluate_mask(frame, mask, confidence)
            result["method"] = score_column.removeprefix("risk_")
            return result
    return None


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    oof = pd.read_csv(ROOT / config["oof_predictions"])
    metadata = pd.read_csv(ROOT / config["development_metadata"])
    keep = [
        "sample_id", "text_context_v2", "context_quality", "small_prediction",
        "small_confidence", "small_margin", "small_entropy",
    ]
    frame = oof.merge(metadata[keep], on="sample_id", how="left", validate="one_to_one")
    if bool(config["clean_only"]):
        frame = frame.loc[frame["oof_clean"].fillna(False).astype(bool)].copy()
    frame = frame.reset_index(drop=True)
    if frame["text_context_v2"].isna().any():
        raise RuntimeError("Missing development metadata after merge")
    if frame.groupby("report_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("A report appears in more than one SEC-BERT outer fold")
    frame["sec_error"] = ~frame["sec_prediction"].eq(frame["target_level1"])
    frame = add_features(frame)
    frame["risk_confidence"] = 1.0 - frame["sec_confidence"]
    frame["risk_margin"] = 1.0 - frame["sec_margin"]
    frame["risk_entropy"] = frame["sec_entropy"]
    frame["risk_class_aware_logistic"] = crossfit_risk(frame, config)
    rng = np.random.default_rng(int(config["random_seed"]))
    frame["risk_random"] = rng.random(len(frame))
    frame["risk_oracle"] = frame["sec_error"].astype(float)

    methods = ["random", "confidence", "margin", "entropy", "class_aware_logistic", "oracle"]
    deployable = [method for method in methods if method != "oracle"]
    budgets = sorted(set(float(x) for x in config["referral_budgets"]))
    rows = []
    for method in methods:
        for budget in budgets:
            metrics = evaluate_mask(
                frame, referral_mask(frame, f"risk_{method}", budget),
                float(config["risk_confidence_level"]),
            )
            metrics.update({"method": method, "requested_referral_budget": budget})
            rows.append(metrics)
    budget_table = pd.DataFrame(rows)

    curves = pd.concat(
        [risk_curve(frame, f"risk_{method}") for method in methods], ignore_index=True
    )
    auc = curves.groupby("method")["selective_risk"].mean().rename("aurc_discrete").reset_index()

    error = frame["sec_error"].astype(int).to_numpy()
    learned_score = frame["risk_class_aware_logistic"].to_numpy()
    discrimination = {
        "class_aware_logistic_error_roc_auc": float(roc_auc_score(error, learned_score)),
        "class_aware_logistic_error_average_precision": float(average_precision_score(error, learned_score)),
        "error_prevalence": float(error.mean()),
    }

    ci = {}
    primary_budget = 0.05
    for method in deployable:
        ci[method] = grouped_bootstrap(frame, f"risk_{method}", primary_budget, config)

    certified = []
    for method in deployable:
        point = first_certified_point(frame, f"risk_{method}", config)
        if point is not None:
            certified.append(point)

    class_rows = []
    for method in deployable:
        referred = referral_mask(frame, f"risk_{method}", primary_budget)
        temporary = frame.assign(referred=referred)
        for label, group in temporary.groupby("sec_prediction", sort=True):
            class_rows.append({
                "method": method,
                "predicted_class": label,
                "samples": int(len(group)),
                "referral_rate": float(group["referred"].mean()),
                "error_rate": float(group["sec_error"].mean()),
            })

    table_dir = ROOT / "results/tables"
    metrics_dir = ROOT / "results/metrics"
    report_dir = ROOT / "results/reports"
    figure_dir = ROOT / "results/figures"
    for directory in (table_dir, metrics_dir, report_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_dir / "sec_bert_risk_router_oof_v1.csv", index=False, encoding="utf-8-sig")
    budget_table.to_csv(table_dir / "sec_bert_risk_router_budgets_v1.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(table_dir / "sec_bert_risk_coverage_curve_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(class_rows).to_csv(
        table_dir / "sec_bert_risk_router_class_referral_v1.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(7.2, 4.8))
    for method in ["confidence", "margin", "entropy", "class_aware_logistic", "random"]:
        group = curves.loc[(curves["method"] == method) & curves["coverage"].ge(0.80)]
        plt.plot(group["coverage"], group["selective_accuracy"], label=method.replace("_", " "))
    plt.axhline(float(config["target_selective_accuracy"]), color="black", linestyle="--", linewidth=1)
    plt.axvline(0.95, color="grey", linestyle=":", linewidth=1)
    plt.xlabel("Automatic coverage")
    plt.ylabel("Selective accuracy")
    plt.xlim(0.80, 1.00)
    plt.ylim(0.90, 0.98)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "sec_bert_risk_coverage_v1.png", dpi=220)
    plt.close()

    payload = {
        "version": config["version"],
        "scope": {
            "samples": int(len(frame)), "reports": int(frame["report_id"].nunique()),
            "errors": int(frame["sec_error"].sum()), "base_accuracy": float(1.0 - frame["sec_error"].mean()),
        },
        "crossfit": {
            "folds": sorted(frame["outer_fold"].astype(int).unique().tolist()),
            "group": "report_id", "features": NUMERIC + CATEGORICAL,
        },
        "discrimination": discrimination,
        "aurc": auc.to_dict(orient="records"),
        "primary_5pct_report_cluster_bootstrap": ci,
        "development_first_sample_level_binomial_screened_points": certified,
        "interpretation_guardrail": (
            "The binomial upper bound treats samples as independent and is only a development screening statistic. "
            "Report-cluster bootstrap intervals at 5% referral still cross the 95% accuracy target. "
            "Therefore the current system is risk-aware, not yet risk-controlled. The deployment threshold must "
            "be selected on calibration and evaluated unchanged on later data."
        ),
    }
    (metrics_dir / "sec_bert_risk_router_v1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    primary = budget_table.loc[budget_table["requested_referral_budget"].eq(primary_budget)].copy()
    primary = primary.sort_values("selective_accuracy", ascending=False)
    lines = [
        "# SEC-BERT risk router v1（development OOF）", "",
        f"分析範圍為 clean report-grouped OOF：n={len(frame):,}、reports={frame['report_id'].nunique():,}、"
        f"base accuracy={1.0-frame['sec_error'].mean():.4f}。learned router 的每筆分數均來自未見過該 report 的 router。", "",
        "## 5% referral 工作點", "",
        "| 方法 | Coverage | Selective accuracy | Sample-level binomial risk upper 95% | Error capture | Referral precision |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.method} | {row.coverage:.4f} | {row.selective_accuracy:.4f} | "
            f"{row.selective_risk_upper_95:.4f} | {row.error_capture_rate:.4f} | {row.referral_precision:.4f} |"
        )
    lines.extend(["", "## 判讀限制", "", (
        "本結果只比較 development OOF 的風險排序能力。表中的二項分布上界把樣本視為獨立，只能作篩選；"
        "5% referral 下按 report 聚類的 bootstrap 區間仍跨過95% accuracy目標，因此目前只能稱為risk-aware，"
        "不能稱為risk-controlled。下一步必須在獨立 calibration split 上選擇校準方法與5%／10%閾值，"
        "再原封不動套用於 retrospective test 與未來 temporal holdout。"
    ), ""])
    (report_dir / "sec_bert_risk_router_v1_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
