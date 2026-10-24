from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"
REPORTS = ROOT / "results" / "reports"
OUTPUT_ROOT = Path(os.environ.get("LABEL_AMBIGUITY_OUTPUT_DIR", str(ROOT)))
OUTPUT_TABLES = OUTPUT_ROOT / "tables"
OUTPUT_METRICS = OUTPUT_ROOT / "metrics"
OUTPUT_REPORTS = OUTPUT_ROOT / "reports"

PRIMARY_THRESHOLD = 0.5691283109492091
SAFETY_THRESHOLD = 0.8602307794324395

CONTEXT_LABELS = {
    "original_sufficient": "original_sufficient",
    "same_report_issuer_audit_area_preamble": "context_added",
    "reference_without_available_preamble": "context_unresolved",
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})


def context_metrics(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for raw_name, group in frame.groupby("context_quality", dropna=False):
        name = CONTEXT_LABELS.get(str(raw_name), str(raw_name))
        correct = as_bool(group["correct"])
        primary = group["calibrated_margin"].astype(float) < PRIMARY_THRESHOLD
        safety = group["calibrated_margin"].astype(float) < SAFETY_THRESHOLD
        errors = ~correct
        rows.append(
            {
                "context_group": name,
                "raw_context_quality": raw_name,
                "samples": len(group),
                "reports": group["report_id"].nunique(),
                "accuracy": accuracy_score(group["target_level1"], group["prediction"]),
                "macro_f1_12class": f1_score(
                    group["target_level1"], group["prediction"], labels=labels, average="macro", zero_division=0
                ),
                "mean_calibrated_margin": group["calibrated_margin"].astype(float).mean(),
                "primary_referral_rate": primary.mean(),
                "primary_error_capture": ((primary & errors).sum() / errors.sum()) if errors.sum() else np.nan,
                "safety_referral_rate": safety.mean(),
                "safety_error_capture": ((safety & errors).sum() / errors.sum()) if errors.sum() else np.nan,
                "errors": int(errors.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("context_group").reset_index(drop=True)


def ambiguity_feature_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    oof["sec_correct_bool"] = as_bool(oof["sec_correct"])
    oof["sec_error_bool"] = ~oof["sec_correct_bool"]
    cutoff = oof["sec_margin"].astype(float).quantile(0.05, interpolation="higher")
    oof["referred_5pct"] = oof["sec_margin"].astype(float) <= cutoff
    conditions = {
        "context_needs_context": oof["context_reference"].astype(float) > 0,
        "context_unresolved": oof["context_quality"].eq("reference_without_available_preamble"),
        "context_added": oof["context_quality"].eq("same_report_issuer_audit_area_preamble"),
        "multi_topic": oof["multi_topic"].astype(float) > 0,
        "crosscut_hits_ge_2": oof["crosscut_hits"].astype(float) >= 2,
        "tfidf_disagreement": oof["tfidf_disagree"].astype(float) > 0,
    }
    rows = []
    for feature, mask in conditions.items():
        for level, selected in [("present", mask), ("absent", ~mask)]:
            g = oof.loc[selected]
            rows.append(
                {
                    "feature": feature,
                    "level": level,
                    "samples": len(g),
                    "reports": g["report_id"].nunique(),
                    "error_rate": g["sec_error_bool"].mean(),
                    "mean_sec_margin": g["sec_margin"].astype(float).mean(),
                    "referral_rate_at_global_5pct": g["referred_5pct"].mean(),
                }
            )
    result = pd.DataFrame(rows)
    lifts = {}
    for feature in conditions:
        pair = result[result["feature"].eq(feature)].set_index("level")
        absent = float(pair.loc["absent", "error_rate"])
        present = float(pair.loc["present", "error_rate"])
        lifts[feature] = present / absent if absent else np.nan
    result["error_rate_ratio_present_vs_absent"] = result["feature"].map(lifts)
    return result


def confusion_pairs(scored: pd.DataFrame) -> pd.DataFrame:
    errors = scored.loc[~as_bool(scored["correct"])].copy()
    return (
        errors.groupby(["target_level1", "prediction"], dropna=False)
        .agg(
            errors=("sample_id", "size"),
            reports=("report_id", "nunique"),
            mean_calibrated_margin=("calibrated_margin", "mean"),
            context_unresolved=(
                "context_quality",
                lambda x: int((x == "reference_without_available_preamble").sum()),
            ),
        )
        .reset_index()
        .sort_values(["errors", "target_level1"], ascending=[False, True])
    )


def rag_takeover_analysis(oof: pd.DataFrame) -> pd.DataFrame:
    rag_path = ROOT / "data" / "interim" / "pcaob_rag_pilot_v1_audit.csv"
    rag = pd.read_csv(rag_path)
    rag = rag[rag["condition"].eq("report_rag") & as_bool(rag["completed"])].copy()
    features = oof[
        [
            "sample_id",
            "sec_correct",
            "context_quality",
            "context_reference",
            "multi_topic",
            "crosscut_hits",
            "tfidf_disagree",
        ]
    ].drop_duplicates("sample_id")
    merged = rag.merge(features, on="sample_id", how="inner", suffixes=("", "_oof"))
    sec_correct = as_bool(merged["sec_correct"])
    rag_correct = as_bool(merged["prediction_correct"])
    merged["sec_rag_outcome"] = np.select(
        [~sec_correct & rag_correct, sec_correct & ~rag_correct, sec_correct & rag_correct],
        ["beneficial", "harmful", "both_correct"],
        default="both_wrong",
    )
    conditions = {
        "all": pd.Series(True, index=merged.index),
        "context_needs_context": merged["context_reference"].astype(float) > 0,
        "context_unresolved": merged["context_quality_oof"].eq("reference_without_available_preamble")
        if "context_quality_oof" in merged.columns
        else merged["context_quality"].eq("reference_without_available_preamble"),
        "multi_topic": merged["multi_topic"].astype(float) > 0,
        "crosscut_hits_ge_2": merged["crosscut_hits"].astype(float) >= 2,
        "tfidf_disagreement": merged["tfidf_disagree"].astype(float) > 0,
    }
    rows = []
    for feature, mask in conditions.items():
        g = merged.loc[mask]
        counts = g["sec_rag_outcome"].value_counts()
        rows.append(
            {
                "feature_present": feature,
                "samples": len(g),
                "beneficial": int(counts.get("beneficial", 0)),
                "harmful": int(counts.get("harmful", 0)),
                "both_correct": int(counts.get("both_correct", 0)),
                "both_wrong": int(counts.get("both_wrong", 0)),
                "net": int(counts.get("beneficial", 0) - counts.get("harmful", 0)),
            }
        )
    return pd.DataFrame(rows)


def fmt_pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{100 * value:.2f}%"


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(TABLES / "sec_bert_frozen_test_scored_v1.csv")
    oof = pd.read_csv(TABLES / "sec_bert_risk_router_oof_v1.csv")
    oof = oof[as_bool(oof["oof_clean"])].copy()
    labels = sorted(scored["target_level1"].unique().tolist())

    context = context_metrics(scored, labels)
    ambiguity = ambiguity_feature_metrics(oof)
    confusions = confusion_pairs(scored)
    rag = rag_takeover_analysis(oof)

    context.to_csv(OUTPUT_TABLES / "sec_bert_context_quality_test_v1.csv", index=False)
    ambiguity.to_csv(OUTPUT_TABLES / "sec_bert_ambiguity_features_oof_v1.csv", index=False)
    confusions.to_csv(OUTPUT_TABLES / "sec_bert_confusion_pairs_test_v1.csv", index=False)
    rag.to_csv(OUTPUT_TABLES / "sec_bert_rag_takeover_ambiguity_v1.csv", index=False)

    metrics = {
        "version": "label-ambiguity-analysis-v1-20261013",
        "definitions": {
            "correctness": "agreement_with_PCAOB_derived_target_level1",
            "not_claimed": "objective_professional_validation_of_the_audit_area",
            "primary_threshold": PRIMARY_THRESHOLD,
            "safety_threshold": SAFETY_THRESHOLD,
        },
        "context_quality_test": context.to_dict(orient="records"),
        "ambiguity_features_clean_oof": ambiguity.to_dict(orient="records"),
        "top_test_confusion_pairs": confusions.head(15).to_dict(orient="records"),
        "report_rag_relative_to_sec_bert": rag.to_dict(orient="records"),
        "guardrail": "All associations are descriptive. Context and ambiguity flags are not expert-adjudicated ground truth.",
    }
    (OUTPUT_METRICS / "label_ambiguity_analysis_v1.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    context_lines = []
    for row in context.to_dict(orient="records"):
        context_lines.append(
            f"| {row['context_group']} | {row['samples']} | {row['accuracy']:.4f} | "
            f"{row['macro_f1_12class']:.4f} | {row['mean_calibrated_margin']:.4f} | "
            f"{fmt_pct(row['primary_referral_rate'])} | {fmt_pct(row['primary_error_capture'])} | "
            f"{fmt_pct(row['safety_referral_rate'])} | {fmt_pct(row['safety_error_capture'])} |"
        )

    present = ambiguity[ambiguity["level"].eq("present")]
    feature_lines = []
    for row in present.to_dict(orient="records"):
        feature_lines.append(
            f"| {row['feature']} | {row['samples']} | {fmt_pct(row['error_rate'])} | "
            f"{row['mean_sec_margin']:.4f} | {fmt_pct(row['referral_rate_at_global_5pct'])} | "
            f"{row['error_rate_ratio_present_vs_absent']:.2f} |"
        )

    confusion_lines = []
    for row in confusions.head(12).to_dict(orient="records"):
        confusion_lines.append(
            f"| {row['target_level1']} | {row['prediction']} | {row['errors']} | "
            f"{row['reports']} | {row['mean_calibrated_margin']:.4f} | {row['context_unresolved']} |"
        )

    rag_lines = []
    for row in rag.to_dict(orient="records"):
        rag_lines.append(
            f"| {row['feature_present']} | {row['samples']} | {row['beneficial']} | "
            f"{row['harmful']} | {row['net']} |"
        )

    report = f"""# 标签含义、上下文质量与歧义分析 v1

## 正式任务定义

本文预测的是由PCAOB原始Audit Area经过名称标准化和研究者映射得到的`target_level1`粗粒度类别。所谓预测正确，仅表示模型输出与该PCAOB-derived研究标签一致，即$\\hat y_i=y_i^{{PCAOB-derived}}$；它不表示模型独立发现了审计缺失，也不构成对客观唯一audit area的专业验证。

风险路由器估计的同样是“SEC-BERT预测与PCAOB-derived标签不一致的风险”。该风险可能来自模型错误、文本证据不足、多主题竞争、taxonomy边界或标签不一致。被转交样本应称为high-risk or potentially ambiguous cases，而不是直接称为模型错题。

## Retrospective test按上下文质量分组

| Context group | n | Accuracy | Macro-F1（固定12类） | Mean margin | 5%点referral | 5%点error capture | 10%点referral | 10%点error capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(context_lines)}

这些结果是描述性分组分析。小组macro-F1使用固定12类标签集合，因此在样本量较小、部分标签缺席的组中会较低。不能把context flag解释为人工确认的歧义ground truth。

## Clean development OOF中的歧义特征

全局5% referral由最低SEC-BERT margin定义。`error-rate ratio`比较特征存在组与不存在组。

| Feature present | n | Error rate | Mean SEC margin | Referral rate at global 5% | Error-rate ratio |
|---|---:|---:|---:|---:|---:|
{chr(10).join(feature_lines)}

这些关联用于说明哪些可观测信号与PCAOB-label disagreement同时出现，不用于证明错误的因果来源。

## SEC-BERT test主要混淆对

| PCAOB-derived target | SEC-BERT prediction | Errors | Reports | Mean margin | Context unresolved |
|---|---|---:|---:|---:|---:|
{chr(10).join(confusion_lines)}

该表是后续single-author exploratory adjudication的抽样框架。正式机制分类应判断信息是否充分、是否存在多个合理类别以及是否依赖完整报告；不得据此改写PCAOB标签并重算主结果。

## Report-RAG相对SEC-BERT的描述性结果

| Feature present | n | Beneficial | Harmful | Net |
|---|---:|---:|---:|---:|
{chr(10).join(rag_lines)}

RAG样本按早期small-model outcome刻意分层抽取，且只有120笔，因此这里只用于描述harmful takeover机制，不估计总体发生率。

## Masked-text敏感性的位置

早期TF-IDF实验中，遮蔽明确audit-area词组和AS引用后，taxonomy v1 macro-F1从0.8148降至0.5945；taxonomy v2中TF-IDF/context由0.8536降至0.5585。该结果说明任务显著依赖直接领域词汇，同时遮蔽后仍保留高于majority的可学习信号。它应被表述为任务可识别性和词汇依赖的敏感性分析，而不是SEC-BERT的模型消融或深层审计推理证明。

## 解释边界

观察到的错误率综合了模型失误、输入文本不完整、粗粒度taxonomy边界和潜在标签不一致。当前研究没有测试审计师能否修正所有转交案例，也没有测量人工时间或最终人机系统准确率。因此，97.67%的结果只能表述为：转交8.54%的高风险案例后，剩余91.46%自动接受样本与PCAOB-derived标签的一致率为97.67%。
"""
    (OUTPUT_REPORTS / "label_ambiguity_analysis_v1_zh.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
