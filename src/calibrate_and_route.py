"""Calibrate frozen baselines and evaluate selective routing without using test labels for selection."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "calibration_v1.json"
METRICS_PATH = ROOT / "results" / "metrics" / "calibration_routing_metrics_v1.json"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "artifacts" / "models"
REPORT_PATH = ROOT / "results" / "reports" / "calibration_routing_summary_zh.md"

OKABE_ITO = {"original": "#0072B2", "masked": "#D55E00"}


def ece_score(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 15) -> float:
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    indices = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        one_hot[row, indices[label]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def risk_coverage(errors: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-confidence, kind="stable")
    sorted_errors = errors[order].astype(float)
    accepted = np.arange(1, len(errors) + 1)
    cumulative_errors = np.cumsum(sorted_errors)
    coverage = accepted / len(errors)
    selective_risk = cumulative_errors / accepted
    generalized_risk = cumulative_errors / len(errors)
    return pd.DataFrame(
        {
            "coverage": coverage,
            "selective_risk": selective_risk,
            "generalized_risk": generalized_risk,
            "accepted": accepted,
        }
    )


def aurc_metrics(errors: np.ndarray, confidence: np.ndarray) -> tuple[float, float]:
    curve = risk_coverage(errors, confidence)
    return float(curve["selective_risk"].mean()), float(curve["generalized_risk"].mean())


def reliability_points(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int = 15
) -> pd.DataFrame:
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            rows.append(
                {
                    "bin": index + 1,
                    "lower": lower,
                    "upper": upper,
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metrics(
    frame: pd.DataFrame,
    errors: np.ndarray,
    confidence: np.ndarray,
    thresholds: dict[float, float],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    reports = frame["report_id"].unique()
    positions = {report: np.flatnonzero(frame["report_id"].to_numpy() == report) for report in reports}
    values: dict[str, list[float]] = {"aurc": [], "augrc": [], "error_rate": []}
    for budget in thresholds:
        values[f"risk_budget_{budget:.2f}"] = []
    for _ in range(iterations):
        sampled_reports = rng.choice(reports, size=len(reports), replace=True)
        sampled = np.concatenate([positions[report] for report in sampled_reports])
        sample_errors = errors[sampled]
        sample_confidence = confidence[sampled]
        aurc, augrc = aurc_metrics(sample_errors, sample_confidence)
        values["aurc"].append(aurc)
        values["augrc"].append(augrc)
        values["error_rate"].append(float(sample_errors.mean()))
        for budget, threshold in thresholds.items():
            accepted = sample_confidence >= threshold
            risk = float(sample_errors[accepted].mean()) if accepted.any() else np.nan
            values[f"risk_budget_{budget:.2f}"].append(risk)
    return {
        name: {
            "lower_95": float(np.nanquantile(series, 0.025)),
            "median": float(np.nanquantile(series, 0.5)),
            "upper_95": float(np.nanquantile(series, 0.975)),
        }
        for name, series in values.items()
    }


def random_routing_risk(errors: np.ndarray, accepted_count: int, seed: int, iterations: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    risks = []
    for _ in range(iterations):
        accepted = rng.choice(len(errors), size=accepted_count, replace=False)
        risks.append(float(errors[accepted].mean()))
    return {
        "mean": float(np.mean(risks)),
        "lower_95": float(np.quantile(risks, 0.025)),
        "upper_95": float(np.quantile(risks, 0.975)),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float", "float64"]).columns:
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}")
    shown = shown.astype(str)
    header = "| " + " | ".join(shown.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()]
    return "\n".join([header, divider, *rows])


def save_figure(
    curves: dict[str, pd.DataFrame],
    reliability: dict[str, dict[str, pd.DataFrame]],
    operating: pd.DataFrame,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))

    ax = axes[0]
    for variant, curve in curves.items():
        sampled = curve.iloc[:: max(1, len(curve) // 250)]
        ax.plot(
            sampled["coverage"],
            sampled["selective_risk"],
            color=OKABE_ITO[variant],
            linewidth=1.5,
            label=variant.capitalize(),
        )
        points = operating.loc[operating["variant"].eq(variant)]
        ax.scatter(points["test_coverage"], points["selective_risk"], color=OKABE_ITO[variant], s=14, zorder=3)
    ax.set_xlabel("Coverage (accepted fraction)")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    for index, variant in enumerate(("original", "masked"), start=1):
        ax = axes[index]
        ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1, label="Ideal")
        for status, linestyle, marker in (("uncalibrated", "--", "s"), ("calibrated", "-", "o")):
            points = reliability[variant][status]
            ax.plot(
                points["mean_confidence"],
                points["accuracy"],
                color=OKABE_ITO[variant],
                linestyle=linestyle,
                marker=marker,
                markersize=3,
                linewidth=1.1,
                label=status.capitalize(),
            )
        ax.set_title(variant.capitalize())
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Observed accuracy" if index == 1 else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="lower right")

    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    fig.tight_layout(w_pad=1.2)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"selective_calibration_v1.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["dataset"])
    calibration = data.loc[data["split"].eq("calibration") & data["clean_eval"]].copy()
    test = data.loc[data["split"].eq("test") & data["clean_eval"]].copy()
    y_cal = calibration["target_label"].to_numpy()
    y_test = test["target_label"].to_numpy()

    for directory in (METRICS_PATH.parent, TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORT_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "calibration_n": len(calibration),
        "test_n": len(test),
        "method": config["method"],
        "variants": {},
    }
    prediction_output = test[["sample_id", "report_id", "target_label"]].copy()
    curve_frames = []
    operating_rows = []
    class_coverage_rows = []
    reliability_frames = []
    curves: dict[str, pd.DataFrame] = {}
    reliability_for_plot: dict[str, dict[str, pd.DataFrame]] = {}

    for variant, model_relative in config["models"].items():
        text_column = config["text_columns"][variant]
        base_model = joblib.load(ROOT / model_relative)
        calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method=config["method"])
        calibrator.fit(calibration[text_column], y_cal)
        joblib.dump(calibrator, MODELS_DIR / f"tfidf_lr_{variant}_sigmoid_v1.joblib")

        uncal_prob_cal = base_model.predict_proba(calibration[text_column])
        cal_prob_cal = calibrator.predict_proba(calibration[text_column])
        uncal_prob_test = base_model.predict_proba(test[text_column])
        cal_prob_test = calibrator.predict_proba(test[text_column])
        classes = calibrator.classes_

        thresholds = {
            float(budget): float(np.quantile(cal_prob_cal.max(axis=1), budget, method="higher"))
            for budget in config["escalation_budgets"]
        }
        calibrated_prediction = classes[cal_prob_test.argmax(axis=1)]
        uncalibrated_prediction = base_model.classes_[uncal_prob_test.argmax(axis=1)]
        errors = (calibrated_prediction != y_test).astype(int)
        confidence = cal_prob_test.max(axis=1)
        uncal_errors = (uncalibrated_prediction != y_test).astype(int)
        uncal_confidence = uncal_prob_test.max(axis=1)
        curve = risk_coverage(errors, confidence)
        curve["variant"] = variant
        curve_frames.append(curve)
        curves[variant] = curve

        aurc, augrc = aurc_metrics(errors, confidence)
        uncal_aurc, uncal_augrc = aurc_metrics(uncal_errors, uncal_confidence)
        before = {
            "accuracy": float(accuracy_score(y_test, uncalibrated_prediction)),
            "error_rate": float(uncal_errors.mean()),
            "ece": ece_score(y_test, uncal_prob_test, base_model.classes_, config["ece_bins"]),
            "brier": brier_score(y_test, uncal_prob_test, base_model.classes_),
            "log_loss": float(log_loss(y_test, uncal_prob_test, labels=base_model.classes_)),
            "aurc": uncal_aurc,
            "augrc": uncal_augrc,
        }
        after = {
            "accuracy": float(accuracy_score(y_test, calibrated_prediction)),
            "error_rate": float(errors.mean()),
            "ece": ece_score(y_test, cal_prob_test, classes, config["ece_bins"]),
            "brier": brier_score(y_test, cal_prob_test, classes),
            "log_loss": float(log_loss(y_test, cal_prob_test, labels=classes)),
            "aurc": aurc,
            "augrc": augrc,
        }

        reliability_for_plot[variant] = {}
        for status, probabilities, status_classes in (
            ("uncalibrated", uncal_prob_test, base_model.classes_),
            ("calibrated", cal_prob_test, classes),
        ):
            points = reliability_points(y_test, probabilities, status_classes, config["ece_bins"])
            points["variant"] = variant
            points["status"] = status
            reliability_frames.append(points)
            reliability_for_plot[variant][status] = points

        budget_results = {}
        for offset, (budget, threshold) in enumerate(thresholds.items()):
            accepted = confidence >= threshold
            routed = ~accepted
            selective_risk = float(errors[accepted].mean()) if accepted.any() else np.nan
            generalized_risk = float(errors[accepted].sum() / len(errors))
            random = random_routing_risk(errors, int(accepted.sum()), config["bootstrap_seed"] + offset)
            row = {
                "variant": variant,
                "target_escalation_budget": budget,
                "calibration_threshold": threshold,
                "calibration_escalation_rate": float((cal_prob_cal.max(axis=1) < threshold).mean()),
                "test_escalation_rate": float(routed.mean()),
                "test_coverage": float(accepted.mean()),
                "accepted": int(accepted.sum()),
                "routed": int(routed.sum()),
                "selective_risk": selective_risk,
                "generalized_risk": generalized_risk,
                "total_test_errors": int(errors.sum()),
                "accepted_errors": int(errors[accepted].sum()),
                "routed_errors": int(errors[routed].sum()),
                "routed_error_capture_rate": float(errors[routed].sum() / errors.sum()) if errors.sum() else 0.0,
                "routed_error_precision": float(errors[routed].mean()) if routed.any() else np.nan,
                "random_routing_mean_risk": random["mean"],
                "random_routing_lower_95": random["lower_95"],
                "random_routing_upper_95": random["upper_95"],
            }
            operating_rows.append(row)
            budget_results[f"{budget:.2f}"] = row
            prediction_output[f"{variant}_route_{int(budget * 100)}pct"] = routed
            for label in classes:
                label_mask = y_test == label
                class_coverage_rows.append(
                    {
                        "variant": variant,
                        "target_escalation_budget": budget,
                        "class": label,
                        "class_n": int(label_mask.sum()),
                        "class_coverage": float(accepted[label_mask].mean()) if label_mask.any() else np.nan,
                        "class_error_rate_accepted": float(errors[label_mask & accepted].mean())
                        if (label_mask & accepted).any()
                        else np.nan,
                    }
                )

        prediction_output[f"{variant}_prediction"] = calibrated_prediction
        prediction_output[f"{variant}_confidence"] = confidence
        ci = bootstrap_metrics(
            test,
            errors,
            confidence,
            thresholds,
            int(config["bootstrap_iterations"]),
            int(config["bootstrap_seed"]),
        )
        results["variants"][variant] = {
            "uncalibrated": before,
            "calibrated": after,
            "thresholds_from_calibration": {f"{key:.2f}": value for key, value in thresholds.items()},
            "operating_points": budget_results,
            "report_cluster_bootstrap_95_ci": ci,
        }

    operating = pd.DataFrame(operating_rows)
    pd.concat(curve_frames, ignore_index=True).to_csv(TABLES_DIR / "risk_coverage_points_v1.csv", index=False)
    pd.concat(reliability_frames, ignore_index=True).to_csv(TABLES_DIR / "reliability_points_v1.csv", index=False)
    operating.to_csv(TABLES_DIR / "routing_operating_points_v1.csv", index=False)
    class_coverage = pd.DataFrame(class_coverage_rows)
    class_coverage.to_csv(TABLES_DIR / "routing_class_coverage_v1.csv", index=False)
    prediction_output.to_csv(TABLES_DIR / "calibrated_test_predictions_v1.csv", index=False)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_figure(curves, reliability_for_plot, operating)

    original = results["variants"]["original"]
    masked = results["variants"]["masked"]
    operating_display = operating[
        [
            "variant",
            "target_escalation_budget",
            "test_escalation_rate",
            "selective_risk",
            "routed_error_capture_rate",
            "routed_error_precision",
            "random_routing_mean_risk",
        ]
    ].copy()
    class_20 = class_coverage.loc[class_coverage["target_escalation_budget"].eq(0.20)]
    disparity_rows = []
    for variant, group in class_20.groupby("variant"):
        lowest = group.loc[group["class_coverage"].idxmin()]
        highest = group.loc[group["class_coverage"].idxmax()]
        worst_error = group.loc[group["class_error_rate_accepted"].idxmax()]
        disparity_rows.append(
            {
                "variant": variant,
                "lowest_coverage_class": lowest["class"],
                "lowest_coverage": lowest["class_coverage"],
                "highest_coverage_class": highest["class"],
                "highest_coverage": highest["class_coverage"],
                "coverage_gap": highest["class_coverage"] - lowest["class_coverage"],
                "worst_accepted_error_class": worst_error["class"],
                "worst_accepted_error_rate": worst_error["class_error_rate_accepted"],
            }
        )
    disparity = pd.DataFrame(disparity_rows)
    report = f"""# 校準與 Selective Routing 結果 v1

## 實驗約束

- Sigmoid calibrator 僅使用 {len(calibration):,} 個 `clean_eval = true` calibration 樣本。
- 最終評估使用 {len(test):,} 個乾淨 test 樣本。
- 10%／20%／30% 路由門檻由 calibration set 分位數固定；test labels 沒有參與模型、校準器或門檻選擇。
- 95% CI 使用按 `report_id` 聚類的 {config['bootstrap_iterations']} 次 bootstrap。

## 校準前後

| 文本 | 狀態 | Accuracy | ECE | Brier | AURC | AUGRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 原文 | 未校準 | {original['uncalibrated']['accuracy']:.4f} | {original['uncalibrated']['ece']:.4f} | {original['uncalibrated']['brier']:.4f} | {original['uncalibrated']['aurc']:.4f} | {original['uncalibrated']['augrc']:.4f} |
| 原文 | Sigmoid | {original['calibrated']['accuracy']:.4f} | {original['calibrated']['ece']:.4f} | {original['calibrated']['brier']:.4f} | {original['calibrated']['aurc']:.4f} | {original['calibrated']['augrc']:.4f} |
| 遮蔽 | 未校準 | {masked['uncalibrated']['accuracy']:.4f} | {masked['uncalibrated']['ece']:.4f} | {masked['uncalibrated']['brier']:.4f} | {masked['uncalibrated']['aurc']:.4f} | {masked['uncalibrated']['augrc']:.4f} |
| 遮蔽 | Sigmoid | {masked['calibrated']['accuracy']:.4f} | {masked['calibrated']['ece']:.4f} | {masked['calibrated']['brier']:.4f} | {masked['calibrated']['aurc']:.4f} | {masked['calibrated']['augrc']:.4f} |

## 固定預算路由

{markdown_table(operating_display)}

`routed_error_capture_rate` 表示被路由樣本捕獲了全部小模型錯誤的比例；`routed_error_precision` 表示被路由樣本中真正錯誤的比例。

## 20% 預算的類別 coverage 差異

{markdown_table(disparity)}

全局門檻對不同類別產生明顯不同的 coverage。這不是可忽略的副作用：下一階段需要比較全局門檻與 class-conditional thresholds，並逐類報告 coverage 和 accepted error rate。

## 解讀原則

- `selective_risk` 是留給小模型直接處理的樣本錯誤率；越低越好。
- `generalized_risk` 是未被路由、且仍然錯誤的樣本佔全部測試樣本比例。
- AURC 與 AUGRC 越低越好。AUGRC 依 generalized risk 定義，能同時反映分類能力和信心排序。
- 本階段只證明 confidence routing 能否挑出較難樣本，尚未假設 LLM 能修正被路由樣本。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
