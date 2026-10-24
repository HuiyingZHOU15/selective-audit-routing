"""Calibrate the frozen taxonomy-v2 TF-IDF model and freeze routing decisions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "core_experiment_v2.json"
MODEL_OUT = ROOT / "artifacts" / "models" / "tfidf_lr_context_sigmoid_taxonomy_v2.joblib"
PREDICTIONS_OUT = ROOT / "results" / "tables" / "core_v2_test_predictions.csv"
METRICS_OUT = ROOT / "results" / "metrics" / "core_v2_calibration_routing.json"
REPORT_OUT = ROOT / "results" / "reports" / "core_v2_calibration_routing_zh.md"


def ece_score(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int) -> float:
    prediction = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def brier_score(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    positions = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        one_hot[row, positions[label]] = 1
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def model_metrics(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, bins: int) -> dict:
    prediction = classes[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "brier": brier_score(y_true, probabilities, classes),
        "ece": ece_score(y_true, probabilities, classes, bins),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["dataset"])
    calibration = data.loc[data["split"].eq("calibration") & data["clean_eval"]].copy()
    test = data.loc[data["split"].eq("test") & data["clean_eval"]].copy().reset_index(drop=True)
    target = config["target"]
    text_column = config["text_column"]
    frozen = joblib.load(ROOT / config["model"])
    classes = np.asarray(frozen.classes_)

    uncalibrated_cal = frozen.predict_proba(calibration[text_column].fillna(""))
    uncalibrated_test = frozen.predict_proba(test[text_column].fillna(""))
    # Reserve reports inside the calibration split for choosing whether Platt
    # scaling is actually useful.  This prevents the clean test set from
    # silently becoming a calibration-method selection set.
    split = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=config["seed"])
    fit_index, select_index = next(
        split.split(calibration, calibration[target], groups=calibration["report_id"])
    )
    calibration_fit = calibration.iloc[fit_index]
    calibration_select = calibration.iloc[select_index]
    candidate = CalibratedClassifierCV(FrozenEstimator(frozen), method=config["calibration_method"])
    candidate.fit(calibration_fit[text_column].fillna(""), calibration_fit[target])
    candidate_select = candidate.predict_proba(calibration_select[text_column].fillna(""))
    raw_select = frozen.predict_proba(calibration_select[text_column].fillna(""))
    selection_raw = model_metrics(
        calibration_select[target].to_numpy(), raw_select, classes, config["ece_bins"]
    )
    selection_sigmoid = model_metrics(
        calibration_select[target].to_numpy(), candidate_select, classes, config["ece_bins"]
    )
    # Log loss is the primary proper scoring rule; ECE breaks practically exact ties.
    use_sigmoid = (selection_sigmoid["log_loss"], selection_sigmoid["ece"]) < (
        selection_raw["log_loss"],
        selection_raw["ece"],
    )
    calibrated = CalibratedClassifierCV(FrozenEstimator(frozen), method=config["calibration_method"])
    calibrated.fit(calibration[text_column].fillna(""), calibration[target])
    calibrated_cal = calibrated.predict_proba(calibration[text_column].fillna(""))
    calibrated_test = calibrated.predict_proba(test[text_column].fillna(""))
    calibrated_classes = np.asarray(calibrated.classes_)
    if not np.array_equal(classes, calibrated_classes):
        raise RuntimeError("Class order changed during calibration.")

    selected_model = calibrated if use_sigmoid else frozen
    selected_cal = calibrated_cal if use_sigmoid else uncalibrated_cal
    selected_test = calibrated_test if use_sigmoid else uncalibrated_test
    selected_method = "sigmoid" if use_sigmoid else "uncalibrated"
    calibration_confidence = selected_cal.max(axis=1)
    thresholds = {
        float(budget): float(np.quantile(calibration_confidence, budget, method="linear"))
        for budget in config["routing_budgets"]
    }
    prediction = classes[selected_test.argmax(axis=1)]
    confidence = selected_test.max(axis=1)
    output = test[
        ["sample_id", "report_id", "target_level1", "target_level2", "audit_areas", "context_quality"]
    ].copy()
    output["small_prediction"] = prediction
    output["small_confidence"] = confidence
    output["small_correct"] = output["small_prediction"].eq(output["target_level1"])
    operating = []
    for budget, threshold in thresholds.items():
        suffix = int(round(budget * 100))
        # Enforce the stated review/LLM budget exactly by ranking the current
        # unlabeled batch.  Stable sorting makes ties reproducible; labels are
        # used below only for retrospective evaluation.
        route = np.zeros(len(output), dtype=bool)
        routed_count = int(round(budget * len(output)))
        route[np.argsort(confidence, kind="stable")[:routed_count]] = True
        output[f"route_{suffix}pct"] = route
        accepted = ~route
        operating.append(
            {
                "budget": budget,
                "calibration_quantile_threshold": threshold,
                "realized_batch_cutoff": float(confidence[route].max()),
                "routed_count": int(route.sum()),
                "routed_fraction": float(route.mean()),
                "accepted_accuracy": float(output.loc[accepted, "small_correct"].mean()),
                "routed_small_accuracy": float(output.loc[route, "small_correct"].mean()),
                "captured_small_errors": int((route & ~output["small_correct"]).sum()),
                "captured_error_fraction": float(
                    (route & ~output["small_correct"]).sum() / (~output["small_correct"]).sum()
                ),
            }
        )

    for path in (MODEL_OUT, PREDICTIONS_OUT, METRICS_OUT, REPORT_OUT):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_OUT)
    output.to_csv(PREDICTIONS_OUT, index=False, encoding="utf-8")
    payload = {
        "version": config["version"],
        "calibration_samples": len(calibration),
        "selection_samples": len(calibration_select),
        "selection_raw": selection_raw,
        "selection_sigmoid": selection_sigmoid,
        "selected_method": selected_method,
        "test_samples": len(test),
        "uncalibrated_test": model_metrics(test[target].to_numpy(), uncalibrated_test, classes, config["ece_bins"]),
        "calibrated_test": model_metrics(test[target].to_numpy(), calibrated_test, classes, config["ece_bins"]),
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "operating_points": operating,
    }
    METRICS_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "\n".join(
        f"| {row['budget']:.0%} | {row['routed_count']} | {row['routed_fraction']:.2%} | "
        f"{row['accepted_accuracy']:.4f} | {row['routed_small_accuracy']:.4f} | {row['captured_error_fraction']:.2%} |"
        for row in operating
    )
    report = f"""# Taxonomy v2 校準與路由凍結報告

## 校準結果

- Clean calibration：{len(calibration)} 筆；clean test：{len(test)} 筆。
- 校準集內以 report 分組留出 {len(calibration_select)} 筆，只用來選擇原始分數或 sigmoid；選中：**{selected_method}**。
- 未校準：accuracy {payload['uncalibrated_test']['accuracy']:.4f}，macro-F1 {payload['uncalibrated_test']['macro_f1']:.4f}，ECE {payload['uncalibrated_test']['ece']:.4f}。
- Sigmoid 校準：accuracy {payload['calibrated_test']['accuracy']:.4f}，macro-F1 {payload['calibrated_test']['macro_f1']:.4f}，ECE {payload['calibrated_test']['ece']:.4f}。

## 預先凍結的操作點

| 目標預算 | 路由數 | 實際比例 | 小模型已接受準確率 | 路由子集小模型準確率 | 捕捉錯誤比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

校準分位門檻作為漂移診斷保留；主實驗按 test 批次的置信度排序精確執行預算。排序不使用 test 標籤。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
