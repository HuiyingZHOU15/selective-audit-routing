"""Train and evaluate majority and TF-IDF logistic-regression baselines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline_v1.json"
METRICS_DIR = ROOT / "results" / "metrics"
TABLES_DIR = ROOT / "results" / "tables"
REPORT_PATH = ROOT / "results" / "reports" / "baseline_results_v1.md"
MODEL_DIR = ROOT / "artifacts" / "models"


def expected_calibration_error(y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray, bins: int = 15) -> float:
    predicted_index = probabilities.argmax(axis=1)
    predictions = classes[predicted_index]
    confidence = probabilities.max(axis=1)
    correct = (predictions == y_true.to_numpy()).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier(y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        one_hot[row, class_to_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def score_model(model: Pipeline, frame: pd.DataFrame, text_column: str) -> tuple[dict, np.ndarray, np.ndarray]:
    prediction = model.predict(frame[text_column])
    probabilities = model.predict_proba(frame[text_column])
    labels = model.named_steps["classifier"].classes_
    metrics = {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(frame["target_label"], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(frame["target_label"], prediction)),
        "macro_f1": float(f1_score(frame["target_label"], prediction, average="macro")),
        "weighted_f1": float(f1_score(frame["target_label"], prediction, average="weighted")),
        "log_loss": float(log_loss(frame["target_label"], probabilities, labels=labels)),
        "brier": multiclass_brier(frame["target_label"], probabilities, labels),
        "ece_15_bin": expected_calibration_error(frame["target_label"], probabilities, labels),
    }
    return metrics, prediction, probabilities


def build_pipeline(config: dict, c_value: float) -> Pipeline:
    tfidf = config["tfidf"]
    lr = config["logistic_regression"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(tfidf["ngram_range"]),
                    min_df=tfidf["min_df"],
                    max_df=tfidf["max_df"],
                    max_features=tfidf["max_features"],
                    sublinear_tf=tfidf["sublinear_tf"],
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=lr["class_weight"],
                    max_iter=lr["max_iter"],
                    solver=lr["solver"],
                    random_state=20260927,
                ),
            ),
        ]
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["dataset"])
    train = data.loc[data["split"].eq("train")].copy()
    validation = data.loc[data["split"].eq("validation") & data["clean_eval"]].copy()
    train_plus_validation = data.loc[data["split"].isin(["train", "validation"])].copy()
    test_all = data.loc[data["split"].eq("test")].copy()
    test_clean = test_all.loc[test_all["clean_eval"]].copy()

    for directory in (METRICS_DIR, TABLES_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    majority_label = train["target_label"].value_counts().idxmax()
    majority_prediction = np.repeat(majority_label, len(test_clean))
    majority_metrics = {
        "label": majority_label,
        "n": int(len(test_clean)),
        "accuracy": float(accuracy_score(test_clean["target_label"], majority_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(test_clean["target_label"], majority_prediction)),
        "macro_f1": float(f1_score(test_clean["target_label"], majority_prediction, average="macro")),
        "weighted_f1": float(f1_score(test_clean["target_label"], majority_prediction, average="weighted")),
    }

    results: dict[str, object] = {"majority": majority_metrics, "variants": {}}
    prediction_table = test_all[["sample_id", "report_id", "target_label", "clean_eval"]].copy()
    prediction_table["majority_prediction"] = majority_label

    report_lines = [
        "# Baseline Results v1",
        "",
        "Headline metrics use the leakage-controlled test subset (`clean_eval = true`).",
        "",
        "## Majority baseline",
        "",
        f"- Predicted label: `{majority_label}`",
        f"- Accuracy: {majority_metrics['accuracy']:.4f}",
        f"- Macro-F1: {majority_metrics['macro_f1']:.4f}",
        "",
    ]

    for variant, text_column in config["text_variants"].items():
        selection_rows = []
        for c_value in config["c_values"]:
            candidate = build_pipeline(config, float(c_value))
            candidate.fit(train[text_column], train["target_label"])
            validation_metrics, _, _ = score_model(candidate, validation, text_column)
            selection_rows.append({"C": float(c_value), **validation_metrics})
        selection = pd.DataFrame(selection_rows).sort_values(["macro_f1", "C"], ascending=[False, True])
        best_c = float(selection.iloc[0]["C"])
        selection.to_csv(TABLES_DIR / f"baseline_{variant}_validation_search.csv", index=False)

        final_model = build_pipeline(config, best_c)
        final_model.fit(train_plus_validation[text_column], train_plus_validation["target_label"])
        clean_metrics, clean_prediction, clean_probabilities = score_model(final_model, test_clean, text_column)
        all_metrics, all_prediction, all_probabilities = score_model(final_model, test_all, text_column)
        classes = final_model.named_steps["classifier"].classes_

        class_report = pd.DataFrame(
            classification_report(
                test_clean["target_label"], clean_prediction, labels=classes, output_dict=True, zero_division=0
            )
        ).transpose()
        class_report.to_csv(TABLES_DIR / f"baseline_{variant}_class_report.csv")
        pd.DataFrame(
            confusion_matrix(test_clean["target_label"], clean_prediction, labels=classes),
            index=classes,
            columns=classes,
        ).to_csv(TABLES_DIR / f"baseline_{variant}_confusion_matrix.csv")

        prediction_table[f"{variant}_prediction"] = all_prediction
        prediction_table[f"{variant}_confidence"] = all_probabilities.max(axis=1)
        joblib.dump(final_model, MODEL_DIR / f"tfidf_lr_{variant}_v1.joblib")
        results["variants"][variant] = {
            "best_C": best_c,
            "validation_search": selection_rows,
            "test_clean": clean_metrics,
            "test_unfiltered": all_metrics,
        }
        report_lines.extend(
            [
                f"## TF-IDF + Logistic Regression — {variant}",
                "",
                f"- Selected C: {best_c:g}",
                f"- Clean test n: {clean_metrics['n']}",
                f"- Accuracy: {clean_metrics['accuracy']:.4f}",
                f"- Macro-F1: {clean_metrics['macro_f1']:.4f}",
                f"- Weighted-F1: {clean_metrics['weighted_f1']:.4f}",
                f"- ECE (15 bins, uncalibrated): {clean_metrics['ece_15_bin']:.4f}",
                f"- Brier score: {clean_metrics['brier']:.4f}",
                f"- Unfiltered test Macro-F1: {all_metrics['macro_f1']:.4f}",
                "",
            ]
        )

    original_f1 = results["variants"]["original"]["test_clean"]["macro_f1"]
    masked_f1 = results["variants"]["masked"]["test_clean"]["macro_f1"]
    report_lines.extend(
        [
            "## Leakage sensitivity",
            "",
            f"Masking audit-area phrases and explicit AS references changes clean-test Macro-F1 by {masked_f1 - original_f1:+.4f}.",
            "This delta is descriptive; confidence intervals and paired testing belong to the next evaluation stage.",
            "",
            "The calibration split was not used in model fitting or hyperparameter selection.",
        ]
    )

    prediction_table.to_csv(TABLES_DIR / "baseline_test_predictions_v1.csv", index=False)
    (METRICS_DIR / "baseline_metrics_v1.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

