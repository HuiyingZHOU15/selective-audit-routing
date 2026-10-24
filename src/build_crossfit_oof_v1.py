"""Generate leakage-controlled nested out-of-fold predictions for PCAOB development data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/crossfit_v1.json"


def component_groups(frame: pd.DataFrame) -> pd.Series:
    # report_id is the true independent unit. Joining reports transitively by
    # repeated templates creates giant components, so duplicates are handled as
    # auditable clean-OOF exclusions after balanced report-group assignment.
    return frame["report_id"].astype(str)


def make_pipeline(config: dict, c_value: float = 1.0) -> Pipeline:
    tfidf = config["tfidf"]
    classifier = config["classifier"]
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
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=classifier["class_weight"],
                    max_iter=classifier["max_iter"],
                    random_state=config["random_seed"],
                ),
            ),
        ]
    )


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["dataset"])
    frame = data.loc[data["split"].isin(config["development_splits"])].copy().reset_index(drop=True)
    frame["leakage_component_id"] = component_groups(frame)
    target = config["target"]
    text_column = config["text_column"]
    labels = np.asarray(sorted(frame[target].unique()))
    groups = frame["leakage_component_id"].to_numpy()
    y = frame[target].to_numpy()
    texts = frame[text_column].fillna("").astype(str).to_numpy()

    outer = StratifiedGroupKFold(
        n_splits=config["outer_folds"], shuffle=True, random_state=config["random_seed"]
    )
    oof_probabilities = np.full((len(frame), len(labels)), np.nan, dtype=float)
    oof_raw_probabilities = np.full((len(frame), len(labels)), np.nan, dtype=float)
    fold_number = np.full(len(frame), -1, dtype=int)
    fold_rows = []

    for fold, (train_index, hold_index) in enumerate(outer.split(texts, y, groups), start=1):
        inner = StratifiedGroupKFold(
            n_splits=config["inner_folds"],
            shuffle=True,
            random_state=config["random_seed"] + fold,
        )
        inner_splits = list(inner.split(texts[train_index], y[train_index], groups[train_index]))
        search = GridSearchCV(
            estimator=make_pipeline(config),
            param_grid={"classifier__C": config["classifier"]["c_grid"]},
            scoring="f1_macro",
            cv=inner_splits,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(texts[train_index], y[train_index])
        best_c = float(search.best_params_["classifier__C"])
        raw_probabilities = search.best_estimator_.predict_proba(texts[hold_index])
        if not np.array_equal(np.asarray(search.best_estimator_.classes_), labels):
            raise RuntimeError(f"Raw class order mismatch in fold {fold}")

        # CalibratedClassifierCV refits every complete text pipeline only on its
        # permitted inner-training fold. No vocabulary or calibration target is
        # learned from the outer holdout.
        calibrated = CalibratedClassifierCV(
            estimator=make_pipeline(config, best_c),
            method=config["calibration_method"],
            cv=inner_splits,
            n_jobs=-1,
        )
        calibrated.fit(texts[train_index], y[train_index])
        probabilities = calibrated.predict_proba(texts[hold_index])
        if not np.array_equal(np.asarray(calibrated.classes_), labels):
            raise RuntimeError(f"Class order mismatch in fold {fold}")
        oof_probabilities[hold_index] = probabilities
        oof_raw_probabilities[hold_index] = raw_probabilities
        fold_number[hold_index] = fold
        prediction = labels[probabilities.argmax(axis=1)]
        fold_rows.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "holdout_samples": len(hold_index),
                "train_components": int(pd.Series(groups[train_index]).nunique()),
                "holdout_components": int(pd.Series(groups[hold_index]).nunique()),
                "best_C": best_c,
                "inner_best_macro_f1": float(search.best_score_),
                "holdout_accuracy": float(accuracy_score(y[hold_index], prediction)),
                "holdout_macro_f1": float(f1_score(y[hold_index], prediction, average="macro", zero_division=0)),
            }
        )
        print(json.dumps(fold_rows[-1]), flush=True)

    if np.isnan(oof_probabilities).any() or np.isnan(oof_raw_probabilities).any() or (fold_number < 1).any():
        raise RuntimeError("OOF matrix is incomplete")
    prediction = labels[oof_probabilities.argmax(axis=1)]
    sorted_probability = np.sort(oof_probabilities, axis=1)
    frame["outer_fold"] = fold_number
    frame["small_prediction"] = prediction
    frame["small_confidence"] = oof_probabilities.max(axis=1)
    frame["small_margin"] = sorted_probability[:, -1] - sorted_probability[:, -2]
    frame["small_entropy"] = entropy(oof_probabilities)
    raw_sorted = np.sort(oof_raw_probabilities, axis=1)
    frame["small_raw_confidence"] = oof_raw_probabilities.max(axis=1)
    frame["small_raw_margin"] = raw_sorted[:, -1] - raw_sorted[:, -2]
    frame["small_raw_entropy"] = entropy(oof_raw_probabilities)
    frame["small_correct"] = frame["small_prediction"].eq(frame[target])
    frame["small_probability_json"] = [
        json.dumps(dict(zip(labels, row)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in oof_probabilities
    ]
    frame["crossfit_version"] = config["version"]
    frame["oof_clean"] = True
    leakage_reasons = {index: set() for index in frame.index}
    cross_fold_exact_groups = 0
    for _, duplicate_group in frame.groupby("description_norm", sort=False):
        positions = duplicate_group.index.to_numpy()
        if len(positions) > 1 and len(set(fold_number[positions])) > 1:
            cross_fold_exact_groups += 1
            for position in positions:
                leakage_reasons[position].add("exact_duplicate_across_outer_fold")
    near = pd.read_csv(ROOT / config["near_duplicates"])
    position_by_sample = {str(sample_id): index for index, sample_id in enumerate(frame["sample_id"])}
    cross_fold_near_pairs = 0
    for row in near.itertuples():
        left = position_by_sample.get(str(row.sample_id))
        right = position_by_sample.get(str(row.neighbor_sample_id))
        if left is None or right is None or fold_number[left] == fold_number[right]:
            continue
        cross_fold_near_pairs += 1
        for position in (left, right):
            leakage_reasons[position].add("audited_near_duplicate_across_outer_fold")
    frame["oof_leakage_reason"] = [" | ".join(sorted(leakage_reasons[index])) for index in frame.index]
    frame["oof_clean"] = frame["oof_leakage_reason"].eq("")

    keep = [
        "sample_id", "report_id", "Registration ID", "Firm Names", "Inspection Year",
        "Issuer Reference Key", "target_level1", "target_level2", "audit_areas", "context_quality",
        "context_needs_context", "context_added", "text_context_v2", "description_norm",
        "leakage_component_id", "outer_fold", "small_prediction", "small_confidence",
        "small_margin", "small_entropy", "small_correct", "small_probability_json", "crossfit_version",
        "small_raw_confidence", "small_raw_margin", "small_raw_entropy",
        "oof_clean", "oof_leakage_reason",
    ]
    output = ROOT / config["oof_output"]
    manifest_path = ROOT / config["fold_manifest"]
    metrics_path = ROOT / config["metrics_output"]
    report_path = ROOT / config["report_output"]
    for path in (output, manifest_path, metrics_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].sort_values("sample_id").to_csv(output, index=False, encoding="utf-8")
    frame[["sample_id", "report_id", "leakage_component_id", "outer_fold"]].sort_values("sample_id").to_csv(
        manifest_path, index=False, encoding="utf-8"
    )
    metrics = {
        "version": config["version"],
        "samples": len(frame),
        "reports": int(frame["report_id"].nunique()),
        "leakage_components": int(frame["leakage_component_id"].nunique()),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, oof_probabilities, labels=labels)),
        "small_errors": int((prediction != y).sum()),
        "clean_oof_samples": int(frame["oof_clean"].sum()),
        "cross_fold_near_duplicate_pairs": cross_fold_near_pairs,
        "cross_fold_exact_duplicate_groups": cross_fold_exact_groups,
        "folds": fold_rows,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    fold_frame = pd.DataFrame(fold_rows)
    fold_table = "\n".join(
        [
            "| " + " | ".join(fold_frame.columns) + " |",
            "| " + " | ".join(["---"] * len(fold_frame.columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in fold_frame.to_numpy().tolist()],
        ]
    )
    report = f"""# PCAOB development nested OOF v1

- 樣本：{len(frame):,}；reports：{metrics['reports']:,}；leakage-connected components：{metrics['leakage_components']:,}。
- OOF accuracy：{metrics['accuracy']:.4f}；macro-F1：{metrics['macro_f1']:.4f}；balanced accuracy：{metrics['balanced_accuracy']:.4f}。
- OOF 小模型錯誤：{metrics['small_errors']:,}。
- 每個 outer fold 的 TF-IDF vocabulary、classifier、inner model selection 與 sigmoid calibration 均只使用該 fold 的 training portion。
- `report_id` 是不可拆分的獨立單位。將跨報告模板作傳遞閉包會形成 1,113 筆 exact component 與 3,083 筆 near component，破壞 fold 可比性，因此不採用。
- 跨 fold exact groups 共 {cross_fold_exact_groups:,} 組，audited near pairs 共 {cross_fold_near_pairs:,} 對；相關樣本標記 `oof_clean=false`，從 clean router 訓練中排除並保留供敏感度分析。

## Folds

{fold_table}
"""
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
