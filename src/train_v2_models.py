"""Compare TF-IDF and frozen MiniLM representations on the frozen v2 taxonomy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from huggingface_hub import snapshot_download
from transformers import AutoModel, BertTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "modeling_v2.json"
METRICS_PATH = ROOT / "results" / "metrics" / "model_comparison_v2.json"
PREDICTIONS_PATH = ROOT / "results" / "tables" / "model_comparison_test_predictions_v2.csv"
CLASS_PATH = ROOT / "results" / "tables" / "model_comparison_class_metrics_v2.csv"
REPORT_PATH = ROOT / "results" / "reports" / "model_comparison_v2_zh.md"
MODEL_DIR = ROOT / "artifacts" / "models"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings"


def classifier(config: dict, c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight=config["classifier"]["class_weight"],
        max_iter=config["classifier"]["max_iter"],
        random_state=config["classifier"]["random_seed"],
    )


def score_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def tune_classifier(x_train, y_train, x_validation, y_validation, config: dict):
    trials = []
    best = None
    for c_value in config["classifier"]["c_grid"]:
        model = classifier(config, c_value)
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        scores = score_metrics(y_validation, prediction)
        trial = {"C": c_value, **scores}
        trials.append(trial)
        key = (scores["macro_f1"], scores["accuracy"], -c_value)
        if best is None or key > best[0]:
            best = (key, c_value)
    assert best is not None
    return float(best[1]), trials


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * expanded, dim=1)
    denominator = torch.clamp(expanded.sum(dim=1), min=1e-9)
    return functional.normalize(summed / denominator, p=2, dim=1)


def encode_texts(texts: list[str], config: dict, cache_path: Path, sample_ids: np.ndarray) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if np.array_equal(cached["sample_ids"].astype(str), sample_ids.astype(str)):
            return cached["embeddings"]

    model_name = config["embedding"]["model"]
    snapshot = Path(snapshot_download(model_name, local_files_only=True))
    vocab_candidates = sorted(snapshot.parent.glob("*/vocab.txt"))
    if not vocab_candidates:
        raise RuntimeError(f"No cached vocabulary found under {snapshot.parent}")
    tokenizer = BertTokenizer(vocab=str(vocab_candidates[0]), do_lower_case=True)
    if tokenizer.vocab_size < 30_000:
        raise RuntimeError(f"Tokenizer vocabulary is incomplete: {tokenizer.vocab_size}")
    model = AutoModel.from_pretrained(snapshot, local_files_only=True)
    model.eval()
    torch.set_num_threads(config["embedding"]["torch_threads"])
    batches = []
    batch_size = config["embedding"]["batch_size"]
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config["embedding"]["max_length"],
                return_tensors="pt",
            )
            output = model(**tokens)
            batches.append(mean_pool(output.last_hidden_state, tokens["attention_mask"]).cpu().numpy())
            if start == 0 or (start // batch_size + 1) % 25 == 0:
                print(f"  encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    embeddings = np.vstack(batches).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, sample_ids=sample_ids.astype(str), embeddings=embeddings)
    print(f"  embedding time: {time.perf_counter() - started:.1f}s")
    return embeddings


def bootstrap_intervals(
    frame: pd.DataFrame, prediction: np.ndarray, iterations: int, seed: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    reports = frame["report_id"].unique()
    positions = {report: np.flatnonzero(frame["report_id"].to_numpy() == report) for report in reports}
    values = {"accuracy": [], "macro_f1": []}
    y_true = frame["target_level1"].to_numpy()
    for _ in range(iterations):
        sampled_reports = rng.choice(reports, size=len(reports), replace=True)
        sampled = np.concatenate([positions[report] for report in sampled_reports])
        values["accuracy"].append(accuracy_score(y_true[sampled], prediction[sampled]))
        values["macro_f1"].append(
            f1_score(y_true[sampled], prediction[sampled], average="macro", zero_division=0)
        )
    return {
        metric: {
            "lower_95": float(np.quantile(scores, 0.025)),
            "median": float(np.quantile(scores, 0.5)),
            "upper_95": float(np.quantile(scores, 0.975)),
        }
        for metric, scores in values.items()
    }


def paired_accuracy_bootstrap(
    frame: pd.DataFrame,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Report B minus A accuracy difference under report-level cluster resampling."""
    rng = np.random.default_rng(seed)
    reports = frame["report_id"].unique()
    positions = {report: np.flatnonzero(frame["report_id"].to_numpy() == report) for report in reports}
    truth = frame["target_level1"].to_numpy()
    differences = []
    for _ in range(iterations):
        sampled_reports = rng.choice(reports, size=len(reports), replace=True)
        sampled = np.concatenate([positions[report] for report in sampled_reports])
        accuracy_a = accuracy_score(truth[sampled], prediction_a[sampled])
        accuracy_b = accuracy_score(truth[sampled], prediction_b[sampled])
        differences.append(accuracy_b - accuracy_a)
    return {
        "difference": float(accuracy_score(truth, prediction_b) - accuracy_score(truth, prediction_a)),
        "lower_95": float(np.quantile(differences, 0.025)),
        "median": float(np.quantile(differences, 0.5)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "probability_b_better": float(np.mean(np.asarray(differences) > 0)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include="number").columns:
        shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    shown = shown.astype(str)
    return "\n".join(
        [
            "| " + " | ".join(shown.columns) + " |",
            "| " + " | ".join(["---"] * len(shown.columns)) + " |",
            *["| " + " | ".join(row) + " |" for row in shown.to_numpy().tolist()],
        ]
    )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["dataset"])
    target = config["target"]
    train = data["split"].eq("train")
    validation = data["split"].eq("validation") & data["clean_eval"]
    final_train = data["split"].isin(["train", "validation"])
    test = data["split"].eq("test") & data["clean_eval"]
    test_frame = data.loc[test].copy().reset_index(drop=True)
    y_train = data.loc[train, target].to_numpy()
    y_validation = data.loc[validation, target].to_numpy()
    y_final = data.loc[final_train, target].to_numpy()
    y_test = test_frame[target].to_numpy()

    for path in (METRICS_PATH, PREDICTIONS_PATH, CLASS_PATH, REPORT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    predictions = test_frame[["sample_id", "report_id", target, "context_quality"]].copy()
    class_rows = []
    trial_log = {}
    for variant, text_column in config["variants"].items():
        print(f"TF-IDF: {variant}")
        tfidf_config = config["tfidf"]
        vectorizer = TfidfVectorizer(
            ngram_range=tuple(tfidf_config["ngram_range"]),
            min_df=tfidf_config["min_df"],
            max_df=tfidf_config["max_df"],
            max_features=tfidf_config["max_features"],
            sublinear_tf=tfidf_config["sublinear_tf"],
        )
        x_train = vectorizer.fit_transform(data.loc[train, text_column].fillna(""))
        x_validation = vectorizer.transform(data.loc[validation, text_column].fillna(""))
        best_c, trials = tune_classifier(x_train, y_train, x_validation, y_validation, config)
        trial_log[f"tfidf_{variant}"] = trials
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier(config, best_c))])
        pipeline.fit(data.loc[final_train, text_column].fillna(""), y_final)
        prediction = pipeline.predict(test_frame[text_column].fillna(""))
        joblib.dump(pipeline, MODEL_DIR / f"tfidf_lr_{variant}_taxonomy_v2.joblib")
        scores = score_metrics(y_test, prediction)
        results.append(
            {
                "representation": "TF-IDF",
                "variant": variant,
                "best_C": best_c,
                **scores,
                "bootstrap": bootstrap_intervals(
                    test_frame, prediction, config["bootstrap_iterations"], config["bootstrap_seed"]
                ),
            }
        )
        predictions[f"tfidf_{variant}"] = prediction

    sample_ids = data["sample_id"].astype(str).to_numpy()
    for variant, text_column in config["variants"].items():
        print(f"MiniLM: {variant}")
        cache_tag = config["embedding"]["cache_tag"]
        embeddings = encode_texts(
            data[text_column].fillna("").astype(str).tolist(),
            config,
            EMBEDDING_DIR / f"minilm_{variant}_{cache_tag}_taxonomy_v2.npz",
            sample_ids,
        )
        best_c, trials = tune_classifier(
            embeddings[train.to_numpy()],
            y_train,
            embeddings[validation.to_numpy()],
            y_validation,
            config,
        )
        trial_log[f"minilm_{variant}"] = trials
        model = classifier(config, best_c)
        model.fit(embeddings[final_train.to_numpy()], y_final)
        prediction = model.predict(embeddings[test.to_numpy()])
        joblib.dump(
            {"classifier": model, "embedding_model": config["embedding"], "text_column": text_column},
            MODEL_DIR / f"minilm_lr_{variant}_taxonomy_v2.joblib",
        )
        scores = score_metrics(y_test, prediction)
        results.append(
            {
                "representation": "MiniLM",
                "variant": variant,
                "best_C": best_c,
                **scores,
                "bootstrap": bootstrap_intervals(
                    test_frame, prediction, config["bootstrap_iterations"], config["bootstrap_seed"]
                ),
            }
        )
        predictions[f"minilm_{variant}"] = prediction

    for result in results:
        column = f"{result['representation'].lower().replace('-', '')}_{result['variant']}"
        prediction = predictions[column]
        for label in sorted(data[target].unique()):
            mask = y_test == label
            class_rows.append(
                {
                    "representation": result["representation"],
                    "variant": result["variant"],
                    "target_level1": label,
                    "n": int(mask.sum()),
                    "recall": float(np.mean(prediction[mask] == y_test[mask])) if mask.any() else None,
                }
            )

    result_frame = pd.DataFrame([{key: value for key, value in row.items() if key != "bootstrap"} for row in results])
    subgroup_rows = []
    subgroup_masks = {
        "all": np.ones(len(test_frame), dtype=bool),
        "context_added": test_frame["context_quality"].eq("same_report_issuer_audit_area_preamble").to_numpy(),
        "context_unresolved": test_frame["context_quality"].eq("reference_without_available_preamble").to_numpy(),
        "original_sufficient": test_frame["context_quality"].eq("original_sufficient").to_numpy(),
    }
    prediction_columns = [
        column
        for column in predictions.columns
        if column.startswith("tfidf_") or column.startswith("minilm_")
    ]
    for subgroup, mask in subgroup_masks.items():
        for column in prediction_columns:
            subgroup_rows.append(
                {
                    "subgroup": subgroup,
                    "n": int(mask.sum()),
                    "model": column,
                    "accuracy": float(np.mean(predictions.loc[mask, column].to_numpy() == y_test[mask])),
                }
            )
    subgroup_frame = pd.DataFrame(subgroup_rows)
    comparison_specs = [
        ("TF-IDF: context minus original", "tfidf_original", "tfidf_context"),
        ("MiniLM: context minus original", "minilm_original", "minilm_context"),
        ("Context: MiniLM minus TF-IDF", "tfidf_context", "minilm_context"),
        ("Masked context: MiniLM minus TF-IDF", "tfidf_context_masked", "minilm_context_masked"),
    ]
    paired_comparisons = {
        name: paired_accuracy_bootstrap(
            test_frame,
            predictions[column_a].to_numpy(),
            predictions[column_b].to_numpy(),
            config["bootstrap_iterations"],
            config["bootstrap_seed"],
        )
        for name, column_a, column_b in comparison_specs
    }
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")
    pd.DataFrame(class_rows).to_csv(CLASS_PATH, index=False, encoding="utf-8")
    payload = {
        "version": config["version"],
        "dataset": config["dataset"],
        "target": target,
        "selection_metric": config["selection_metric"],
        "models": results,
        "validation_trials": trial_log,
        "paired_accuracy_comparisons": paired_comparisons,
        "context_subgroups": subgroup_rows,
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    best = max(results, key=lambda row: (row["macro_f1"], row["accuracy"]))
    paired_frame = pd.DataFrame(
        [{"comparison": name, **values} for name, values in paired_comparisons.items()]
    )
    subgroup_wide = subgroup_frame.pivot(index=["subgroup", "n"], columns="model", values="accuracy").reset_index()
    report = f"""# Taxonomy v2 小模型比較

## 結果

{markdown_table(result_frame)}

最佳配置為 **{best['representation']} / {best['variant']}**：test accuracy={best['accuracy']:.4f}，macro-F1={best['macro_f1']:.4f}，balanced accuracy={best['balanced_accuracy']:.4f}。

## 配對差異（B − A）

{markdown_table(paired_frame)}

## 上下文完整性子集

{markdown_table(subgroup_wide)}

## 實驗控制

- 所有模型使用相同的凍結 report-level split 與 12 類 taxonomy v2。
- `C` 只在 clean validation set 上選擇；選定後使用 train + validation 重訓。
- calibration split 完全保留，沒有參與選模或最終擬合。
- test headline metrics 只使用 `clean_eval = true`。
- MiniLM encoder 完全凍結；本階段測量 representation 的獨立價值，不進行端到端微調。
- 95% 區間使用 report-level cluster bootstrap，避免把同一報告的 finding 當成完全獨立樣本。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(result_frame.sort_values("macro_f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
