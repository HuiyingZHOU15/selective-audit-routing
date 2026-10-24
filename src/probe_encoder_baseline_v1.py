"""Frozen-encoder linear probes for pre-specified PCAOB baselines.

This is deliberately labeled a linear probe, not end-to-end fine-tuning.  It
tests whether a modern general or SEC-domain representation merits a later GPU
fine-tuning experiment under the same frozen report-grouped split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from train_encoder_baseline_v1 import report_bootstrap


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/encoder_baseline_v1.json"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1.0)
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)


def encode(texts: list[str], tokenizer, model, max_length: int, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**tokens).last_hidden_state
            outputs.append(mean_pool(hidden, tokens["attention_mask"]).cpu().numpy())
            batch_number = start // batch_size + 1
            if batch_number == 1 or batch_number % 20 == 0:
                print(
                    f"encoded={min(start + batch_size, len(texts))}/{len(texts)} "
                    f"elapsed_s={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    return np.vstack(outputs).astype(np.float32)


def scores(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=["general_xsmall", "sec_domain"])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["models"][args.model_key]
    frame = pd.read_csv(ROOT / config["dataset"])
    text_column = config["text_column"]
    texts = frame[text_column].fillna("").astype(str).tolist()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = ROOT / f"artifacts/embeddings/{args.model_key}_context_linear_probe_v1.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if embedding_path.exists():
        cached = np.load(embedding_path, allow_pickle=False)
        if np.array_equal(cached["sample_ids"].astype(str), frame["sample_id"].astype(str).to_numpy()):
            embeddings = cached["embeddings"]
        else:
            raise ValueError("Embedding cache sample IDs do not match the frozen dataset")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_config["model_name"], cache_dir=cache_dir)
        # CPU float16 kernels are prohibitively slow on this machine.  The
        # conversion changes only numerical precision, not pretrained weights.
        model = model.float().to("cpu")
        torch.set_num_threads(min(10, torch.get_num_threads()))
        embeddings = encode(texts, tokenizer, model, int(config["max_length"]), args.batch_size)
        np.savez_compressed(
            embedding_path,
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            embeddings=embeddings,
        )

    train = frame["split"].eq("train").to_numpy()
    validation = (frame["split"].eq("validation") & frame["clean_eval"].fillna(False).astype(bool)).to_numpy()
    final_train = frame["split"].isin(["train", "validation"]).to_numpy()
    test = (frame["split"].eq("test") & frame["clean_eval"].fillna(False).astype(bool)).to_numpy()
    target = frame[config["target"]].astype(str).to_numpy()

    trials = []
    best = None
    for c_value in [0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 64.0]:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            random_state=int(config["seed"]),
        )
        classifier.fit(embeddings[train], target[train])
        prediction = classifier.predict(embeddings[validation])
        result = {"C": c_value, **scores(target[validation], prediction)}
        trials.append(result)
        key = (result["macro_f1"], result["accuracy"], -c_value)
        if best is None or key > best[0]:
            best = (key, c_value)
    assert best is not None

    classifier = LogisticRegression(
        C=float(best[1]),
        class_weight="balanced",
        max_iter=3000,
        random_state=int(config["seed"]),
    )
    classifier.fit(embeddings[final_train], target[final_train])
    prediction = classifier.predict(embeddings[test])
    probabilities = classifier.predict_proba(embeddings[test])
    test_scores = scores(target[test], prediction)

    test_frame = frame.loc[test, ["sample_id", "report_id", config["target"], "context_quality"]].reset_index(drop=True)
    reference = pd.read_csv(ROOT / "results/tables/model_comparison_test_predictions_v2.csv")
    reference = reference[["sample_id", "tfidf_context"]]
    aligned = test_frame.merge(reference, on="sample_id", how="left", validate="one_to_one")
    paired = report_bootstrap(
        aligned,
        prediction,
        aligned["tfidf_context"].to_numpy(),
        iterations=1000,
        seed=int(config["seed"]),
    )

    output = test_frame.copy()
    output["prediction"] = prediction
    output["confidence"] = probabilities.max(axis=1)
    output["correct"] = output["prediction"].eq(output[config["target"]])
    stem = f"encoder_{args.model_key}_linear_probe_v1"
    table_path = ROOT / f"results/tables/{stem}_test_predictions.csv"
    metrics_path = ROOT / f"results/metrics/{stem}.json"
    report_path = ROOT / f"results/reports/{stem}_zh.md"
    for path in [table_path, metrics_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(table_path, index=False, encoding="utf-8-sig")
    payload = {
        "version": config["version"],
        "experiment_type": "frozen_encoder_linear_probe_not_fine_tuning",
        "model_key": args.model_key,
        "model": model_config,
        "embedding_shape": list(embeddings.shape),
        "best_C_on_clean_validation": float(best[1]),
        "validation_trials": trials,
        "retrospective_test": test_scores,
        "paired_vs_tfidf_context": paired,
        "elapsed_seconds": float(time.perf_counter() - started),
        "calibration_used": False,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        f"""# PCAOB frozen-encoder linear probe：{args.model_key}

這是 **frozen encoder + logistic regression linear probe**，不是端到端 fine-tuning。

- Model: `{model_config['model_name']}`
- Clean retrospective test accuracy: {test_scores['accuracy']:.4f}
- Clean retrospective test macro-F1: {test_scores['macro_f1']:.4f}
- Accuracy difference versus TF-IDF/context: {paired['accuracy_difference_encoder_minus_tfidf']:+.4f}
- Report-cluster bootstrap 95% CI: [{paired['lower_95']:+.4f}, {paired['upper_95']:+.4f}]

本結果只決定是否值得投入GPU作完整fine-tuning。既有test已被查看，不能作確認性證據；calibration未使用。
""",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
