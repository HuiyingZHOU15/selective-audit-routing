"""Generate report-grouped SEC-BERT OOF predictions and re-score the RAG pilot.

The five outer folds are frozen in pcaob_development_oof_v1.csv.  Each fold is
predicted by a model trained only on the other four folds.  Completed folds are
saved immediately, so rerunning after interruption resumes instead of starting
over.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/encoder_sec_domain_oof_v1.json"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.texts[index], "label": self.labels[index]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collator(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        tokens = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return tokens

    return collate


@torch.inference_mode()
def predict(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        logits.append(model(**batch).logits.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def score(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["target_level1"].astype(str).to_numpy()
    prediction = frame["sec_prediction"].astype(str).to_numpy()
    return {
        "samples": int(len(frame)),
        "reports": int(frame["report_id"].astype(str).nunique()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "errors": int((truth != prediction).sum()),
    }


def rag_comparison(oof: pd.DataFrame, rag_path: Path) -> tuple[pd.DataFrame, dict]:
    if not rag_path.exists():
        return pd.DataFrame(), {"status": "rag_pilot_not_in_bundle"}
    rag = pd.read_csv(rag_path)
    rag = rag.loc[
        rag["condition"].isin(["direct", "report_rag"])
        & rag["completed"].fillna(False).astype(bool)
    ].copy()
    columns = [
        "sample_id", "report_id", "target_level1", "outer_fold", "oof_clean",
        "sec_prediction", "sec_confidence", "sec_margin", "sec_entropy", "sec_correct",
    ]
    # The frozen RAG audit already carries TF-IDF-era OOF metadata such as
    # outer_fold/oof_clean.  Drop overlapping non-key columns so the SEC-BERT
    # OOF fields keep their canonical names instead of receiving _x/_y suffixes.
    merge_keys = ["sample_id", "report_id", "target_level1"]
    overlap = sorted((set(rag.columns) & set(columns)) - set(merge_keys))
    if overlap:
        rag = rag.drop(columns=overlap)
    paired = rag.merge(oof[columns], on=merge_keys, how="inner")
    paired["backend_correct"] = paired["predicted_label"].eq(paired["target_level1"])
    paired["beneficial_vs_sec"] = (~paired["sec_correct"]) & paired["backend_correct"]
    paired["harmful_vs_sec"] = paired["sec_correct"] & (~paired["backend_correct"])

    result = {"status": "complete", "conditions": {}}
    for condition, group in paired.groupby("condition", sort=True):
        condition_result = {}
        for scope, scoped in {
            "all": group,
            "clean_oof": group.loc[group["oof_clean"].fillna(False).astype(bool)],
        }.items():
            benefit = int(scoped["beneficial_vs_sec"].sum())
            harm = int(scoped["harmful_vs_sec"].sum())
            condition_result[scope] = {
                "samples": int(len(scoped)),
                "sec_errors": int((~scoped["sec_correct"]).sum()),
                "backend_accuracy": float(scoped["backend_correct"].mean()) if len(scoped) else None,
                "beneficial": benefit,
                "harmful": harm,
                "net": benefit - harm,
                "both_wrong": int(((~scoped["sec_correct"]) & (~scoped["backend_correct"])).sum()),
            }
        result["conditions"][condition] = condition_result
    return paired, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts/hf_cache"))
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Kaggle OOF run")
    device = torch.device("cuda")
    print(f"device=cuda gpu={torch.cuda.get_device_name(0)}", flush=True)

    frame = pd.read_csv(ROOT / config["dataset"])
    fold_column = config["fold_column"]
    target = config["target"]
    text_column = config["text_column"]
    frame[text_column] = frame[text_column].fillna("").astype(str)
    folds = sorted(frame[fold_column].dropna().astype(int).unique().tolist())
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected locked folds 1..5, found {folds}")
    if frame.groupby("report_id")[fold_column].nunique().max() != 1:
        raise ValueError("A report appears in more than one outer fold")
    labels = sorted(frame[target].dropna().astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    frame["label_id"] = frame[target].map(label_to_id).astype(int)

    table_dir = ROOT / "results/tables"
    metrics_dir = ROOT / "results/metrics"
    report_dir = ROOT / "results/reports"
    for directory in (table_dir, metrics_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    partial_path = table_dir / "encoder_sec_domain_oof_v1.partial.csv"
    output_path = table_dir / "encoder_sec_domain_oof_v1.csv"
    fold_metrics_path = metrics_dir / "encoder_sec_domain_oof_folds_v1.json"

    if partial_path.exists():
        partial = pd.read_csv(partial_path)
    else:
        partial = pd.DataFrame()
    completed_folds: set[int] = set()
    if not partial.empty:
        for fold in folds:
            expected = int(frame[fold_column].eq(fold).sum())
            actual = int(partial[fold_column].eq(fold).sum())
            if actual == expected:
                completed_folds.add(fold)
            elif actual:
                raise ValueError(f"Partial fold {fold} has {actual}/{expected} rows")
    print(f"resume_completed_folds={sorted(completed_folds)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=args.cache_dir)
    collator = make_collator(tokenizer, int(config["max_length"]))
    fold_metrics = []
    started_all = time.perf_counter()
    for fold in folds:
        if fold in completed_folds:
            print(f"fold={fold} already complete; skipping", flush=True)
            continue
        set_seed(int(config["seed"]) + fold)
        train = frame.loc[frame[fold_column].ne(fold)].reset_index(drop=True)
        hold = frame.loc[frame[fold_column].eq(fold)].reset_index(drop=True)
        train_loader = DataLoader(
            TextDataset(train[text_column].tolist(), train["label_id"].tolist()),
            batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collator,
            num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(int(config["seed"]) + fold),
        )
        hold_loader = DataLoader(
            TextDataset(hold[text_column].tolist(), hold["label_id"].tolist()),
            batch_size=int(config["batch_size"]) * 2, shuffle=False, collate_fn=collator,
            num_workers=0, pin_memory=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=len(labels), id2label=id_to_label,
            label2id=label_to_id, cache_dir=args.cache_dir,
        ).to(device)
        counts = train["label_id"].value_counts().reindex(range(len(labels)), fill_value=1).to_numpy(float)
        weights = torch.tensor(len(train) / (len(labels) * counts), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        accumulation = int(config["gradient_accumulation"])
        updates_per_epoch = int(np.ceil(len(train_loader) / accumulation))
        total_updates = updates_per_epoch * int(config["epochs"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, max(1, int(total_updates * float(config["warmup_ratio"]))), total_updates
        )
        started_fold = time.perf_counter()
        update = 0
        history = []
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            for step, batch in enumerate(train_loader, start=1):
                y = batch.pop("labels").to(device, non_blocking=True)
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                logits = model(**batch).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, y, weight=weights) / accumulation
                loss.backward()
                running += float(loss.item()) * accumulation
                if step % accumulation == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    update += 1
                    if update == 1 or update % 100 == 0:
                        print(
                            f"fold={fold} epoch={epoch} update={update}/{total_updates} "
                            f"loss={running / step:.4f} elapsed_s={time.perf_counter()-started_fold:.1f}",
                            flush=True,
                        )
            history.append({"epoch": epoch, "train_loss": running / len(train_loader)})

        logits, hold_y = predict(model, hold_loader, device)
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
        prediction_id = probabilities.argmax(axis=1)
        prediction = np.asarray([id_to_label[int(index)] for index in prediction_id])
        sorted_p = np.sort(probabilities, axis=1)
        result = hold[[
            "sample_id", "report_id", target, fold_column, "oof_clean", "oof_leakage_reason"
        ]].copy()
        result["sec_prediction"] = prediction
        result["sec_confidence"] = probabilities.max(axis=1)
        result["sec_margin"] = sorted_p[:, -1] - sorted_p[:, -2]
        clipped = np.clip(probabilities, 1e-15, 1.0)
        result["sec_entropy"] = -np.sum(clipped * np.log(clipped), axis=1)
        result["sec_correct"] = result["sec_prediction"].eq(result[target])
        result["sec_probability_json"] = [
            json.dumps(
                dict(zip(labels, map(float, row))),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in probabilities
        ]
        partial = pd.concat([partial, result], ignore_index=True)
        partial = partial.drop_duplicates("sample_id", keep="last").sort_values("sample_id")
        partial.to_csv(partial_path, index=False, encoding="utf-8")
        fold_row = {
            "fold": fold, "train_samples": len(train), "holdout_samples": len(hold),
            "train_reports": int(train["report_id"].nunique()),
            "holdout_reports": int(hold["report_id"].nunique()),
            "accuracy": float(accuracy_score(hold_y, prediction_id)),
            "macro_f1": float(f1_score(hold_y, prediction_id, average="macro", zero_division=0)),
            "elapsed_seconds": float(time.perf_counter() - started_fold), "history": history,
        }
        fold_metrics.append(fold_row)
        fold_metrics_path.write_text(json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(fold_row, ensure_ascii=False), flush=True)
        del model, optimizer, scheduler, train_loader, hold_loader
        gc.collect(); torch.cuda.empty_cache()

    oof = pd.read_csv(partial_path).sort_values("sample_id")
    if len(oof) != len(frame) or oof["sample_id"].nunique() != len(frame):
        raise RuntimeError(f"OOF incomplete: {len(oof)}/{len(frame)}")
    oof.to_csv(output_path, index=False, encoding="utf-8")
    clean = oof.loc[oof["oof_clean"].fillna(False).astype(bool)].copy()
    paired, rag_metrics = rag_comparison(oof, ROOT / config["rag_pilot"])
    paired_path = table_dir / "encoder_sec_domain_vs_rag_pilot_v1.csv"
    if not paired.empty:
        paired.to_csv(paired_path, index=False, encoding="utf-8")
    payload = {
        "version": config["version"], "model": config["model_name"],
        "fold_policy": config["selection_rule"], "all_oof": score(oof),
        "clean_oof": score(clean), "rag_pilot_vs_sec_oof": rag_metrics,
        "elapsed_seconds_this_invocation": float(time.perf_counter() - started_all),
    }
    metrics_path = metrics_dir / "encoder_sec_domain_oof_v1.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = report_dir / "encoder_sec_domain_oof_v1_zh.md"
    report_path.write_text(
        "# SEC-BERT report-grouped OOF v1\n\n"
        f"- All OOF: n={payload['all_oof']['samples']}, accuracy={payload['all_oof']['accuracy']:.4f}, "
        f"macro-F1={payload['all_oof']['macro_f1']:.4f}.\n"
        f"- Clean OOF: n={payload['clean_oof']['samples']}, accuracy={payload['clean_oof']['accuracy']:.4f}, "
        f"macro-F1={payload['clean_oof']['macro_f1']:.4f}.\n"
        "- 每筆預測均來自未見過該 report 的 SEC-BERT；固定3 epochs，outer holdout未用於模型選擇。\n\n"
        "## Existing RAG pilot re-analysis\n\n```json\n"
        + json.dumps(rag_metrics, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
