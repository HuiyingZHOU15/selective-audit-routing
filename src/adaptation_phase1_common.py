"""Shared, test-embargoed utilities for the Phase 1 v1.1 experiment.

Training and tuning code must consume a development-only table containing only
``train`` and ``validation`` rows.  The retrospective test is opened solely by
the separately gated Phase 6 evaluator.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
PROHIBITED_TUNING_SPLITS = frozenset({"calibration", "test"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_hash_manifest(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "sha256_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (root / "sha256_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def set_reproducibility(seed: int, deterministic: bool = True) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "pythonhashseed_at_process_start": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if hasattr(torch.backends, "cudnn") else None,
    }


def environment_payload() -> dict:
    try:
        import peft
        peft_version = peft.__version__
    except ImportError:
        peft_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def load_development_table(
    path: Path,
    expected_sha256: str | None = None,
    expected_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Load a physically development-only table and reject any other split."""
    path = path.resolve()
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"Development table hash mismatch: {path}")
    frame = pd.read_csv(path, low_memory=False)
    if "split" not in frame:
        raise ValueError("Development table has no split column")
    observed = set(frame["split"].dropna().astype(str).unique())
    forbidden = observed & PROHIBITED_TUNING_SPLITS
    if forbidden:
        raise RuntimeError(f"Test embargo violation: development input contains {sorted(forbidden)}")
    if not observed or not observed <= ALLOWED_DEVELOPMENT_SPLITS:
        raise RuntimeError(f"Unexpected development splits: {sorted(observed)}")
    if expected_counts:
        counts = frame["split"].value_counts().to_dict()
        if counts != expected_counts:
            raise ValueError(f"Unexpected development split sizes: {counts}")
    if frame.groupby("report_id")["split"].nunique().max() != 1:
        raise ValueError("A report crosses development partitions")
    return frame


def split_development(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["split"].eq("train")].copy().reset_index(drop=True)
    validation = frame.loc[
        frame["split"].eq("validation") & frame["clean_eval"].fillna(False).astype(bool)
    ].copy().reset_index(drop=True)
    if not len(train) or not len(validation):
        raise ValueError("Train or clean validation partition is empty")
    return train, validation


def label_order(frame: pd.DataFrame, target: str) -> list[str]:
    labels = sorted(frame[target].dropna().astype(str).unique())
    if len(labels) != 12:
        raise ValueError(f"Expected 12 labels, found {len(labels)}")
    return labels


def balanced_class_weights(label_ids: Iterable[int], num_labels: int) -> np.ndarray:
    values = np.asarray(list(label_ids), dtype=int)
    counts = np.bincount(values, minlength=num_labels).astype(float)
    if np.any(counts == 0):
        raise ValueError("Every class must occur in the training partition")
    weights = len(values) / (num_labels * counts)
    # The sample-frequency-weighted mean is exactly one.
    if not np.isclose(np.sum(weights * counts) / len(values), 1.0):
        raise AssertionError("Balanced class weights are not sample-mean normalized")
    return weights


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class": [
            {
                "label_id": int(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def report_cluster_bootstrap(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    label_ids: list[int],
    iterations: int,
    seed: int,
) -> dict:
    truth = frame["label_id"].to_numpy(int)
    reports = frame["report_id"].astype(str).unique()
    report_array = frame["report_id"].astype(str).to_numpy()
    positions = {report: np.flatnonzero(report_array == report) for report in reports}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled_reports = rng.choice(reports, size=len(reports), replace=True)
        sampled = np.concatenate([positions[report] for report in sampled_reports])
        values.append(
            f1_score(truth[sampled], y_pred[sampled], labels=label_ids, average="macro", zero_division=0)
        )
    return {
        "metric": "macro_f1",
        "iterations": iterations,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best_score: float = -np.inf
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, epoch: int, score: float) -> tuple[bool, bool]:
        """Return ``(is_new_best, should_stop)`` using the frozen rule."""
        improvement = score - self.best_score
        is_new_best = self.best_epoch == 0 or score > self.best_score
        meaningful_improvement = self.best_epoch == 0 or improvement >= self.min_delta
        if is_new_best:
            self.best_score = score
            self.best_epoch = epoch
        if meaningful_improvement:
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return is_new_best, self.stale_epochs >= self.patience


def select_adapter_candidate(trials: list[dict], tolerance: float = 0.005) -> dict:
    """Apply the frozen validation-only adapter selection rule."""
    order = {name: index for index, name in enumerate(
        ["LORA_QV_R8", "LORA_QV_R16", "LORA_QKV_R16", "LORA_QKVO_R16"]
    )}
    eligible = [row for row in trials if row.get("status", "complete") == "complete"]
    if not eligible:
        raise ValueError("No eligible adapter trial")
    maximum = max(float(row["best_validation_macro_f1"]) for row in eligible)
    efficient = [
        row for row in eligible
        if maximum - float(row["best_validation_macro_f1"]) <= tolerance + 1e-12
    ]
    return min(
        efficient,
        key=lambda row: (
            int(row["trainable_parameters"]),
            int(row["peak_cuda_memory_bytes"]),
            float(row["training_time_seconds"]),
            order[row["candidate_id"]],
        ),
    )


def checkpoint_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


