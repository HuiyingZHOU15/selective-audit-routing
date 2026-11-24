"""Checkpoint-driven Phase 6 evaluator with a validation-only acceptance mode."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from adaptation_phase1_common import classification_metrics, load_development_table, sha256_file, write_hash_manifest
from train_adaptation_phase1 import attach_masked_text, load_checkpoint, loader, predict


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_freeze(freeze: dict) -> None:
    if freeze.get("status") != "phase6a_rules_frozen_test_gate_closed":
        raise RuntimeError("Phase 6 rules are not frozen with the test gate closed")
    if freeze.get("expected_official_runs") != 24 or len(freeze.get("runs", [])) != 24:
        raise RuntimeError("Phase 6 freeze does not contain all 24 official runs")
    if freeze.get("test_gate", {}).get("closed") is not True:
        raise RuntimeError("Phase 6 freeze does not declare a closed test gate")


def validate_test_gate(args, freeze: dict) -> None:
    # Must complete before any test-bearing file is opened.
    if not args.confirm_retrospective_test:
        raise RuntimeError("Retrospective test access requires --confirm-retrospective-test")
    if args.phase6a_acceptance is None:
        raise RuntimeError("Retrospective test access requires --phase6a-acceptance")
    acceptance = read_json(args.phase6a_acceptance)
    required = freeze["test_gate"]["required_phase6a_acceptance_status"]
    if acceptance.get("status") != required:
        raise RuntimeError("Phase 6A acceptance status does not open the test gate")
    if acceptance.get("evaluator_sha256") != script_sha256():
        raise RuntimeError("Evaluator code differs from the accepted Phase 6A evaluator")
    if acceptance.get("phase6_freeze_sha256") != sha256_file(args.phase6_freeze):
        raise RuntimeError("Phase 6 freeze differs from the accepted configuration")


def label_order_from_freeze(freeze: dict) -> list[str]:
    tiers = freeze["frequency_tiers_from_train_only"]
    labels = sorted(tiers["high"] + tiers["medium"] + tiers["low"])
    if len(labels) != 12 or len(set(labels)) != 12:
        raise RuntimeError("Frozen frequency tiers do not define exactly 12 labels")
    return labels


def load_partition(args, freeze: dict) -> pd.DataFrame:
    if args.partition == "validation":
        frame = load_development_table(
            args.dataset, expected_sha256=freeze["development_dry_run"]["development_sha256"]
        )
        selected = frame.loc[
            frame["split"].eq("validation") & frame["clean_eval"].fillna(False).astype(bool)
        ].copy().reset_index(drop=True)
        expected_rows = freeze["development_dry_run"]["clean_validation_rows"]
    else:
        validate_test_gate(args, freeze)
        if sha256_file(args.dataset) != freeze["test_gate"]["dataset_sha256"]:
            raise ValueError("Retrospective dataset hash mismatch")
        frame = pd.read_csv(args.dataset, low_memory=False)
        selected = frame.loc[
            frame["split"].eq("test") & frame["clean_eval"].fillna(False).astype(bool)
        ].copy().reset_index(drop=True)
        expected_rows = freeze["test_gate"]["clean_test_rows"]
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} clean {args.partition} rows, found {len(selected)}")
    if selected["sample_id"].duplicated().any():
        raise ValueError("Evaluation partition contains duplicate sample IDs")
    return selected


def verify_checkpoint(run_dir: Path, spec: dict) -> None:
    for item in spec["checkpoint_files"]:
        path = run_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Checkpoint integrity failure: {path}")


def tier_metrics(metrics: dict, labels: list[str], freeze: dict) -> dict:
    per_class = {labels[row["label_id"]]: row for row in metrics["per_class"]}
    output = {}
    for tier in ("high", "medium", "low"):
        members = freeze["frequency_tiers_from_train_only"][tier]
        output[tier] = {
            "labels": members,
            "macro_f1": float(np.mean([per_class[label]["f1"] for label in members])),
            "macro_recall": float(np.mean([per_class[label]["recall"] for label in members])),
        }
    return output


def evaluate_run(args, freeze: dict, spec: dict, base_frame: pd.DataFrame, labels: list[str]) -> dict:
    run_dir = args.official_root / spec["run_relative_path"]
    verify_checkpoint(run_dir, spec)
    config = read_json(run_dir / "resolved_config.json")
    if config["run_id"] != spec["run_id"] or config["method"] != spec["method"]:
        raise RuntimeError(f"Resolved configuration identity mismatch: {spec['run_id']}")
    config = dict(config)
    config["local_files_only"] = True
    if args.cache_dir is not None:
        config["cache_dir"] = str(args.cache_dir)

    frame = base_frame.copy()
    if spec["text_condition"] == "masked":
        expected_mask = (
            freeze["development_dry_run"]["masked_development_sha256"]
            if args.partition == "validation"
            else args.masked_table_sha256
        )
        frame = attach_masked_text(frame, args.masked_table, expected_mask)
        text_column = "text_audit_area_masked_phase1"
    else:
        text_column = "text_original"
    frame[text_column] = frame[text_column].fillna("").astype(str)
    label_to_id = {label: index for index, label in enumerate(labels)}
    frame["label_id"] = frame[config.get("target", "target_level1")].map(label_to_id)
    if frame["label_id"].isna().any():
        raise ValueError("Evaluation partition contains labels outside the frozen order")
    frame["label_id"] = frame["label_id"].astype(int)

    device = torch.device(args.device)
    checkpoint = run_dir / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    load_started = time.perf_counter()
    model = load_checkpoint(config, checkpoint, device, len(labels))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    data_loader = loader(
        frame, tokenizer, text_column, int(config.get("max_length", 256)), int(args.batch_size),
        False, int(config["seed"]),
    )

    for _ in range(args.warmup_passes):
        predict(model, data_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings, logits, y_true = [], None, None
    for _ in range(args.timed_passes):
        started = time.perf_counter()
        current_logits, current_y = predict(model, data_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        if logits is None:
            logits, y_true = current_logits, current_y
        elif not np.array_equal(current_logits, logits):
            raise RuntimeError(f"Repeated inference is not deterministic: {spec['run_id']}")
    assert logits is not None and y_true is not None
    y_pred = logits.argmax(axis=1)
    metrics = classification_metrics(y_true, y_pred, list(range(len(labels))))
    metrics["frequency_tiers"] = tier_metrics(metrics, labels, freeze)

    validation_replay = None
    if args.partition == "validation":
        frozen = pd.read_csv(run_dir / "validation_predictions.csv").sort_values("sample_id").reset_index(drop=True)
        replay = frame.assign(prediction_id=y_pred).sort_values("sample_id").reset_index(drop=True)
        if frozen["sample_id"].astype(str).tolist() != replay["sample_id"].astype(str).tolist():
            raise RuntimeError(f"Frozen validation membership mismatch: {spec['run_id']}")
        prediction_match = bool(
            np.array_equal(frozen["prediction_id"].to_numpy(int), replay["prediction_id"].to_numpy(int))
        )
        validation_replay = {"prediction_ids_exact_match": prediction_match}
        if not prediction_match:
            raise RuntimeError(f"Reloaded validation predictions differ: {spec['run_id']}")

    output = args.output_root / spec["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = frame[["sample_id", "report_id"]].copy()
    prediction_frame["label_id"] = y_true
    prediction_frame["prediction_id"] = y_pred
    prediction_frame["logits_json"] = [
        json.dumps(list(map(float, row)), separators=(",", ":")) for row in logits
    ]
    prediction_frame.to_csv(output / f"{args.partition}_predictions.csv", index=False, encoding="utf-8")
    payload = {
        "run_id": spec["run_id"], "condition_id": spec["condition_id"], "seed": spec["seed"],
        "method": spec["method"], "text_condition": spec["text_condition"], "loss": spec["loss"],
        "evaluation_mode": "validation_dry_run" if args.partition == "validation" else "retrospective_test",
        "partition": args.partition, "rows": int(len(frame)),
        "test_previously_inspected": args.partition == "test", "calibration_loaded": False,
        "model_parameters_frozen_for_inference": True, "metrics": metrics,
        "validation_replay": validation_replay, "load_seconds": load_seconds,
        "warmup_passes": args.warmup_passes, "timed_passes": args.timed_passes,
        "timed_seconds": timings,
        "median_examples_per_second": len(frame) / statistics.median(timings),
        "evaluator_sha256": script_sha256(), "phase6_freeze_sha256": sha256_file(args.phase6_freeze),
    }
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_hash_manifest(output)
    del model, tokenizer, data_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase6-freeze", type=Path, required=True)
    parser.add_argument("--phase6a-acceptance", type=Path)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--masked-table", type=Path, required=True)
    parser.add_argument("--masked-table-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partition", choices=["validation", "test"], default="validation")
    parser.add_argument("--confirm-retrospective-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--timed-passes", type=int, default=1)
    parser.add_argument("--run-id", action="append")
    args = parser.parse_args()
    if args.partition == "test" and not args.masked_table_sha256:
        raise RuntimeError("Test evaluation requires --masked-table-sha256")
    if args.warmup_passes < 1 or args.timed_passes < 1:
        raise ValueError("At least one warm-up and one timed pass are required")

    freeze = read_json(args.phase6_freeze)
    validate_freeze(freeze)
    frame = load_partition(args, freeze)  # test gate is enforced inside before file access
    labels = label_order_from_freeze(freeze)
    selected = freeze["runs"]
    if args.run_id:
        wanted = set(args.run_id)
        selected = [spec for spec in selected if spec["run_id"] in wanted]
        if {spec["run_id"] for spec in selected} != wanted:
            raise ValueError("One or more requested run IDs are not frozen")
    results = [evaluate_run(args, freeze, spec, frame, labels) for spec in selected]
    summary = {
        "status": "evaluation_complete", "partition": args.partition,
        "runs_completed": len(results), "runs_requested": len(selected),
        "test_scored": args.partition == "test", "calibration_loaded": False,
        "evaluator_sha256": script_sha256(), "phase6_freeze_sha256": sha256_file(args.phase6_freeze),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "execution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

