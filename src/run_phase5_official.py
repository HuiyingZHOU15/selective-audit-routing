"""Execute frozen Phase 5 configs as resumable, seed-blocked subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_phase5_official import canonical_json_sha256, load_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SEEDS = {20261114, 20261115, 20261116}
EXPECTED_LRS = {"sec_lora": 0.0002, "sec_full": 0.00002, "bert_full": 0.00003}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "official":
        raise RuntimeError(f"Non-official config supplied: {config.get('run_id')}")
    if config.get("evaluate_test") is not False or config.get("load_calibration") is not False:
        raise RuntimeError("Phase 5 config violates the test/calibration embargo")
    if int(config.get("seed", -1)) not in OFFICIAL_SEEDS:
        raise RuntimeError("Unexpected Phase 5 seed")
    if config.get("method") not in EXPECTED_LRS:
        raise RuntimeError("Unexpected Phase 5 method")
    if float(config.get("learning_rate")) != EXPECTED_LRS[config["method"]]:
        raise RuntimeError("Phase 5 learning rate differs from the frozen selection")
    if config.get("method") == "sec_lora" and config.get("adapter_candidate_id") != "LORA_QKVO_R16":
        raise RuntimeError("Phase 5 LoRA structure differs from the frozen selection")
    if config.get("method") != "sec_lora" and config.get("adapter_candidate_id") is not None:
        raise RuntimeError("Full fine-tuning config unexpectedly contains an adapter")
    if config.get("text_condition") not in {"original", "masked"} or config.get("loss") not in {"weighted", "unweighted"}:
        raise RuntimeError("Unexpected Phase 5 condition")
    if config.get("device") != "cuda":
        raise RuntimeError("Formal Phase 5 runs require CUDA")
    if "SELECTED" in json.dumps(config, ensure_ascii=False):
        raise RuntimeError("Unresolved placeholder in Phase 5 config")


def verify_hash_manifest(run_root: Path) -> bool:
    path = run_root / "sha256_manifest.json"
    if not path.is_file():
        return False
    try:
        rows = load_json(path)
        if not isinstance(rows, list) or not rows:
            return False
        for row in rows:
            target = run_root / row["path"]
            if not target.is_file() or target.stat().st_size != int(row["bytes"]):
                return False
            if sha256_file(target) != row["sha256"]:
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def is_valid_completion(config: dict[str, Any], run_root: Path) -> bool:
    metrics_path = run_root / "metrics.json"
    resolved_path = run_root / "resolved_config.json"
    predictions_path = run_root / "validation_predictions.csv"
    checkpoint = run_root / "checkpoint"
    if not all((metrics_path.is_file(), resolved_path.is_file(), predictions_path.is_file(), checkpoint.is_dir())):
        return False
    try:
        metrics = load_json(metrics_path)
        resolved = load_json(resolved_path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        resolved == config
        and metrics.get("run_id") == config.get("run_id")
        and metrics.get("stage") == "official"
        and metrics.get("test_scored") is False
        and metrics.get("calibration_loaded") is False
        and metrics.get("config_sha256") == canonical_json_sha256(config)
        and isinstance(metrics.get("validation_metrics", {}).get("macro_f1"), (int, float))
        and verify_hash_manifest(run_root)
    )


def append_attempt(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def execute_manifest(
    manifest_path: Path,
    runs_root: Path,
    project_root: Path = ROOT,
    python_executable: str = sys.executable,
    seed: int | None = None,
    continue_on_failure: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "official_configs_frozen" or manifest.get("expected_runs") != 24:
        raise RuntimeError("Invalid or incomplete Phase 5 config manifest")
    if manifest.get("test_scored") is not False or manifest.get("calibration_loaded") is not False:
        raise RuntimeError("Phase 5 manifest violates the embargo")
    if seed is not None and seed not in OFFICIAL_SEEDS:
        raise ValueError(f"seed must be one of {sorted(OFFICIAL_SEEDS)}")
    records = sorted(manifest["runs"], key=lambda item: int(item["order"]))
    if seed is not None:
        records = [record for record in records if int(record["seed"]) == seed]
    expected = 8 if seed is not None else 24
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} selected configs, found {len(records)}")

    config_dir = manifest_path.parent
    runs_root.mkdir(parents=True, exist_ok=True)
    attempts_path = runs_root / "scheduler_attempts.jsonl"
    statuses = []
    for record in records:
        config_path = config_dir / record["config_path"]
        if sha256_file(config_path) != record["config_file_sha256"]:
            raise RuntimeError(f"Config file hash mismatch: {config_path.name}")
        config = load_json(config_path)
        validate_config(config)
        if canonical_json_sha256(config) != record["config_canonical_sha256"]:
            raise RuntimeError(f"Canonical config hash mismatch: {config_path.name}")
        run_root = runs_root / str(config["run_id"])
        if is_valid_completion(config, run_root):
            statuses.append({"order": record["order"], "run_id": config["run_id"], "status": "skipped_complete"})
            continue
        run_root.mkdir(parents=True, exist_ok=True)
        attempt = {
            "order": record["order"],
            "run_id": config["run_id"],
            "seed": config["seed"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_canonical_sha256": canonical_json_sha256(config),
        }
        command = [
            python_executable,
            str(project_root / "src/train_adaptation_phase1.py"),
            "--config",
            str(config_path),
            "--output-root",
            str(run_root),
        ]
        process_environment = os.environ.copy()
        process_environment["PYTHONHASHSEED"] = str(config["seed"])
        completed = subprocess.run(command, cwd=project_root, env=process_environment, check=False)
        attempt["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        attempt["return_code"] = int(completed.returncode)
        attempt["status"] = "complete" if completed.returncode == 0 and is_valid_completion(config, run_root) else "failed"
        append_attempt(attempts_path, attempt)
        statuses.append({"order": record["order"], "run_id": config["run_id"], "status": attempt["status"]})
        if attempt["status"] == "failed" and not continue_on_failure:
            break
    complete = sum(row["status"] in {"complete", "skipped_complete"} for row in statuses)
    summary = {
        "version": "phase5-official-execution-summary-v1.1",
        "stage": "official",
        "seed_filter": seed,
        "expected_runs": expected,
        "attempted_or_seen_runs": len(statuses),
        "complete_runs": complete,
        "failed_runs": sum(row["status"] == "failed" for row in statuses),
        "test_scored": False,
        "calibration_loaded": False,
        "runs": statuses,
    }
    (runs_root / "execution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    summary = execute_manifest(
        args.manifest,
        args.runs_root,
        args.project_root.resolve(),
        args.python_executable,
        seed=args.seed,
        continue_on_failure=not args.stop_on_failure,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

