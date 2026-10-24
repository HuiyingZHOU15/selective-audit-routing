"""Run resumable Jev Choice classification on frozen taxonomy-v2 samples."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from llm_common import ROOT

CONFIG_PATH = ROOT / "configs" / "jev_core_v2.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def completed(path) -> set[tuple[str, str]]:
    result = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    result.add((str(item["sample_id"]), str(item["prompt_bundle_sha256"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return result


def invoke(config: dict, key: str, text: str, definitions: dict[str, str]) -> tuple[dict, float]:
    payload = {
        "model": config["model"],
        "state": text,
        "questions": {
            "audit_area": {
                "type": "choice",
                "instructions": (
                    "Choose the principal PCAOB inspection finding audit-area category. "
                    "Prefer a specific account or topic category over the cross-cutting reporting category when clearly supported."
                ),
                "criteria": definitions,
            }
        },
    }
    request = urllib.request.Request(
        config["endpoint"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid.uuid4()),
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=config["request_timeout_seconds"]) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body, round((time.perf_counter() - started) * 1000, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified-per-class", type=int)
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    candidate_key = "calibration_candidate_table" if args.calibration else "candidate_table"
    response_key = "calibration_response_log" if args.calibration else "response_log"
    candidates = pd.read_csv(ROOT / config[candidate_key])
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    output = ROOT / config[response_key]
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed(output)
    pending = candidates.loc[~candidates.apply(
        lambda row: (str(row["sample_id"]), str(row["prompt_bundle_sha256"])) in done, axis=1
    )]
    if args.stratified_per_class is not None:
        if args.calibration:
            raise SystemExit("--stratified-per-class is only supported for the test smoke run")
        counts = pending.groupby("target_level1").size()
        if (counts < args.stratified_per_class).any():
            raise SystemExit(f"Insufficient pending samples for: {counts[counts < args.stratified_per_class].to_dict()}")
        pending = (
            pending.groupby("target_level1", group_keys=False)
            .sample(n=args.stratified_per_class, random_state=20260927)
            .sort_values(["target_level1", "sample_id"])
        )
        selection_path = ROOT / "data" / "interim" / "jev_stratified_smoke_v2.csv"
        pending[["sample_id", "prompt_bundle_sha256", "target_level1"]].to_csv(
            selection_path, index=False, encoding="utf-8"
        )
        print(f"stratified_selection={selection_path}", flush=True)
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(f"candidates={len(candidates)} already_complete={len(done)} selected_now={len(pending)}", flush=True)
    if not args.run:
        return
    key = os.getenv(config["api_key_env"])
    if not key:
        raise SystemExit(f"Missing {config['api_key_env']}")
    for number, (_, row) in enumerate(pending.iterrows(), start=1):
        error = None
        for attempt in range(1, config["max_attempts"] + 1):
            try:
                response, latency = invoke(config, key, str(row["text"]), definitions)
                answer = response["answers"]["audit_area"]
                record = {
                    "sample_id": str(row["sample_id"]),
                    "prompt_bundle_sha256": str(row["prompt_bundle_sha256"]),
                    "status": "ok",
                    "attempt": attempt,
                    "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_model": config["model"],
                    "response_model": response.get("model"),
                    "prediction": answer.get("choice"),
                    "confidence": answer.get("confidence"),
                    "probabilities": answer.get("probabilities"),
                    "usage": response.get("usage", {}),
                    "latency_ms": latency,
                }
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"OK {number}/{len(pending)} sample={row['sample_id']}", flush=True)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < config["max_attempts"]:
                    time.sleep(2 ** attempt)
        else:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "sample_id": str(row["sample_id"]),
                    "prompt_bundle_sha256": str(row["prompt_bundle_sha256"]),
                    "status": "error",
                    "error": error,
                }, ensure_ascii=False) + "\n")
            print(f"ERROR sample={row['sample_id']} {error}", flush=True)


if __name__ == "__main__":
    main()
