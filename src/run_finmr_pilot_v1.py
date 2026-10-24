"""Resumable DeepSeek runner for the FinMR paired pilot."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from llm_common import ROOT
from prepare_finmr_candidates_v1 import load_config, prepare


class FinMRDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extracted_value: str
    calculated_value: str


SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_value": {"type": "string"},
        "calculated_value": {"type": "string"},
    },
    "required": ["extracted_value", "calculated_value"],
    "additionalProperties": False,
}


def completed(path) -> set[tuple[str, str]]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    done.add((item["request_id"], item["prompt_bundle_sha256"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition", action="append")
    args = parser.parse_args()
    config = load_config()
    candidate_path = ROOT / config["candidate_table"]
    candidates = pd.read_csv(candidate_path) if candidate_path.exists() else prepare()
    if args.condition:
        candidates = candidates.loc[candidates["condition"].isin(args.condition)].copy()
    response_path = ROOT / config["response_log"]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed(response_path)
    pending = candidates.loc[~candidates.apply(
        lambda row: (row["request_id"], row["prompt_bundle_sha256"]) in done, axis=1
    )]
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(f"candidates={len(candidates)} pending_selected={len(pending)}", flush=True)
    if not args.run:
        return
    key = os.getenv(config["api_key_env"])
    if not key:
        raise SystemExit(f"Missing {config['api_key_env']}")
    system = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=key, base_url=config["base_url"], timeout=config["request_timeout_seconds"])
    successes = 0
    for _, row in pending.iterrows():
        last_error = None
        for attempt in range(1, config["max_attempts"] + 1):
            try:
                started = time.perf_counter()
                response = client.responses.create(
                    model=config["model"], instructions=system, input=str(row["user_prompt"]),
                    temperature=config["temperature"], max_output_tokens=config["max_output_tokens"],
                    reasoning={"effort": config["reasoning_effort"]}, store=False,
                    text={"format": {"type": "json_schema", "name": "finmr_result", "strict": True, "schema": SCHEMA}},
                )
                latency = round((time.perf_counter() - started) * 1000, 1)
                decision = FinMRDecision.model_validate_json(response.output_text)
                dumped = response.model_dump(mode="json")
                usage = dumped.get("usage") or {}
                record = {
                    "request_id": row["request_id"], "sample_id": row["sample_id"], "condition": row["condition"],
                    "status": "ok", "attempt": attempt, "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provider": config["provider"], "requested_model": config["model"], "response_model": dumped.get("model"),
                    "latency_ms": latency, "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0), "decision": decision.model_dump(mode="json"),
                    "raw_output": response.output_text, "prompt_bundle_sha256": row["prompt_bundle_sha256"],
                    "max_output_tokens": config["max_output_tokens"],
                }
                with response_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                successes += 1
                print(f"OK {successes}/{len(pending)} {row['request_id']}", flush=True)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < config["max_attempts"]:
                    time.sleep(2 ** attempt)
        else:
            with response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "request_id": row["request_id"], "sample_id": row["sample_id"], "condition": row["condition"],
                    "status": "error", "error": last_error, "attempt": config["max_attempts"],
                    "requested_at_utc": datetime.now(timezone.utc).isoformat(), "prompt_bundle_sha256": row["prompt_bundle_sha256"],
                }, ensure_ascii=False) + "\n")
            print(f"ERROR {row['request_id']}: {last_error}", flush=True)


if __name__ == "__main__":
    main()
