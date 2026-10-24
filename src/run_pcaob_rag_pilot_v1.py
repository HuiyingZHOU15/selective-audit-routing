"""Run resumable DeepSeek calls for the PCAOB RAG feasibility pilot."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_common import ROOT
from prepare_pcaob_rag_pilot_v1 import load_config, prepare_candidates


class PilotDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_label: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str
    evidence_source_id: str | None
    reason: str
    standard_citation: str | None
    abstain: bool

    @model_validator(mode="after")
    def validate_decision(self) -> "PilotDecision":
        if self.abstain and self.predicted_label is not None:
            raise ValueError("predicted_label must be null when abstaining")
        if not self.abstain and (not self.predicted_label or not self.evidence_quote.strip() or not self.evidence_source_id):
            raise ValueError("label, evidence quote, and source are required")
        return self


def schema(labels: list[str]) -> dict:
    nullable_label = {"anyOf": [{"type": "string", "enum": labels}, {"type": "null"}]}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "predicted_label": nullable_label,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_quote": {"type": "string"},
            "evidence_source_id": nullable_string,
            "reason": {"type": "string"},
            "standard_citation": nullable_string,
            "abstain": {"type": "boolean"},
        },
        "required": ["predicted_label", "confidence", "evidence_quote", "evidence_source_id", "reason", "standard_citation", "abstain"],
        "additionalProperties": False,
    }


def completed(path) -> set[tuple[str, str]]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    done.add((str(item["request_id"]), str(item["prompt_bundle_sha256"])))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def usage_value(usage: dict, *names: str) -> int:
    for name in names:
        if usage.get(name) is not None:
            return int(usage[name])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--condition", action="append")
    args = parser.parse_args()
    config = load_config()
    path = ROOT / config["candidate_table"]
    candidates = pd.read_csv(path) if path.exists() else prepare_candidates(config)
    if args.condition:
        candidates = candidates.loc[candidates["condition"].isin(args.condition)].copy()
    response_path = ROOT / config["response_log"]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed(response_path)
    pending = candidates.loc[~candidates.apply(lambda row: (str(row["request_id"]), str(row["prompt_bundle_sha256"])) in done, axis=1)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(f"candidates={len(candidates)} pending_selected={len(pending)}", flush=True)
    if not args.run:
        return
    key = os.getenv(config["api_key_env"])
    if not key:
        raise SystemExit(f"Missing {config['api_key_env']}; no calls sent")
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    labels = list(definitions)
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
                    text={"format": {"type": "json_schema", "name": "pcaob_rag_pilot", "strict": True, "schema": schema(labels)}},
                )
                latency = round((time.perf_counter() - started) * 1000, 1)
                decision = PilotDecision.model_validate_json(response.output_text)
                dumped = response.model_dump(mode="json")
                usage = dumped.get("usage") or {}
                record = {
                    "request_id": row["request_id"], "sample_id": row["sample_id"], "condition": row["condition"],
                    "status": "ok", "attempt": attempt, "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provider": config["provider"], "requested_model": config["model"], "response_model": dumped.get("model"),
                    "latency_ms": latency, "input_tokens": usage_value(usage, "input_tokens", "prompt_tokens"),
                    "output_tokens": usage_value(usage, "output_tokens", "completion_tokens"),
                    "total_tokens": usage_value(usage, "total_tokens"), "decision": decision.model_dump(mode="json"),
                    "raw_output": response.output_text, "prompt_bundle_sha256": row["prompt_bundle_sha256"],
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
