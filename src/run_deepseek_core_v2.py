"""Run the resumable all-test DeepSeek core experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI

from llm_common import LLMDecision, ROOT, decision_schema, render_user_prompt
from prepare_deepseek_core_v2 import load_config_v2, prepare_candidates


def completed_requests(path) -> set[tuple[str, str]]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    done.add((str(item["sample_id"]), str(item.get("prompt_bundle_sha256", ""))))
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
    args = parser.parse_args()
    config = load_config_v2()
    path = ROOT / config["candidate_table"]
    candidates = pd.read_csv(path) if path.exists() else prepare_candidates()
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    labels = list(definitions)
    system = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    response_path = ROOT / config["response_log"]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_requests(response_path)
    pending = candidates.loc[~candidates.apply(lambda r: (str(r["sample_id"]), str(r["prompt_bundle_sha256"])) in done, axis=1)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    print(f"candidates={len(candidates)} complete={len(candidates)-len(pending)} selected_now={len(pending)}", flush=True)
    if not args.run:
        return
    key = os.getenv(config["api_key_env"])
    if not key:
        raise SystemExit(f"Missing {config['api_key_env']}; no calls sent.")
    client = OpenAI(api_key=key, base_url=config["base_url"], timeout=config["request_timeout_seconds"])
    successes = 0
    for _, row in pending.iterrows():
        ordered = json.loads(row["label_order_json"])
        prompt = render_user_prompt(str(row["text"]), ordered, definitions)
        last_error = None
        for attempt in range(1, config["max_attempts"] + 1):
            try:
                started = time.perf_counter()
                response = client.responses.create(
                    model=config["model"], instructions=system, input=prompt,
                    temperature=config["temperature"], max_output_tokens=config["max_output_tokens"],
                    reasoning={"effort": config["reasoning_effort"]}, store=False,
                    text={"format": {"type": "json_schema", "name": "audit_finding_classification", "strict": True, "schema": decision_schema(labels)}},
                )
                latency = round((time.perf_counter() - started) * 1000, 1)
                decision = LLMDecision.model_validate_json(response.output_text)
                dumped = response.model_dump(mode="json")
                usage = dumped.get("usage") or {}
                record = {
                    "sample_id": str(row["sample_id"]), "status": "ok", "attempt": attempt,
                    "requested_at_utc": datetime.now(timezone.utc).isoformat(), "provider": config["provider"],
                    "requested_model": config["model"], "response_id": dumped.get("id"), "response_model": dumped.get("model"),
                    "latency_ms": latency, "input_tokens": usage_value(usage, "input_tokens", "prompt_tokens"),
                    "output_tokens": usage_value(usage, "output_tokens", "completion_tokens"), "total_tokens": usage_value(usage, "total_tokens"),
                    "raw_output": response.output_text, "decision": decision.model_dump(mode="json"),
                    "user_prompt_sha256": row["user_prompt_sha256"], "prompt_bundle_sha256": row["prompt_bundle_sha256"],
                }
                with response_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                successes += 1
                if successes % 25 == 0 or successes == len(pending):
                    print(f"OK {successes}/{len(pending)}", flush=True)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < config["max_attempts"]:
                    time.sleep(2 ** attempt)
        else:
            failure = {"sample_id": str(row["sample_id"]), "status": "error", "attempt": config["max_attempts"], "error": last_error,
                       "requested_at_utc": datetime.now(timezone.utc).isoformat(), "prompt_bundle_sha256": row["prompt_bundle_sha256"]}
            with response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"ERROR {row['sample_id']}: {last_error}", flush=True)


if __name__ == "__main__":
    main()
