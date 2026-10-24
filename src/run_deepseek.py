"""Run the frozen routed-sample batch through DeepSeek with resumable JSONL logging."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from openai import OpenAI

from llm_common import (
    LLMDecision,
    ROOT,
    decision_schema,
    load_config,
    render_user_prompt,
)
from prepare_deepseek_batch import prepare_candidates


def completed_requests(path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    requests = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if item.get("status") == "ok":
                requests.add((str(item["sample_id"]), str(item.get("prompt_bundle_sha256", ""))))
        except (json.JSONDecodeError, KeyError):
            continue
    return requests


def usage_value(usage: dict, *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if value is not None:
            return int(value)
    return 0


def invoke(client: OpenAI, config: dict, system_prompt: str, user_prompt: str, labels: list[str]):
    started = time.perf_counter()
    response = client.responses.create(
        model=config["model"],
        instructions=system_prompt,
        input=user_prompt,
        temperature=config["temperature"],
        max_output_tokens=config["max_output_tokens"],
        reasoning={"effort": config["reasoning_effort"]},
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "audit_finding_classification",
                "strict": True,
                "schema": decision_schema(labels),
            }
        },
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    raw = response.output_text
    decision = LLMDecision.model_validate_json(raw)
    if decision.predicted_label is not None and decision.predicted_label not in labels:
        raise ValueError(f"Unknown label: {decision.predicted_label}")
    dumped = response.model_dump(mode="json")
    usage = dumped.get("usage") or {}
    return decision, raw, dumped, usage, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Actually call the paid DeepSeek API.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new successful samples.")
    args = parser.parse_args()

    config = load_config()
    candidate_path = ROOT / config["candidate_table"]
    candidates = pd.read_csv(candidate_path) if candidate_path.exists() else prepare_candidates()
    taxonomy = json.loads((ROOT / config["taxonomy"]).read_text(encoding="utf-8"))
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    labels = taxonomy["target_labels"]
    system_prompt = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    response_path = ROOT / config["response_log"]
    response_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_requests(response_path)
    is_done = candidates.apply(
        lambda row: (str(row["sample_id"]), str(row["prompt_bundle_sha256"])) in done,
        axis=1,
    )
    pending = candidates.loc[~is_done]
    if args.limit is not None:
        pending = pending.head(args.limit)

    completed_current = int(is_done.sum())
    print(f"Candidates={len(candidates)} completed_current_prompt={completed_current} selected_now={len(pending)}")
    if not args.run:
        print("Dry run only. Add --run to call DeepSeek.")
        return

    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise SystemExit(f"Missing {config['api_key_env']}; no API requests were sent.")
    client = OpenAI(api_key=api_key, base_url=config["base_url"], timeout=config["request_timeout_seconds"])

    successful = 0
    for _, row in pending.iterrows():
        ordered_labels = json.loads(row["label_order_json"])
        user_prompt = render_user_prompt(str(row["text"]), ordered_labels, definitions)
        last_error = None
        for attempt in range(1, config["max_attempts"] + 1):
            try:
                decision, raw, response_dump, usage, latency_ms = invoke(
                    client, config, system_prompt, user_prompt, labels
                )
                record = {
                    "sample_id": str(row["sample_id"]),
                    "status": "ok",
                    "attempt": attempt,
                    "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provider": config["provider"],
                    "requested_model": config["model"],
                    "response_id": response_dump.get("id"),
                    "response_model": response_dump.get("model"),
                    "latency_ms": latency_ms,
                    "input_tokens": usage_value(usage, "input_tokens", "prompt_tokens"),
                    "output_tokens": usage_value(usage, "output_tokens", "completion_tokens"),
                    "total_tokens": usage_value(usage, "total_tokens"),
                    "raw_output": raw,
                    "decision": decision.model_dump(mode="json"),
                    "user_prompt_sha256": row["user_prompt_sha256"],
                    "prompt_bundle_sha256": row["prompt_bundle_sha256"],
                }
                with response_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                successful += 1
                print(f"OK {row['sample_id']} ({successful}/{len(pending)})")
                break
            except Exception as exc:  # API and schema errors are both retryable in the MVP.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < config["max_attempts"]:
                    time.sleep(2 ** attempt)
        else:
            failure = {
                "sample_id": str(row["sample_id"]),
                "status": "error",
                "attempt": config["max_attempts"],
                "requested_at_utc": datetime.now(timezone.utc).isoformat(),
                "provider": config["provider"],
                "requested_model": config["model"],
                "error": last_error,
                "user_prompt_sha256": row["user_prompt_sha256"],
                "prompt_bundle_sha256": row["prompt_bundle_sha256"],
            }
            with response_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"ERROR {row['sample_id']}: {last_error}")


if __name__ == "__main__":
    main()
