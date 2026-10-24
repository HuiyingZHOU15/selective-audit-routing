"""Freeze development-only Direct and structured-RAG FinMR prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from finmr_structured import retrieve


ROOT = Path(__file__).resolve().parents[1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config() -> dict:
    return json.loads((ROOT / "configs" / "finmr_pilot_v1.json").read_text(encoding="utf-8"))


def structured_prompt(evidence_text: str) -> str:
    evidence = "\n".join(
        line for line in evidence_text.splitlines()
        if not line.startswith("DETERMINISTIC_SUM:")
    )
    return f"""Use only the retrieved XBRL evidence below.

The reported value is the TARGET fact for the requested period.
The calculated value is the exact sum of every CHILD_FACT value multiplied by its weight.
Return numeric values as strings in this exact JSON form:
{{"extracted_value":"...", "calculated_value":"..."}}

RETRIEVED EVIDENCE
{evidence}
"""


def prepare() -> pd.DataFrame:
    config = load_config()
    system = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    source = pd.read_parquet(ROOT / config["dataset_dir"] / f"{config['source_split']}.parquet")
    rows = []
    for item in source.to_dict("records"):
        retrieval = retrieve(item["query"])
        if item["dqc_id"] != config["eligible_dqc"] or retrieval.status != "complete":
            continue
        sample_id = f"finmr-{config['source_split']}-{int(item['id']):04d}"
        gold = item["answer"]
        deterministic = str(retrieval.calculated_value) if retrieval.calculated_value is not None else None
        prompts = {
            "direct": item["query"],
            "structured_rag": structured_prompt(retrieval.evidence_text),
        }
        for condition, user_prompt in prompts.items():
            rows.append({
                "sample_id": sample_id,
                "source_id": int(item["id"]),
                "split": config["source_split"],
                "dqc_id": item["dqc_id"],
                "condition": condition,
                "request_id": f"{sample_id}::{condition}",
                "gold_extracted_value": str(gold["extracted_value"]),
                "gold_calculated_value": str(gold["calculated_value"]),
                "deterministic_extracted_value": retrieval.reported_fact.value_text if retrieval.reported_fact else None,
                "deterministic_calculated_value": deterministic,
                "target": retrieval.question.target,
                "period": retrieval.question.period_text,
                "child_count": len(retrieval.children),
                "query_chars": len(item["query"]),
                "evidence_chars": len(retrieval.evidence_text),
                "user_prompt": user_prompt,
                "user_prompt_sha256": digest(user_prompt),
                "system_prompt_sha256": digest(system),
                "prompt_bundle_sha256": digest(
                    system
                    + f"\nGENERATION_CONFIG:model={config['model']};temperature={config['temperature']};reasoning={config['reasoning_effort']};max_output_tokens={config['max_output_tokens']}"
                    + "\n\n"
                    + user_prompt
                ),
            })
    candidates = pd.DataFrame(rows).sort_values(["source_id", "condition"]).reset_index(drop=True)
    output = ROOT / config["candidate_table"]
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False, encoding="utf-8")
    print(json.dumps({
        "samples": int(candidates["sample_id"].nunique()),
        "requests": len(candidates),
        "conditions": candidates["condition"].value_counts().to_dict(),
        "median_direct_chars": float(candidates.loc[candidates.condition.eq("direct"), "user_prompt"].str.len().median()),
        "median_structured_chars": float(candidates.loc[candidates.condition.eq("structured_rag"), "user_prompt"].str.len().median()),
    }, indent=2))
    return candidates


if __name__ == "__main__":
    prepare()
