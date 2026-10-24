"""Freeze the 20% selective-routing candidates and deterministic DeepSeek prompts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from llm_common import ROOT, load_config, render_user_prompt, sha256_text


def prepare_candidates() -> pd.DataFrame:
    config = load_config()
    data = pd.read_csv(ROOT / config["dataset"])
    routes = pd.read_csv(ROOT / config["routing_table"])
    taxonomy = json.loads((ROOT / config["taxonomy"]).read_text(encoding="utf-8"))
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    system_prompt = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    labels = taxonomy["target_labels"]

    routed = routes.loc[routes[config["route_column"]].astype(bool)].copy()
    columns = ["sample_id", "report_id", "target_label", "audit_area_norm", config["text_column"]]
    merged = routed.merge(data[columns], on=["sample_id", "report_id", "target_label"], validate="one_to_one")
    merged = merged.sort_values("sample_id").reset_index(drop=True)

    rows = []
    for position, row in merged.iterrows():
        rng = np.random.default_rng(config["prompt_seed"] + position)
        ordered_labels = list(rng.permutation(labels))
        user_prompt = render_user_prompt(str(row[config["text_column"]]), ordered_labels, definitions)
        rows.append(
            {
                "sample_id": row["sample_id"],
                "report_id": row["report_id"],
                "target_label": row["target_label"],
                "raw_audit_area": row["audit_area_norm"],
                "small_prediction": row[config["small_prediction_column"]],
                "small_confidence": row[config["small_confidence_column"]],
                "text": row[config["text_column"]],
                "label_order_json": json.dumps(ordered_labels, ensure_ascii=False),
                "prompt_seed": config["prompt_seed"] + position,
                "system_prompt_sha256": sha256_text(system_prompt),
                "user_prompt_sha256": sha256_text(user_prompt),
                "prompt_bundle_sha256": sha256_text(system_prompt + "\n\n" + user_prompt),
                "prompt_version": config["version"],
            }
        )

    candidates = pd.DataFrame(rows)
    output = ROOT / config["candidate_table"]
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False, encoding="utf-8")
    print(f"Prepared {len(candidates)} routed candidates: {output}")
    return candidates


if __name__ == "__main__":
    prepare_candidates()
