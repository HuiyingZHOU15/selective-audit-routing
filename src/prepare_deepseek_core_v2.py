"""Freeze all clean-test prompts for the taxonomy-v2 core experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_common import ROOT, render_user_prompt, sha256_text

CONFIG_PATH = ROOT / "configs" / "llm_deepseek_core_v2.json"


def load_config_v2() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def prepare_candidates() -> pd.DataFrame:
    config = load_config_v2()
    data = pd.read_csv(ROOT / config["dataset"])
    routes = pd.read_csv(ROOT / config["routing_table"])
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    labels = list(definitions)
    system_prompt = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    clean_test = data.loc[data["split"].eq("test") & data["clean_eval"]].copy()
    keep = ["sample_id", "report_id", config["target_column"], "target_level2", "audit_areas", config["text_column"]]
    merged = routes.merge(clean_test[keep], on=["sample_id", "report_id", config["target_column"], "target_level2", "audit_areas"], validate="one_to_one")
    merged = merged.sort_values("sample_id").reset_index(drop=True)
    rows = []
    for position, row in merged.iterrows():
        ordered_labels = list(np.random.default_rng(config["prompt_seed"] + position).permutation(labels))
        prompt = render_user_prompt(str(row[config["text_column"]]), ordered_labels, definitions)
        item = row.to_dict()
        item.update({
            "text": row[config["text_column"]],
            "label_order_json": json.dumps(ordered_labels, ensure_ascii=False),
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(prompt),
            "prompt_bundle_sha256": sha256_text(system_prompt + "\n\n" + prompt),
            "prompt_version": config["version"],
        })
        rows.append(item)
    output = pd.DataFrame(rows)
    path = ROOT / config["candidate_table"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8")
    print(f"Prepared {len(output)} clean-test prompts: {path}")
    return output


if __name__ == "__main__":
    prepare_candidates()
