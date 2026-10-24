"""Prepare every clean PCAOB development OOF sample for resumable Jev scoring."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from llm_common import ROOT, sha256_text


CONFIG_PATH = ROOT / "configs/jev_oof_development_v1.json"
OOF_PATH = ROOT / "data/processed/pcaob_development_oof_v1.csv"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    data = pd.read_csv(OOF_PATH)
    candidates = data.loc[data["oof_clean"].astype(bool)].copy()
    candidates["text"] = candidates["text_context_v2"]
    criteria_json = json.dumps(definitions, ensure_ascii=False, sort_keys=True)
    candidates["prompt_bundle_sha256"] = candidates["text"].map(
        lambda text: sha256_text(config["prompt_version"] + "\n" + criteria_json + "\n" + str(text))
    )
    candidates = candidates.sample(frac=1.0, random_state=config["sampling_seed"]).reset_index(drop=True)
    candidates.insert(0, "request_order", np.arange(1, len(candidates) + 1))
    keep = [
        "request_order", "sample_id", "report_id", "outer_fold", "target_level1", "target_level2",
        "audit_areas", "context_quality", "context_needs_context", "context_added", "text",
        "small_prediction", "small_confidence", "small_margin", "small_entropy", "small_correct",
        "small_raw_confidence", "small_raw_margin", "small_raw_entropy",
        "prompt_bundle_sha256",
    ]
    output = ROOT / config["candidate_table"]
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates[keep].to_csv(output, index=False, encoding="utf-8")
    print(json.dumps({
        "candidates": len(candidates),
        "small_errors": int((~candidates["small_correct"].astype(bool)).sum()),
        "reports": int(candidates["report_id"].nunique()),
        "folds": int(candidates["outer_fold"].nunique()),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
