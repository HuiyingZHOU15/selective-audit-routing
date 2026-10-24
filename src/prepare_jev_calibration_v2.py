"""Prepare clean calibration samples and frozen small-model predictions for Jev."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from llm_common import ROOT, sha256_text


def main() -> None:
    config = json.loads((ROOT / "configs/jev_core_v2.json").read_text(encoding="utf-8"))
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / "data/processed/model_dataset_v2.csv")
    frame = data.loc[data["split"].eq("calibration") & data["clean_eval"]].copy().reset_index(drop=True)
    model = joblib.load(ROOT / "artifacts/models/tfidf_lr_context_taxonomy_v2.joblib")
    probabilities = model.predict_proba(frame["text_context_v2"].fillna(""))
    classes = np.asarray(model.classes_)
    frame["small_prediction"] = classes[probabilities.argmax(axis=1)]
    frame["small_confidence"] = probabilities.max(axis=1)
    frame["text"] = frame["text_context_v2"]
    criteria_json = json.dumps(definitions, ensure_ascii=False, sort_keys=True)
    frame["prompt_bundle_sha256"] = frame["text"].map(
        lambda text: sha256_text(config["version"] + "\n" + criteria_json + "\n" + str(text))
    )
    keep = [
        "sample_id", "report_id", "target_level1", "target_level2", "audit_areas", "context_quality",
        "text", "small_prediction", "small_confidence", "prompt_bundle_sha256",
    ]
    output = ROOT / config["calibration_candidate_table"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(output, index=False, encoding="utf-8")
    print(f"Prepared {len(frame)} calibration candidates: {output}")


if __name__ == "__main__":
    main()
