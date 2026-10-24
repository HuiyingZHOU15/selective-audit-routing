"""Select a weighted 300-sample Jev pilot from clean nested-OOF PCAOB outcomes."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from llm_common import ROOT, sha256_text


CONFIG_PATH = ROOT / "configs/jev_oof_pilot_v1.json"
OOF_PATH = ROOT / "data/processed/pcaob_development_oof_v1.csv"


def allocate(populations: pd.Series, total: int, minimum: int = 2) -> dict[str, int]:
    populations = populations.astype(int).sort_index()
    allocation = pd.Series(0, index=populations.index, dtype=int)
    for label in populations.index:
        allocation[label] = min(minimum, populations[label])
    remaining = total - int(allocation.sum())
    if remaining < 0:
        raise ValueError("Quota is smaller than the requested per-class minimum")
    capacity = populations - allocation
    while remaining:
        active = capacity[capacity > 0]
        if active.empty:
            raise ValueError("Insufficient population for requested quota")
        ideal = remaining * active / active.sum()
        proposed = np.floor(ideal).astype(int).clip(upper=active)
        if proposed.sum() == 0:
            order = (ideal - np.floor(ideal)).sort_values(ascending=False).index
            for label in order:
                if remaining == 0:
                    break
                if capacity[label] > 0:
                    allocation[label] += 1
                    capacity[label] -= 1
                    remaining -= 1
            continue
        allocation.loc[proposed.index] += proposed
        capacity.loc[proposed.index] -= proposed
        remaining -= int(proposed.sum())
    return allocation.to_dict()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    data = pd.read_csv(OOF_PATH)
    clean = data.loc[data["oof_clean"].astype(bool)].copy()
    clean["correctness_stratum"] = np.where(clean["small_correct"].astype(bool), "small_correct", "small_error")
    quotas = {"small_error": config["small_error_quota"], "small_correct": config["small_correct_quota"]}
    selected = []
    allocation_rows = []
    for stratum, quota in quotas.items():
        pool = clean.loc[clean["correctness_stratum"].eq(stratum)]
        populations = pool.groupby("target_level1").size()
        allocation = allocate(populations, quota, minimum=2)
        for offset, (label, sample_count) in enumerate(sorted(allocation.items())):
            group = pool.loc[pool["target_level1"].eq(label)]
            sample = group.sample(
                n=sample_count,
                random_state=config["sampling_seed"] + offset + (10_000 if stratum == "small_error" else 0),
            ).copy()
            sample["stratum_population"] = len(group)
            sample["stratum_sample_size"] = sample_count
            sample["inclusion_probability"] = sample_count / len(group)
            sample["sampling_weight"] = len(group) / sample_count
            selected.append(sample)
            allocation_rows.append({
                "correctness_stratum": stratum,
                "target_level1": label,
                "population": len(group),
                "sample_size": sample_count,
                "inclusion_probability": sample_count / len(group),
            })
    pilot = pd.concat(selected, ignore_index=True)
    if len(pilot) != config["sample_size"] or pilot["sample_id"].duplicated().any():
        raise RuntimeError("Pilot selection is incomplete or duplicated")
    criteria_json = json.dumps(definitions, ensure_ascii=False, sort_keys=True)
    pilot["text"] = pilot["text_context_v2"]
    pilot["prompt_bundle_sha256"] = pilot["text"].map(
        lambda text: sha256_text(config["version"] + "\n" + criteria_json + "\n" + str(text))
    )
    pilot = pilot.sample(frac=1.0, random_state=config["sampling_seed"]).reset_index(drop=True)
    pilot.insert(0, "request_order", np.arange(1, len(pilot) + 1))
    keep = [
        "request_order", "sample_id", "report_id", "outer_fold", "target_level1", "target_level2",
        "audit_areas", "context_quality", "text", "small_prediction", "small_confidence",
        "small_margin", "small_entropy", "small_correct", "correctness_stratum",
        "stratum_population", "stratum_sample_size", "inclusion_probability", "sampling_weight",
        "prompt_bundle_sha256",
    ]
    output = ROOT / config["candidate_table"]
    allocation_output = ROOT / "results/tables/jev_oof_pilot_v1_sampling.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    allocation_output.parent.mkdir(parents=True, exist_ok=True)
    pilot[keep].to_csv(output, index=False, encoding="utf-8")
    pd.DataFrame(allocation_rows).to_csv(allocation_output, index=False, encoding="utf-8")
    print(json.dumps({
        "selected": len(pilot),
        "small_error": int(pilot["correctness_stratum"].eq("small_error").sum()),
        "small_correct": int(pilot["correctness_stratum"].eq("small_correct").sum()),
        "classes": int(pilot["target_level1"].nunique()),
        "folds": int(pilot["outer_fold"].nunique()),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
