"""Audit the weighted Jev OOF pilot and estimate development-population event rates."""

from __future__ import annotations

import json

import pandas as pd

from llm_common import ROOT


CONFIG_PATH = ROOT / "configs/jev_oof_pilot_v1.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    rows = [json.loads(line) for line in (ROOT / config["response_log"]).read_text(encoding="utf-8").splitlines()]
    responses = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    responses = responses.drop_duplicates(["sample_id", "prompt_bundle_sha256"], keep="last")
    audit = candidates.merge(responses, on=["sample_id", "prompt_bundle_sha256"], how="left", validate="one_to_one")
    if audit["prediction"].isna().any():
        raise RuntimeError(f"Missing successful responses: {int(audit['prediction'].isna().sum())}")
    audit["llm_prediction"] = audit["prediction"]
    audit["llm_correct"] = audit["llm_prediction"].eq(audit["target_level1"])
    audit["routing_outcome"] = "neutral_both_wrong"
    audit.loc[audit["small_correct"] & audit["llm_correct"], "routing_outcome"] = "neutral_both_correct"
    audit.loc[~audit["small_correct"] & audit["llm_correct"], "routing_outcome"] = "beneficial"
    audit.loc[audit["small_correct"] & ~audit["llm_correct"], "routing_outcome"] = "harmful"
    raw = audit["routing_outcome"].value_counts().rename_axis("routing_outcome").reset_index(name="pilot_count")
    weighted = audit.groupby("routing_outcome")["sampling_weight"].sum().rename("estimated_population_count")
    raw = raw.merge(weighted, on="routing_outcome", how="outer").fillna(0)
    population = float(audit["sampling_weight"].sum())
    raw["estimated_population_rate"] = raw["estimated_population_count"] / population
    output = ROOT / "results/tables/jev_oof_pilot_v1_audit.csv"
    stats_output = ROOT / "results/tables/jev_oof_pilot_v1_outcomes.csv"
    metrics_output = ROOT / "results/metrics/jev_oof_pilot_v1.json"
    for path in (output, stats_output, metrics_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False, encoding="utf-8")
    raw.to_csv(stats_output, index=False, encoding="utf-8")
    metrics = {
        "version": config["version"],
        "pilot_samples": len(audit),
        "weighted_population": population,
        "response_models": audit["response_model"].value_counts().to_dict(),
        "median_latency_ms": float(audit["latency_ms"].median()),
        "outcomes": raw.to_dict(orient="records"),
    }
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
