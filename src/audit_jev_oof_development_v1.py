"""Create full comparative outcomes for the clean PCAOB development OOF population."""

from __future__ import annotations

import json

import pandas as pd

from llm_common import ROOT


CONFIG_PATH = ROOT / "configs/jev_oof_development_v1.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    records = [json.loads(line) for line in (ROOT / config["response_log"]).read_text(encoding="utf-8").splitlines()]
    responses = pd.DataFrame([record for record in records if record.get("status") == "ok"])
    responses = responses.drop_duplicates(["sample_id", "prompt_bundle_sha256"], keep="last")
    audit = candidates.merge(responses, on=["sample_id", "prompt_bundle_sha256"], how="left", validate="one_to_one")
    missing = int(audit["prediction"].isna().sum())
    if missing:
        raise RuntimeError(f"Missing successful responses: {missing}")
    audit["llm_prediction"] = audit["prediction"]
    audit["llm_confidence"] = audit["confidence"]
    audit["llm_correct"] = audit["llm_prediction"].eq(audit["target_level1"])
    audit["routing_outcome"] = "neutral_both_wrong"
    audit.loc[audit["small_correct"] & audit["llm_correct"], "routing_outcome"] = "neutral_both_correct"
    audit.loc[~audit["small_correct"] & audit["llm_correct"], "routing_outcome"] = "beneficial"
    audit.loc[audit["small_correct"] & ~audit["llm_correct"], "routing_outcome"] = "harmful"
    audit["utility_equal_error_cost"] = audit["routing_outcome"].map({
        "beneficial": 1.0,
        "harmful": -1.0,
        "neutral_both_correct": 0.0,
        "neutral_both_wrong": 0.0,
    })
    counts = audit["routing_outcome"].value_counts().rename_axis("routing_outcome").reset_index(name="samples")
    counts["rate"] = counts["samples"] / len(audit)
    output = ROOT / "data/processed/pcaob_development_comparative_outcomes_v1.csv"
    counts_output = ROOT / "results/tables/pcaob_development_outcome_counts_v1.csv"
    metrics_output = ROOT / "results/metrics/pcaob_development_outcomes_v1.json"
    for path in (output, counts_output, metrics_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False, encoding="utf-8")
    counts.to_csv(counts_output, index=False, encoding="utf-8")
    usage = audit["usage"].map(lambda value: value if isinstance(value, dict) else {})
    metrics = {
        "version": config["version"],
        "samples": len(audit),
        "reports": int(audit["report_id"].nunique()),
        "response_models": audit["response_model"].value_counts().to_dict(),
        "median_latency_ms": float(audit["latency_ms"].median()),
        "outcomes": counts.to_dict(orient="records"),
        "net_utility_if_always_jev": int(audit["utility_equal_error_cost"].sum()),
    }
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
