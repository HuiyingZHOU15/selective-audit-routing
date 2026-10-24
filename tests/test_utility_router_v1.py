import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_router_uses_only_pre_call_features():
    metrics = json.loads((ROOT / "results/metrics/utility_router_v1.json").read_text(encoding="utf-8"))
    features = metrics["features_available_before_llm_call"]
    joined = features["numeric"] + features["categorical"]
    assert not any(name.startswith("llm_") or name.startswith("jev_") for name in joined)
    assert "target_level1" not in joined
    assert "small_correct" not in joined


def test_router_population_and_budget():
    development = pd.read_csv(ROOT / "results/tables/utility_router_development_oof_v1.csv")
    calibration = pd.read_csv(ROOT / "results/tables/utility_router_calibration_v1.csv")
    test = pd.read_csv(ROOT / "results/tables/utility_router_test_v1.csv")
    assert len(development) == 4613
    assert len(calibration) == 608
    assert len(test) == 890
    assert development[["router_p_benefit_oof", "router_p_harm_oof"]].notna().all().all()
    assert calibration["utility_route"].sum() <= round(0.20 * len(calibration))
    assert test["utility_route"].sum() <= round(0.20 * len(test))


def test_test_status_is_retrospective():
    metrics = json.loads((ROOT / "results/metrics/utility_router_v1.json").read_text(encoding="utf-8"))
    assert metrics["test_status"] == "retrospective_after_prior_inspection"
