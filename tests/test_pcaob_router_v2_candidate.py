from __future__ import annotations

import numpy as np
import pandas as pd

from train_pcaob_router_v2_candidate import FEATURES, exact_top_fraction, scenario_weights


def test_exact_top_fraction_uses_exact_ceiling_budget() -> None:
    selected = exact_top_fraction(pd.Series([0.2, 0.9, 0.1, 0.8, 0.3]), 0.4)
    assert selected.tolist() == [False, True, False, True, False]


def test_router_features_exclude_outcome_and_llm_signals() -> None:
    forbidden = {
        "small_correct",
        "target_level1",
        "prediction_correct",
        "predicted_label",
        "confidence",
        "rag_correct",
        "benefit",
        "harm",
    }
    assert forbidden.isdisjoint(FEATURES)


def test_scenario_weights_match_requested_small_error_prevalence() -> None:
    frame = pd.DataFrame({"small_wrong": [1, 1, 0, 0, 0]})
    weights = scenario_weights(frame, 0.12)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[:2].sum(), 0.12)
    assert np.isclose(weights[2:].sum(), 0.88)
