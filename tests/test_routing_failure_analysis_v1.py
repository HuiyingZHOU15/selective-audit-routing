import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_routing_failure_v1 import budget_curve, outcome_summary, top_budget_route


def test_top_budget_route_uses_highest_scores_and_exact_count():
    route = top_budget_route(np.array([.1, .8, .3, .7]), .5)
    assert route.tolist() == [False, True, False, True]


def test_outcome_summary_uses_benefit_minus_harm():
    rows = []
    for partition in ["Development OOF", "Calibration", "Retrospective test"]:
        for index, outcome in enumerate(["beneficial", "harmful", "neutral_both_correct", "neutral_both_wrong"]):
            rows.append({"partition": partition, "routing_outcome": outcome, "outcome_utility": [1, -1, 0, 0][index], "report_id": str(index)})
    summary = outcome_summary(pd.DataFrame(rows))
    assert (summary["beneficial_rate"] == .25).all()
    assert (summary["harmful_rate"] == .25).all()
    assert (summary["always_jev_net_per_100"] == 0).all()


def test_budget_curve_net_matches_routed_utilities():
    rows = []
    for partition in ["Development OOF", "Calibration", "Retrospective test"]:
        for score, utility in zip([.9, .8, .2, .1], [1, -1, 1, -1]):
            rows.append({"partition": partition, "router_score": score, "outcome_utility": utility})
    curve = budget_curve(pd.DataFrame(rows), .5, .5)
    half = curve.loc[curve["budget"].eq(.5)]
    assert (half["calls"] == 2).all()
    assert (half["net_correct"] == 0).all()
