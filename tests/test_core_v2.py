import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrate_route_v2 import brier_score, ece_score  # noqa: E402


def test_calibration_metrics_are_zero_for_perfect_predictions():
    classes = np.array(["a", "b"])
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array(["a", "b"])
    assert brier_score(target, probabilities, classes) == 0
    assert ece_score(target, probabilities, classes, 5) == 0


def test_routing_output_has_nested_budgets_after_generation():
    path = ROOT / "results" / "tables" / "core_v2_test_predictions.csv"
    if not path.exists():
        return
    import pandas as pd

    frame = pd.read_csv(path)
    route10 = frame["route_10pct"].astype(bool)
    route20 = frame["route_20pct"].astype(bool)
    route30 = frame["route_30pct"].astype(bool)
    assert (~route10 | route20).all()
    assert (~route20 | route30).all()
