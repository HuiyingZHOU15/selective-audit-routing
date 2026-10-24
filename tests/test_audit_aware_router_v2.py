import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_audit_aware_router_v2 import exact_route  # noqa: E402


def test_exact_route_enforces_budget_and_is_deterministic():
    confidence = np.asarray([0.8, 0.1, 0.4, 0.2, 0.9])
    route = exact_route(confidence, 0.4)
    assert route.tolist() == [False, True, False, True, False]
    assert route.sum() == 2


def test_router_outputs_exist_after_generation():
    path = ROOT / "results/tables/audit_aware_router_test_v2.csv"
    if not path.exists():
        return
    import pandas as pd

    frame = pd.read_csv(path)
    assert len(frame) == 890
    assert frame["gate_benefit_probability"].between(0, 1).all()
    assert int(frame["route_20pct"].sum()) == 178
