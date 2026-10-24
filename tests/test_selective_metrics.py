import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrate_and_route import aurc_metrics, risk_coverage


def test_perfect_predictions_have_zero_aurc_and_augrc():
    errors = np.zeros(5, dtype=int)
    confidence = np.linspace(0.5, 0.9, 5)
    aurc, augrc = aurc_metrics(errors, confidence)
    assert aurc == 0.0
    assert augrc == 0.0


def test_generalized_risk_equals_coverage_times_selective_risk():
    errors = np.array([0, 1, 0, 1])
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    curve = risk_coverage(errors, confidence)
    assert np.allclose(curve["generalized_risk"], curve["coverage"] * curve["selective_risk"])


def test_confident_errors_worsen_selective_area():
    errors = np.array([1, 0, 0, 0])
    bad_confidence = np.array([0.9, 0.8, 0.7, 0.6])
    good_confidence = np.array([0.6, 0.9, 0.8, 0.7])
    bad_aurc, bad_augrc = aurc_metrics(errors, bad_confidence)
    good_aurc, good_augrc = aurc_metrics(errors, good_confidence)
    assert bad_aurc > good_aurc
    assert bad_augrc > good_augrc
