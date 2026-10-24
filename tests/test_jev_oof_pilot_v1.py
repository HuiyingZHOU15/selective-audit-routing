from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/interim/jev_oof_pilot_v1_candidates.csv"


def test_pilot_is_complete_unique_and_weighted():
    data = pd.read_csv(CANDIDATES)
    assert len(data) == 300
    assert data["sample_id"].is_unique
    assert data["correctness_stratum"].value_counts().to_dict() == {"small_error": 180, "small_correct": 120}
    assert data["target_level1"].nunique() == 12
    assert data["outer_fold"].nunique() == 5
    assert data["inclusion_probability"].between(0, 1, inclusive="both").all()
    assert (data["sampling_weight"] >= 1).all()


def test_pilot_contains_only_clean_oof_samples():
    pilot = pd.read_csv(CANDIDATES)
    oof = pd.read_csv(ROOT / "data/processed/pcaob_development_oof_v1.csv")
    clean = set(oof.loc[oof["oof_clean"].astype(bool), "sample_id"])
    assert set(pilot["sample_id"]).issubset(clean)
