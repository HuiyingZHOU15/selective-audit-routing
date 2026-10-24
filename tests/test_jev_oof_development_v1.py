from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/interim/jev_oof_development_v1_candidates.csv"


def test_full_candidate_population_matches_clean_oof():
    candidates = pd.read_csv(CANDIDATES)
    oof = pd.read_csv(ROOT / "data/processed/pcaob_development_oof_v1.csv")
    clean = oof.loc[oof["oof_clean"].astype(bool)]
    assert len(candidates) == len(clean) == 4613
    assert candidates["sample_id"].is_unique
    assert set(candidates["sample_id"]) == set(clean["sample_id"])
    assert candidates["outer_fold"].nunique() == 5


def test_pilot_prompt_hashes_are_reusable_in_full_population():
    full = pd.read_csv(CANDIDATES).set_index("sample_id")
    pilot = pd.read_csv(ROOT / "data/interim/jev_oof_pilot_v1_candidates.csv").set_index("sample_id")
    common = full.loc[pilot.index]
    assert common["prompt_bundle_sha256"].equals(pilot["prompt_bundle_sha256"])
