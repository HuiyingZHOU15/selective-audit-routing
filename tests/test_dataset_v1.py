import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/releases/pcaob_selective_routing_v1"


def load_data() -> pd.DataFrame:
    return pd.read_csv(RELEASE / "pcaob_selective_routing_v1.csv")


def test_release_manifest_hashes():
    import hashlib

    manifest = json.loads((RELEASE / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        payload = (RELEASE / name).read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]


def test_sample_and_report_split_integrity():
    data = load_data()
    assert len(data) == 6607
    assert data["sample_id"].is_unique
    assert data.groupby("report_id")["group_split"].nunique().max() == 1
    assert data["audit_area_count"].eq(1).all()


def test_temporal_policy():
    data = load_data()
    years = data["Inspection Year"].astype(int)
    assert data.loc[years <= 2023, "temporal_split"].eq("development").all()
    assert data.loc[years == 2024, "temporal_split"].eq("calibration").all()
    assert data.loc[years >= 2025, "temporal_split"].eq("retrospective_test").all()
    assert data["temporal_confirmatory_status"].eq("retrospective_only_not_confirmatory").all()


def test_comparative_outcomes_are_complete_and_consistent():
    data = load_data()
    observed = data["routing_outcome"].ne("not_observed")
    assert observed.sum() == 6111
    assert data.loc[observed, ["small_correct", "llm_correct"]].notna().all().all()
    for row in data.loc[observed].itertuples():
        if row.small_correct and row.llm_correct:
            expected = "neutral_both_correct"
        elif (not row.small_correct) and row.llm_correct:
            expected = "beneficial"
        elif row.small_correct and (not row.llm_correct):
            expected = "harmful"
        else:
            expected = "neutral_both_wrong"
        assert row.routing_outcome == expected


def test_test_outcomes_are_not_marked_for_training():
    data = load_data()
    test_rows = data["outcome_partition"].eq("test")
    assert test_rows.sum() == 890
    assert data.loc[test_rows, "outcome_usage"].eq("retrospective_evaluation_only").all()


def test_development_outcomes_are_crossfit_training_labels():
    data = load_data()
    development = data["outcome_partition"].eq("development_oof")
    assert development.sum() == 4613
    assert data.loc[development, "outcome_usage"].eq("router_training_crossfit").all()
