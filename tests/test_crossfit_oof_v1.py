import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "data/processed/pcaob_development_oof_v1.csv"
FOLDS = ROOT / "results/tables/pcaob_crossfit_fold_manifest_v1.csv"


def test_crossfit_population_and_uniqueness():
    data = pd.read_csv(OOF)
    assert len(data) == 4974
    assert data["sample_id"].is_unique
    assert data["outer_fold"].between(1, 5).all()
    assert data[["small_prediction", "small_confidence", "small_margin", "small_entropy"]].notna().all().all()
    assert data[["small_raw_confidence", "small_raw_margin", "small_raw_entropy"]].notna().all().all()
    assert data["oof_clean"].notna().all()


def test_reports_and_leakage_components_do_not_cross_folds():
    folds = pd.read_csv(FOLDS)
    assert folds.groupby("report_id")["outer_fold"].nunique().max() == 1
    assert folds.groupby("leakage_component_id")["outer_fold"].nunique().max() == 1


def test_probability_vectors_are_valid_and_match_prediction():
    data = pd.read_csv(OOF)
    for row in data.itertuples():
        probabilities = json.loads(row.small_probability_json)
        assert abs(sum(probabilities.values()) - 1.0) < 1e-8
        assert row.small_prediction == max(probabilities, key=probabilities.get)
        assert row.small_correct == (row.small_prediction == row.target_level1)


def test_every_sample_is_predicted_by_a_model_that_did_not_see_its_component():
    folds = pd.read_csv(FOLDS)
    components_by_fold = {
        fold: set(group["leakage_component_id"])
        for fold, group in folds.groupby("outer_fold")
    }
    for fold, held_components in components_by_fold.items():
        train_components = set(folds.loc[folds["outer_fold"].ne(fold), "leakage_component_id"])
        assert held_components.isdisjoint(train_components)
