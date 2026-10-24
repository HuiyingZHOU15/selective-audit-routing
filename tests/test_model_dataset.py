from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "model_dataset_v1.csv"
NEAR = ROOT / "results" / "tables" / "near_duplicate_report.csv"


def test_reports_do_not_cross_splits():
    data = pd.read_csv(DATASET)
    assert data.groupby("report_id")["split"].nunique().max() == 1


def test_all_target_labels_exist_in_every_split():
    data = pd.read_csv(DATASET)
    expected = set(data["target_label"])
    for _, group in data.groupby("split"):
        assert set(group["target_label"]) == expected


def test_clean_rows_have_no_exact_cross_split_duplicates():
    data = pd.read_csv(DATASET)
    clean = data.loc[data["clean_eval"]]
    assert clean.groupby("description_norm")["split"].nunique().max() == 1


def test_clean_rows_have_no_audited_near_duplicate_cross_split_pairs():
    data = pd.read_csv(DATASET).set_index("sample_id")
    near = pd.read_csv(NEAR)
    for pair in near.itertuples():
        if pair.sample_id not in data.index or pair.neighbor_sample_id not in data.index:
            continue
        left = data.loc[pair.sample_id]
        right = data.loc[pair.neighbor_sample_id]
        assert not (left["clean_eval"] and right["clean_eval"] and left["split"] != right["split"])
