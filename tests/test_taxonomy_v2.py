import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_taxonomy_v2 import assign_level1, normalize_label  # noqa: E402


def test_alias_normalization_and_level_assignment():
    config = json.loads((ROOT / "configs" / "taxonomy_v2.json").read_text(encoding="utf-8"))
    assert normalize_label("Accounts Receivables", config["explicit_aliases"]) == "Accounts Receivable"
    assert normalize_label("Business Combination", config["explicit_aliases"]) == "Business Combinations"
    assert assign_level1("Accounts Receivable", config["level1_rules"]) == "Revenue and Receivables"
    assert assign_level1("Goodwill and Intangible Assets", config["level1_rules"]) == "Long-Lived and Intangible Assets"


def test_v2_preserves_v1_population_and_splits():
    v1 = pd.read_csv(ROOT / "data" / "processed" / "model_dataset_v1.csv")
    v2 = pd.read_csv(ROOT / "data" / "processed" / "model_dataset_v2.csv")
    assert len(v2) == len(v1)
    joined = v1[["sample_id", "split", "clean_eval"]].merge(
        v2[["sample_id", "split", "clean_eval"]], on="sample_id", suffixes=("_v1", "_v2"), validate="one_to_one"
    )
    assert joined["split_v1"].eq(joined["split_v2"]).all()
    assert joined["clean_eval_v1"].eq(joined["clean_eval_v2"]).all()
    assert v2["target_level1"].notna().all()
    assert v2["target_level2"].notna().all()
    assert v2["text_context_v2"].notna().all()
    assert v2["text_context_masked_v2"].notna().all()
    assert v2.loc[v2["context_added"], "text_context_v2"].str.startswith("Shared report context:").all()
    assert not v2["text_context_masked_v2"].str.contains(r"\bAS\s+\d{3,4}", case=False, regex=True).any()


def test_context_provenance_stays_inside_group():
    audit = pd.read_csv(ROOT / "results" / "tables" / "taxonomy_v2_context_audit.csv")
    added = audit.loc[audit["context_added"]]
    assert len(added) > 0
    source = audit[["sample_id", "report_id", "issuer_reference_key", "raw_audit_area"]].rename(
        columns={
            "sample_id": "context_source_sample_id",
            "report_id": "source_report_id",
            "issuer_reference_key": "source_issuer_reference_key",
            "raw_audit_area": "source_raw_audit_area",
        }
    )
    checked = added.merge(source, on="context_source_sample_id", validate="many_to_one")
    assert checked["report_id"].eq(checked["source_report_id"]).all()
    assert checked["issuer_reference_key"].astype(str).eq(checked["source_issuer_reference_key"].astype(str)).all()
    assert checked["raw_audit_area"].eq(checked["source_raw_audit_area"]).all()
