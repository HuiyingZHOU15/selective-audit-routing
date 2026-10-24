import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fetch_pcaob_standards_v1 import extract_paragraphs
from prepare_pcaob_rag_pilot_v1 import standard_ids, top_texts
from run_pcaob_rag_pilot_v1 import PilotDecision, schema


def test_standard_ids():
    assert standard_ids("AS 1105 | AS 2201") == ["1105", "2201"]


def test_top_texts_prefers_overlap():
    ranked = top_texts("inventory valuation", ["revenue recognition", "inventory valuation reserve"], 1)
    assert ranked[0][0] == 1


def test_extract_standard_paragraphs():
    html = "<main><p>.01 This is a sufficiently long audit paragraph for testing.</p><p>.02 Another sufficiently long audit paragraph for testing.</p></main>"
    paragraphs = extract_paragraphs("AS1105", html, "https://example.test")
    assert [item["paragraph"] for item in paragraphs] == [".01", ".02"]


def test_decision_and_schema():
    labels = ["Revenue and Receivables"]
    decision = PilotDecision(
        predicted_label=labels[0], confidence=0.8, evidence_quote="revenue",
        evidence_source_id="target", reason="topic", standard_citation=None, abstain=False,
    )
    assert decision.predicted_label == labels[0]
    assert schema(labels)["additionalProperties"] is False
