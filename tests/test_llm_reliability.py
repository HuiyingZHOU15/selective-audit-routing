import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_common import LLMDecision, evidence_match, render_user_prompt  # noqa: E402


def test_frozen_candidates_match_routed_test_set():
    config = json.loads((ROOT / "configs" / "llm_deepseek_v1.json").read_text(encoding="utf-8"))
    routes = pd.read_csv(ROOT / config["routing_table"])
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    expected = routes.loc[routes[config["route_column"]].astype(bool), "sample_id"].astype(str)
    assert len(candidates) == len(expected) == 167
    assert set(candidates["sample_id"].astype(str)) == set(expected)
    assert candidates["user_prompt_sha256"].nunique() == len(candidates)
    assert candidates["prompt_bundle_sha256"].nunique() == len(candidates)


def test_prompt_renderer_has_no_ground_truth_parameter():
    prompt = render_user_prompt("The firm did not test journal entries.", ["Other", "Journal Entries"], {
        "Other": "None fits.",
        "Journal Entries": "Journal-entry testing.",
    })
    assert "FINDING TEXT" in prompt
    assert "target_label" not in prompt


def test_exact_and_normalized_evidence_checks():
    text = "The firm did not sufficiently test journal entries."
    assert evidence_match(text, "test journal entries") == (True, True)
    assert evidence_match(text, "test   journal entries") == (False, True)
    assert evidence_match(text, "invented evidence") == (False, False)


def test_closed_book_decision_schema():
    decision = LLMDecision.model_validate(
        {
            "predicted_label": "Journal Entries",
            "confidence": 0.9,
            "evidence_quote": "journal entries",
            "reason": "Direct topic match.",
            "standard_citation": None,
            "abstain": False,
        }
    )
    assert decision.predicted_label == "Journal Entries"
