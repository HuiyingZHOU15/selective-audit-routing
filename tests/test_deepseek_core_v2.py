import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_common import render_user_prompt  # noqa: E402


def test_prompt_does_not_contain_ground_truth_field():
    definitions = json.loads((ROOT / "configs" / "llm_label_definitions_v2.json").read_text(encoding="utf-8"))
    prompt = render_user_prompt("Example finding text.", list(definitions), definitions)
    assert "target_level1" not in prompt
    assert "Example finding text." in prompt


def test_candidates_are_exact_clean_test_population_if_generated():
    path = ROOT / "data" / "interim" / "deepseek_core_v2_candidates.csv"
    if path.exists():
        frame = pd.read_csv(path)
        assert len(frame) == 890
        assert frame["sample_id"].is_unique
        assert {"route_10pct", "route_20pct", "route_30pct"}.issubset(frame.columns)
