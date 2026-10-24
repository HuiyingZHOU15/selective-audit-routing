import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_v2_models import mean_pool, score_metrics  # noqa: E402


def test_mean_pool_masks_padding_and_normalizes():
    states = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(states, mask)
    expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_score_metrics_are_bounded():
    scores = score_metrics(np.array(["a", "b", "b"]), np.array(["a", "a", "b"]))
    assert set(scores) == {"accuracy", "macro_f1", "balanced_accuracy"}
    assert all(0 <= value <= 1 for value in scores.values())


def test_modeling_config_preserves_calibration_split():
    config = json.loads((ROOT / "configs" / "modeling_v2.json").read_text(encoding="utf-8"))
    assert "calibration remains untouched" in config["final_fit"]
    assert config["target"] == "target_level1"
    assert config["embedding"]["cache_tag"] == "explicit_bert_vocab_v1"
