"""Plot RQ1 model comparison and lexical-cue masking sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
MASKING_METRICS = ROOT / "results" / "masking_v2" / "metrics" / "sec_bert_masking_v2.json"
FIGURES = ROOT / "results" / "figures"

NAVY = "#173B63"
TEAL = "#00796B"
AMBER = "#B26A00"
SKY = "#56B4E9"
SLATE = "#64748B"
GRID = "#D8E1EA"


def load_data() -> tuple[
    dict[str, list[float]], dict[str, tuple[float, float]], float
]:
    comparison = json.loads((METRICS / "model_comparison_v2.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        (METRICS / "sec_bert_frozen_retrospective_test_v1.json").read_text(encoding="utf-8")
    )
    masking = json.loads(MASKING_METRICS.read_text(encoding="utf-8"))
    lookup = {
        (row["representation"], row["variant"]): row for row in comparison["models"]
    }
    tfidf = lookup[("TF-IDF", "context")]
    minilm = lookup[("MiniLM", "context")]
    secbert = frozen["probability_metrics"]["temperature_scaled"]
    model_metrics = {
        "TF-IDF": [tfidf["accuracy"], tfidf["macro_f1"], tfidf["balanced_accuracy"]],
        "MiniLM": [minilm["accuracy"], minilm["macro_f1"], minilm["balanced_accuracy"]],
        "SEC-BERT": [secbert["accuracy"], secbert["macro_f1"], secbert["balanced_accuracy"]],
    }
    masked = {
        "TF-IDF": (tfidf["macro_f1"], lookup[("TF-IDF", "context_masked")]["macro_f1"]),
        "MiniLM": (minilm["macro_f1"], lookup[("MiniLM", "context_masked")]["macro_f1"]),
        "SEC-BERT": (
            masking["original_model"]["test_original_input"]["macro_f1"],
            masking["masked_model"]["test_masked_input"]["macro_f1"],
        ),
    }
    secbert_unadapted = masking["original_model"]["test_masked_input"]["macro_f1"]
    return model_metrics, masked, secbert_unadapted


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SLATE)
    ax.tick_params(colors="#25364A", labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.85, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    model_metrics, masked, secbert_unadapted = load_data()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#25364A",
            "text.color": "#182433",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 3.45), constrained_layout=True)

    # Panel A: three evaluation metrics for the three frozen model families.
    models = list(model_metrics)
    x = np.arange(len(models))
    width = 0.22
    series = [
        ("Accuracy", 0, NAVY),
        ("Macro-F1", 1, TEAL),
        ("Balanced accuracy", 2, AMBER),
    ]
    for offset, (label, idx, color) in zip((-width, 0, width), series):
        values = [model_metrics[m][idx] for m in models]
        bars = ax1.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.6,
                rotation=90,
                color="#25364A",
            )
    ax1.set_xticks(x, models)
    # Keep an intentionally empty upper band for the required upper-right legend.
    ax1.set_ylim(0, 1.30)
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.set_ylabel("Test score")
    ax1.set_title("a  Model performance", loc="left", fontsize=10.5)
    ax1.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        framealpha=1,
        fontsize=7.1,
        borderpad=0.45,
        handlelength=1.7,
    )
    style_axis(ax1)

    # Panel B: paired sensitivity test after masking explicit lexical cues.
    positions = np.array([0, 1])
    cue_colors = {"TF-IDF": NAVY, "MiniLM": SKY, "SEC-BERT": AMBER}
    cue_markers = {"TF-IDF": "o", "MiniLM": "s", "SEC-BERT": "^"}
    left_offsets = {"TF-IDF": 0.004, "MiniLM": -0.025, "SEC-BERT": 0.022}
    right_offsets = {"TF-IDF": 0.034, "MiniLM": -0.034, "SEC-BERT": -0.006}
    for name, values in masked.items():
        ax2.plot(
            positions,
            values,
            color=cue_colors[name],
            marker=cue_markers[name],
            markersize=6,
            linewidth=2.0,
            label=name,
            zorder=3,
        )
        ax2.text(
            -0.045,
            values[0] + left_offsets[name],
            f"{values[0]:.3f}",
            ha="right",
            va="bottom",
            fontsize=7.0,
            color=cue_colors[name],
        )
        ax2.text(
            1.035,
            values[1] + right_offsets[name],
            f"{values[1]:.3f}",
            ha="left",
            va="bottom",
            fontsize=7.0,
            color=cue_colors[name],
        )
    ax2.scatter(
        [1], [secbert_unadapted], marker="X", s=55, color="#A61B1B",
        edgecolor="white", linewidth=0.6, zorder=5, label="SEC-BERT, no retraining",
    )
    ax2.text(
        1.035, secbert_unadapted - 0.004, f"{secbert_unadapted:.3f}",
        ha="left", va="center", fontsize=7.0, color="#A61B1B", fontweight="bold",
    )
    ax2.set_xlim(-0.18, 1.42)
    ax2.set_ylim(0.20, 1.16)
    ax2.set_xticks(positions, ["Original cues", "Masked cues\n(matched training)"])
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("b  Lexical-cue sensitivity", loc="left", fontsize=10.5)
    ax2.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        framealpha=1,
        fontsize=6.8,
        borderpad=0.5,
    )
    style_axis(ax2)

    fig.suptitle(
        "Classifier quality and dependence on explicit audit-area cues",
        fontsize=11.3,
        fontweight="bold",
        color="#173B63",
    )
    pdf_path = FIGURES / "model_comparison_sensitivity_journal_v1.pdf"
    png_path = args.png_dir / "model_comparison_sensitivity_journal_v1.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
