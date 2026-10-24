"""Plot descriptive diagnostic value of observable ambiguity signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

NAVY = "#173B63"
TEAL = "#00796B"
AMBER = "#B26A00"
SLATE = "#8A98A8"
GRID = "#D8E1EA"

NAMES = {
    "tfidf_disagreement": "TF–IDF disagrees",
    "multi_topic": "Multiple audit topics",
    "context_unresolved": "Context unresolved",
    "context_needs_context": "Context reference present",
    "crosscut_hits_ge_2": "≥2 cross-cutting cues",
    "context_added": "Report context restored",
}


def style_axis(ax: plt.Axes, face: str, edge: str) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_facecolor(face)
    ax.patch.set_edgecolor(edge)
    ax.patch.set_linewidth(1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#64748B")
    ax.tick_params(colors="#25364A", labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    payload = json.loads(
        (METRICS / "label_ambiguity_analysis_v1.json").read_text(encoding="utf-8")
    )
    records = payload["ambiguity_features_clean_oof"]
    grouped: dict[str, dict[str, dict]] = {}
    for row in records:
        grouped.setdefault(row["feature"], {})[row["level"]] = row

    features = sorted(
        grouped,
        key=lambda key: grouped[key]["present"]["error_rate_ratio_present_vs_absent"],
        reverse=True,
    )
    y = np.arange(len(features))
    bar_h = 0.30

    present_error = np.array([100 * grouped[f]["present"]["error_rate"] for f in features])
    absent_error = np.array([100 * grouped[f]["absent"]["error_rate"] for f in features])
    present_ref = np.array([100 * grouped[f]["present"]["referral_rate_at_global_5pct"] for f in features])
    absent_ref = np.array([100 * grouped[f]["absent"]["referral_rate_at_global_5pct"] for f in features])
    ratios = np.array([grouped[f]["present"]["error_rate_ratio_present_vs_absent"] for f in features])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelcolor": "#25364A",
            "text.color": "#182433",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 3.55), constrained_layout=True)

    # Panel A: descriptive error association.
    ax1.barh(y - bar_h / 2, present_error, height=bar_h, color=TEAL,
             label="Feature present", zorder=3)
    ax1.barh(y + bar_h / 2, absent_error, height=bar_h, color=SLATE,
             label="Feature absent", zorder=3)
    for i, (pval, aval, ratio) in enumerate(zip(present_error, absent_error, ratios)):
        ax1.text(pval + 0.45, i - bar_h / 2, f"{pval:.1f}", va="center", ha="left",
                 fontsize=6.1, color=TEAL, fontweight="bold")
        ax1.text(aval + 0.45, i + bar_h / 2, f"{aval:.1f}", va="center", ha="left",
                 fontsize=6.1, color="#667585", fontweight="bold")
        ax1.text(41.8, i, f"RR {ratio:.2f}×", va="center", ha="right",
                 fontsize=6.1, color=AMBER, fontweight="bold")
    ax1.set_yticks(y, [NAMES[f] for f in features], fontsize=6.6)
    ax1.set_xlim(0, 43)
    ax1.set_ylim(len(features) - 0.55, -1.05)
    ax1.set_xlabel("SEC-BERT disagreement rate (%)")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#C9D6E3",
               fontsize=6.3, ncol=2, columnspacing=0.8, handlelength=1.3)
    ax1.set_title(
        "a  Association with label disagreement", loc="left", fontsize=9.5,
        fontweight="bold", pad=11,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#EAF2F8", "edgecolor": "#C9D6E3"},
    )
    style_axis(ax1, "#F7FAFC", "#C9D6E3")

    # Panel B: how the same groups are represented in the global 5% queue.
    ax2.barh(y - bar_h / 2, present_ref, height=bar_h, color=TEAL,
             label="Feature present", zorder=3)
    ax2.barh(y + bar_h / 2, absent_ref, height=bar_h, color=SLATE,
             label="Feature absent", zorder=3)
    for i, (pval, aval) in enumerate(zip(present_ref, absent_ref)):
        ax2.text(pval + 0.45, i - bar_h / 2, f"{pval:.1f}", va="center", ha="left",
                 fontsize=6.1, color=TEAL, fontweight="bold")
        ax2.text(aval + 0.45, i + bar_h / 2, f"{aval:.1f}", va="center", ha="left",
                 fontsize=6.1, color="#667585", fontweight="bold")
    ax2.set_yticks(y, [NAMES[f] for f in features], fontsize=6.6)
    ax2.set_xlim(0, 36)
    ax2.set_ylim(len(features) - 0.55, -1.05)
    ax2.set_xlabel("Referred within group at global 5% budget (%)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DCC9A8",
               fontsize=6.3, ncol=2, columnspacing=0.8, handlelength=1.3)
    ax2.set_title(
        "b  Allocation of the fixed referral queue", loc="left", fontsize=9.5,
        fontweight="bold", pad=11,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FBF3E6", "edgecolor": "#DCC9A8"},
    )
    style_axis(ax2, "#FCFAF6", "#DCC9A8")

    fig.suptitle("Cross-model disagreement dominates simple text-based ambiguity flags",
                 fontsize=10.8, fontweight="bold", color=NAVY)
    pdf_path = FIGURES / "ambiguity_signal_diagnostics_journal_v1.pdf"
    png_path = args.png_dir / "ambiguity_signal_diagnostics_journal_v1.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
