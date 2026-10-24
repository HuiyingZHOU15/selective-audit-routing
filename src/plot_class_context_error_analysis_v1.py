"""Plot context-stratified routing signals and the leading SEC-BERT confusions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

NAVY = "#173B63"
TEAL = "#00796B"
AMBER = "#B26A00"
SKY = "#56B4E9"
SLATE = "#64748B"
GRID = "#D8E1EA"


SHORT = {
    "Investments and Financial Instruments": "Investments",
    "Cash and Other Assets": "Cash / other assets",
    "Credit, Loans, Deposits and Insurance": "Credit / loans",
    "Business Combinations and Consolidation": "Business combinations",
    "Reporting, Controls, Estimates and Audit Procedures": "Reporting / controls",
    "Related Parties and Significant Transactions": "Related parties",
    "Debt, Equity and Liabilities": "Debt / equity",
    "Long-Lived and Intangible Assets": "Long-lived / intangibles",
}


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SLATE)
    ax.tick_params(colors="#25364A", labelsize=7.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.85, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    parser.add_argument("--context", type=Path, default=TABLES / "sec_bert_context_quality_test_v1.csv")
    parser.add_argument("--confusion", type=Path, default=TABLES / "sec_bert_confusion_pairs_test_v1.csv")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    context = pd.read_csv(args.context)
    confusion = pd.read_csv(args.confusion).head(8).copy()
    if "high_coverage_referral_rate" in context:
        context = context.rename(columns={
            "high_coverage_referral_rate": "primary_referral_rate",
            "conservative_referral_rate": "safety_referral_rate",
        })

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
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.15, 4.15), gridspec_kw={"width_ratios": [1.18, 1.00]},
        constrained_layout=True,
    )

    # Panel A: context flags versus empirical risk and frozen referral behavior.
    group_order = ["context_added", "context_unresolved", "original_sufficient"]
    c = context.set_index("context_group").loc[group_order]
    base_labels = ["Context restored", "Context unresolved", "Original sufficient"]
    labels = [
        f"{name}\nn={int(n)}; errors={int(errors)}"
        for name, n, errors in zip(base_labels, c["samples"], c["errors"])
    ]
    y = np.array([2, 1, 0], dtype=float)
    offsets = [0.25, 0.0, -0.25]
    measures = [
        ("Observed error rate", 100 * (1 - c["accuracy"].to_numpy()), NAVY, "o"),
        ("High-coverage referral", 100 * c["primary_referral_rate"].to_numpy(), TEAL, "s"),
        ("Conservative referral", 100 * c["safety_referral_rate"].to_numpy(), AMBER, "^"),
    ]
    for offset, (name, values, color, marker) in zip(offsets, measures):
        ax1.scatter(values, y + offset, s=34, color=color, marker=marker, label=name, zorder=4)
        for value, yy in zip(values, y + offset):
            ax1.text(value + 0.52, yy, f"{value:.1f}", va="center", ha="left", fontsize=6.7, color=color)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, 15.0)
    ax1.set_ylim(-0.55, 4.15)
    ax1.set_xlabel("Share of subgroup (%)")
    ax1.set_title(
        "a  Context flags and routing", loc="left", fontsize=10.5, pad=30,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EAF2F8", "edgecolor": "#C9D6E3"},
    )
    for tick in ax1.get_yticklabels():
        tick.set_multialignment("right")
    ax1.legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor=GRID,
        framealpha=1, fontsize=6.8, borderpad=0.45, handletextpad=0.45,
    )
    style_axis(ax1, "x")
    ax1.set_facecolor("#F7FAFC")
    ax1.patch.set_edgecolor("#C9D6E3")
    ax1.patch.set_linewidth(1.0)

    # Panel B: most frequent directional class confusions and their margins.
    confusion = confusion.iloc[::-1].reset_index(drop=True)
    pair_labels = [
        f"{SHORT.get(t, t)}  →  {SHORT.get(p, p)}"
        for t, p in zip(confusion["target_level1"], confusion["prediction"])
    ]
    yy = np.arange(len(confusion))
    bars = ax2.barh(yy, confusion["errors"], height=0.58, color=NAVY, edgecolor="white", zorder=3)
    for bar, value in zip(bars, confusion["errors"]):
        ax2.text(value + 0.10, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=7.0, color=NAVY, fontweight="bold")
    ax2.set_yticks(yy, pair_labels)
    ax2.set_xlim(0, 7.2)
    ax2.set_ylim(-0.7, len(confusion) + 2.2)
    ax2.set_xlabel("Number of test disagreements")
    ax2.set_title(
        "b  Leading directional confusions", loc="left", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FBF3E6", "edgecolor": "#DCC9A8"},
    )
    style_axis(ax2, "x")
    ax2.set_facecolor("#FCFAF6")
    ax2.patch.set_edgecolor("#DCC9A8")
    ax2.patch.set_linewidth(1.0)

    ax2_margin = ax2.twiny()
    ax2_margin.scatter(
        confusion["mean_calibrated_margin"], yy + 0.20, s=34, color=SKY,
        edgecolor="white", linewidth=0.6, zorder=5,
    )
    ax2_margin.set_xlim(0, 1.18)
    ax2_margin.set_xlabel("Mean calibrated margin", color=SKY, fontsize=8)
    ax2_margin.tick_params(axis="x", colors=SKY, labelsize=7.2)
    ax2_margin.spines["top"].set_color(SKY)
    ax2_margin.spines[["left", "right", "bottom"]].set_visible(False)
    ax2.legend(
        handles=[
            Patch(facecolor=NAVY, label="Disagreement count"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=SKY,
                   markeredgecolor="white", markersize=6, label="Mean margin"),
        ],
        loc="upper right", frameon=True, facecolor="white", edgecolor=GRID,
        framealpha=1, fontsize=6.8, borderpad=0.45,
    )

    fig.suptitle(
        "Where classification risk concentrates",
        fontsize=11.3, fontweight="bold", color=NAVY,
    )
    pdf_path = args.output_dir / f"class_context_error_analysis_journal_{args.version}.pdf"
    png_path = args.png_dir / f"class_context_error_analysis_journal_{args.version}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
