"""Plot dataset construction and final 12-class distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
    "Business Combinations and Consolidation": "Business combinations",
    "Cash and Other Assets": "Cash / other assets",
    "Credit, Loans, Deposits and Insurance": "Credit / loans",
    "Debt, Equity and Liabilities": "Debt / equity",
    "Expenses, Compensation and Benefits": "Expenses / compensation",
    "Income Taxes": "Income taxes",
    "Inventory and Cost of Sales": "Inventory / cost of sales",
    "Investments and Financial Instruments": "Investments",
    "Long-Lived and Intangible Assets": "Long-lived / intangibles",
    "Related Parties and Significant Transactions": "Related parties",
    "Reporting, Controls, Estimates and Audit Procedures": "Reporting / controls",
    "Revenue and Receivables": "Revenue / receivables",
}


def add_stage(ax: plt.Axes, y: float, width: float, color: str, count: str, label: str, note: str = "") -> None:
    x = 0.08 + (0.78 - width) / 2
    box = FancyBboxPatch(
        (x, y), width, 0.17,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=color, edgecolor="white", linewidth=1.1,
        transform=ax.transAxes, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + 0.107, count, transform=ax.transAxes,
            ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    ax.text(x + width / 2, y + 0.055, label, transform=ax.transAxes,
            ha="center", va="center", color="white", fontsize=7.2)
    if note:
        ax.text(x + width / 2, y + 0.018, note, transform=ax.transAxes,
                ha="center", va="center", color="white", fontsize=5.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(TABLES / "taxonomy_v2_counts.csv")
    totals = counts.groupby("target_level1", as_index=False)["samples"].sum()
    assert int(totals["samples"].sum()) == 6607
    totals["label"] = totals["target_level1"].map(SHORT)
    totals = totals.sort_values("samples")

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
        1, 2, figsize=(7.15, 4.15), gridspec_kw={"width_ratios": [0.92, 1.18]},
        constrained_layout=True,
    )

    # Panel A: auditable construction funnel.
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_facecolor("#F7FAFC")
    ax1.set_title(
        "a  Dataset construction", loc="left", fontsize=10.5, pad=12,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EAF2F8", "edgecolor": "#C9D6E3"},
    )
    add_stage(ax1, 0.72, 0.78, NAVY, "14,043", "official source rows")
    add_stage(ax1, 0.43, 0.66, TEAL, "7,467", "finding-level records")
    add_stage(ax1, 0.06, 0.55, AMBER, "6,607", "primary single-label task", "650 reports; 391 firms")
    for x, y1, y2 in [(0.47, 0.71, 0.61), (0.30, 0.42, 0.24)]:
        ax1.add_patch(FancyArrowPatch(
            (x, y1), (x, y2), transform=ax1.transAxes,
            arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=SLATE,
        ))
    ax1.text(0.64, 0.34, "860 excluded", transform=ax1.transAxes,
             ha="center", va="center", fontsize=7.4, color=SLATE, fontweight="bold")
    ax1.text(0.64, 0.295, "615 multi-label  ·  245 target conflicts", transform=ax1.transAxes,
             ha="center", va="center", fontsize=6.2, color=NAVY)
    ax1.add_patch(FancyBboxPatch(
        (0.015, 0.02), 0.97, 0.92, transform=ax1.transAxes,
        boxstyle="round,pad=0.012", facecolor="none", edgecolor="#C9D6E3", linewidth=1.0,
    ))

    # Panel B: final target distribution.
    n = len(totals)
    colors = [SKY] * n
    colors[-1] = NAVY
    colors[0] = AMBER
    bars = ax2.barh(totals["label"], totals["samples"], color=colors, edgecolor="white", height=0.68, zorder=3)
    for bar, value in zip(bars, totals["samples"]):
        ax2.text(value + 28, bar.get_y() + bar.get_height() / 2, f"{int(value):,}",
                 ha="left", va="center", fontsize=7.0, color=NAVY, fontweight="bold")
    ax2.set_xlim(0, 2500)
    ax2.set_xlabel("Findings in modeling population")
    ax2.set_title(
        "b  Final 12-class distribution", loc="left", fontsize=10.5, pad=12,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FBF3E6", "edgecolor": "#DCC9A8"},
    )
    ax2.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.85, zorder=0)
    ax2.tick_params(colors="#25364A", labelsize=7.2)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(SLATE)
    ax2.set_facecolor("#FCFAF6")
    ax2.patch.set_edgecolor("#DCC9A8")
    ax2.patch.set_linewidth(1.0)

    fig.suptitle("From public inspection rows to the modeling dataset",
                 fontsize=11.3, fontweight="bold", color=NAVY)
    pdf_path = FIGURES / "dataset_construction_distribution_journal_v1.pdf"
    png_path = args.png_dir / "dataset_construction_distribution_journal_v1.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
