"""Compare risk-ranking scores using AURC and fixed-budget error capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

NAVY = "#173B63"
TEAL = "#00796B"
AMBER = "#B26A00"
SKY = "#56B4E9"
GRAY = "#9AA5B1"
SLATE = "#64748B"
GRID = "#D8E1EA"

DISPLAY = {
    "random": "Random",
    "class_aware_logistic": "Learned router",
    "entropy": "Entropy",
    "confidence": "Confidence",
    "margin": "Top-two margin",
    "oracle": "Oracle bound",
}
COLORS = {
    "random": GRAY,
    "class_aware_logistic": AMBER,
    "entropy": SKY,
    "confidence": "#4C86B8",
    "margin": NAVY,
    "oracle": TEAL,
}
MARKERS = {
    "random": "x",
    "class_aware_logistic": "D",
    "entropy": "^",
    "confidence": "s",
    "margin": "o",
    "oracle": "*",
}


def style_axis(ax: plt.Axes, face: str, edge: str) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.85, zorder=0)
    ax.tick_params(colors="#25364A", labelsize=7.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SLATE)
    ax.set_facecolor(face)
    ax.patch.set_edgecolor(edge)
    ax.patch.set_linewidth(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((METRICS / "sec_bert_risk_router_v1.json").read_text(encoding="utf-8"))
    budgets = pd.read_csv(TABLES / "sec_bert_risk_router_budgets_v1.csv")
    fixed = budgets[np.isclose(budgets["requested_referral_budget"], 0.05)].set_index("method")
    aurc = {row["method"]: row["aurc_discrete"] for row in metrics["aurc"]}
    bootstrap = metrics["primary_5pct_report_cluster_bootstrap"]
    order = ["random", "class_aware_logistic", "entropy", "confidence", "margin", "oracle"]
    y = np.arange(len(order))

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 3.55), constrained_layout=True)

    # Panel A: global ranking quality.
    for yy, method in zip(y, order):
        value = aurc[method]
        ax1.hlines(yy, 0.002, value, color="#CBD5DF", linewidth=1.2, zorder=1)
        ax1.scatter(value, yy, s=46 if method != "oracle" else 70, color=COLORS[method],
                    marker=MARKERS[method], zorder=4)
        ax1.text(value * 1.08, yy, f"{value:.4f}", va="center", ha="left",
                 fontsize=6.8, color=COLORS[method], fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlim(0.0018, 0.095)
    ax1.set_ylim(-0.7, len(order) + 1.4)
    ax1.set_yticks(y, [DISPLAY[m] for m in order])
    ax1.set_xlabel("Discrete AURC (log scale; lower is better)")
    ax1.set_title(
        "a  Global risk–coverage ranking", loc="left", fontsize=10.5, pad=12,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EAF2F8", "edgecolor": "#C9D6E3"},
    )
    ax1.annotate("better", xy=(0.0023, len(order) + 0.65), xytext=(0.012, len(order) + 0.65),
                 arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.2},
                 ha="left", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
    style_axis(ax1, "#F7FAFC", "#C9D6E3")

    # Panel B: practical fixed-budget outcome with clustered uncertainty.
    for yy, method in zip(y, order):
        observed = 100 * fixed.loc[method, "error_capture_rate"]
        label_x = observed + 1.7
        if method in bootstrap:
            low, _, high = [100 * x for x in bootstrap[method]["error_capture_rate"]]
            ax2.errorbar(observed, yy, xerr=[[observed - low], [high - observed]], fmt="none",
                         ecolor=COLORS[method], elinewidth=1.5, capsize=3, zorder=2)
            # Keep the numeric label clear of both the marker and the CI cap.
            label_x = high + 2.0
        ax2.scatter(observed, yy, s=46 if method != "oracle" else 70, color=COLORS[method],
                    marker=MARKERS[method], zorder=4)
        ax2.text(label_x, yy, f"{observed:.1f}%", va="center", ha="left",
                 fontsize=6.8, color=COLORS[method], fontweight="bold")
    ax2.set_xlim(0, 86)
    ax2.set_ylim(-0.7, len(order) + 1.4)
    ax2.set_yticks(y, [DISPLAY[m] for m in order])
    ax2.set_xlabel("Disagreements captured at 5% referral")
    ax2.set_title(
        "b  Fixed-budget error capture", loc="left", fontsize=10.5, pad=12,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FBF3E6", "edgecolor": "#DCC9A8"},
    )
    ax2.text(84, len(order) + 0.65, "95% report-cluster bootstrap CI",
             ha="right", va="center", fontsize=6.7, color=SLATE)
    style_axis(ax2, "#FCFAF6", "#DCC9A8")

    fig.suptitle("Simple uncertainty scores match or exceed the learned router",
                 fontsize=11.3, fontweight="bold", color=NAVY)
    pdf_path = FIGURES / "risk_score_comparison_journal_v1.pdf"
    png_path = args.png_dir / "risk_score_comparison_journal_v1.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
