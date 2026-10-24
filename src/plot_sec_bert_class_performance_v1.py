"""Create a journal-ready classwise performance and confusion-matrix figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

NAVY = "#173B63"
BLUE = "#4C86B8"
SKY = "#56B4E9"
TEAL = "#00796B"
AMBER = "#B26A00"
SLATE = "#64748B"
GRID = "#D8E1EA"

LABELS = [
    "Business Combinations and Consolidation",
    "Cash and Other Assets",
    "Credit, Loans, Deposits and Insurance",
    "Debt, Equity and Liabilities",
    "Expenses, Compensation and Benefits",
    "Income Taxes",
    "Inventory and Cost of Sales",
    "Investments and Financial Instruments",
    "Long-Lived and Intangible Assets",
    "Related Parties and Significant Transactions",
    "Reporting, Controls, Estimates and Audit Procedures",
    "Revenue and Receivables",
]

SHORT = {
    LABELS[0]: "Business combinations",
    LABELS[1]: "Cash / other assets",
    LABELS[2]: "Credit / loans / deposits",
    LABELS[3]: "Debt / equity / liabilities",
    LABELS[4]: "Expenses / compensation",
    LABELS[5]: "Income taxes",
    LABELS[6]: "Inventory / cost of sales",
    LABELS[7]: "Investments / instruments",
    LABELS[8]: "Long-lived / intangibles",
    LABELS[9]: "Related parties / transactions",
    LABELS[10]: "Reporting / controls / estimates",
    LABELS[11]: "Revenue / receivables",
}

ABBR = ["BC", "Cash", "Credit", "Debt", "Exp", "Tax", "Inv", "Invest", "LLIA", "RP", "Report", "Rev"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    parser.add_argument("--scored", type=Path, default=TABLES / "sec_bert_frozen_test_scored_v1.csv")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scored)
    y_true = df["target_level1"]
    y_pred = df["prediction"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

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

    fig = plt.figure(figsize=(7.15, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.16], wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel A: metric profiles, with labels ordered by test support.
    order = np.argsort(support)
    yy = np.arange(len(LABELS))
    for pos, idx in enumerate(order):
        ax1.hlines(pos, min(precision[idx], recall[idx], f1[idx]),
                   max(precision[idx], recall[idx], f1[idx]), color="#CBD5DF", lw=1.1, zorder=1)
    ax1.scatter(precision[order], yy, marker="s", s=24, color=BLUE, label="Precision", zorder=3)
    ax1.scatter(recall[order], yy, marker="^", s=27, color=SKY, label="Recall", zorder=3)
    ax1.scatter(f1[order], yy, marker="o", s=25, color=NAVY, label="F1", zorder=4)
    ax1.set_yticks(
        yy,
        [f"{SHORT[LABELS[i]]}  (n={support[i]})" for i in order],
        fontsize=6.2,
    )
    ax1.set_xlim(0.42, 1.015)
    ax1.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.set_xlabel("Classwise score")
    ax1.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#C9D6E3",
               fontsize=6.5, ncol=3, handletextpad=0.35, columnspacing=0.7)
    ax1.set_title(
        "a  Per-class precision, recall and F1", loc="left", fontsize=9.4, fontweight="bold", pad=10,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#EAF2F8", "edgecolor": "#C9D6E3"},
    )
    ax1.set_facecolor("#F7FAFC")
    ax1.patch.set_edgecolor("#C9D6E3")
    ax1.patch.set_linewidth(1.0)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(SLATE)
    ax1.tick_params(colors="#25364A", labelsize=6.5)

    # Panel B: row-normalized confusion matrix. Only material off-diagonals are annotated.
    cmap = LinearSegmentedColormap.from_list("audit_blue", ["#FFFFFF", "#DCEAF4", "#7FB3D5", NAVY])
    image = ax2.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = cm_norm[i, j]
            if i == j or value >= 0.05:
                ax2.text(j, i, f"{100 * value:.0f}", ha="center", va="center",
                         fontsize=5.0, fontweight="bold" if i == j else "normal",
                         color="white" if value >= 0.58 else "#25364A")
    ax2.set_xticks(np.arange(len(LABELS)), ABBR, rotation=48, ha="right", fontsize=5.7)
    ax2.set_yticks(np.arange(len(LABELS)), ABBR, fontsize=5.7)
    ax2.set_xlabel("Predicted class")
    ax2.set_ylabel("PCAOB-derived target class")
    ax2.set_title(
        "b  Row-normalized confusion matrix (%)", loc="left", fontsize=9.4, fontweight="bold", pad=10,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#FBF3E6", "edgecolor": "#DCC9A8"},
    )
    ax2.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=0.7)
    ax2.tick_params(which="minor", bottom=False, left=False)
    for spine in ax2.spines.values():
        spine.set_color("#DCC9A8")
        spine.set_linewidth(1.0)
    cbar = fig.colorbar(image, ax=ax2, fraction=0.040, pad=0.025)
    cbar.set_label("Within-target share", fontsize=6.5)
    cbar.set_ticks([0, 0.5, 1.0], labels=["0%", "50%", "100%"])
    cbar.ax.tick_params(labelsize=5.8)

    fig.suptitle("SEC-BERT performance is strong but uneven across the 12-class taxonomy",
                 fontsize=10.8, fontweight="bold", color=NAVY)
    pdf_path = args.output_dir / f"sec_bert_class_performance_journal_{args.version}.pdf"
    png_path = args.png_dir / f"sec_bert_class_performance_journal_{args.version}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
