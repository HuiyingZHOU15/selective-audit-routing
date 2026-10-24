"""Plot class-level referral allocation and captured disagreements."""

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
GRAY = "#A7B0BA"
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


def style_axis(ax: plt.Axes, face: str, edge: str) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.85, zorder=0)
    ax.tick_params(colors="#25364A", labelsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SLATE)
    ax.set_facecolor(face)
    ax.patch.set_edgecolor(edge)
    ax.patch.set_linewidth(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=FIGURES)
    parser.add_argument("--scored", type=Path, default=TABLES / "sec_bert_frozen_test_scored_v1.csv")
    parser.add_argument("--metrics", type=Path, default=METRICS / "sec_bert_frozen_retrospective_test_v1.json")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.png_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(args.scored)
    scored["correct"] = scored["correct"].astype(str).str.lower().eq("true")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if "retrospective_test_operating_points" in metrics:
        rows = metrics["retrospective_test_operating_points"]
        thresholds = {"primary_5pct": rows["high_coverage"]["threshold"], "safety_10pct": rows["conservative"]["threshold"]}
    else:
        thresholds = {row["operating_point"]: row["threshold"] for row in metrics["operating_points"]}
    scored["primary"] = scored["calibrated_margin"] < thresholds["primary_5pct"]
    scored["safety"] = scored["calibrated_margin"] < thresholds["safety_10pct"]
    expected = (32, 63) if args.version == "v3" else (38, 76)
    assert int(scored["primary"].sum()) == expected[0]
    assert int(scored["safety"].sum()) == expected[1]

    rows = []
    for prediction, group in scored.groupby("prediction"):
        error = ~group["correct"]
        rows.append(
            {
                "prediction": prediction,
                "label": SHORT[prediction],
                "samples": len(group),
                "errors": int(error.sum()),
                "error_rate": error.mean(),
                "primary_rate": group["primary"].mean(),
                "safety_rate": group["safety"].mean(),
                "primary_captured": int((error & group["primary"]).sum()),
                "safety_captured": int((error & group["safety"]).sum()),
            }
        )
    data = pd.DataFrame(rows).sort_values(["error_rate", "samples"])
    data["safety_increment"] = data["safety_captured"] - data["primary_captured"]
    data["remaining"] = data["errors"] - data["safety_captured"]

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
        1, 2, figsize=(7.15, 4.5), gridspec_kw={"width_ratios": [1.15, 1.00]},
        constrained_layout=True,
    )
    y = np.arange(len(data))

    # Panel A: risk burden versus allocated review rate.
    ax1.hlines(
        y, 100 * data[["error_rate", "primary_rate", "safety_rate"]].min(axis=1),
        100 * data[["error_rate", "primary_rate", "safety_rate"]].max(axis=1),
        color="#C8D3DE", linewidth=1.0, zorder=1,
    )
    ax1.scatter(100 * data["error_rate"], y + 0.18, s=30, color=NAVY, marker="o", label="Observed error rate", zorder=4)
    ax1.scatter(100 * data["primary_rate"], y, s=30, color=TEAL, marker="s", label="High-coverage referral", zorder=4)
    ax1.scatter(100 * data["safety_rate"], y - 0.18, s=34, color=AMBER, marker="^", label="Conservative referral", zorder=4)
    ax1.set_yticks(y, data["label"])
    ax1.set_xlim(0, 55)
    ax1.set_ylim(-0.65, len(data) + 2.5)
    ax1.set_xlabel("Share within predicted class (%)")
    ax1.set_title(
        "a  Risk and referral allocation", loc="left", fontsize=10.5, pad=12,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#EAF2F8", "edgecolor": "#C9D6E3"},
    )
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID,
               framealpha=1, fontsize=6.7, borderpad=0.45)
    style_axis(ax1, "#F7FAFC", "#C9D6E3")

    # Panel B: nested capture under the two frozen policies.
    left_primary = np.zeros(len(data))
    left_increment = data["primary_captured"].to_numpy()
    left_remaining = left_increment + data["safety_increment"].to_numpy()
    ax2.barh(y, data["primary_captured"], color=TEAL, edgecolor="white", height=0.62,
             label="Captured at high coverage", zorder=3)
    ax2.barh(y, data["safety_increment"], left=left_increment, color=AMBER, edgecolor="white",
             height=0.62, label="Additional when conservative", zorder=3)
    ax2.barh(y, data["remaining"], left=left_remaining, color=GRAY, edgecolor="white",
             height=0.62, label="Remaining after conservative", zorder=3)
    for yy, total in zip(y, data["errors"]):
        if total > 0:
            ax2.text(total + 0.18, yy, str(int(total)), va="center", ha="left",
                     fontsize=6.8, color=NAVY, fontweight="bold")
    ax2.set_yticks(y, data["label"])
    ax2.set_xlim(0, max(15, int(data["errors"].max()) + 2))
    ax2.set_ylim(-0.65, len(data) + 2.5)
    ax2.set_xlabel("SEC-BERT disagreements (count)")
    ax2.set_title(
        "b  Captured disagreements", loc="left", fontsize=10.5, pad=12,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FBF3E6", "edgecolor": "#DCC9A8"},
    )
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID,
               framealpha=1, fontsize=6.5, borderpad=0.45)
    style_axis(ax2, "#FCFAF6", "#DCC9A8")

    fig.suptitle("Class-level allocation of the frozen verification queue",
                 fontsize=11.3, fontweight="bold", color=NAVY)
    pdf_path = args.output_dir / f"class_referral_allocation_journal_{args.version}.pdf"
    png_path = args.png_dir / f"class_referral_allocation_journal_{args.version}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
