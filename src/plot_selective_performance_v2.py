"""Create the manuscript's selective-performance figure from frozen result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"


def main() -> None:
    curves = pd.read_csv(TABLES / "sec_bert_risk_coverage_curve_v1.csv")
    frozen = json.loads(
        (METRICS / "sec_bert_frozen_retrospective_test_v1.json").read_text(
            encoding="utf-8"
        )
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    navy = "#173B63"
    blue = "#2E6F9E"
    sky = "#56B4E9"
    teal = "#00796B"
    amber = "#B26A00"
    slate = "#64748B"
    pale = "#EAF3FA"

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), constrained_layout=True)

    # Panel A: all risk scores were selected/evaluated on report-grouped OOF data.
    ax = axes[0]
    styles = {
        "margin": (navy, "-", 2.0, "Top-two margin (selected)"),
        "confidence": (sky, "--", 1.35, "Confidence"),
        "entropy": (teal, ":", 1.55, "Entropy"),
        "class_aware_logistic": (amber, "-.", 1.35, "Learned risk score"),
        "random": (slate, (0, (1, 2)), 1.15, "Random referral"),
    }
    for method, (color, linestyle, width, label) in styles.items():
        group = curves.loc[
            (curves["method"] == method) & curves["coverage"].between(0.80, 1.00)
        ].sort_values("coverage")
        ax.plot(
            group["coverage"] * 100,
            group["selective_accuracy"] * 100,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=label,
            zorder=3 if method == "margin" else 2,
        )

    ax.axhline(95, color="#202020", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(95, color=slate, linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)
    annotation_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88,
                      "pad": 1.2}
    ax.text(80.4, 95.58, "95% accuracy", color="#202020", fontsize=6.8,
            bbox=annotation_box)
    ax.text(95.52, 90.35, "95% coverage", color=slate, fontsize=6.8, rotation=90,
            bbox=annotation_box)
    ax.set_xlim(80, 100)
    # Reserve the upper band for the legend so it never covers a curve.
    ax.set_ylim(90, 100.6)
    ax.set_xlabel("Automatic coverage (%)")
    ax.set_ylabel("Selective accuracy (%)")
    ax.set_title("Development OOF error ranking", loc="left", fontweight="bold")
    ax.legend(frameon=True, ncol=2, loc="upper right", handlelength=2.5,
              columnspacing=1.0, borderaxespad=0.45, fancybox=False,
              facecolor="white", edgecolor="#D7DEE6", framealpha=0.96)

    # Panel B: frozen policy outcomes on the unchanged retrospective test.
    ax = axes[1]
    base_accuracy = frozen["probability_metrics"]["temperature_scaled"]["accuracy"]
    test_points = [
        {
            "name": "Full automation",
            "coverage": 1.0,
            "accuracy": base_accuracy,
            "color": navy,
            "marker": "D",
            "referral": "0% referred",
        }
    ]
    names = {
        "primary_5pct": ("High coverage", teal, "o", "4.27% referred"),
        "safety_10pct": ("Conservative", amber, "s", "8.54% referred"),
    }
    for row in frozen["operating_points"]:
        display, color, marker, referral = names[row["operating_point"]]
        ci = frozen["report_cluster_bootstrap_95"][row["operating_point"]]
        test_points.append(
            {
                "name": display,
                "coverage": row["coverage"],
                "accuracy": row["selective_accuracy"],
                "color": color,
                "marker": marker,
                "referral": referral,
                "coverage_ci": (ci["coverage"][0], ci["coverage"][2]),
                "accuracy_ci": (ci["selective_accuracy"][0], ci["selective_accuracy"][2]),
            }
        )

    order = sorted(test_points, key=lambda item: item["coverage"])
    ax.plot(
        [point["coverage"] * 100 for point in order],
        [point["accuracy"] * 100 for point in order],
        color=slate,
        linewidth=0.85,
        zorder=1,
    )
    for point in test_points:
        x = point["coverage"] * 100
        y = point["accuracy"] * 100
        if "coverage_ci" in point:
            x_low, x_high = point["coverage_ci"]
            y_low, y_high = point["accuracy_ci"]
            ax.errorbar(
                x,
                y,
                xerr=np.array([[x - x_low * 100], [x_high * 100 - x]]),
                yerr=np.array([[y - y_low * 100], [y_high * 100 - y]]),
                fmt="none",
                ecolor=point["color"],
                elinewidth=1.0,
                capsize=2.2,
                zorder=2,
            )
        ax.scatter(
            x,
            y,
            s=46,
            marker=point["marker"],
            facecolor=point["color"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    annotations = {
        "Full automation": (99.15, 92.05, "right"),
        "High coverage": (97.20, 96.20, "left"),
        "Conservative": (92.15, 98.55, "left"),
    }
    for point in test_points:
        tx, ty, align = annotations[point["name"]]
        ax.text(
            tx,
            ty,
            f"{point['name']}\n{point['referral']}",
            ha=align,
            va="center",
            color=point["color"],
            fontsize=7,
            fontweight="bold" if point["name"] != "Full automation" else "normal",
            bbox=annotation_box,
        )

    ax.axhline(95, color="#202020", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvspan(90, 100, color=pale, alpha=0.28, zorder=-2)
    ax.set_xlim(90, 100.5)
    ax.set_ylim(91.5, 99.2)
    ax.set_xlabel("Automatic coverage (%)")
    ax.set_ylabel("Selective accuracy (%)")
    ax.set_title("Frozen retrospective-test policies", loc="left", fontweight="bold")
    ax.text(90.2, 95.48, "95% accuracy", color="#202020", fontsize=6.8,
            bbox=annotation_box)

    for panel, ax in zip(("a", "b"), axes):
        ax.text(
            -0.14,
            1.07,
            panel,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D7DEE6", linewidth=0.45, alpha=0.75)
        ax.tick_params(width=0.65, length=3)

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "sec_bert_selective_performance_journal_v2"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
