from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "isafm_major_revision_2026-11-21" / "03_figure_files"
DATASET = ROOT / "data" / "processed" / "model_dataset_v2.csv"
CLASS_DISTRIBUTION = None
FREEZE = ROOT / "configs" / "adaptation_evaluation_phase1.freeze.json"
CONTRAST_890 = (
    ROOT
    / "results"
    / "phase6b_retrospective_test_aggregation_v1_1"
    / "phase6_test_contrasts_v1_1.csv"
)
CONTRAST_863 = (
    ROOT
    / "results"
    / "phase6_posthoc_near_duplicate_sensitivity_v1_1"
    / "phase6_test_contrast_near_duplicate_sensitivity_v1_1.csv"
)
TIER_SUMMARY = (
    ROOT
    / "results"
    / "phase6b_retrospective_test_final_v1_1"
    / "phase6_test_frequency_tier_summary_v1_1.csv"
)


BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    tiff_path = OUT / f"{stem}_800dpi.tiff"
    fig.savefig(
        tiff_path,
        dpi=800,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    with Image.open(tiff_path) as image:
        image.load()
        if image.mode == "RGBA":
            flattened = Image.new("RGB", image.size, "white")
            flattened.paste(image, mask=image.getchannel("A"))
        else:
            flattened = image.convert("RGB")
        flattened.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(800, 800))
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)


def build_class_distribution() -> None:
    if CLASS_DISTRIBUTION is not None:
        counts = pd.read_csv(CLASS_DISTRIBUTION).rename(
            columns={"target_level1": "class_name", "count": "train_n"}
        )
    else:
        data = pd.read_csv(DATASET, usecols=["split", "target_level1"])
        counts = (
            data.loc[data["split"].eq("train"), "target_level1"]
            .value_counts()
            .rename_axis("class_name")
            .reset_index(name="train_n")
        )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    tier_lookup = {
        class_name: tier.title()
        for tier, classes in freeze["frequency_tiers_from_train_only"].items()
        if isinstance(classes, list)
        for class_name in classes
    }
    counts["tier"] = counts["class_name"].map(tier_lookup)
    counts = counts.sort_values("train_n", ascending=True)

    short = {
        "Reporting, Controls, Estimates and Audit Procedures": "Reporting, controls, estimates,\nand audit procedures",
        "Related Parties and Significant Transactions": "Related parties and\nsignificant transactions",
        "Expenses, Compensation and Benefits": "Expenses, compensation,\nand benefits",
        "Business Combinations and Consolidation": "Business combinations\nand consolidation",
        "Credit, Loans, Deposits and Insurance": "Credit, loans, deposits,\nand insurance",
        "Investments and Financial Instruments": "Investments and\nfinancial instruments",
        "Long-Lived and Intangible Assets": "Long-lived and intangible assets",
        "Inventory and Cost of Sales": "Inventory and cost of sales",
        "Debt, Equity and Liabilities": "Debt, equity, and liabilities",
        "Revenue and Receivables": "Revenue and receivables",
        "Cash and Other Assets": "Cash and other assets",
        "Income Taxes": "Income taxes",
    }
    colors = {"High": BLUE, "Medium": GREEN, "Low": ORANGE}
    hatches = {"High": "", "Medium": "//", "Low": ".."}

    fig, ax = plt.subplots(figsize=(7.1, 4.5), constrained_layout=True)
    y = np.arange(len(counts))
    for i, row in enumerate(counts.itertuples(index=False)):
        ax.barh(
            i,
            row.train_n,
            color=colors[row.tier],
            edgecolor="black",
            linewidth=0.35,
            hatch=hatches[row.tier],
            height=0.72,
        )
        ax.text(row.train_n * 1.05, i, f"{row.train_n:,}", va="center", fontsize=7.2)
    ax.set_yticks(y, [short[name] for name in counts["class_name"]])
    ax.set_xscale("log")
    ax.set_xlim(30, 2100)
    ax.set_xlabel("Training findings (log scale)")
    ax.set_ylabel("")
    style_axis(ax)
    ax.text(
        0.985,
        0.04,
        "Largest / smallest = 1,365 / 43 = 31.7:1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    handles = [
        Patch(facecolor=colors[t], edgecolor="black", linewidth=0.35, hatch=hatches[t], label=t)
        for t in ["High", "Medium", "Low"]
    ]
    ax.legend(
        handles=handles,
        title="Training-frequency tier",
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.69, 1.10),
    )
    save_figure(fig, "class_imbalance_distribution")


def build_forest_plot() -> None:
    main = pd.read_csv(CONTRAST_890).set_index("contrast")
    post = pd.read_csv(CONTRAST_863).set_index("contrast")
    order = [
        "primary_lora_vs_full_original_weighted",
        "lora_weighted_vs_unweighted_original",
        "sec_full_weighted_vs_unweighted_original",
        "sec_full_vs_bert_full_original_weighted",
        "lora_vs_full_matched_masked",
    ]
    labels = [
        "Primary: LoRA weighted − SEC full weighted\n(original)",
        "LoRA weighted − LoRA unweighted\n(original)",
        "SEC full weighted − SEC full unweighted\n(original)",
        "SEC full weighted − BERT full weighted\n(original)",
        "LoRA weighted − SEC full weighted\n(matched masked)",
    ]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(7.1, 4.15), constrained_layout=True)
    ax.axhspan(y[0] - 0.43, y[0] + 0.43, color=SKY, alpha=0.12, zorder=0)
    ax.axvline(0, color="black", linewidth=0.8, zorder=1)

    for yi, key in zip(y, order):
        m = main.loc[key]
        p = post.loc[key]
        ax.errorbar(
            m["delta_mean"],
            yi + 0.10,
            xerr=[[m["delta_mean"] - m["hierarchical_ci95_low"]], [m["hierarchical_ci95_high"] - m["delta_mean"]]],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            markersize=5,
            elinewidth=1.15,
            capsize=2.5,
            zorder=3,
        )
        ax.errorbar(
            p["delta_mean"],
            yi - 0.10,
            xerr=[[p["delta_mean"] - p["hierarchical_ci95_low"]], [p["hierarchical_ci95_high"] - p["delta_mean"]]],
            fmt="o",
            markerfacecolor="white",
            markeredgecolor=GRAY,
            ecolor=GRAY,
            markersize=5,
            elinewidth=0.95,
            capsize=2.5,
            zorder=2,
        )

    ax.set_yticks(y, labels)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.07, 0.07)
    ax.set_xticks(np.arange(-0.06, 0.061, 0.02))
    ax.set_xlabel("Macro-F1 difference (first method minus second)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend = [
        Line2D([0], [0], marker="o", color=BLUE, markerfacecolor=BLUE, linewidth=1.1, label="890 retrospective test"),
        Line2D([0], [0], marker="o", color=GRAY, markerfacecolor="white", linewidth=0.95, label="863 post-hoc sensitivity"),
    ]
    ax.legend(handles=legend, frameon=False, loc="lower right")
    ax.text(
        0.0,
        -0.18,
        "Horizontal lines are unadjusted 95% hierarchical-bootstrap intervals (5,000 replicates).",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#374151",
    )
    save_figure(fig, "prespecified_contrasts_forest")


def build_frequency_tier_panels() -> None:
    data = pd.read_csv(TIER_SUMMARY)
    condition_order = [
        "C01_SEC_LORA_ORIGINAL_WEIGHTED",
        "C04_SEC_FULL_ORIGINAL_WEIGHTED",
        "C07_BERT_FULL_ORIGINAL_WEIGHTED",
    ]
    method_labels = ["SEC-BERT\nLoRA", "SEC-BERT\nfull FT", "BERT-base\nfull FT"]
    tiers = ["high", "medium", "low"]
    colors = {"high": BLUE, "medium": GREEN, "low": ORANGE}
    markers = {"high": "o", "medium": "s", "low": "^"}
    selected = data[data["condition_id"].isin(condition_order)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), sharey=True, constrained_layout=True)
    metrics = [
        ("tier_macro_f1_mean", "A", "Tier Macro-F1"),
        ("tier_macro_recall_mean", "B", "Tier Macro-Recall"),
    ]
    x = np.arange(len(condition_order))
    offsets = {"high": -0.16, "medium": 0.0, "low": 0.16}
    for ax, (metric, panel, ylabel) in zip(axes, metrics):
        for tier in tiers:
            values = []
            for condition in condition_order:
                row = selected[(selected["condition_id"] == condition) & (selected["tier"] == tier)].iloc[0]
                values.append(row[metric])
            ax.plot(
                x + offsets[tier],
                values,
                linestyle="none",
                marker=markers[tier],
                markersize=6,
                markerfacecolor=colors[tier],
                markeredgecolor="black",
                markeredgewidth=0.35,
                color=colors[tier],
                label=tier.title(),
            )
        ax.set_xticks(x, method_labels)
        ax.set_ylim(0.72, 1.0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    axes[1].set_ylabel("")
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=markers[tier],
            markerfacecolor=colors[tier],
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=tier.title(),
        )
        for tier in tiers
    ]
    fig.legend(
        handles=handles,
        title="Training-frequency tier",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
    )
    save_figure(fig, "frequency_tier_f1_recall")


def main() -> None:
    configure_style()
    build_class_distribution()
    build_forest_plot()
    build_frequency_tier_panels()
    print(OUT)


if __name__ == "__main__":
    main()

