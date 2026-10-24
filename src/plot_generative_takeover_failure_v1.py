"""Plot paired LLM/RAG takeover outcomes and harmful-case diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "tables" / "encoder_sec_domain_vs_rag_pilot_v1.csv"
OUTPUT = ROOT / "results" / "figures" / "sec_bert_generative_takeover_failure_v1"


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame = frame.loc[as_bool(frame["oof_clean"])].copy()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    navy = "#173B63"
    blue = "#56B4E9"
    teal = "#00796B"
    amber = "#B26A00"
    gray = "#9AA5B1"
    slate = "#64748B"

    conditions = ["direct", "report_rag"]
    display = ["Direct LLM", "Report-RAG"]
    outcomes = ["beneficial", "harmful", "both_correct", "both_wrong"]
    labels = ["Beneficial", "Harmful", "Both correct", "Both wrong"]
    colors = [teal, amber, blue, gray]

    counts: dict[str, list[int]] = {name: [] for name in outcomes}
    diagnostics = {"broad": [], "exact": [], "high": []}
    harmful_ns: list[int] = []

    broad_label = "Reporting, Controls, Estimates and Audit Procedures"
    for condition in conditions:
        group = frame.loc[frame["condition"].eq(condition)].copy()
        beneficial = as_bool(group["beneficial_vs_sec"])
        harmful = as_bool(group["harmful_vs_sec"])
        backend_correct = as_bool(group["backend_correct"])
        sec_correct = as_bool(group["sec_correct"])

        counts["beneficial"].append(int(beneficial.sum()))
        counts["harmful"].append(int(harmful.sum()))
        counts["both_correct"].append(int((backend_correct & sec_correct).sum()))
        counts["both_wrong"].append(int((~backend_correct & ~sec_correct).sum()))

        harmful_group = group.loc[harmful]
        harmful_ns.append(len(harmful_group))
        diagnostics["broad"].append(
            100 * harmful_group["predicted_label"].eq(broad_label).mean()
        )
        diagnostics["exact"].append(
            100 * as_bool(harmful_group["evidence_exact"]).mean()
        )
        diagnostics["high"].append(
            100 * harmful_group["confidence"].astype(float).ge(0.90).mean()
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.70), constrained_layout=True)

    # Panel A: paired outcome composition, all conditions have n=120.
    ax = axes[0]
    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions), dtype=float)
    for outcome, label, color in zip(outcomes, labels, colors):
        values = np.asarray(counts[outcome])
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.58,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 9:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if color != gray else navy,
                    fontsize=7.4,
                    fontweight="bold",
                )
        bottom += values

    nets = [counts["beneficial"][i] - counts["harmful"][i] for i in range(2)]
    for i, net in enumerate(nets):
        ax.text(
            i,
            123.5,
            f"Net {net:+d}",
            ha="center",
            va="bottom",
            color=amber,
            fontsize=7.7,
            fontweight="bold",
        )
    ax.set_xticks(x, display)
    ax.set_ylabel("Pilot findings (count)")
    # Reserve the upper band for the legend and net-effect annotations.
    ax.set_ylim(0, 154)
    ax.set_title("Paired outcomes relative to SEC-BERT", loc="left", fontweight="bold")
    ax.legend(frameon=True, ncol=2, loc="upper right", columnspacing=1.0,
              handlelength=1.7, borderaxespad=0.45, fancybox=False,
              facecolor="white", edgecolor="#D7DEE6", framealpha=0.96)

    # Panel B: diagnostic properties among harmful takeovers only.
    ax = axes[1]
    categories = [
        "Predicted broad\nReporting class",
        "Evidence exact\nand traceable",
        "Confidence\n≥ 0.90",
    ]
    centers = np.arange(len(categories))
    width = 0.34
    direct_values = [diagnostics[key][0] for key in ("broad", "exact", "high")]
    rag_values = [diagnostics[key][1] for key in ("broad", "exact", "high")]
    bars_direct = ax.bar(
        centers - width / 2,
        direct_values,
        width,
        color=blue,
        edgecolor=navy,
        linewidth=0.65,
        label=f"Direct LLM (n={harmful_ns[0]})",
    )
    bars_rag = ax.bar(
        centers + width / 2,
        rag_values,
        width,
        color="white",
        edgecolor=teal,
        linewidth=1.0,
        hatch="///",
        label=f"Report-RAG (n={harmful_ns[1]})",
    )
    for bars in (bars_direct, bars_rag):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2.2,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.1,
                color=navy,
            )
    ax.set_xticks(centers, categories)
    ax.set_ylabel("Harmful takeovers (%)")
    # Reserve the upper band for the legend, above the 100% bars and labels.
    ax.set_ylim(0, 132)
    ax.set_title("Why grounding did not ensure correctness", loc="left", fontweight="bold")
    ax.legend(frameon=True, loc="upper right", ncol=2, columnspacing=1.0,
              handlelength=1.8, borderaxespad=0.45, fancybox=False,
              facecolor="white", edgecolor="#D7DEE6", framealpha=0.96)

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
        ax.set_axisbelow(True)
        ax.tick_params(width=0.65, length=3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
