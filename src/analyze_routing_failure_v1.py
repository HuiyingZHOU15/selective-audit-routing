"""Diagnose calibration-to-test failure of the PCAOB selective router."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from llm_common import ROOT


CONFIG_PATH = ROOT / "configs/routing_failure_analysis_v1.json"
TABLE_DIR = ROOT / "results/tables"
METRICS_OUT = ROOT / "results/metrics/routing_failure_analysis_v1.json"
REPORT_OUT = ROOT / "results/reports/routing_failure_analysis_v1_zh.md"
FIGURE_PNG = ROOT / "results/figures/routing_failure_analysis_v1.png"
FIGURE_PDF = ROOT / "results/figures/routing_failure_analysis_v1.pdf"

PARTITIONS = ["Development OOF", "Calibration", "Retrospective test"]
OUTCOMES = ["beneficial", "harmful", "neutral_both_correct", "neutral_both_wrong"]
COLORS = {
    "beneficial": "#009E73",
    "harmful": "#D55E00",
    "neutral_both_correct": "#56B4E9",
    "neutral_both_wrong": "#999999",
    "Development OOF": "#0072B2",
    "Calibration": "#E69F00",
    "Retrospective test": "#CC79A7",
}


def load_partitions(config: dict) -> pd.DataFrame:
    specs = [
        ("Development OOF", config["development_table"], "router_utility_oof", "utility_route_20pct"),
        ("Calibration", config["calibration_table"], "router_utility", "utility_route"),
        ("Retrospective test", config["test_table"], "router_utility", "utility_route"),
    ]
    metadata = pd.read_csv(ROOT / config["model_dataset"])[
        ["sample_id", "Registration ID", "Firm Names", "Inspection Year", "split"]
    ]
    frames = []
    for partition, relative_path, score_column, route_column in specs:
        frame = pd.read_csv(ROOT / relative_path).merge(metadata, on="sample_id", how="left", validate="one_to_one")
        frame["partition"] = partition
        frame["router_score"] = frame[score_column].astype(float)
        frame["selected_route"] = frame[route_column].astype(bool)
        frame["outcome_utility"] = np.select(
            [frame["routing_outcome"].eq("beneficial"), frame["routing_outcome"].eq("harmful")],
            [1, -1],
            default=0,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition in PARTITIONS:
        part = frame.loc[frame["partition"].eq(partition)]
        row = {"partition": partition, "n": len(part), "reports": part["report_id"].nunique()}
        for outcome in OUTCOMES:
            count = int(part["routing_outcome"].eq(outcome).sum())
            row[f"{outcome}_n"] = count
            row[f"{outcome}_rate"] = count / len(part)
        row["always_jev_net_per_100"] = 100 * part["outcome_utility"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def top_budget_route(scores: np.ndarray, budget: float) -> np.ndarray:
    count = int(round(len(scores) * budget))
    route = np.zeros(len(scores), dtype=bool)
    if count:
        route[np.argsort(-scores, kind="stable")[:count]] = True
    return route


def budget_curve(frame: pd.DataFrame, maximum: float, step: float) -> pd.DataFrame:
    rows = []
    budgets = np.arange(0.0, maximum + step / 2, step)
    for partition in PARTITIONS:
        part = frame.loc[frame["partition"].eq(partition)].reset_index(drop=True)
        scores = part["router_score"].to_numpy(float)
        utilities = part["outcome_utility"].to_numpy(int)
        for budget in budgets:
            route = top_budget_route(scores, float(budget))
            routed_utility = utilities[route]
            rows.append(
                {
                    "partition": partition,
                    "budget": float(budget),
                    "calls": int(route.sum()),
                    "corrected": int((routed_utility == 1).sum()),
                    "harmed": int((routed_utility == -1).sum()),
                    "net_correct": int(routed_utility.sum()),
                    "net_per_100": float(100 * routed_utility.sum() / len(part)),
                }
            )
    return pd.DataFrame(rows)


def fixed_score_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    development = frame.loc[frame["partition"].eq("Development OOF"), "router_score"].to_numpy(float)
    cutoffs = np.quantile(development, np.arange(.1, 1.0, .1))
    rows = []
    for partition in PARTITIONS:
        part = frame.loc[frame["partition"].eq(partition)].copy()
        part["score_decile"] = np.searchsorted(cutoffs, part["router_score"].to_numpy(float), side="right") + 1
        for decile, group in part.groupby("score_decile", sort=True):
            rows.append(
                {
                    "partition": partition,
                    "score_decile": int(decile),
                    "n": len(group),
                    "mean_score": group["router_score"].mean(),
                    "beneficial_rate": group["routing_outcome"].eq("beneficial").mean(),
                    "harmful_rate": group["routing_outcome"].eq("harmful").mean(),
                    "observed_net_rate": group["outcome_utility"].mean(),
                }
            )
    return pd.DataFrame(rows)


def standardized_mean_difference(reference: pd.Series, other: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    other = pd.to_numeric(other, errors="coerce").dropna().to_numpy(float)
    pooled = np.sqrt((reference.var(ddof=1) + other.var(ddof=1)) / 2)
    return float((other.mean() - reference.mean()) / pooled) if pooled else 0.0


def total_variation(reference: pd.Series, other: pd.Series) -> float:
    categories = reference.astype(str).value_counts(normalize=True).index.union(
        other.astype(str).value_counts(normalize=True).index
    )
    p = reference.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    q = other.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    return float(0.5 * np.abs(p - q).sum())


def feature_drift(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "small_raw_confidence", "small_raw_margin", "small_raw_entropy",
        "text_log_chars", "text_log_words", "account_topic_hits", "crosscut_hits",
    ]
    categorical = ["target_level1", "small_prediction", "context_quality", "Inspection Year"]
    reference = frame.loc[frame["partition"].eq("Development OOF")]
    rows = []
    for partition in ["Calibration", "Retrospective test"]:
        other = frame.loc[frame["partition"].eq(partition)]
        rows.extend(
            {"partition": partition, "feature": column, "measure": "standardized_mean_difference", "value": standardized_mean_difference(reference[column], other[column])}
            for column in numeric
        )
        rows.extend(
            {"partition": partition, "feature": column, "measure": "total_variation_distance", "value": total_variation(reference[column], other[column])}
            for column in categorical
        )
    return pd.DataFrame(rows)


def class_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["partition", "target_level1"], observed=True)
    rows = []
    for (partition, target), group in grouped:
        rows.append(
            {
                "partition": partition,
                "target_level1": target,
                "n": len(group),
                "beneficial_n": int(group["routing_outcome"].eq("beneficial").sum()),
                "harmful_n": int(group["routing_outcome"].eq("harmful").sum()),
                "beneficial_rate": group["routing_outcome"].eq("beneficial").mean(),
                "harmful_rate": group["routing_outcome"].eq("harmful").mean(),
                "always_jev_net_rate": group["outcome_utility"].mean(),
            }
        )
    return pd.DataFrame(rows)


def discrimination(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition in PARTITIONS:
        part = frame.loc[frame["partition"].eq(partition)]
        score = part["router_score"].to_numpy(float)
        for event in ["beneficial", "harmful"]:
            target = part["routing_outcome"].eq(event).to_numpy(int)
            oriented_score = score if event == "beneficial" else -score
            rows.append(
                {
                    "partition": partition,
                    "event": event,
                    "events": int(target.sum()),
                    "prevalence": float(target.mean()),
                    "roc_auc": float(roc_auc_score(target, oriented_score)),
                    "average_precision": float(average_precision_score(target, oriented_score)),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_selected(frame: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows = []
    draws_by_partition = {}
    for partition in PARTITIONS:
        part = frame.loc[frame["partition"].eq(partition)].copy()
        report_stats = part.assign(
            selected_utility=part["selected_route"].astype(int) * part["outcome_utility"]
        ).groupby("report_id", sort=False).agg(n=("sample_id", "size"), selected_utility=("selected_utility", "sum"))
        cluster_n = report_stats["n"].to_numpy(float)
        cluster_utility = report_stats["selected_utility"].to_numpy(float)
        sampled_indices = rng.integers(0, len(report_stats), size=(iterations, len(report_stats)))
        draws = cluster_utility[sampled_indices].sum(axis=1) / cluster_n[sampled_indices].sum(axis=1)
        draws_by_partition[partition] = draws
        observed = part.loc[part["selected_route"], "outcome_utility"].sum() / len(part)
        rows.append(
            {
                "partition": partition,
                "observed_accuracy_delta": float(observed),
                "bootstrap_mean": float(draws.mean()),
                "ci95_low": float(np.quantile(draws, .025)),
                "ci95_high": float(np.quantile(draws, .975)),
                "probability_positive": float((draws > 0).mean()),
                "calls": int(part["selected_route"].sum()),
                "corrected": int(((part["selected_route"]) & part["routing_outcome"].eq("beneficial")).sum()),
                "harmed": int(((part["selected_route"]) & part["routing_outcome"].eq("harmful")).sum()),
            }
        )
    return pd.DataFrame(rows), draws_by_partition


def bootstrap_partition_differences(draws: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for left, right in [("Calibration", "Development OOF"), ("Retrospective test", "Calibration"), ("Retrospective test", "Development OOF")]:
        difference = draws[left] - draws[right]
        rows.append(
            {
                "contrast": f"{left} minus {right}",
                "mean_difference": float(difference.mean()),
                "ci95_low": float(np.quantile(difference, .025)),
                "ci95_high": float(np.quantile(difference, .975)),
                "probability_positive": float((difference > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def make_figure(outcomes: pd.DataFrame, curves: pd.DataFrame, deciles: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    bottom = np.zeros(len(outcomes))
    labels = {"beneficial": "Beneficial", "harmful": "Harmful", "neutral_both_correct": "Both correct", "neutral_both_wrong": "Both wrong"}
    for outcome in OUTCOMES:
        values = 100 * outcomes[f"{outcome}_rate"].to_numpy()
        ax.bar(outcomes["partition"], values, bottom=bottom, color=COLORS[outcome], label=labels[outcome], width=.68)
        bottom += values
    ax.set_ylabel("Share of findings (%)")
    ax.set_title("A  Comparative outcome composition", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False, ncol=2, loc="lower center")

    ax = axes[0, 1]
    for partition in PARTITIONS:
        group = curves.loc[curves["partition"].eq(partition)]
        ax.plot(100 * group["budget"], group["net_per_100"], color=COLORS[partition], label=partition, linewidth=1.8)
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.axvline(20, color="#777777", linestyle="--", linewidth=.8)
    ax.set_xlabel("Jev call budget (%)")
    ax.set_ylabel("Net corrections per 100 findings")
    ax.set_title("B  Utility–budget curves", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for partition in PARTITIONS:
        group = deciles.loc[deciles["partition"].eq(partition)]
        ax.plot(group["score_decile"], 100 * group["observed_net_rate"], marker="o", markersize=3, color=COLORS[partition], label=partition)
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Router-score decile (development cut points)")
    ax.set_ylabel("Observed net utility (%)")
    ax.set_title("C  Score-to-utility stability", loc="left", fontweight="bold")

    ax = axes[1, 1]
    positions = np.arange(len(bootstrap))
    values = 100 * bootstrap["observed_accuracy_delta"].to_numpy()
    lower = values - 100 * bootstrap["ci95_low"].to_numpy()
    upper = 100 * bootstrap["ci95_high"].to_numpy() - values
    ax.errorbar(values, positions, xerr=np.vstack([lower, upper]), fmt="o", color="#0072B2", ecolor="#444444", capsize=3)
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_yticks(positions, bootstrap["partition"])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change (percentage points; report bootstrap 95% CI)")
    ax.set_title("D  Selected-policy uncertainty", loc="left", fontweight="bold")

    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def write_report(
    outcomes: pd.DataFrame,
    curves: pd.DataFrame,
    drift: pd.DataFrame,
    discrimination_table: pd.DataFrame,
    bootstrap: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    outcome_index = outcomes.set_index("partition")
    boot_index = bootstrap.set_index("partition")
    strongest = drift.assign(abs_value=drift["value"].abs()).sort_values("abs_value", ascending=False).head(6)
    best_budgets = curves.loc[curves.groupby("partition")["net_correct"].idxmax()].set_index("partition")
    lines = [
        "# PCAOB Routing Failure Analysis v1",
        "",
        "## 結論先行",
        "",
        "失效不是因為測試集完全變成另一種文本，而是兩個問題疊加：**可受益案例的基準率下降／有害案例上升**，以及**路由分數在最高分區間未能保持排序能力**。因此，calibration 上選出的閾值沒有跨到 retrospective test。這支持把目前結果寫成明確的失效條件，而不是宣稱已得到穩健的正收益路由器。",
        "",
        "- Development OOF 的 beneficial/harmful 比率為 " + fmt_pct(outcome_index.loc["Development OOF", "beneficial_rate"]) + "/" + fmt_pct(outcome_index.loc["Development OOF", "harmful_rate"]) + "；retrospective test 變為 " + fmt_pct(outcome_index.loc["Retrospective test", "beneficial_rate"]) + "/" + fmt_pct(outcome_index.loc["Retrospective test", "harmful_rate"]) + "。",
        "- 固定策略在 development、calibration、test 的淨修正分別為 " + ", ".join(f"{p}: {int(boot_index.loc[p, 'corrected'])}-{int(boot_index.loc[p, 'harmed'])}={int(boot_index.loc[p, 'corrected']-boot_index.loc[p, 'harmed']):+d}" for p in PARTITIONS) + "。",
        "- Test 的 accuracy change 為 " + fmt_pct(boot_index.loc["Retrospective test", "observed_accuracy_delta"]) + "，report-cluster bootstrap 95% CI [" + fmt_pct(boot_index.loc["Retrospective test", "ci95_low"]) + ", " + fmt_pct(boot_index.loc["Retrospective test", "ci95_high"]) + "]；不支持穩健正收益。",
        "",
        "## 1. Comparative outcome 漂移",
        "",
        "| Partition | n | Reports | Beneficial | Harmful | Both correct | Both wrong | Always-Jev net/100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in outcomes.iterrows():
        lines.append(
            f"| {row['partition']} | {int(row['n'])} | {int(row['reports'])} | {row['beneficial_rate']:.2%} | {row['harmful_rate']:.2%} | {row['neutral_both_correct_rate']:.2%} | {row['neutral_both_wrong_rate']:.2%} | {row['always_jev_net_per_100']:.2f} |"
        )
    lines += [
        "",
        "Test 的主要不利變化是 beneficial 減少、harmful 增加；即使不知道路由分數，Jev 相對小模型的先驗效用也更差。",
        "",
        "## 2. 排序能力與閾值穩定性",
        "",
        "| Partition | Event | Events | Prevalence | ROC-AUC | Average precision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in discrimination_table.iterrows():
        lines.append(f"| {row['partition']} | {row['event']} | {int(row['events'])} | {row['prevalence']:.2%} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} |")
    lines += [
        "",
        "每個分割事後可達到的最佳 budget（只作診斷，不可視為有效選模結果）如下：",
        "",
        "| Partition | Best budget | Net corrections | Net/100 |",
        "|---|---:|---:|---:|",
    ]
    for partition in PARTITIONS:
        row = best_budgets.loc[partition]
        lines.append(f"| {partition} | {row['budget']:.0%} | {int(row['net_correct'])} | {row['net_per_100']:.2f} |")
    lines += [
        "",
        "Test 曲線與 calibration 曲線不一致，表示單一 calibration 閾值對少量 beneficial/harmful events 很敏感；這不是把閾值再微調一次就能可靠解決的問題。",
        "",
        "## 3. 可觀測特徵漂移",
        "",
        "最大六個漂移量（數值特徵為 standardized mean difference；類別特徵為 total variation distance）：",
        "",
        "| Partition | Feature | Measure | Value |",
        "|---|---|---|---:|",
    ]
    for _, row in strongest.iterrows():
        lines.append(f"| {row['partition']} | {row['feature']} | {row['measure']} | {row['value']:.3f} |")
    lines += [
        "",
        "這些漂移量用來定位可能來源，不能單獨證明因果。類別內 event 數仍少，class-by-class 結果應作描述性證據。",
        "",
        "## 4. Report-cluster bootstrap",
        "",
        "| Partition | Calls | Corrected | Harmed | Accuracy delta | 95% CI | P(delta>0) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bootstrap.iterrows():
        lines.append(f"| {row['partition']} | {int(row['calls'])} | {int(row['corrected'])} | {int(row['harmed'])} | {row['observed_accuracy_delta']:.3%} | [{row['ci95_low']:.3%}, {row['ci95_high']:.3%}] | {row['probability_positive']:.3f} |")
    lines += [
        "",
        "分割間差異的 bootstrap 對比：",
        "",
        "| Contrast | Mean difference | 95% CI | P(>0) |",
        "|---|---:|---:|---:|",
    ]
    for _, row in contrasts.iterrows():
        lines.append(f"| {row['contrast']} | {row['mean_difference']:.3%} | [{row['ci95_low']:.3%}, {row['ci95_high']:.3%}] | {row['probability_positive']:.3f} |")
    lines += [
        "",
        "## 5. 對論文與下一輪實驗的含義",
        "",
        "1. 現在可以可靠地寫出：RAG 式 uncertainty/benefit routing 在 PCAOB 審計文本上不是直接可移植；其收益依賴 LLM 相對優勢的穩定基準率與 score-to-utility calibration。",
        "2. 不應用 retrospective test 重新選 threshold。應保留它作已檢視的回顧性測試，另建真正未接觸的 confirmatory set。",
        "3. 下一版路由器應優先處理不對稱風險：加入 conservative abstention、harm upper bound 或分層最低事件數；不是增加複雜模型。",
        "4. 外部資料集（CAM 或 AuditBench）的角色是驗證失效條件是否重現，而不是混入 PCAOB 後掩蓋分布差異。",
        "",
        "## 6. 產物與可重現性",
        "",
        "- 圖：`results/figures/routing_failure_analysis_v1.pdf` 與 `.png`",
        "- 表：`results/tables/routing_failure_*.csv`",
        "- 指標：`results/metrics/routing_failure_analysis_v1.json`",
        "- bootstrap 以 report 為抽樣單位，避免把同一 report 的 findings 當作獨立樣本。",
        "- retrospective test 已在先前迭代中被查看，因此所有 test 分析均為描述性、非 confirmatory。",
    ]
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = load_partitions(config)
    outcomes = outcome_summary(frame)
    curves = budget_curve(frame, config["maximum_budget"], config["budget_step"])
    deciles = fixed_score_deciles(frame)
    drift = feature_drift(frame)
    classes = class_outcomes(frame)
    discrimination_table = discrimination(frame)
    bootstrap, draws = cluster_bootstrap_selected(frame, config["bootstrap_iterations"], config["random_seed"])
    contrasts = bootstrap_partition_differences(draws)

    outputs = {
        "routing_failure_outcome_summary_v1.csv": outcomes,
        "routing_failure_budget_curves_v1.csv": curves,
        "routing_failure_score_deciles_v1.csv": deciles,
        "routing_failure_feature_drift_v1.csv": drift,
        "routing_failure_class_outcomes_v1.csv": classes,
        "routing_failure_discrimination_v1.csv": discrimination_table,
        "routing_failure_bootstrap_v1.csv": bootstrap,
        "routing_failure_bootstrap_contrasts_v1.csv": contrasts,
    }
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(TABLE_DIR / filename, index=False)

    make_figure(outcomes, curves, deciles, bootstrap)
    write_report(outcomes, curves, drift, discrimination_table, bootstrap, contrasts)

    metrics = {
        "version": config["version"],
        "test_status": config["test_status"],
        "partitions": outcomes.to_dict(orient="records"),
        "selected_policy_bootstrap": bootstrap.to_dict(orient="records"),
        "selected_policy_contrasts": contrasts.to_dict(orient="records"),
        "posthoc_best_budgets": curves.loc[curves.groupby("partition")["net_correct"].idxmax()].to_dict(orient="records"),
        "discrimination": discrimination_table.to_dict(orient="records"),
        "bootstrap_unit": "report_id",
        "bootstrap_iterations": config["bootstrap_iterations"],
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
