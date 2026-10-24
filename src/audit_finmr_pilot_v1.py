"""Evaluate Direct, structured-RAG, and deterministic verified FinMR outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finmr_structured import decimal_value
from llm_common import ROOT
from prepare_finmr_candidates_v1 import load_config


def numeric_equal(left, right) -> bool:
    return decimal_value(str(left)) == decimal_value(str(right))


def load_responses(path: Path) -> pd.DataFrame:
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("status") == "ok":
                latest[(item["request_id"], item["prompt_bundle_sha256"])] = item
    rows = []
    for item in latest.values():
        rows.append({
            "request_id": item["request_id"], "sample_id": item["sample_id"], "condition": item["condition"],
            "prompt_bundle_sha256": item["prompt_bundle_sha256"], "latency_ms": item["latency_ms"],
            "input_tokens": item["input_tokens"], "output_tokens": item["output_tokens"], **item["decision"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    responses = load_responses(ROOT / config["response_log"])
    audit = candidates.merge(responses, on=["request_id", "sample_id", "condition", "prompt_bundle_sha256"], how="left")
    audit["completed"] = audit["extracted_value"].notna() & audit["calculated_value"].notna()
    audit["extraction_correct"] = audit.apply(lambda row: numeric_equal(row["extracted_value"], row["gold_extracted_value"]) if row["completed"] else False, axis=1)
    audit["calculation_correct"] = audit.apply(lambda row: numeric_equal(row["calculated_value"], row["gold_calculated_value"]) if row["completed"] else False, axis=1)
    audit["joint_correct"] = audit["extraction_correct"] & audit["calculation_correct"]
    observed = audit.loc[audit["completed"]].copy()

    # Deterministic verified backend is derived once per sample from query-only evidence.
    base = candidates.drop_duplicates("sample_id")
    verified = pd.DataFrame({
        "sample_id": base["sample_id"],
        "condition": "verified_structured",
        "extraction_correct": base.apply(lambda row: numeric_equal(row["deterministic_extracted_value"], row["gold_extracted_value"]), axis=1),
        "calculation_correct": base.apply(lambda row: numeric_equal(row["deterministic_calculated_value"], row["gold_calculated_value"]), axis=1),
    })
    verified["joint_correct"] = verified["extraction_correct"] & verified["calculation_correct"]

    condition_metrics = {}
    for condition, group in observed.groupby("condition"):
        condition_metrics[condition] = {
            "n": len(group), "joint_accuracy": float(group["joint_correct"].mean()),
            "extraction_accuracy": float(group["extraction_correct"].mean()),
            "calculation_accuracy": float(group["calculation_correct"].mean()),
            "mean_input_tokens": float(group["input_tokens"].mean()),
            "mean_latency_ms": float(group["latency_ms"].mean()),
        }
    condition_metrics["verified_structured"] = {
        "n": len(verified), "joint_accuracy": float(verified["joint_correct"].mean()),
        "extraction_accuracy": float(verified["extraction_correct"].mean()),
        "calculation_accuracy": float(verified["calculation_correct"].mean()),
        "mean_input_tokens": 0.0, "mean_latency_ms": None,
    }

    direct = observed.loc[observed.condition.eq("direct"), ["sample_id", "joint_correct"]].rename(columns={"joint_correct": "direct_correct"})
    paired = {}
    for condition, group in [("structured_rag", observed.loc[observed.condition.eq("structured_rag")]), ("verified_structured", verified)]:
        merged = group[["sample_id", "joint_correct"]].merge(direct, on="sample_id", how="inner")
        beneficial = int((merged["joint_correct"] & ~merged["direct_correct"]).sum())
        harmful = int((~merged["joint_correct"] & merged["direct_correct"]).sum())
        paired[condition] = {"paired_n": len(merged), "beneficial": beneficial, "harmful": harmful, "net": beneficial - harmful}

    metrics = {"version": config["version"], "conditions": condition_metrics, "paired_vs_direct": paired}
    table_path = ROOT / config["audit_table"]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(table_path, index=False, encoding="utf-8")
    (ROOT / config["metrics"]).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# FinMR structured-RAG feasibility pilot v1", "", "| condition | n | Joint ACC | extraction | calculation | mean input tokens | mean latency ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition, value in condition_metrics.items():
        latency = "—" if value["mean_latency_ms"] is None else f"{value['mean_latency_ms']:.1f}"
        lines.append(f"| {condition} | {value['n']} | {value['joint_accuracy']:.2%} | {value['extraction_accuracy']:.2%} | {value['calculation_accuracy']:.2%} | {value['mean_input_tokens']:.1f} | {latency} |")
    lines += ["", "| compared with direct | paired n | beneficial | harmful | net |", "|---|---:|---:|---:|---:|"]
    for condition, value in paired.items():
        lines.append(f"| {condition} | {value['paired_n']} | {value['beneficial']} | {value['harmful']} | {value['net']:+d} |")
    (ROOT / config["report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
