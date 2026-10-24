"""Audit structured outputs and paired net benefit for the PCAOB RAG pilot."""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from llm_common import ROOT, evidence_match
from prepare_pcaob_rag_pilot_v1 import load_config


def load_responses(path) -> pd.DataFrame:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("status") == "ok":
            latest[(item["request_id"], item["prompt_bundle_sha256"])] = item
    rows = []
    for item in latest.values():
        rows.append({**{k: item.get(k) for k in ["request_id", "sample_id", "condition", "latency_ms", "input_tokens", "output_tokens", "total_tokens", "prompt_bundle_sha256"]}, **item["decision"]})
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    candidates = pd.read_csv(ROOT / config["candidate_table"])
    responses = load_responses(ROOT / config["response_log"])
    audit = candidates.merge(responses, on=["request_id", "sample_id", "condition", "prompt_bundle_sha256"], how="left", validate="one_to_one")
    audit["completed"] = audit["predicted_label"].notna() | audit["abstain"].fillna(False)
    audit["prediction_correct"] = audit["predicted_label"].eq(audit["target_level1"]) & ~audit["abstain"].fillna(False)
    exact, normalized, valid_source = [], [], []
    for _, row in audit.iterrows():
        source_map = json.loads(row["source_map_json"])
        source_id = row.get("evidence_source_id")
        valid = isinstance(source_id, str) and source_id in source_map
        valid_source.append(valid)
        if valid:
            ex, norm = evidence_match(source_map[source_id], str(row.get("evidence_quote") or ""))
        else:
            ex, norm = False, False
        exact.append(ex)
        normalized.append(norm)
    audit["evidence_source_valid"] = valid_source
    audit["evidence_exact"] = exact
    audit["evidence_normalized"] = normalized
    audit["takeover_eligible"] = audit["prediction_correct"] & audit["evidence_normalized"]

    # Keep the full frozen candidate table in the audit artifact, but compute
    # interim/final metrics only from completed API responses.  This makes the
    # same audit command safe for smoke tests and resumable partial runs.
    observed = audit.loc[audit["completed"]].copy()
    metrics = {"version": config["version"], "samples": int(observed["sample_id"].nunique()), "conditions": {}}
    for condition, group in observed.groupby("condition"):
        metrics["conditions"][condition] = {
            "n": int(len(group)), "completed": int(group["completed"].sum()),
            "accuracy": float(group["prediction_correct"].mean()),
            "evidence_exact_rate": float(group["evidence_exact"].mean()),
            "evidence_valid_rate": float(group["evidence_normalized"].mean()),
            "abstain_rate": float(group["abstain"].fillna(False).mean()),
            "mean_input_tokens": float(group["input_tokens"].dropna().mean()) if group["input_tokens"].notna().any() else None,
            "mean_latency_ms": float(group["latency_ms"].dropna().mean()) if group["latency_ms"].notna().any() else None,
        }

    direct = observed.loc[observed["condition"].eq("direct"), ["sample_id", "prediction_correct"]].rename(columns={"prediction_correct": "direct_correct"})
    paired = {}
    for condition in ["report_rag", "standards_rag", "combined_rag"]:
        group = observed.loc[observed["condition"].eq(condition)].merge(direct, on="sample_id", how="inner")
        beneficial = int((group["prediction_correct"] & ~group["direct_correct"]).sum())
        harmful = int((~group["prediction_correct"] & group["direct_correct"]).sum())
        paired[condition] = {"paired_n": int(len(group)), "beneficial": beneficial, "harmful": harmful, "net": beneficial - harmful}
    metrics["paired_vs_direct"] = paired

    table_path = ROOT / config["audit_table"]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(table_path, index=False, encoding="utf-8")
    metrics_path = ROOT / config["metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# PCAOB RAG feasibility pilot v1", "",
        f"- 已凍結樣本：{metrics['samples']} 筆 development-only finding。", "",
        "## 條件結果", "",
        "| 條件 | n | accuracy | evidence valid | abstain |", "|---|---:|---:|---:|---:|",
    ]
    for condition, values in metrics["conditions"].items():
        lines.append(f"| {condition} | {values['n']} | {values['accuracy']:.4f} | {values['evidence_valid_rate']:.4f} | {values['abstain_rate']:.4f} |")
    lines += ["", "## 相對 Direct DeepSeek 的配對結果", "", "| RAG條件 | paired n | beneficial | harmful | net |", "|---|---:|---:|---:|---:|"]
    for condition, values in paired.items():
        lines.append(f"| {condition} | {values['paired_n']} | {values['beneficial']} | {values['harmful']} | {values['net']} |")
    report_path = ROOT / config["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
