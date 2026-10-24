"""Measure structured parser coverage on FinMR_Sub without using gold in retrieval."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from finmr_structured import decimal_value, retrieve


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "finauditing_finmr_sub_v1"


def same_numeric(left: str | Decimal | None, right: str | Decimal | None) -> bool:
    if left is None or right is None:
        return False
    return decimal_value(str(left)) == decimal_value(str(right))


def main() -> None:
    rows = []
    for split in ("train", "test"):
        frame = pd.read_parquet(DATA / f"{split}.parquet")
        for item in frame.to_dict("records"):
            result = retrieve(item["query"])
            gold = item["answer"]
            rows.append({
                "split": split,
                "id": int(item["id"]),
                "dqc_id": item["dqc_id"],
                "target": result.question.target,
                "status": result.status,
                "child_count": len(result.children),
                "reported_match": same_numeric(result.reported_fact.value_text if result.reported_fact else None, gold["extracted_value"]),
                "calculated_match": same_numeric(result.calculated_value, gold["calculated_value"]),
                "query_chars": len(item["query"]),
                "evidence_chars": len(result.evidence_text),
            })
    audit = pd.DataFrame(rows)
    out_csv = ROOT.parent / "rag_pilot_runtime" / "finmr_parser_v1_audit.csv"
    audit.to_csv(out_csv, index=False, encoding="utf-8")
    summary = {
        "n": len(audit),
        "reported_match_rate": float(audit["reported_match"].mean()),
        "calculated_match_rate": float(audit["calculated_match"].mean()),
        "complete_rate": float(audit["status"].eq("complete").mean()),
        "median_query_chars": float(audit["query_chars"].median()),
        "median_evidence_chars": float(audit["evidence_chars"].median()),
        "by_dqc": audit.groupby("dqc_id").agg(
            n=("id", "size"),
            reported_match=("reported_match", "mean"),
            calculated_match=("calculated_match", "mean"),
            complete=("status", lambda values: (values == "complete").mean()),
            median_evidence_chars=("evidence_chars", "median"),
        ).to_dict("index"),
    }
    out_json = ROOT.parent / "rag_pilot_runtime" / "finmr_parser_v1_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
