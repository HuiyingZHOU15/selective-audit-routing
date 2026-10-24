"""Download, freeze, and inspect the official FinAuditing FinMR_Sub dataset."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "finauditing_finmr_sub_v1"
HF_CACHE = Path(os.environ.get("FINMR_HF_CACHE", ROOT.parent / "rag_pilot_runtime" / "hf_cache"))
DATASET = "TheFinAI/FinMR_Sub"
SOURCE_REVISION = "main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, revision=SOURCE_REVISION, cache_dir=str(HF_CACHE))
    files = []
    split_summary = {}
    for split, table in dataset.items():
        path = OUT / f"{split}.parquet"
        table.to_parquet(str(path))
        lengths = [len(str(value)) for value in table["query"]]
        split_summary[split] = {
            "rows": len(table),
            "columns": table.column_names,
            "query_chars_min": min(lengths),
            "query_chars_median": sorted(lengths)[len(lengths) // 2],
            "query_chars_max": max(lengths),
        }
        files.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "dataset": DATASET,
        "requested_revision": SOURCE_REVISION,
        "license": "CC-BY-4.0 (per official dataset card)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": split_summary,
        "files": files,
        "upstream_code_commit": "cee2b4e563d8d6cadb94cd0138163bb72e8ee0e7",
        "upstream_code_url": "https://github.com/The-FinAI/FinAuditing",
        "upstream_dataset_url": "https://huggingface.co/datasets/TheFinAI/FinMR_Sub",
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
