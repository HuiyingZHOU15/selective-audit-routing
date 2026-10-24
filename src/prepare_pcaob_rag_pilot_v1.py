"""Create a frozen, development-only PCAOB RAG feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_common import ROOT, sha256_text


CONFIG_PATH = ROOT / "configs" / "pcaob_rag_pilot_v1.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def stable_random(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    def key(sample_id: object) -> str:
        return hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest()
    return frame.assign(_random_key=frame["sample_id"].map(key)).sort_values("_random_key")


def balanced_one_per_report(frame: pd.DataFrame, n: int, seed: int, excluded_reports: set[str]) -> pd.DataFrame:
    pool = frame.loc[~frame["report_id"].astype(str).isin(excluded_reports)].copy()
    pool = stable_random(pool, seed).drop_duplicates("report_id", keep="first")
    groups = {label: group.copy() for label, group in pool.groupby("target_level1", sort=True)}
    labels = sorted(groups)
    selected = []
    offset = 0
    while len(selected) < n and labels:
        label = labels[offset % len(labels)]
        group = groups[label]
        if group.empty:
            labels.remove(label)
            continue
        selected.append(group.iloc[0])
        groups[label] = group.iloc[1:]
        offset += 1
    if len(selected) < n:
        raise ValueError(f"Only {len(selected)} report-independent samples available; requested {n}")
    return pd.DataFrame(selected).drop(columns=["_random_key"], errors="ignore")


def make_sample(config: dict) -> pd.DataFrame:
    oof = pd.read_csv(ROOT / config["development_oof"])
    model = pd.read_csv(ROOT / config["model_dataset"])
    oof = oof.loc[bool_series(oof["oof_clean"])].copy()
    keep = ["sample_id", "text_original", "standard_refs"]
    frame = oof.merge(model[keep], on="sample_id", how="left", validate="one_to_one")
    frame["small_correct"] = bool_series(frame["small_correct"])
    frame["context_needs_context"] = bool_series(frame["context_needs_context"])
    counts = config["sample_counts"]
    used: set[str] = set()

    wrong = balanced_one_per_report(frame.loc[~frame["small_correct"]], counts["small_wrong"], config["seed"] + 1, used)
    wrong["pilot_stratum"] = "small_wrong"
    used.update(wrong["report_id"].astype(str))

    correct = frame.loc[frame["small_correct"]].copy()
    threshold = correct["small_confidence"].quantile(0.25)
    risk_pool = correct.loc[correct["context_needs_context"] | correct["small_confidence"].le(threshold)]
    risk = balanced_one_per_report(risk_pool, counts["small_correct_risk"], config["seed"] + 2, used)
    risk["pilot_stratum"] = "small_correct_risk"
    used.update(risk["report_id"].astype(str))

    control_pool = correct.loc[~correct["sample_id"].isin(risk["sample_id"])]
    control = balanced_one_per_report(control_pool, counts["small_correct_control"], config["seed"] + 3, used)
    control["pilot_stratum"] = "small_correct_control"

    sample = pd.concat([wrong, risk, control], ignore_index=True)
    sample = sample.sort_values(["pilot_stratum", "sample_id"]).reset_index(drop=True)
    sample["pilot_seed"] = config["seed"]
    path = ROOT / config["sample_table"]
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8")
    return sample


def top_texts(query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
    texts = [str(value) for value in texts]
    if not texts:
        return []
    matrix = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1).fit_transform([query] + texts)
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(index), float(scores[index])) for index in order]


def report_blocks(row: pd.Series, development: pd.DataFrame, top_k: int) -> list[dict]:
    peers = development.loc[
        development["report_id"].astype(str).eq(str(row["report_id"]))
        & ~development["sample_id"].astype(str).eq(str(row["sample_id"]))
    ]
    texts = peers["text_context_v2"].fillna("").astype(str).tolist()
    ranked = top_texts(str(row["text_context_v2"]), texts, top_k)
    blocks = []
    for rank, (index, score) in enumerate(ranked, start=1):
        peer = peers.iloc[index]
        blocks.append({
            "source_id": f"report:{peer['sample_id']}",
            "rank": rank,
            "score": round(score, 6),
            "text": str(peer["text_context_v2"]),
        })
    return blocks


def standard_ids(value: object) -> list[str]:
    return sorted(set(re.findall(r"AS\s*(\d{4})", str(value or ""), flags=re.I)))


def standard_blocks(row: pd.Series, corpus: dict, top_k: int) -> list[dict]:
    chunks = []
    for standard_id in standard_ids(row.get("standard_refs")):
        chunks.extend(corpus.get(f"AS{standard_id}", {}).get("paragraphs", []))
    texts = [chunk["text"] for chunk in chunks]
    ranked = top_texts(str(row["text_context_v2"]), texts, top_k)
    blocks = []
    for rank, (index, score) in enumerate(ranked, start=1):
        chunk = chunks[index]
        blocks.append({**chunk, "rank": rank, "score": round(score, 6)})
    return blocks


def render_prompt(row: pd.Series, condition: str, labels: list[str], definitions: dict, blocks: list[dict]) -> str:
    choices = "\n".join(f"- {label}: {definitions[label]}" for label in labels)
    context = "\n\n".join(
        f"<source id=\"{block['source_id']}\">\n{block['text']}\n</source>" for block in blocks
    ) or "(none)"
    return (
        "Choose one audit-area label for the TARGET FINDING, or abstain if evidence is insufficient.\n\n"
        f"CONDITION: {condition}\n\nLABELS:\n{choices}\n\n"
        f"<source id=\"target\">\n{row['text_context_v2']}\n</source>\n\n"
        f"RETRIEVED CONTEXT:\n{context}"
    )


def prepare_candidates(config: dict) -> pd.DataFrame:
    sample_path = ROOT / config["sample_table"]
    sample = pd.read_csv(sample_path) if sample_path.exists() else make_sample(config)
    development = pd.read_csv(ROOT / config["development_oof"])
    development = development.loc[bool_series(development["oof_clean"])].copy()
    definitions = json.loads((ROOT / config["label_definitions"]).read_text(encoding="utf-8"))
    labels = list(definitions)
    system = (ROOT / config["system_prompt"]).read_text(encoding="utf-8").strip()
    cache_path = ROOT / config["standards_cache"]
    corpus = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    rows = []
    for position, (_, row) in enumerate(sample.iterrows()):
        reports = report_blocks(row, development, config["report_top_k"])
        standards = standard_blocks(row, corpus, config["standards_top_k"])
        ordered = list(np.random.default_rng(config["seed"] + position).permutation(labels))
        for condition in config["conditions"]:
            blocks = []
            if condition in {"report_rag", "combined_rag"}:
                blocks.extend(reports)
            if condition in {"standards_rag", "combined_rag"}:
                blocks.extend(standards)
            prompt = render_prompt(row, condition, ordered, definitions, blocks)
            source_map = {"target": str(row["text_context_v2"]), **{b["source_id"]: b["text"] for b in blocks}}
            rows.append({
                **row.to_dict(),
                "condition": condition,
                "request_id": f"{row['sample_id']}::{condition}",
                "label_order_json": json.dumps(ordered, ensure_ascii=False),
                "retrieved_blocks_json": json.dumps(blocks, ensure_ascii=False),
                "source_map_json": json.dumps(source_map, ensure_ascii=False),
                "user_prompt": prompt,
                "user_prompt_sha256": sha256_text(prompt),
                "system_prompt_sha256": sha256_text(system),
                "prompt_bundle_sha256": sha256_text(system + "\n\n" + prompt),
                "prompt_version": config["version"],
                "report_context_count": sum(b["source_id"].startswith("report:") for b in blocks),
                "standards_context_count": sum(b["source_id"].startswith("standard:") for b in blocks),
            })
    output = pd.DataFrame(rows)
    path = ROOT / config["candidate_table"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8")
    print(f"samples={len(sample)} candidates={len(output)} standards_cached={len(corpus)} path={path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.resample or not (ROOT / config["sample_table"]).exists():
        sample = make_sample(config)
        print(sample["pilot_stratum"].value_counts().to_string())
    prepare_candidates(config)


if __name__ == "__main__":
    main()
