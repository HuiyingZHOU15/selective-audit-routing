"""Fetch referenced PCAOB standards from official HTML pages and cache paragraphs."""

from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from llm_common import ROOT
from prepare_pcaob_rag_pilot_v1 import load_config, standard_ids


PARAGRAPH_RE = re.compile(r"(?m)(?=\.(?:[A-Z]?\d{1,3})\s+)")


def extract_paragraphs(standard: str, html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    pieces = PARAGRAPH_RE.split(text)
    paragraphs = []
    for piece in pieces:
        match = re.match(r"\.(?P<num>[A-Z]?\d{1,3})\s+(?P<body>.+)", piece, flags=re.S)
        if not match:
            continue
        body = re.sub(r"\s+", " ", match.group("body")).strip()
        if len(body) < 30:
            continue
        number = f".{match.group('num')}"
        paragraphs.append({
            "source_id": f"standard:{standard}:{number}",
            "standard": standard,
            "paragraph": number,
            "text": f"{standard} {number} {body}",
            "url": url,
        })
    return paragraphs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    config = load_config()
    sample = pd.read_csv(ROOT / config["sample_table"])
    standards = sorted({f"AS{value}" for raw in sample["standard_refs"] for value in standard_ids(raw)})
    path = ROOT / config["standards_cache"]
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    session = requests.Session()
    session.headers["User-Agent"] = "SelectiveAuditRoutingResearch/1.0 (academic reproducibility)"
    for standard in standards:
        if cache.get(standard, {}).get("paragraphs"):
            continue
        url = f"https://pcaobus.org/oversight/standards/auditing-standards/details/{standard}"
        response = session.get(url, timeout=60)
        response.raise_for_status()
        paragraphs = extract_paragraphs(standard, response.text, url)
        if not paragraphs:
            raise RuntimeError(f"No paragraphs parsed for {standard} from {url}")
        cache[standard] = {"url": url, "paragraphs": paragraphs}
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{standard}: {len(paragraphs)} paragraphs", flush=True)
        time.sleep(args.delay)
    print(f"cached={len(cache)} path={path}")


if __name__ == "__main__":
    main()
