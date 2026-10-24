"""Shared prompt, schema, and validation utilities for the DeepSeek reliability experiment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "llm_deepseek_v1.json"


class LLMDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_label: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str
    reason: str
    standard_citation: str | None
    abstain: bool

    @model_validator(mode="after")
    def check_abstention(self) -> "LLMDecision":
        if self.abstain and self.predicted_label is not None:
            raise ValueError("predicted_label must be null when abstain=true")
        if not self.abstain and not self.predicted_label:
            raise ValueError("predicted_label is required when abstain=false")
        if not self.abstain and not self.evidence_quote.strip():
            raise ValueError("evidence_quote is required when abstain=false")
        if self.standard_citation is not None:
            raise ValueError("standard_citation must be null in the closed-book phase")
        return self


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def decision_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "predicted_label": {
                "anyOf": [
                    {"type": "string", "enum": labels},
                    {"type": "null"},
                ]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_quote": {"type": "string"},
            "reason": {"type": "string"},
            "standard_citation": {"type": "null"},
            "abstain": {"type": "boolean"},
        },
        "required": [
            "predicted_label",
            "confidence",
            "evidence_quote",
            "reason",
            "standard_citation",
            "abstain",
        ],
        "additionalProperties": False,
    }


def render_user_prompt(text: str, ordered_labels: list[str], definitions: dict[str, str]) -> str:
    choices = "\n".join(f"- {label}: {definitions[label]}" for label in ordered_labels)
    return (
        "Choose one audit-area label from this list, or abstain if the evidence is insufficient.\n\n"
        f"LABELS (presentation order is randomized):\n{choices}\n\n"
        "FINDING TEXT:\n"
        "<finding>\n"
        f"{text}\n"
        "</finding>"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_evidence(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def evidence_match(text: str, quote: str) -> tuple[bool, bool]:
    quote = quote.strip()
    if not quote:
        return False, False
    exact = quote in text
    normalized = normalize_evidence(quote) in normalize_evidence(text)
    return exact, normalized
