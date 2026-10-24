"""Minimal structured retrieval and deterministic calculation for FinMR queries.

The parser uses only the public query text.  It never reads the gold answer or
the dataset's DQC label when retrieving evidence or computing a candidate.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


SECTION_RE = re.compile(r"^##([^\r\n]+)\s*$", re.MULTILINE)
ATTR_RE = re.compile(r"([\w:.-]+)=[\"']([^\"']*)[\"']")
CONTEXT_RE = re.compile(r"<context\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*>(.*?)</context>", re.DOTALL | re.IGNORECASE)
FACT_RE = re.compile(r"<([\w.-]+:[\w.-]+)\b([^>]*)>([^<>]+)</\1>", re.DOTALL)
LOC_RE = re.compile(r"<link:loc\b([^>]*)/?>", re.IGNORECASE)
ARC_RE = re.compile(r"<link:calculationArc\b([^>]*)/?>", re.IGNORECASE)
CALC_LINK_RE = re.compile(r"<link:calculationLink\b([^>]*)>(.*?)</link:calculationLink>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    target: str
    period_text: str
    start: str | None
    end: str


@dataclass(frozen=True)
class Context:
    context_id: str
    start: str | None
    end: str | None
    instant: str | None
    segment: str

    @property
    def signature(self) -> tuple[str | None, str | None, str | None, str]:
        normalized_segment = re.sub(r"\s+", " ", self.segment).strip()
        return self.start, self.end, self.instant, normalized_segment


@dataclass(frozen=True)
class Fact:
    concept: str
    context_ref: str
    value_text: str
    decimals: str | None
    unit_ref: str | None


@dataclass
class Retrieval:
    question: Question
    reported_fact: Fact | None
    children: list[tuple[Fact, Decimal]]
    calculated_value: Decimal | None
    calc_role: str | None
    evidence_text: str
    status: str


def attributes(text: str) -> dict[str, str]:
    return {key: html.unescape(value) for key, value in ATTR_RE.findall(text)}


def split_sections(query: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(query))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        sections[match.group(1).strip().lower()] = query[match.end():end].strip()
    return sections


def parse_question(query: str) -> Question:
    match = re.search(
        r"Question1:\s*What's the reported value of\s+(\S+)\s+in the Instance document for the period\s+([^?]+)\?",
        query,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("FinMR Question1 was not found")
    target, period = match.group(1), match.group(2).strip()
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", period)
    if len(dates) == 1:
        return Question(target, period, None, dates[0])
    if len(dates) == 2:
        return Question(target, period, dates[0], dates[1])
    raise ValueError(f"Unsupported period: {period}")


def parse_contexts(instance: str) -> dict[str, Context]:
    contexts = {}
    for context_id, body in CONTEXT_RE.findall(instance):
        start = re.search(r"<startDate>([^<]+)</startDate>", body, re.IGNORECASE)
        end = re.search(r"<endDate>([^<]+)</endDate>", body, re.IGNORECASE)
        instant = re.search(r"<instant>([^<]+)</instant>", body, re.IGNORECASE)
        segment = re.search(r"<segment>(.*?)</segment>", body, re.DOTALL | re.IGNORECASE)
        contexts[context_id] = Context(
            context_id=context_id,
            start=start.group(1).strip() if start else None,
            end=end.group(1).strip() if end else None,
            instant=instant.group(1).strip() if instant else None,
            segment=segment.group(1).strip() if segment else "",
        )
    return contexts


def parse_facts(instance: str) -> list[Fact]:
    facts = []
    for concept, raw_attrs, raw_value in FACT_RE.findall(instance):
        attrs = attributes(raw_attrs)
        context_ref = attrs.get("contextRef")
        if not context_ref:
            continue
        facts.append(
            Fact(
                concept=concept,
                context_ref=context_ref,
                value_text=html.unescape(raw_value).strip(),
                decimals=attrs.get("decimals"),
                unit_ref=attrs.get("unitRef"),
            )
        )
    return facts


def concept_from_href(href: str) -> str:
    fragment = href.rsplit("#", 1)[-1]
    if "_" in fragment:
        prefix, local = fragment.split("_", 1)
        return f"{prefix}:{local}"
    return fragment


def parse_calculation_links(calculation: str) -> list[dict]:
    links = []
    for raw_attrs, body in CALC_LINK_RE.findall(calculation):
        link_attrs = attributes(raw_attrs)
        locators = {}
        for loc_raw in LOC_RE.findall(body):
            attrs = attributes(loc_raw)
            label = attrs.get("xlink:label")
            href = attrs.get("xlink:href")
            if label and href:
                locators[label] = concept_from_href(href)
        arcs = []
        for arc_raw in ARC_RE.findall(body):
            attrs = attributes(arc_raw)
            parent = locators.get(attrs.get("xlink:from", ""))
            child = locators.get(attrs.get("xlink:to", ""))
            if not parent or not child:
                continue
            try:
                weight = Decimal(attrs.get("weight", "1"))
            except InvalidOperation:
                continue
            arcs.append((parent, child, weight))
        links.append({"role": link_attrs.get("xlink:role"), "arcs": arcs})
    return links


def decimal_value(text: str) -> Decimal | None:
    normalized = text.strip().replace(",", "")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def period_matches(context: Context, question: Question) -> bool:
    if question.start is None:
        return context.instant == question.end
    return context.start == question.start and context.end == question.end


def retrieve(query: str) -> Retrieval:
    sections = split_sections(query)
    question = parse_question(query)
    instance = sections.get("instance document", "")
    calculation = sections.get("calculation linkbase document", "")
    taxonomy = sections.get("us gaap taxonomy", "")
    contexts = parse_contexts(instance)
    facts = parse_facts(instance)
    facts_by_concept: dict[str, list[Fact]] = {}
    for fact in facts:
        facts_by_concept.setdefault(fact.concept, []).append(fact)

    target_candidates = [
        fact for fact in facts_by_concept.get(question.target, [])
        if fact.context_ref in contexts and period_matches(contexts[fact.context_ref], question)
    ]
    # Consolidated/no-dimension context is the deterministic first choice.
    target_candidates.sort(key=lambda fact: (bool(contexts[fact.context_ref].segment), fact.context_ref))
    reported = target_candidates[0] if target_candidates else None

    selected_children: list[tuple[Fact, Decimal]] = []
    calculated = None
    role = None
    if reported:
        target_context = contexts[reported.context_ref]
        signature = target_context.signature
        relevant_links = []
        for link in parse_calculation_links(calculation):
            direct_arcs = [arc for arc in link["arcs"] if arc[0] == question.target]
            if direct_arcs:
                relevant_links.append((link, direct_arcs))
        # Prefer the link with the largest number of retrievable child facts.
        candidates = []
        for link, arcs in relevant_links:
            child_rows = []
            for _, child_concept, weight in arcs:
                child_candidates = [
                    fact for fact in facts_by_concept.get(child_concept, [])
                    if fact.context_ref in contexts and contexts[fact.context_ref].signature == signature
                ]
                if child_candidates:
                    child_rows.append((child_candidates[0], weight))
            candidates.append((len(child_rows), len(arcs), link.get("role"), child_rows))
        if candidates:
            _, arc_count, role, selected_children = max(candidates, key=lambda value: (value[0], value[1]))
            numeric = [(decimal_value(fact.value_text), weight) for fact, weight in selected_children]
            if selected_children and len(selected_children) == arc_count and all(value is not None for value, _ in numeric):
                calculated = sum((value * weight for value, weight in numeric if value is not None), Decimal(0))

    taxonomy_lines = []
    local = question.target.split(":", 1)[-1]
    for line in taxonomy.splitlines():
        if local.lower() in line.lower() or line.startswith(("Label:", "Type:", "Calculation:")):
            taxonomy_lines.append(line.strip())
    evidence = [
        f"TARGET: {question.target}",
        f"PERIOD: {question.period_text}",
        f"REPORTED_FACT: {reported.concept} context={reported.context_ref} value={reported.value_text}" if reported else "REPORTED_FACT: NOT_FOUND",
        f"CALCULATION_ROLE: {role or 'NOT_FOUND'}",
    ]
    for fact, weight in selected_children:
        evidence.append(f"CHILD_FACT: concept={fact.concept} context={fact.context_ref} value={fact.value_text} weight={weight}")
    evidence.append(f"DETERMINISTIC_SUM: {calculated if calculated is not None else 'NOT_COMPUTABLE'}")
    evidence.extend(f"TAXONOMY: {line}" for line in taxonomy_lines[:20])
    status = "complete" if reported and calculated is not None else ("reported_only" if reported else "missing_reported")
    return Retrieval(question, reported, selected_children, calculated, role, "\n".join(evidence), status)

