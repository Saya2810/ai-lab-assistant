"""Deterministic planning rules (no LLM).

* ``enrich_with_regex``: facts the LLM must not miss or invent (experiment
  numbers, ISO dates), applied to every plan;
* ``heuristic_plan``: the fallback planner when the LLM fails (invalid JSON,
  timeout). It goes through the same catalog validation as the LLM path, so a
  degraded planner can never produce an invalid filter;
* ``is_chitchat``: greetings and thanks need no planning at all.
"""

from __future__ import annotations

import re
from datetime import date

from lab_chatbot.plan import Plan, RawFilters, RawMeasurement, RawPlan
from lab_chatbot.plan_validation import normalize_plan
from lab_chatbot.sql_tools import Catalog
from lab_chatbot.store import norm
from lab_chatbot.units import lookup_unit, parse_number

_ID_PATTERN = re.compile(
    r"(?:experiment(?:s|e)?|exp\.?|versuch(?:e)?|eintrag|entry|#)\s*[:#]?\s*(\d+(?:\s*(?:,|und|and|&)\s*\d+)*)",
    re.I,
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

_COUNT_WORDS = re.compile(r"\b(how many|number of|count|wie viele|anzahl|wieviele)\b", re.I)
_LIST_WORDS = re.compile(
    r"\b(list|show all|welche (?:experimente|versuche|einträge|messungen)|which (?:experiments|entries|runs)|"
    r"zeige alle|zeig alle|liste)\b",
    re.I,
)
_CHITCHAT = re.compile(
    r"^\s*(hi|hallo|hey|moin|danke|thanks|thank you|thx|guten (?:morgen|tag|abend)|"
    r"was kannst du|what can you do|help|hilfe)\b[\s!.?]*$",
    re.I,
)

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_UNIT = r"[A-Za-zµμ°Å%][A-Za-zµμ°Å%/^\-\d]{0,9}"
_OP_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "between",
        re.compile(
            rf"(?:zwischen|between)\s+(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?\s*"
            rf"(?:und|and|bis|to|-|–)\s*(?P<b>{_NUM})\s*(?P<ub>{_UNIT})?",
            re.I,
        ),
    ),
    (">=", re.compile(rf"(?:>=|≥|mindestens|at least)\s*(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?", re.I)),
    ("<=", re.compile(rf"(?:<=|≤|höchstens|at most|bis zu|up to)\s*(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?", re.I)),
    (
        ">",
        re.compile(
            rf"(?:>|über|mehr als|größer als|grösser als|höher als|above|over|greater than|more than|higher than)"
            rf"\s*(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?",
            re.I,
        ),
    ),
    (
        "<",
        re.compile(
            rf"(?:<|unter|weniger als|kleiner als|niedriger als|below|under|less than|lower than|fewer than)"
            rf"\s*(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?",
            re.I,
        ),
    ),
    ("=", re.compile(rf"(?:=|genau|exakt|exactly|gleich|equal to|bei|at)\s*(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?", re.I)),
]
_AGG_WORDS: list[tuple[str, re.Pattern]] = [
    ("max", re.compile(r"\b(maxim\w*|höchst\w*|größt\w*|grösst\w*|highest|largest|biggest|max)\b", re.I)),
    ("min", re.compile(r"\b(minim\w*|niedrigst\w*|kleinst\w*|geringst\w*|lowest|smallest|min)\b", re.I)),
    ("avg", re.compile(r"\b(durchschnitt\w*|mittelwert\w*|mittler\w*|average|mean|avg)\b", re.I)),
    ("sum", re.compile(r"\b(summe|gesamt\w*|insgesamt|total|sum)\b", re.I)),
]
_GROUP_WORDS = re.compile(
    r"\b(?:pro|per|je|nach|by)\s+(tag|status|kategorie|category|autor|author|monat|month)\b", re.I
)
_GROUP_MAP = {
    "tag": "tag", "status": "status", "kategorie": "category", "category": "category",
    "autor": "author", "author": "author", "monat": "month", "month": "month",
}  # fmt: skip


def is_chitchat(question: str) -> bool:
    return bool(_CHITCHAT.match(question))


def extract_ids(question: str) -> list[int]:
    ids: list[int] = []
    for match in _ID_PATTERN.finditer(question):
        ids += [int(n) for n in re.findall(r"\d+", match.group(1))]
    return list(dict.fromkeys(ids))


def enrich_with_regex(plan: Plan) -> Plan:
    """Cheap, deterministic facts the LLM must not be able to miss or invent."""
    ids = extract_ids(plan.question)
    if ids and not plan.filters.experiment_ids:
        plan.filters.experiment_ids = ids
    dates = _ISO_DATE.findall(plan.question)
    if dates and not (plan.filters.date_from or plan.filters.date_to):
        valid = []
        for d in dates:
            try:
                valid.append(date.fromisoformat(d).isoformat())
            except ValueError:
                pass
        if valid:
            plan.filters.date_from, plan.filters.date_to = min(valid), max(valid)
    return plan


# ---------------------------------------------------------------------------
# heuristic fallback planner
# ---------------------------------------------------------------------------


def _unit_or_none(token: str | None) -> str | None:
    return token if token and lookup_unit(token) else None


def _heuristic_measurement(question: str, catalog: Catalog) -> RawMeasurement | None:
    """Find "<measurement name> ... <op> <number> <unit>" with plain rules."""
    lowered = norm(question)
    name = None
    for m in sorted(catalog.measurements, key=lambda m: -len(m.name)):
        # prefix match so German plurals/inflections ("Wellenlängen") still hit
        if re.search(rf"(?<!\w){re.escape(norm(m.name))}", lowered):
            name = m.name
            break
    if name is None:
        return None
    raw = RawMeasurement(name=name)
    for op, pattern in _OP_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        a = parse_number(match.group("a"))
        if a is None:
            continue
        raw.op, raw.value, raw.unit = op, a, _unit_or_none(match.group("ua"))
        if op == "between":
            b = parse_number(match.group("b"))
            if b is None:
                raw.op, raw.value = None, None
                continue
            raw.value2 = b
            raw.unit = raw.unit or _unit_or_none(match.group("ub"))
        break
    for agg, pattern in _AGG_WORDS:
        if pattern.search(question):
            raw.agg = agg
            break
    return raw


def heuristic_plan(question: str, catalog: Catalog, default_language: str = "de") -> Plan:
    """Rule-based plan used when the planner LLM fails (invalid JSON, timeout, ...)."""
    lowered = norm(question)
    filters = RawFilters()
    for tag in catalog.tags:
        if re.search(rf"(?<!\w){re.escape(norm(tag))}(?!\w)", lowered):
            filters.tags_all.append(tag)
    for attr, options in (("category", catalog.categories), ("status", catalog.statuses)):
        for option in options:
            if re.search(rf"(?<!\w){re.escape(norm(option))}(?!\w)", lowered):
                setattr(filters, attr, option)
                break

    measurement = _heuristic_measurement(question, catalog)
    group_match = _GROUP_WORDS.search(question)
    if is_chitchat(question):
        intent = "chitchat"
    elif _COUNT_WORDS.search(question):
        intent = "count"
    elif measurement and measurement.agg and not _LIST_WORDS.search(question):
        intent = "aggregate"
    elif _LIST_WORDS.search(question):
        intent = "list"
    else:
        intent = "search"

    raw = RawPlan(
        intent=intent,
        semantic_query=question,
        filters=filters,
        measurement=measurement if intent != "chitchat" else None,
        group_by=_GROUP_MAP[group_match.group(1).lower()] if group_match else None,
    )
    plan = normalize_plan(raw, catalog, question, default_language)
    plan.source = "heuristic"
    plan.notes.insert(0, "Planner LLM unavailable or invalid; used keyword rules.")
    return enrich_with_regex(plan)
