"""Turn an untrusted ``RawPlan`` into a ``Plan`` that only uses what really exists.

Two checks, both deterministic:

* ``normalize_plan``: every tag, category, status, author and measurement name
  is snapped to a real catalog entry (exact, unique substring or close match) or
  dropped with a note; dates must parse; a comparison without a unit gets the
  measurement's usual unit.
* ``ground_filters``: a filter value must also be *mentioned* by the user. Small
  planner models sometimes invent filters, which then silently match nothing.
"""

from __future__ import annotations

import difflib
import re
from datetime import date
from typing import Sequence

from lab_chatbot.language import detect_language
from lab_chatbot.plan import AGGS, INTENTS, Plan, RawFilters, RawPlan
from lab_chatbot.retrieval import fold, query_tokens
from lab_chatbot.sql_tools import GROUP_BYS, OPS, Catalog, Filters, MeasurementCond
from lab_chatbot.store import norm
from lab_chatbot.units import lookup_unit


def snap(value: str, options: Sequence[str], cutoff: float = 0.8) -> str | None:
    """Map ``value`` to one real option (exact, unique-substring, or close match)."""
    key = norm(value)
    keyed = {norm(o): o for o in options}
    if key in keyed:
        return keyed[key]
    if len(key) >= 4:
        contained = [o for k, o in keyed.items() if key in k or k in key]
        if len(contained) == 1:
            return contained[0]
    close = difflib.get_close_matches(key, list(keyed), n=1, cutoff=cutoff)
    return keyed[close[0]] if close else None


def _valid_date(value: str | None, notes: list[str]) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        notes.append(f"Ignored invalid date '{value}'.")
        return None


def normalize_plan(
    raw: RawPlan, catalog: Catalog, question: str, default_language: str = "de"
) -> Plan:
    notes: list[str] = []
    f = raw.filters or RawFilters()
    intent = raw.intent if raw.intent in INTENTS else "search"

    def snap_all(values: list[str], options: list[str], label: str) -> list[str]:
        result = []
        for value in values:
            match = snap(value, options)
            if match:
                if match not in result:
                    result.append(match)
            else:
                notes.append(f"Ignored unknown {label} '{value}'.")
        return result

    filters = Filters(
        date_from=_valid_date(f.date_from, notes),
        date_to=_valid_date(f.date_to, notes),
        tags_all=snap_all(f.tags_all, catalog.tag_options(), "tag"),
        tags_any=snap_all(f.tags_any, catalog.tag_options(), "tag"),
        experiment_ids=sorted(set(f.experiment_ids)),
    )
    if filters.date_from and filters.date_to and filters.date_from > filters.date_to:
        filters.date_from, filters.date_to = filters.date_to, filters.date_from
        notes.append("Swapped date range (from was after to).")
    for attr, options, label in (
        ("category", catalog.category_options(), "category"),
        ("status", catalog.status_options(), "status"),
        ("author", catalog.author_options(), "author"),
    ):
        wanted = getattr(f, attr)
        if wanted:
            match = snap(wanted, options)
            if match:
                setattr(filters, attr, match)
            else:
                notes.append(f"Ignored unknown {label} '{wanted}'.")

    measurement = None
    agg = None
    m = raw.measurement
    if m and m.name.strip():
        known = catalog.measurement(m.name)
        if known is None:
            close = snap(m.name, catalog.measurement_names())
            known = catalog.measurement(close) if close else None
        if known is None:
            notes.append(f"Ignored unknown measurement '{m.name}'.")
        else:
            op = m.op if m.op in OPS and m.value is not None else None
            if op == "between" and m.value2 is None:
                op = None
            unit = m.unit
            if op and not unit and known.unit:
                unit = known.unit
                notes.append(f"No unit given; assumed {known.unit} for {known.name}.")
            if op and unit and not lookup_unit(unit):
                notes.append(f"Unit '{unit}' is not recognised; comparing raw numbers.")
            # unit is kept even without a comparison: aggregates use it as display unit
            measurement = MeasurementCond(
                known.name,
                op,
                m.value if op else None,
                m.value2 if op == "between" else None,
                unit,
            )
            agg = m.agg if m.agg in AGGS else None

    if intent == "aggregate" and measurement is None:
        notes.append("Aggregate without a known measurement; falling back to text search.")
        intent = "search"
    group_by = raw.group_by if raw.group_by in GROUP_BYS else None
    if intent != "count":
        group_by = None

    return Plan(
        intent=intent,
        question=question,
        semantic_query=(raw.semantic_query or "").strip() or question,
        filters=filters,
        measurement=measurement,
        agg=agg,
        group_by=group_by,
        order="date_asc" if raw.order == "date_asc" else "date_desc",
        notes=notes,
        source="llm",
        language=detect_language(question, default_language),
    )


# ---------------------------------------------------------------------------
# grounding: the user must have asked for a filter
# ---------------------------------------------------------------------------


def _value_named(value: str, folded_text: str, tokens: list[str]) -> bool:
    """Does the user's text name ``value`` (accent/case-insensitive, inflection- and typo-tolerant)?"""
    folded = fold(value)
    if not folded.strip() or folded in folded_text:
        return True
    for word in re.findall(r"\w+", folded):
        if len(word) < 3:
            continue
        for token in tokens:
            if word == token:
                return True
            if len(word) >= 4 and len(token) >= 4:
                head = max(4, min(len(word), len(token)) - 2)
                if word[:head] == token[:head]:
                    return True
                if difflib.SequenceMatcher(None, word, token).ratio() >= 0.84:
                    return True
    return False


def ground_filters(plan: Plan, question: str, history: Sequence[dict] | None = None) -> Plan:
    """Drop tag/category/status/author/id filters the user never mentioned.

    Small planner models sometimes copy half the catalog into the filters (seen:
    every tag plus a status for "Bei welchem Experiment gab es Rauschen bei
    Probe A3?") or take experiment ids from the prompt's examples. Such a filter
    silently returns nothing. A filter value must therefore appear in the
    question or in one of the last user turns (follow-ups such as "und davon nur
    die abgeschlossenen?"); otherwise it is removed and the search runs wider
    instead of coming back empty.
    """
    recent = [m.get("content", "") for m in (history or []) if m.get("role") == "user"][-2:]
    text = " ".join([*recent, question])
    folded_text, tokens = fold(text), [fold(t) for t in query_tokens(text)]
    f = plan.filters

    def keep(value: str | None, label: str) -> bool:
        if not value or _value_named(value, folded_text, tokens):
            return True
        plan.notes.append(f"Dropped {label} '{value}': not mentioned in the question.")
        return False

    f.tags_all = [v for v in f.tags_all if keep(v, "tag")]
    f.tags_any = [v for v in f.tags_any if keep(v, "tag")]
    numbers = {int(n) for n in re.findall(r"(?<![\w.,])(\d+)(?![\w.,]\d)", text)}
    invented = [i for i in f.experiment_ids if i not in numbers]
    if invented:
        f.experiment_ids = [i for i in f.experiment_ids if i in numbers]
        plan.notes.append(f"Dropped experiment ids {invented}: not in the question.")
    for attr in ("category", "status", "author"):
        if not keep(getattr(f, attr), attr):
            setattr(f, attr, None)
    return plan
