"""Query planner: decides *how* a question is answered.

The planner is the "intelligent integration" of SQL and RAG. For every
question it produces a small, validated ``Plan``:

    count / aggregate / list   -> exact answers from SQL (no LLM can corrupt a number)
    search                     -> RAG over the text, optionally restricted by SQL
                                  filters ("wavelength > 500 nm" first, then meaning)
    chitchat                   -> greeting / meta question

One LLM call (structured JSON output, temperature 0) proposes the plan. The
proposal is then *validated against what actually exists* in the index: tags,
categories, statuses and measurement names are snapped to real catalog entries
(or dropped with a note), dates must parse, units are converted. If the LLM
fails or returns garbage, a deterministic heuristic planner takes over, so
the chatbot degrades instead of breaking.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from lab_chatbot.errors import BackendUnavailable
from lab_chatbot.llm import LLM
from lab_chatbot.sql_tools import (
    GROUP_BYS,
    OPS,
    Catalog,
    Filters,
    MeasurementCond,
    load_catalog,
)
from lab_chatbot.store import Store, norm
from lab_chatbot.units import lookup_unit, parse_number

log = logging.getLogger(__name__)

INTENTS = ("search", "count", "aggregate", "list", "chitchat")
AGGS = ("min", "max", "avg", "sum", "count")

# ---------------------------------------------------------------------------
# language (only used for the deterministic, non-LLM answers)
# ---------------------------------------------------------------------------

_DE = set(
    "der die das und ist wie viele welche welcher welches was wann mit von im den dem ein eine nicht "
    "für wurde wurden gab hat haben bei zeige zeig alle mir ich gibt experimente experiment versuch "
    "versuche eintrag einträge letzte letzten diese dieser durchschnitt höchste niedrigste wo wer".split()
)  # fmt: skip
_EN = set(
    "the and is are how many which what when with from of in a an not for was were has have at show "
    "all me i there experiments experiment entry entries last this average highest lowest where who".split()
)  # fmt: skip


# Words valid in both languages ("in", "was", "an", ...) say nothing about the language.
_COMMON = (_DE & _EN) | {"in", "an"}
_DE, _EN = _DE - _COMMON, _EN - _COMMON


def detect_language(text: str, default: str = "de") -> str:
    tokens = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(t in _DE for t in tokens)
    en = sum(t in _EN for t in tokens)
    if de == en:
        return default
    return "de" if de > en else "en"


# ---------------------------------------------------------------------------
# the validated plan
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    intent: str
    question: str
    semantic_query: str
    filters: Filters = field(default_factory=Filters)
    measurement: MeasurementCond | None = None
    agg: str | None = None
    group_by: str | None = None
    order: str = "date_desc"
    limit: int = 20
    notes: list[str] = field(default_factory=list)
    source: str = "llm"  # "llm" | "heuristic"
    language: str = "de"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# raw (untrusted) LLM output
# ---------------------------------------------------------------------------


def _blank_to_none(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _as_list(value):
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


class RawFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")
    date_from: str | None = None
    date_to: str | None = None
    tags_all: list[str] = []
    tags_any: list[str] = []
    category: str | None = None
    status: str | None = None
    author: str | None = None
    experiment_ids: list[int] = []

    @field_validator("date_from", "date_to", "category", "status", "author", mode="before")
    @classmethod
    def _blank(cls, v):
        return _blank_to_none(v)

    @field_validator("tags_all", "tags_any", mode="before")
    @classmethod
    def _lists(cls, v):
        return [str(x) for x in _as_list(v)]

    @field_validator("experiment_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        ids: list[int] = []
        for item in _as_list(v):
            ids += [int(n) for n in re.findall(r"\d+", str(item))]
        return ids


class RawMeasurement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    op: str | None = None
    value: float | None = None
    value2: float | None = None
    unit: str | None = None
    agg: str | None = None

    @field_validator("op", "unit", "agg", mode="before")
    @classmethod
    def _blank(cls, v):
        return _blank_to_none(v)

    @field_validator("value", "value2", mode="before")
    @classmethod
    def _numbers(cls, v):
        v = _blank_to_none(v)
        return parse_number(v) if isinstance(v, str) else v


class RawPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    intent: str = "search"
    semantic_query: str = ""
    filters: RawFilters | None = None
    measurement: RawMeasurement | None = None
    group_by: str | None = None
    order: str | None = None

    @field_validator("group_by", "order", mode="before")
    @classmethod
    def _blank(cls, v):
        return _blank_to_none(v)


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "semantic_query": {"type": "string"},
        "filters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
                "tags_all": {"type": "array", "items": {"type": "string"}},
                "tags_any": {"type": "array", "items": {"type": "string"}},
                "category": {"type": "string"},
                "status": {"type": "string"},
                "author": {"type": "string"},
                "experiment_ids": {"type": "array", "items": {"type": "integer"}},
            },
        },
        "measurement": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "op": {"type": "string", "enum": ["", ">", ">=", "<", "<=", "=", "between"]},
                "value": {"type": "number"},
                "value2": {"type": "number"},
                "unit": {"type": "string"},
                "agg": {"type": "string", "enum": ["", *AGGS]},
            },
        },
        "group_by": {"type": "string", "enum": ["", *sorted(GROUP_BYS)]},
        "order": {"type": "string", "enum": ["date_desc", "date_asc"]},
    },
    "required": ["intent", "semantic_query"],
}

PLANNER_SYSTEM = """\
You are the query planner of a lab-notebook assistant. You do NOT answer the question.
You translate it into a JSON plan that decides how it is answered. Output JSON only.

Fields:
- intent:
  "count"     = how many experiments (optionally per tag/category/status/author/month)
  "aggregate" = min / max / average / sum of ONE numeric measurement
  "list"      = which/list experiments that match filters, no explanation needed
  "search"    = anything that needs the TEXT of the entries: why, how, what happened,
                explain, summarise, problems, results, comparisons, details of an experiment
  "chitchat"  = greeting, thanks, or a question about what you can do
- semantic_query: the question rewritten as a STANDALONE search phrase in the user's language
  (resolve "it", "that one", "the previous experiment" using the conversation). Keep only the
  content words; leave out what the filters already express.
- filters: only what the user actually asked for.
  Use ONLY tag/category/status/measurement names that appear in the CATALOG, spelled exactly
  as there. If the user names something that is not in the catalog, leave the filter out and
  keep the words in semantic_query.
  Dates are ISO (YYYY-MM-DD) and inclusive; resolve relative dates ("last week",
  "in October", "yesterday") from TODAY.
  experiment_ids: numbers the user refers to as experiment / Versuch / Eintrag / #.
- measurement: for numeric conditions or aggregates. name from the catalog. op is one of
  > >= < <= = between (leave empty for no condition). value (and value2 for between) are
  plain numbers; unit is the unit the USER said (empty if none).
  For "aggregate", set agg to min/max/avg/sum/count.
- group_by: only for "count": tag, category, status, author or month; otherwise empty.
- order: date_desc (newest first, default) or date_asc.

If the question mixes a numeric condition with a text question (e.g. "why was the noise high
in the runs above 500 nm?") use intent "search" AND fill measurement: the numbers filter the
experiments, the text is then searched inside them.
"""

PLANNER_EXAMPLES = """\
Examples (catalog names are placeholders):
Q: Wie viele Experimente haben den Tag "raman"?
{"intent":"count","semantic_query":"","filters":{"tags_all":["raman"]}}
Q: Was war die maximale Belichtungszeit?
{"intent":"aggregate","semantic_query":"","measurement":{"name":"Exposure","agg":"max"}}
Q: Welche Messungen hatten eine Wellenlänge über 600 nm?
{"intent":"list","semantic_query":"","measurement":{"name":"Wavelength","op":">","value":600,"unit":"nm"}}
Q: Warum war das Rauschen bei Probe A3 so hoch?
{"intent":"search","semantic_query":"Rauschen Probe A3 Ursache","filters":{}}
Q: Danke!
{"intent":"chitchat","semantic_query":""}
"""


def build_planner_messages(
    question: str,
    catalog: Catalog,
    today: date,
    history: Sequence[dict] | None,
) -> list[dict]:
    lines = [
        f"TODAY: {today.isoformat()} ({today.strftime('%A')})",
        "",
        "CATALOG (what exists in this lab notebook):",
        f"- experiments: {catalog.n_experiments}"
        + (f", dated {catalog.date_min} to {catalog.date_max}" if catalog.date_min else ""),
        f"- tags: {', '.join(catalog.tags) or '(none)'}",
        f"- categories: {', '.join(catalog.categories) or '(none)'}",
        f"- statuses: {', '.join(catalog.statuses) or '(none)'}",
        "- measurement names (usual unit): "
        + (
            "; ".join(f"{m.name} ({m.unit or 'no unit'})" for m in catalog.measurements)
            or "(none)"
        ),
    ]
    if history:
        lines += ["", "CONVERSATION SO FAR:"]
        for message in history[-4:]:
            lines.append(f"{message['role'].upper()}: {str(message['content'])[:400]}")
    lines += ["", f"QUESTION: {question}", "", "Return the JSON plan."]
    return [
        {"role": "system", "content": PLANNER_SYSTEM + "\n" + PLANNER_EXAMPLES},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ---------------------------------------------------------------------------
# validation against the catalog
# ---------------------------------------------------------------------------


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


_ID_PATTERN = re.compile(
    r"(?:experiment(?:s|e)?|exp\.?|versuch(?:e)?|eintrag|entry|#)\s*[:#]?\s*(\d+(?:\s*(?:,|und|and|&)\s*\d+)*)",
    re.I,
)
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def extract_ids(question: str) -> list[int]:
    ids: list[int] = []
    for match in _ID_PATTERN.finditer(question):
        ids += [int(n) for n in re.findall(r"\d+", match.group(1))]
    return list(dict.fromkeys(ids))


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
        tags_all=snap_all(f.tags_all, catalog.tags, "tag"),
        tags_any=snap_all(f.tags_any, catalog.tags, "tag"),
        experiment_ids=sorted(set(f.experiment_ids)),
    )
    if filters.date_from and filters.date_to and filters.date_from > filters.date_to:
        filters.date_from, filters.date_to = filters.date_to, filters.date_from
        notes.append("Swapped date range (from was after to).")
    for attr, options, label in (
        ("category", catalog.categories, "category"),
        ("status", catalog.statuses, "status"),
        ("author", catalog.authors, "author"),
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
            close = snap(m.name, [c.name for c in catalog.measurements])
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
# heuristic fallback
# ---------------------------------------------------------------------------

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
    """Rule-based plan used when the planner LLM fails (invalid JSON, timeout, ...).

    Deliberately simple, but it goes through the same catalog validation as the
    LLM path, so a degraded planner can never produce an invalid filter.
    """
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
    if _CHITCHAT.match(question):
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


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in planner output")
    return json.loads(text[start : end + 1])


class Planner:
    def __init__(
        self,
        llm: LLM,
        store: Store,
        today: Callable[[], date] = date.today,
        default_language: str = "de",
    ):
        self.llm = llm
        self.store = store
        self.today = today
        self.default_language = default_language

    def _ask_llm(self, messages: list[dict]) -> RawPlan:
        try:
            output = self.llm.complete(messages, schema=PLAN_SCHEMA)
        except BackendUnavailable:
            raise
        except Exception as exc:  # backend rejected the schema: retry with plain JSON mode
            log.info("Structured planner output failed (%s); retrying with format=json", exc)
            output = self.llm.complete(messages, schema="json")
        return RawPlan.model_validate(_parse_json(output))

    def plan(self, question: str, history: Sequence[dict] | None = None) -> Plan:
        catalog = load_catalog(self.store)
        if _CHITCHAT.match(question):
            return Plan(
                "chitchat", question, "", source="heuristic",
                language=detect_language(question, self.default_language),
            )  # fmt: skip
        try:
            raw = self._ask_llm(
                build_planner_messages(question, catalog, self.today(), history)
            )
            plan = normalize_plan(raw, catalog, question, self.default_language)
        except BackendUnavailable:
            raise
        except Exception as exc:
            log.warning("Planner fell back to heuristics: %s", exc)
            plan = heuristic_plan(question, catalog, self.default_language)
        return enrich_with_regex(plan)
