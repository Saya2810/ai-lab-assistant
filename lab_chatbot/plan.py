"""The plan of how a question is answered: the validated ``Plan`` and the raw LLM output.

``RawPlan`` and friends describe what the planner LLM returns. They are
*untrusted*: lenient about types (a string where a list belongs, "1,5" as a
number), and every value is checked against the catalog afterwards
(``plan_validation.normalize_plan``). ``Plan`` is the trusted result the
engine works with.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from pydantic import BaseModel, ConfigDict, field_validator

from lab_chatbot.sql_tools import GROUP_BYS, Filters, MeasurementCond
from lab_chatbot.units import parse_number

INTENTS = ("search", "count", "aggregate", "list", "chitchat")
STRUCTURED_INTENTS = ("count", "aggregate", "list")  # answered from SQL, without the LLM
AGGS = ("min", "max", "avg", "sum", "count")


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

    def describe_filters(self) -> str:
        """Filters and measurement condition in words ("–" when there are none)."""
        parts = self.filters.describe()
        if self.measurement and self.measurement.op:
            parts.append(self.measurement.describe())
        return "; ".join(parts) or "–"


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


# JSON schema for Ollama's structured output (``format=``)
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
