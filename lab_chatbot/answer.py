"""What the engine returns: the answer, its numbered sources and exact tables.

``Engine.ask_stream`` yields ``Event`` tuples ``(kind, payload)``:

    ("images",  [str])      text read from attached images
    ("plan",    Plan)       how the question was understood
    ("tables",  [Table])    exact SQL results
    ("sources", [Source])   numbered passages the answer may cite
    ("token",   str)        a piece of the streamed answer text
    ("final",   Answer)     always last
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lab_chatbot.plan import Plan

Event = tuple[str, Any]


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[list]


@dataclass
class Source:
    n: int
    experiment_id: int  # 0 for an uploaded image
    title: str
    date: str
    text: str
    dense: float | None = None
    url: str | None = None


@dataclass
class Answer:
    question: str
    text: str
    plan: Plan | None = None
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_llm: bool = False
    latencies: dict[str, float] = field(default_factory=dict)

    @property
    def cited_sources(self) -> list[Source]:
        wanted = set(self.cited)
        return [s for s in self.sources if s.n in wanted]
