"""A small golden-set evaluator, so every change can be measured.

Golden file (JSON list)::

    [
      {"question": "Wie viele Experimente haben den Tag raman?",
       "expect_intent": "count", "expect_contains": ["2"]},
      {"question": "Warum war das Rauschen bei Probe A3 hoch?",
       "expect_ids": [102]}
    ]

* ``expect_ids``: retrieval check; hit@k and MRR over the ranked experiment ids
  (cheap: planner + retrieval only, no answer generation).
* ``expect_intent``: did the planner route the question correctly?
* ``expect_contains``: substrings the final answer text must contain
  (needs ``full=True`` because it runs the whole pipeline incl. the LLM).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lab_chatbot.engine import Engine


@dataclass
class Case:
    question: str
    expect_ids: list[int] = field(default_factory=list)
    expect_intent: str | None = None
    expect_contains: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: Case
    intent: str = ""
    ids: list[int] = field(default_factory=list)
    rank: int | None = None
    hit: bool | None = None
    intent_ok: bool | None = None
    contains_ok: bool | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        checks = [self.hit, self.intent_ok, self.contains_ok]
        return self.error is None and all(c is not False for c in checks)


def load_cases(path: str | Path) -> list[Case]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Case(
            question=item["question"],
            expect_ids=[int(i) for i in item.get("expect_ids", [])],
            expect_intent=item.get("expect_intent"),
            expect_contains=[str(s) for s in item.get("expect_contains", [])],
        )
        for item in raw
    ]


def evaluate(engine: Engine, cases: list[Case], full: bool = False) -> tuple[list[CaseResult], dict]:
    results: list[CaseResult] = []
    for case in cases:
        result = CaseResult(case)
        try:
            if full and case.expect_contains:
                answer = engine.ask(case.question)
                result.intent = answer.plan.intent if answer.plan else ""
                haystack = answer.text + " " + " ".join(
                    str(cell) for table in answer.tables for row in table.rows for cell in row
                )
                result.ids = [s.experiment_id for s in answer.sources]
                result.contains_ok = all(s.lower() in haystack.lower() for s in case.expect_contains)
            else:
                plan, ids, _ = engine.retrieve_only(case.question)
                result.intent, result.ids = plan.intent, ids
            if case.expect_intent:
                result.intent_ok = result.intent == case.expect_intent
            if case.expect_ids:
                ranks = [result.ids.index(i) + 1 for i in case.expect_ids if i in result.ids]
                result.rank = min(ranks) if ranks else None
                result.hit = result.rank is not None
        except Exception as exc:  # one broken case must not stop the run
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)

    with_ids = [r for r in results if r.case.expect_ids and r.error is None]
    with_intent = [r for r in results if r.case.expect_intent and r.error is None]
    with_text = [r for r in results if r.contains_ok is not None]
    summary = {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "errors": sum(r.error is not None for r in results),
        "hit_rate": (sum(bool(r.hit) for r in with_ids) / len(with_ids)) if with_ids else None,
        "mrr": (sum(1 / r.rank for r in with_ids if r.rank) / len(with_ids)) if with_ids else None,
        "intent_accuracy": (sum(bool(r.intent_ok) for r in with_intent) / len(with_intent))
        if with_intent
        else None,
        "contains_accuracy": (sum(bool(r.contains_ok) for r in with_text) / len(with_text))
        if with_text
        else None,
    }
    return results, summary
