"""Query planner: decides *how* a question is answered.

The planner is the "intelligent integration" of SQL and RAG. For every
question it produces a small, validated ``Plan`` (lab_chatbot/plan.py):

    count / aggregate / list   -> exact answers from SQL (no LLM can corrupt a number)
    search                     -> RAG over the text, optionally restricted by SQL
                                  filters ("wavelength > 500 nm" first, then meaning)
    chitchat                   -> greeting / meta question

One LLM call (structured JSON output, temperature 0) proposes the plan. The
proposal is then validated against what actually exists in the index and
against what the user actually said (plan_validation.py). If the LLM fails or
returns garbage, deterministic rules take over (plan_rules.py), so the chatbot
degrades instead of breaking.

    question -> catalog for this question -> LLM -> normalize_plan -> ground_filters
                                              \\-> (on failure) heuristic_plan
             -> enrich_with_regex (experiment numbers, ISO dates)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Callable, Sequence

from lab_chatbot.errors import BackendUnavailable
from lab_chatbot.language import detect_language
from lab_chatbot.llm import LLM
from lab_chatbot.plan import PLAN_SCHEMA, Plan, RawPlan
from lab_chatbot.plan_rules import enrich_with_regex, heuristic_plan, is_chitchat
from lab_chatbot.plan_validation import fix_routing, ground_filters, normalize_plan
from lab_chatbot.sql_tools import Catalog, load_catalog
from lab_chatbot.store import Store

log = logging.getLogger(__name__)

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
- keywords: for "search" only: 4-8 other words a lab notebook might use for the same
  content: synonyms, the German AND the English word, related technical terms (for
  "Schaden an der Probe": "verbrannt", "beschädigt", "damaged", "burnt"). Single words or
  short terms. Never filter values, never invented sample names or numbers.
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
Examples (catalog names are placeholders; never copy them, use the CATALOG):
Q: Wie viele Einträge sind mit "hplc" getaggt?
{"intent":"count","semantic_query":"","filters":{"tags_all":["hplc"]}}
Q: What was the average flow rate?
{"intent":"aggregate","semantic_query":"","measurement":{"name":"Flow rate","agg":"avg"}}
Q: Which runs had a temperature below 4 °C?
{"intent":"list","semantic_query":"","measurement":{"name":"Temperature","op":"<","value":4,"unit":"°C"}}
Q: Warum ist die Säule beim zweiten Lauf verstopft?
{"intent":"search","semantic_query":"Säule verstopft zweiter Lauf Ursache","keywords":["Verstopfung","clogged","column","Druckanstieg","pressure"],"filters":{}}
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

    def plan(self, question: str, history: Sequence[dict] | None = None) -> Plan:
        if is_chitchat(question):
            return Plan(
                "chitchat", question, "", source="heuristic",
                language=detect_language(question, self.default_language),
            )  # fmt: skip
        catalog = load_catalog(self.store, question=question)
        try:
            raw = self._ask_llm(
                build_planner_messages(question, catalog, self.today(), history)
            )
            plan = normalize_plan(raw, catalog, question, self.default_language)
            plan = fix_routing(ground_filters(plan, question, history), question)
        except BackendUnavailable:
            raise
        except Exception as exc:
            log.warning("Planner fell back to heuristics: %s", exc)
            plan = heuristic_plan(question, catalog, self.default_language)
        return enrich_with_regex(plan)

    def _ask_llm(self, messages: list[dict]) -> RawPlan:
        try:
            output = self.llm.complete(messages, schema=PLAN_SCHEMA)
        except BackendUnavailable:
            raise
        except Exception as exc:  # backend rejected the schema: retry with plain JSON mode
            log.info("Structured planner output failed (%s); retrying with format=json", exc)
            output = self.llm.complete(messages, schema="json")
        return RawPlan.model_validate(_parse_json(output))
