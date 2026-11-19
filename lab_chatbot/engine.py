"""The chat engine: plan -> (SQL | hybrid retrieval) -> grounded answer.

    question
       |
    Planner ------------- one LLM call, validated against the real catalog
       |
       +-- count / aggregate / list ->  exact SQL result, rendered by templates (no LLM)
       +-- search -> SQL filters narrow the candidate experiments (optional)
       |             hybrid retrieval ranks chunks inside them
       |             no relevant chunk -> fixed "nothing found" answer (no LLM call)
       |             else LLM answers from numbered sources, streamed
       +-- chitchat -> help text

``ask_stream`` yields ``(kind, payload)`` events so a UI can show the plan, the
exact tables and the sources *before* the answer text finishes streaming.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.embeddings import Embedder
from lab_chatbot.errors import BackendUnavailable
from lab_chatbot.llm import LLM
from lab_chatbot.planner import Plan, Planner, detect_language
from lab_chatbot.prompts import ANSWER_SYSTEM, build_answer_prompt, t
from lab_chatbot.rerank import Reranker
from lab_chatbot.retrieval import Hit, hybrid_search
from lab_chatbot.sql_tools import (
    aggregate_measurement,
    candidate_ids,
    count_experiments,
    list_experiments,
    measurement_values,
)
from lab_chatbot.store import Store
from lab_chatbot.telemetry import QueryLog
from lab_chatbot.units import format_number, format_quantity
from lab_chatbot.vision import ImageReader
from shared.config import experiment_url

Event = tuple[str, Any]


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[list]


@dataclass
class Source:
    n: int
    experiment_id: int
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


@dataclass
class _Structured:
    text: str
    tables: list[Table] = field(default_factory=list)
    experiment_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# small helpers (module level so they can be tested alone)
# ---------------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def parse_citations(text: str, n_sources: int) -> list[int]:
    cited: list[int] = []
    for match in _CITATION.finditer(text):
        for number in re.findall(r"\d+", match.group(1)):
            value = int(number)
            if 1 <= value <= n_sources and value not in cited:
                cited.append(value)
    return sorted(cited)


def _numbers(text: str) -> set[str]:
    found = set()
    for token in _NUMBER.findall(text):
        value = token.replace(",", ".")
        try:
            found.add(f"{float(value):g}")
        except ValueError:
            continue
    return found


def unsupported_numbers(answer: str, evidence: Sequence[str]) -> list[str]:
    """Numbers in the answer that appear nowhere in the evidence.

    A cheap guard against the classic local-LLM failure of "improving" a
    value. Citation markers, list numbering and small integers (0-20: counts
    and ordinals the model legitimately writes itself) are ignored.
    """
    cleaned = _CITATION.sub(" ", answer)
    cleaned = re.sub(r"(?m)^\s*\d+[.)]\s", " ", cleaned)
    allowed: set[str] = set()
    for chunk in evidence:
        allowed |= _numbers(chunk)
    missing = []
    for token in _NUMBER.findall(cleaned):
        try:
            value = float(token.replace(",", "."))
        except ValueError:
            continue
        key = f"{value:g}"
        if key in allowed:
            continue
        if value == int(value) and 0 <= value <= 20:
            continue
        if token not in missing:
            missing.append(token)
    return missing


@contextmanager
def _timer(store: dict[str, float], key: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - started) * 1000, 1)


def _clean_history(history: Sequence[dict] | None, turns: int) -> list[dict]:
    if not history:
        return []
    messages = [
        {"role": m["role"], "content": str(m["content"])[:1500]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return messages[-(turns * 2) :]


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class Engine:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        llm: LLM,
        config: ChatbotConfig,
        planner: Planner | None = None,
        image_reader: ImageReader | None = None,
        query_log: QueryLog | None = None,
        reranker: Reranker | None = None,
    ):
        self.reranker = reranker
        self.image_reader = image_reader
        self.query_log = query_log or (QueryLog(store) if config.query_log else None)
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.config = config
        self.planner = planner or Planner(
            llm, store, default_language=config.default_language
        )

    # ----------------------------------------------------------- public API

    def ask(
        self,
        question: str,
        history: Sequence[dict] | None = None,
        images: Sequence[bytes] | None = None,
    ) -> Answer:
        final = None
        for kind, payload in self.ask_stream(question, history, images):
            if kind == "final":
                final = payload
        assert final is not None
        return final

    def ask_stream(
        self,
        question: str,
        history: Sequence[dict] | None = None,
        images: Sequence[bytes] | None = None,
    ) -> Iterator[Event]:
        """Events of ``_ask_stream``; additionally writes one telemetry row per question."""
        started = time.perf_counter()
        latencies: dict[str, float] = {}
        try:
            for event in self._ask_stream(question, history, images, started, latencies):
                if event[0] == "final":
                    self._record(event[1], bool(images and any(images)))
                yield event
        except Exception as exc:
            latencies["total"] = round((time.perf_counter() - started) * 1000, 1)
            self._record(None, bool(images and any(images)), (question or "").strip(),
                         latencies, type(exc).__name__)  # fmt: skip
            raise

    def _record(self, answer, had_images, question="", latencies=None, error="") -> None:
        if self.query_log is None:
            return
        plan = answer.plan if answer else None
        question = answer.question if answer else question
        self.query_log.record(
            question=question,
            language=(plan.language if plan else detect_language(question, self.config.default_language)),
            route=plan.intent if plan else "",
            used_llm=bool(answer and answer.used_llm),
            n_sources=len(answer.sources) if answer else 0,
            n_cited=len(answer.cited) if answer else 0,
            latencies=answer.latencies if answer else latencies,
            error=error,
            chat_model=getattr(self.llm, "model", "") or self.config.chat_model,
            embed_model=getattr(self.embedder, "model", "") or self.config.embed_model,
            had_images=had_images,
        )  # fmt: skip

    def _ask_stream(
        self,
        question: str,
        history: Sequence[dict] | None,
        images: Sequence[bytes] | None,
        started: float,
        latencies: dict[str, float],
    ) -> Iterator[Event]:
        question = (question or "").strip()
        language = detect_language(question, self.config.default_language)
        images = [img for img in (images or []) if img]
        if images:
            yield from self._ask_with_images(question, history, images, language, started, latencies)
            return

        def finish(answer: Answer) -> Event:
            latencies["total"] = round((time.perf_counter() - started) * 1000, 1)
            answer.latencies = latencies
            return ("final", answer)

        if not question:
            yield finish(Answer(question, t(language, "empty_question")))
            return
        if self.store.experiment_count() == 0:
            yield finish(Answer(question, t(language, "empty_index")))
            return

        recent = _clean_history(history, self.config.history_turns)
        with _timer(latencies, "plan"):
            plan = self.planner.plan(question, recent)
        yield ("plan", plan)

        if plan.intent == "chitchat":
            text = t(plan.language, "help", experiments=self.store.experiment_count())
            yield finish(Answer(question, text, plan))
            return

        if plan.intent in ("count", "aggregate", "list"):
            with _timer(latencies, "sql"):
                result = self._run_structured(plan)
            yield ("tables", result.tables)
            yield finish(Answer(question, result.text, plan, tables=result.tables))
            return

        # ---- search (RAG, optionally narrowed by SQL)
        with _timer(latencies, "retrieve"):
            candidates = candidate_ids(self.store, plan.filters, plan.measurement)
            if candidates is not None and not candidates:
                text = t(plan.language, "no_experiments", filters=self._describe(plan))
                yield finish(Answer(question, text, plan))
                return
            hits = self._retrieve(plan, candidates)
            tables = self._search_tables(plan, candidates)
        if tables:
            yield ("tables", tables)

        if not hits:
            if candidates:
                text = t(
                    plan.language, "no_passages_but_filters",
                    filters=self._describe(plan), n=len(candidates),
                )  # fmt: skip
                if not tables:
                    tables = [self._matches_table(plan, candidates)]
                    yield ("tables", tables)
            else:
                text = t(plan.language, "no_passages")
            yield finish(Answer(question, text, plan, tables=tables))
            return

        sources = self._build_sources(hits)
        yield ("sources", sources)

        facts = self._facts_text(plan, tables)
        messages = [{"role": "system", "content": ANSWER_SYSTEM}, *recent]
        messages.append(
            {
                "role": "user",
                "content": build_answer_prompt(
                    question, plan.semantic_query, sources, facts, plan.language
                ),
            }
        )
        parts: list[str] = []
        with _timer(latencies, "generate"):
            for piece in self.llm.stream(messages):
                parts.append(piece)
                yield ("token", piece)
        text = "".join(parts).strip()

        cited = parse_citations(text, len(sources))
        warnings = []
        evidence = [s.text for s in sources] + [facts, question]
        missing = unsupported_numbers(text, evidence)
        if missing:
            warnings.append(t(plan.language, "warn_numbers", numbers=", ".join(missing[:8])))
        if not cited and text:
            warnings.append(t(plan.language, "warn_uncited"))
        yield finish(
            Answer(question, text, plan, sources, cited, tables, warnings, used_llm=True)
        )

    def _ask_with_images(
        self,
        question: str,
        history: Sequence[dict] | None,
        images: Sequence[bytes],
        language: str,
        started: float,
        latencies: dict[str, float],
    ) -> Iterator[Event]:
        """Question with attached images: read them, then answer from image + notebook.

        The image text is always a source. The notebook is searched with the question plus
        the beginning of the image text, so "which experiment does this plot belong to?"
        finds entries that mention the same sample names or axis labels.
        """

        def finish(answer: Answer) -> Event:
            latencies["total"] = round((time.perf_counter() - started) * 1000, 1)
            answer.latencies = latencies
            return ("final", answer)

        question = question or t(language, "image_question")
        if self.image_reader is None:
            raise BackendUnavailable("No vision model configured (LAB_VISION_MODEL).")

        image_sources: list[Source] = []
        with _timer(latencies, "vision"):
            for position, content in enumerate(images, start=1):
                text = self.image_reader.read(content).strip()
                if text:
                    image_sources.append(
                        Source(
                            n=len(image_sources) + 1,
                            experiment_id=0,
                            title=t(language, "image_title", n=position),
                            date="",
                            text=text[: self.config.max_context_chars // 2],
                        )
                    )
        yield ("images", [s.text for s in image_sources])
        if not image_sources:
            yield finish(Answer(question, t(language, "image_unreadable")))
            return

        recent = _clean_history(history, self.config.history_turns)
        plan = None
        hits: list[Hit] = []
        if self.store.experiment_count() > 0:
            with _timer(latencies, "plan"):
                plan = self.planner.plan(question, recent)
            yield ("plan", plan)
            with _timer(latencies, "retrieve"):
                candidates = candidate_ids(self.store, plan.filters, plan.measurement)
                if candidates is None or candidates:
                    query = f"{plan.semantic_query or question}\n{image_sources[0].text[:400]}"
                    hits = hybrid_search(
                        self.store, self.embedder, query,
                        experiment_ids=candidates, k=self.config.top_k,
                        min_dense=self.config.min_dense, reranker=self.reranker,
                    )  # fmt: skip

        notebook = self._build_sources(hits)
        for source in notebook:
            source.n += len(image_sources)
        sources = image_sources + notebook
        yield ("sources", sources)

        messages = [{"role": "system", "content": ANSWER_SYSTEM}, *recent]
        semantic = plan.semantic_query if plan else question
        messages.append(
            {"role": "user", "content": build_answer_prompt(question, semantic, sources, language=language)}
        )
        parts: list[str] = []
        with _timer(latencies, "generate"):
            for piece in self.llm.stream(messages):
                parts.append(piece)
                yield ("token", piece)
        text = "".join(parts).strip()
        cited = parse_citations(text, len(sources))
        warnings = []
        missing = unsupported_numbers(text, [s.text for s in sources] + [question])
        if missing:
            warnings.append(t(language, "warn_numbers", numbers=", ".join(missing[:8])))
        if not cited and text:
            warnings.append(t(language, "warn_uncited"))
        yield finish(Answer(question, text, plan, sources, cited, [], warnings, used_llm=True))

    def retrieve_only(self, question: str, history: Sequence[dict] | None = None):
        """Plan + ranked experiment ids, no generation. Used by ``eval`` / ``search``."""
        plan = self.planner.plan(question, _clean_history(history, self.config.history_turns))
        if plan.intent in ("count", "aggregate", "list"):
            return plan, self._run_structured(plan).experiment_ids, []
        if plan.intent == "chitchat":
            return plan, [], []
        candidates = candidate_ids(self.store, plan.filters, plan.measurement)
        hits = self._retrieve(plan, candidates)
        return plan, list(dict.fromkeys(h.experiment_id for h in hits)), hits

    # ------------------------------------------------------------- internals

    def _describe(self, plan: Plan) -> str:
        parts = plan.filters.describe()
        if plan.measurement and plan.measurement.op:
            parts.append(plan.measurement.describe())
        return "; ".join(parts) or "–"

    def _retrieve(self, plan: Plan, candidates: set[int] | None) -> list[Hit]:
        cfg = self.config
        query = plan.semantic_query or plan.question
        explicit = set(plan.filters.experiment_ids)
        if explicit:
            # The user named the experiment(s): small ones are fetched whole, so
            # "summarise experiment 12" works even though no word "matches".
            whole = self.store.chunks_of(sorted(explicit & (candidates or explicit)))
            if 0 < len(whole) <= cfg.top_k + 2:
                return whole
        hits = hybrid_search(
            self.store, self.embedder, query,
            experiment_ids=candidates, k=cfg.top_k, min_dense=cfg.min_dense,
            reranker=self.reranker,
        )  # fmt: skip
        if not hits and explicit:
            hits = self.store.chunks_of(sorted(explicit & (candidates or explicit)))[: cfg.top_k]
        return hits

    def _build_sources(self, hits: Sequence[Hit]) -> list[Source]:
        budget = self.config.max_context_chars
        sources: list[Source] = []
        used = 0
        for hit in hits:
            if sources and used + len(hit.text) > budget:
                continue  # skip what does not fit, a later (shorter) chunk still might
            used += len(hit.text)
            sources.append(
                Source(
                    n=len(sources) + 1,
                    experiment_id=hit.experiment_id,
                    title=hit.title,
                    date=hit.date,
                    text=hit.text,
                    dense=hit.dense,
                    url=experiment_url(hit.experiment_id, self.config.site_url),
                )
            )
        return sources

    def _search_tables(self, plan: Plan, candidates: set[int] | None) -> list[Table]:
        """Exact measurement values behind a numeric filter (shown and given to the LLM)."""
        if not (plan.measurement and plan.measurement.op and candidates):
            return []
        rows = measurement_values(
            self.store, plan.filters, plan.measurement, plan.order, limit=30
        )
        return [self._values_table(plan, rows)] if rows else []

    def _facts_text(self, plan: Plan, tables: Sequence[Table]) -> str:
        lines = []
        for table in tables:
            if table.title != t(plan.language, "table_values"):
                continue
            lines.append(f"Matching measurements ({self._describe(plan)}):")
            for row in table.rows:
                lines.append(f"- Experiment {row[0]} ({row[2]}): {row[3]}")
        return "\n".join(lines)

    # ---- tables

    def _values_table(self, plan: Plan, rows: list[dict]) -> Table:
        lang = plan.language
        return Table(
            t(lang, "table_values"),
            [t(lang, "col_id"), t(lang, "col_title"), t(lang, "col_date"), t(lang, "col_value")],
            [
                [
                    r["experiment_id"], r["title"], r["date"],
                    f"{r['name']}: "
                    + (
                        format_quantity(r["value_num"], r["unit"])
                        if r["value_num"] is not None
                        else r["value_text"]
                    ),
                ]
                for r in rows
            ],
        )  # fmt: skip

    def _experiments_table(self, plan: Plan, rows: list[dict]) -> Table:
        lang = plan.language
        return Table(
            t(lang, "table_matches"),
            [t(lang, k) for k in ("col_id", "col_title", "col_date", "col_tags", "col_category", "col_status", "col_link")],
            [
                [
                    r["id"], r["title"], r["date"], r["tags"] or "", r["category"] or "",
                    r["status"] or "", experiment_url(r["id"], self.config.site_url),
                ]
                for r in rows
            ],
        )  # fmt: skip

    def _matches_table(self, plan: Plan, candidates: set[int]) -> Table:
        from lab_chatbot.sql_tools import Filters

        result = list_experiments(
            self.store, Filters(experiment_ids=sorted(candidates)), None, plan.order, plan.limit
        )
        return self._experiments_table(plan, result.rows)

    # ---- structured (no LLM) answers

    def _run_structured(self, plan: Plan) -> _Structured:
        if plan.intent == "count":
            return self._count(plan)
        if plan.intent == "aggregate":
            return self._aggregate(plan)
        return self._list(plan)

    def _count(self, plan: Plan) -> _Structured:
        lang = plan.language
        rows = count_experiments(self.store, plan.filters, plan.measurement, plan.group_by)
        if plan.group_by:
            if not rows:
                return _Structured(t(lang, "count_grouped_empty"))
            label = t(lang, f"group_{plan.group_by}")
            table = Table(
                t(lang, "table_counts"), [label, t(lang, "col_count")],
                [[r["group"], r["count"]] for r in rows],
            )  # fmt: skip
            return _Structured(t(lang, "count_grouped", group=label), [table])
        n = rows[0]["count"]
        text = t(lang, "count_one") if n == 1 else t(lang, "count_many", n=n)
        described = self._describe(plan)
        if described != "–":
            text += f"\n\n_{t(lang, 'filters_used')}: {described}_"
        return _Structured(text)

    def _list(self, plan: Plan) -> _Structured:
        lang = plan.language
        result = list_experiments(
            self.store, plan.filters, plan.measurement, plan.order, plan.limit
        )
        if not result.rows:
            return _Structured(t(lang, "no_experiments", filters=self._describe(plan)))
        key = "listed" if result.total <= plan.limit else "listed_limited"
        text = t(lang, key, total=result.total, limit=plan.limit)
        described = self._describe(plan)
        if described != "–":
            text += f"\n\n_{t(lang, 'filters_used')}: {described}_"
        tables = [self._experiments_table(plan, result.rows)]
        if plan.measurement:
            values = measurement_values(
                self.store, plan.filters, plan.measurement, plan.order, limit=plan.limit
            )
            if values:
                tables.append(self._values_table(plan, values))
        return _Structured(text, tables, [r["id"] for r in result.rows])

    def _aggregate(self, plan: Plan) -> _Structured:
        lang = plan.language
        cond = plan.measurement
        groups = aggregate_measurement(self.store, plan.filters, cond, unit=cond.unit)
        if not groups:
            return _Structured(t(lang, "no_values", name=cond.name))
        lines = []
        for group in groups:
            unit = group["unit"]
            lines.append(
                t(lang, "agg_header", name=group["name"], n_values=group["n_values"],
                  n_experiments=group["n_experiments"])
            )  # fmt: skip
            for stat in ("min", "max", "avg", "sum"):
                value = format_quantity(group[stat], unit)
                suffix = ""
                if stat in ("min", "max"):
                    suffix = f" ({t(lang, 'experiment')} {group[f'{stat}_experiment']})"
                line = f"- {t(lang, stat)}: {value}{suffix}"
                lines.append(f"**{line[2:]}**" if plan.agg == stat else line)
            if plan.agg == "count":
                lines.append(f"- **{t(lang, 'count')}: {group['n_values']}**")
            lines.append("")
        values = measurement_values(self.store, plan.filters, cond, plan.order, limit=30)
        tables = [self._values_table(plan, values)] if values else []
        ids = list(dict.fromkeys(v["experiment_id"] for v in values))
        return _Structured("\n".join(lines).strip(), tables, ids)
