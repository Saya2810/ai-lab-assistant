"""The chat engine: plan -> (SQL | hybrid retrieval) -> grounded answer.

    question (+ optional images)
       |
    Planner ------------- one LLM call, validated against the real catalog (planner.py)
       |
       +-- count / aggregate / list ->  exact SQL result, rendered by templates (structured.py)
       +-- search -> SQL filters narrow the candidate experiments (optional)
       |             hybrid retrieval ranks chunks inside them (retrieval.py)
       |             no relevant chunk -> fixed "nothing found" answer (no LLM call)
       |             else LLM answers from numbered sources, streamed, then checked
       |             for citations and invented numbers (answer_checks.py)
       +-- chitchat -> help text

With images the vision model reads them first; the image text becomes a source
and also steers the notebook search.

``ask_stream`` yields ``(kind, payload)`` events (see answer.py) so a UI can
show the plan, the exact tables and the sources *before* the answer text
finishes streaming. Every answered question is recorded in the query log.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator, Iterator, Sequence

from lab_chatbot.answer import Answer, Event, Source, Table
from lab_chatbot.answer_checks import check_answer
from lab_chatbot.config import ChatbotConfig
from lab_chatbot.embeddings import Embedder
from lab_chatbot.errors import BackendUnavailable
from lab_chatbot.language import detect_language
from lab_chatbot.llm import LLM
from lab_chatbot.plan import STRUCTURED_INTENTS, Plan
from lab_chatbot.planner import Planner
from lab_chatbot.prompts import ANSWER_SYSTEM, build_answer_prompt, t
from lab_chatbot.rerank import Reranker
from lab_chatbot.retrieval import Hit, hybrid_search
from lab_chatbot.scope import Scope
from lab_chatbot.sql_tools import Filters, candidate_ids
from lab_chatbot.store import Store
from lab_chatbot.structured import StructuredAnswers
from lab_chatbot.telemetry import QueryLog
from lab_chatbot.vision import ImageReader
from shared.config import experiment_url

__all__ = ["Answer", "Engine", "Event", "Source", "Table"]

Generated = Generator[Event, None, tuple[str, list[int], list[str]]]


class _Turn:
    """Timing of one question: per-step latencies and the final event."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.latencies: dict[str, float] = {}

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.latencies[name] = round((time.perf_counter() - started) * 1000, 1)

    def finish(self, answer: Answer) -> Event:
        self.latencies["total"] = self.elapsed_ms()
        answer.latencies = self.latencies
        return ("final", answer)


def _restrict(plan: Plan, scope: Scope | None) -> None:
    """Apply the user's scope to the plan: it becomes part of every SQL query."""
    if scope is not None and not scope.is_empty():
        plan.filters.scope = scope


def _clean_history(history: Sequence[dict] | None, turns: int) -> list[dict]:
    if not history:
        return []
    messages = [
        {"role": m["role"], "content": str(m["content"])[:1500]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return messages[-(turns * 2) :]


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
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.config = config
        self.planner = planner or Planner(llm, store, default_language=config.default_language)
        self.image_reader = image_reader
        self.reranker = reranker
        self.query_log = query_log or (QueryLog(store) if config.query_log else None)
        self.structured = StructuredAnswers(store, config.site_url)

    # ----------------------------------------------------------- public API

    def ask(
        self,
        question: str,
        history: Sequence[dict] | None = None,
        images: Sequence[bytes] | None = None,
        scope: Scope | None = None,
    ) -> Answer:
        final = None
        for kind, payload in self.ask_stream(question, history, images, scope):
            if kind == "final":
                final = payload
        assert final is not None
        return final

    def ask_stream(
        self,
        question: str,
        history: Sequence[dict] | None = None,
        images: Sequence[bytes] | None = None,
        scope: Scope | None = None,
    ) -> Iterator[Event]:
        """Answer events, ending with ``("final", Answer)``; one query-log row per question.

        ``scope`` limits every answer to part of the notebook (projects, tags, ...).
        """
        turn = _Turn()
        question = (question or "").strip()
        images = [img for img in (images or []) if img]
        scope = scope if scope is not None and not scope.is_empty() else None
        try:
            if images:
                events = self._answer_images(question, history, images, turn, scope)
            else:
                events = self._answer_text(question, history, turn, scope)
            for event in events:
                if event[0] == "final":
                    self._record(event[1], bool(images))
                yield event
        except Exception as exc:
            turn.latencies["total"] = turn.elapsed_ms()
            self._record(None, bool(images), question, turn.latencies, type(exc).__name__)
            raise

    def retrieve_only(
        self, question: str, history: Sequence[dict] | None = None, scope: Scope | None = None
    ):
        """Plan + ranked experiment ids, no generation. Used by ``eval`` / ``search``."""
        plan = self.planner.plan(question, _clean_history(history, self.config.history_turns))
        _restrict(plan, scope)
        if plan.intent in STRUCTURED_INTENTS:
            result = self._structured(plan)
            if result is not None:
                return plan, result.experiment_ids, []
        if plan.intent == "chitchat":
            return plan, [], []
        candidates = candidate_ids(self.store, plan.filters, plan.measurement)
        hits = self._retrieve(plan, candidates)
        return plan, list(dict.fromkeys(h.experiment_id for h in hits)), hits

    # ------------------------------------------------------------ text path

    def _answer_text(
        self, question: str, history: Sequence[dict] | None, turn: _Turn, scope: Scope | None
    ) -> Iterator[Event]:
        language = detect_language(question, self.config.default_language)
        if not question:
            yield turn.finish(Answer(question, t(language, "empty_question")))
            return
        if self.store.experiment_count() == 0:
            yield turn.finish(Answer(question, t(language, "empty_index")))
            return
        if self._scope_is_empty(scope):
            yield turn.finish(Answer(question, t(language, "scope_empty", scope=scope.describe(language))))
            return

        recent = _clean_history(history, self.config.history_turns)
        with turn.step("plan"):
            plan = self.planner.plan(question, recent)
        _restrict(plan, scope)
        yield ("plan", plan)

        if plan.intent == "chitchat":
            text = t(plan.language, "help", experiments=self.store.experiment_count())
            yield turn.finish(Answer(question, text, plan))
            return

        if plan.intent in STRUCTURED_INTENTS:
            with turn.step("sql"):
                result = self._structured(plan)
            if result is not None:
                yield ("tables", result.tables)
                yield turn.finish(Answer(question, result.text, plan, tables=result.tables))
                return

        # ---- search (RAG, optionally narrowed by SQL)
        with turn.step("retrieve"):
            candidates = candidate_ids(self.store, plan.filters, plan.measurement)
            if candidates is not None and not candidates:
                text = t(plan.language, "no_experiments", filters=plan.describe_filters())
                yield turn.finish(Answer(question, text, plan))
                return
            hits = self._retrieve(plan, candidates)
            tables = self.structured.search_tables(plan, candidates)
        if tables:
            yield ("tables", tables)

        if not hits:
            answer = self._nothing_found(question, plan, candidates, tables)
            if answer.tables and not tables:
                yield ("tables", answer.tables)
            yield turn.finish(answer)
            return

        sources = self._sources(hits)
        yield ("sources", sources)
        facts = self.structured.facts_text(plan, tables)
        prompt = build_answer_prompt(question, plan.semantic_query, sources, facts, plan.language)
        text, cited, warnings = yield from self._generate(
            turn, recent, prompt, sources, [facts, question], plan.language
        )
        yield turn.finish(
            Answer(question, text, plan, sources, cited, tables, warnings, used_llm=True)
        )

    def _nothing_found(
        self, question: str, plan: Plan, candidates: set[int] | None, tables: list[Table]
    ) -> Answer:
        if not candidates:
            return Answer(question, t(plan.language, "no_passages"), plan, tables=tables)
        text = t(
            plan.language, "no_passages_but_filters",
            filters=plan.describe_filters(), n=len(candidates),
        )  # fmt: skip
        # the filters did match: at least show which experiments they matched
        return Answer(question, text, plan, tables=tables or [self.structured.matches_table(plan, candidates)])

    # ----------------------------------------------------------- image path

    def _answer_images(
        self,
        question: str,
        history: Sequence[dict] | None,
        images: Sequence[bytes],
        turn: _Turn,
        scope: Scope | None,
    ) -> Iterator[Event]:
        """Question with attached images: read them, then answer from image + notebook.

        The image text is always a source. The notebook is searched with the question plus
        the beginning of the image text, so "which experiment does this plot belong to?"
        finds entries that mention the same sample names or axis labels.
        """
        language = detect_language(question, self.config.default_language)
        question = question or t(language, "image_question")
        if self.image_reader is None:
            raise BackendUnavailable("No vision model configured (LAB_VISION_MODEL).")

        with turn.step("vision"):
            image_sources = self._read_images(images, language)
        yield ("images", [s.text for s in image_sources])
        if not image_sources:
            yield turn.finish(Answer(question, t(language, "image_unreadable")))
            return

        recent = _clean_history(history, self.config.history_turns)
        plan = None
        hits: list[Hit] = []
        if self.store.experiment_count() > 0 and not self._scope_is_empty(scope):
            with turn.step("plan"):
                plan = self.planner.plan(question, recent)
            _restrict(plan, scope)
            yield ("plan", plan)
            with turn.step("retrieve"):
                candidates = candidate_ids(self.store, plan.filters, plan.measurement)
                if candidates is None or candidates:
                    query = f"{plan.semantic_query or question}\n{image_sources[0].text[:400]}"
                    hits = self._search(query, candidates, keywords=plan.keywords)

        notebook = self._sources(hits, first_number=len(image_sources) + 1)
        sources = image_sources + notebook
        yield ("sources", sources)

        semantic = plan.semantic_query if plan else question
        prompt = build_answer_prompt(question, semantic, sources, language=language)
        text, cited, warnings = yield from self._generate(
            turn, recent, prompt, sources, [question], language
        )
        yield turn.finish(Answer(question, text, plan, sources, cited, [], warnings, used_llm=True))

    def _read_images(self, images: Sequence[bytes], language: str) -> list[Source]:
        sources: list[Source] = []
        for position, content in enumerate(images, start=1):
            text = self.image_reader.read(content).strip()
            if text:
                sources.append(
                    Source(
                        n=len(sources) + 1,
                        experiment_id=0,
                        title=t(language, "image_title", n=position),
                        date="",
                        text=text[: self.config.max_context_chars // 2],
                    )
                )
        return sources

    # ---------------------------------------------------------- shared steps

    def _scope_is_empty(self, scope: Scope | None) -> bool:
        """True when a scope is set but matches no experiment at all."""
        return scope is not None and not candidate_ids(self.store, Filters(scope=scope))

    def _generate(
        self,
        turn: _Turn,
        recent: list[dict],
        prompt: str,
        sources: Sequence[Source],
        extra_evidence: Sequence[str],
        language: str,
    ) -> Generated:
        """Stream the answer as ``token`` events; returns ``(text, cited, warnings)``."""
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            *recent,
            {"role": "user", "content": prompt},
        ]
        parts: list[str] = []
        with turn.step("generate"):
            for piece in self.llm.stream(messages):
                parts.append(piece)
                yield ("token", piece)
        text = "".join(parts).strip()
        cited, warnings = check_answer(text, sources, extra_evidence, language)
        return text, cited, warnings

    def _structured(self, plan: Plan):
        """The SQL answer, or None when an aggregate found no values at all.

        Small planner models route text questions ("Wie dicht waren die Zellen?") to an
        aggregate over some measurement. When that has no values, the question is
        answered from the text instead of with "no values".
        """
        result = self.structured.answer(plan)
        if result.found or plan.intent != "aggregate":
            return result
        plan.notes.append("The aggregate found no values; searching the text instead.")
        plan.intent, plan.measurement, plan.agg = "search", None, None
        return None

    def _search(
        self,
        query: str,
        candidates: set[int] | None,
        alt_queries: Sequence[str] = (),
        keywords: Sequence[str] = (),
    ) -> list[Hit]:
        return hybrid_search(
            self.store, self.embedder, query,
            experiment_ids=candidates, k=self.config.top_k,
            min_dense=self.config.min_dense, reranker=self.reranker,
            alt_queries=alt_queries, keywords=keywords,
        )  # fmt: skip

    def _retrieve(self, plan: Plan, candidates: set[int] | None) -> list[Hit]:
        top_k = self.config.top_k
        explicit = set(plan.filters.experiment_ids)
        if explicit:
            # The user named the experiment(s): small ones are fetched whole, so
            # "summarise experiment 12" works even though no word "matches".
            whole = self.store.chunks_of(sorted(explicit & (candidates or explicit)))
            if 0 < len(whole) <= top_k + 2:
                return whole
        # the planner's search phrase and the user's own words: whichever matches better
        hits = self._search(
            plan.semantic_query or plan.question, candidates, [plan.question], plan.keywords
        )
        if not hits and explicit:
            hits = self.store.chunks_of(sorted(explicit & (candidates or explicit)))[:top_k]
        return hits

    def _sources(self, hits: Sequence[Hit], first_number: int = 1) -> list[Source]:
        """Numbered sources within the prompt budget (``max_context_chars``)."""
        budget = self.config.max_context_chars
        sources: list[Source] = []
        used = 0
        for hit in hits:
            if sources and used + len(hit.text) > budget:
                continue  # skip what does not fit, a later (shorter) chunk still might
            used += len(hit.text)
            sources.append(
                Source(
                    n=first_number + len(sources),
                    experiment_id=hit.experiment_id,
                    title=hit.title,
                    date=hit.date,
                    text=hit.text,
                    dense=hit.dense,
                    url=experiment_url(hit.experiment_id, self.config.site_url),
                )
            )
        return sources

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
