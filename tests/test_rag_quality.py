"""Semantic search robustness: routing, filter grounding, query expansion, relevance gate."""

from __future__ import annotations

from lab_chatbot.engine import Engine
from lab_chatbot.plan import Plan, RawPlan
from lab_chatbot.plan_validation import _clean_keywords, fix_routing, ground_filters
from lab_chatbot.retrieval import hybrid_search
from lab_chatbot.sql_tools import Filters
from tests.fakes import FakeLLM, plan_json


def plan(intent="search", **filters) -> Plan:
    return Plan(intent, "q", "q", filters=Filters(**filters))


# ---------------------------------------------------------------- routing


def test_a_list_without_filters_becomes_a_text_search():
    assert fix_routing(plan("list"), "Which solvent was used for the NMR measurement?").intent == "search"
    assert fix_routing(plan("list"), "Zeige alle Experimente").intent == "list"
    assert fix_routing(plan("list", tags_all=["raman"]), "Raman?").intent == "list"


def test_an_aggregate_without_values_falls_back_to_the_text(demo_store, embedder, config):
    raw = plan_json("aggregate", measurement={"name": "Laserleistung", "agg": "max"}, filters={"tags_all": ["pcr"]})
    llm = FakeLLM(plan=raw, answer="Annealing bei 58 °C [1].")
    answer = Engine(demo_store, embedder, llm, config).ask("Maximale Laserleistung bei der PCR?")
    assert answer.used_llm and answer.plan.intent == "search"
    assert {s.experiment_id for s in answer.sources} == {103}
    assert any("searching the text" in note for note in answer.plan.notes)


# --------------------------------------------------------------- grounding


def test_a_loosely_mentioned_tag_is_a_search_word_in_a_text_search():
    p = ground_filters(plan("search", tags_all=["laser"]), "Welche Probe nahm durch zu viel Laserleistung Schaden?")
    assert p.filters.tags_all == [] and "laser" in p.keywords
    p = ground_filters(plan("count", tags_all=["laser"]), "Wie viele Messungen zur Laserleistung?")
    assert p.filters.tags_all == ["laser"]  # counting needs the filter
    p = ground_filters(plan("search", tags_all=["raman"]), "Warum rauschen die Raman-Messungen?")
    assert p.filters.tags_all == ["raman"]  # named as a word of its own: a real filter


def test_an_invented_date_range_is_dropped_a_named_one_kept():
    p = ground_filters(plan("search", date_from="2026-09-28", date_to="2026-11-03"), "Viskosität des Polymers?")
    assert (p.filters.date_from, p.filters.date_to) == (None, None)
    p = ground_filters(plan("list", date_from="2026-10-01", date_to="2026-10-31"), "Was lief im Oktober?")
    assert p.filters.date_from == "2026-10-01"


def test_planner_keywords_are_cleaned():
    assert RawPlan.model_validate({"keywords": "verbrannt, burnt"}).keywords == ["verbrannt", " burnt"]
    assert _clean_keywords(["verbrannt", " burnt ", "Burnt", "x", "a" * 80]) == ["verbrannt", "burnt", "a" * 40]


# ----------------------------------------------------------------- search


def test_synonyms_find_other_words_but_only_with_some_meaning(demo_store, embedder):
    # nothing in the notebook says "Schaden"; the planner's synonym "verbrannt" does
    found = hybrid_search(demo_store, embedder, "Schaden Probe", keywords=["verbrannt"], min_dense=0.0)
    assert 105 in {h.experiment_id for h in found}
    strict = hybrid_search(demo_store, embedder, "Schaden", keywords=["verbrannt"], min_dense=0.99)
    assert strict == []  # a synonym alone, without meaning, is not evidence enough


def test_the_users_own_words_always_count(demo_store, embedder):
    hits = hybrid_search(demo_store, embedder, "Probe A3", min_dense=0.99)
    assert hits and hits[0].experiment_id == 102  # identifiers win even when meaning is unclear


def test_the_better_of_two_wordings_counts(demo_store, embedder):
    alone = hybrid_search(demo_store, embedder, "xyzzy plugh", min_dense=0.2)
    both = hybrid_search(demo_store, embedder, "xyzzy plugh", alt_queries=["Rauschen Probe A3"], min_dense=0.2)
    assert not alone and 102 in {h.experiment_id for h in both}
