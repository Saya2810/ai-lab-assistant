from __future__ import annotations

from datetime import date

import pytest

from lab_chatbot.errors import BackendUnavailable
from lab_chatbot.planner import (
    Planner,
    detect_language,
    extract_ids,
    heuristic_plan,
    snap,
)
from lab_chatbot.sql_tools import load_catalog
from tests.fakes import FakeLLM, plan_json

TODAY = date(2026, 11, 12)


@pytest.fixture
def planner_for(demo_store):
    def make(plan):
        llm = FakeLLM(plan=plan)
        return Planner(llm, demo_store, today=lambda: TODAY), llm

    return make


# ---------------------------------------------------------------- LLM path


def test_valid_llm_plan_is_normalised(planner_for):
    planner, _ = planner_for(plan_json("count", filters={"tags_all": ["Raman"]}))
    plan = planner.plan("Wie viele Experimente haben den Tag raman?")
    assert plan.intent == "count" and plan.source == "llm"
    assert plan.filters.tags_all == ["raman"]  # snapped to the real spelling


def test_planner_sees_catalog_date_and_history(planner_for):
    planner, llm = planner_for(plan_json("search", "x"))
    planner.plan("Und davon?", history=[{"role": "user", "content": "Zeige Raman Messungen"}])
    prompt = llm.complete_calls[0]["messages"][-1]["content"]
    for expected in ("2026-11-12", "raman", "Wellenlänge (nm)", "Biologie", "Zeige Raman Messungen"):
        assert expected in prompt
    assert llm.complete_calls[0]["schema"] is not None  # structured output requested


def test_unknown_tag_is_dropped_with_a_note(planner_for):
    planner, _ = planner_for(plan_json("search", "x", filters={"tags_all": ["raman", "quidditch"]}))
    plan = planner.plan("Raman und Quidditch")
    assert plan.filters.tags_all == ["raman"] and any("quidditch" in n for n in plan.notes)


def test_fuzzy_snap_of_tag_category_status(planner_for):
    planner, _ = planner_for(
        plan_json("list", filters={"tags_any": ["spektroskopy"], "category": "messung", "status": "in arbeit"})
    )
    plan = planner.plan("Liste")
    assert plan.filters.tags_any == ["spektroskopie"]
    assert (plan.filters.category, plan.filters.status) == ("Messung", "In Arbeit")


def test_hallucinated_measurement_is_dropped_and_aggregate_degrades_to_search(planner_for):
    planner, _ = planner_for(plan_json("aggregate", measurement={"name": "Flux", "agg": "max"}))
    plan = planner.plan("Maximaler Flux?")
    assert plan.measurement is None and plan.intent == "search"


def test_measurement_with_condition(planner_for):
    planner, _ = planner_for(
        plan_json("list", measurement={"name": "wellenlänge", "op": ">", "value": "600", "unit": "nm"})
    )
    plan = planner.plan("Wellenlänge über 600 nm")
    m = plan.measurement
    assert (m.name, m.op, m.value, m.unit) == ("Wellenlänge", ">", 600.0, "nm")


def test_missing_unit_is_filled_from_catalog_with_note(planner_for):
    planner, _ = planner_for(plan_json("list", measurement={"name": "Wellenlänge", "op": ">", "value": 600}))
    plan = planner.plan("Wellenlänge über 600")
    assert plan.measurement.unit == "nm" and any("assumed" in n for n in plan.notes)


def test_bad_operator_or_missing_value_drops_the_condition(planner_for):
    planner, _ = planner_for(plan_json("list", measurement={"name": "pH", "op": "~=", "value": 7}))
    assert planner.plan("pH").measurement.op is None
    planner, _ = planner_for(plan_json("list", measurement={"name": "pH", "op": "between", "value": 7}))
    assert planner.plan("pH").measurement.op is None


def test_aggregate_keeps_unit_without_condition(planner_for):
    planner, _ = planner_for(
        plan_json("aggregate", measurement={"name": "Belichtungszeit", "agg": "avg", "unit": "ms"})
    )
    plan = planner.plan("Durchschnittliche Belichtungszeit in ms")
    assert plan.agg == "avg" and plan.measurement.unit == "ms" and plan.measurement.op is None


def test_invalid_and_reversed_dates(planner_for):
    planner, _ = planner_for(plan_json("list", filters={"date_from": "2026-10-31", "date_to": "2026-10-01"}))
    plan = planner.plan("Oktober")
    assert (plan.filters.date_from, plan.filters.date_to) == ("2026-10-01", "2026-10-31")
    planner, _ = planner_for(plan_json("list", filters={"date_from": "gestern"}))
    plan = planner.plan("gestern")
    assert plan.filters.date_from is None and plan.notes


def test_group_by_only_for_counts(planner_for):
    planner, _ = planner_for(plan_json("count", group_by="status"))
    assert planner.plan("pro Status").group_by == "status"
    planner, _ = planner_for(plan_json("list", group_by="status"))
    assert planner.plan("Liste").group_by is None
    planner, _ = planner_for(plan_json("count", group_by="title; DROP"))
    assert planner.plan("x").group_by is None


def test_json_in_code_fence_and_chatter_is_accepted(planner_for):
    planner, _ = planner_for('Klar!\n```json\n{"intent": "count", "filters": {}}\n```')
    assert planner.plan("Wie viele?").intent == "count" and planner.plan("Wie viele?").source == "llm"


def test_lenient_types(planner_for):
    planner, _ = planner_for(
        '{"intent":"list","filters":{"tags_all":"raman","experiment_ids":"Experiment 102 und 105"},'
        '"measurement":{"name":"Laserleistung","op":">","value":"1,5"}}'
    )
    plan = planner.plan("x")
    assert plan.filters.tags_all == ["raman"] and plan.filters.experiment_ids == [102, 105]
    assert plan.measurement.value == 1.5


def test_unknown_intent_falls_back_to_search(planner_for):
    planner, _ = planner_for(plan_json("dance"))
    assert planner.plan("x").intent == "search"


def test_regex_adds_experiment_ids_and_dates_the_llm_missed(planner_for):
    planner, _ = planner_for(plan_json("search", "zusammenfassung"))
    plan = planner.plan("Fasse Experiment 12 und 15 zusammen, vom 2026-10-01")
    assert plan.filters.experiment_ids == [12, 15]
    assert plan.filters.date_from == plan.filters.date_to == "2026-10-01"


def test_structured_output_failure_retries_in_json_mode(demo_store):
    calls = []

    class Picky(FakeLLM):
        def complete(self, messages, schema=None):
            calls.append(schema)
            if isinstance(schema, dict):
                raise RuntimeError("model does not support schemas")
            return '{"intent": "count"}'

    plan = Planner(Picky(), demo_store, today=lambda: TODAY).plan("Wie viele?")
    assert plan.intent == "count" and plan.source == "llm" and calls[-1] == "json"


def test_garbage_output_falls_back_to_heuristics(planner_for):
    planner, _ = planner_for("ich bin ein freundlicher Assistent ohne JSON")
    plan = planner.plan("Wie viele Experimente haben den Tag raman?")
    assert plan.source == "heuristic" and plan.intent == "count" and plan.filters.tags_all == ["raman"]


def test_backend_down_is_not_swallowed(planner_for):
    planner, _ = planner_for(BackendUnavailable("ollama down"))
    with pytest.raises(BackendUnavailable):
        planner.plan("Wie viele?")


def test_chitchat_needs_no_llm(planner_for):
    planner, llm = planner_for(plan_json("search"))
    assert planner.plan("Hallo!").intent == "chitchat" and llm.complete_calls == []
    assert planner.plan("Danke").intent == "chitchat"


# -------------------------------------------------------------- heuristics


@pytest.fixture
def catalog(demo_store):
    return load_catalog(demo_store)


@pytest.mark.parametrize(
    "question, intent",
    [
        ("Wie viele Experimente gibt es?", "count"),
        ("How many experiments use raman?", "count"),
        ("Welche Experimente gab es zu raman?", "list"),
        ("Warum war das Rauschen so hoch?", "search"),
        ("Hilfe", "chitchat"),
    ],
)
def test_heuristic_intents(catalog, question, intent):
    assert heuristic_plan(question, catalog).intent == intent


def test_heuristic_numeric_condition(catalog):
    plan = heuristic_plan("Welche Messungen hatten eine Wellenlänge über 600 nm?", catalog)
    m = plan.measurement
    assert plan.intent == "list" and (m.name, m.op, m.value, m.unit) == ("Wellenlänge", ">", 600.0, "nm")


@pytest.mark.parametrize(
    "question, op, value, value2",
    [
        ("Laserleistung mindestens 50 mW", ">=", 50, None),
        ("Laserleistung höchstens 20 mW", "<=", 20, None),
        ("Laserleistung unter 30 mW", "<", 30, None),
        ("Laserleistung genau 100 mW", "=", 100, None),
        ("Laserleistung zwischen 10 und 60 mW", "between", 10, 60),
        ("Laserleistung more than 1,5 W", ">", 1.5, None),
    ],
)
def test_heuristic_operators(catalog, question, op, value, value2):
    m = heuristic_plan(question, catalog).measurement
    assert (m.op, m.value, m.value2) == (op, value, value2)


def test_heuristic_unknown_unit_word_is_not_a_unit(catalog):
    m = heuristic_plan("Wellenlänge über 600 Proben", catalog).measurement
    assert m.value == 600 and m.unit == "nm"  # 'Proben' ignored, catalog default used


def test_heuristic_aggregates(catalog):
    plan = heuristic_plan("Was war die maximale Belichtungszeit bei raman?", catalog)
    assert plan.intent == "aggregate" and plan.agg == "max" and plan.filters.tags_all == ["raman"]
    assert heuristic_plan("Durchschnittliche Laserleistung", catalog).agg == "avg"
    assert heuristic_plan("Kleinste Wellenlänge", catalog).agg == "min"
    assert heuristic_plan("Gesamtzahl Zyklen: Summe", catalog).agg == "sum"


def test_heuristic_inflected_measurement_name(catalog):
    assert heuristic_plan("Welche Wellenlängen gab es über 600 nm?", catalog).measurement.name == "Wellenlänge"


def test_heuristic_group_by(catalog):
    plan = heuristic_plan("Wie viele Experimente gibt es pro Status?", catalog)
    assert plan.intent == "count" and plan.group_by == "status"
    assert heuristic_plan("Wie viele pro Monat?", catalog).group_by == "month"


def test_heuristic_category_and_status(catalog):
    plan = heuristic_plan("Wie viele Experimente in der Kategorie Biologie sind In Arbeit?", catalog)
    assert plan.filters.category == "Biologie" and plan.filters.status == "In Arbeit"


# ------------------------------------------------------------------ helpers


def test_extract_ids():
    assert extract_ids("Fasse Experiment 12 zusammen") == [12]
    assert extract_ids("Vergleiche Experimente 3, 4 und 5") == [3, 4, 5]
    assert extract_ids("#42 und exp. 7") == [42, 7]
    assert extract_ids("Probe A3 bei 785 nm") == []


def test_snap():
    options = ["Laserleistung", "Wellenlänge"]
    assert snap("wellenlänge", options) == "Wellenlänge"
    assert snap("laser", options) == "Laserleistung"  # unique substring
    assert snap("Quark", options) is None


def test_detect_language():
    assert detect_language("Wie viele Experimente gab es?") == "de"
    assert detect_language("How many experiments were there?") == "en"
    assert detect_language("A3 532", default="en") == "en"
