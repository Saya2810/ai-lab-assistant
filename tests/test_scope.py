"""Limiting the chatbot to part of the notebook (projects, tags, categories, statuses)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from lab_chatbot.cli import scope_from_args
from lab_chatbot.engine import Engine
from lab_chatbot.scope import Scope, scope_options
from lab_chatbot.sql_tools import Filters, candidate_ids
from tests.fakes import FakeLLM, plan_json


@pytest.fixture
def scoped_store(demo_store):
    """The demo notebook with projects: 102 and 105 belong to "Projekt Raman", 105 also to "Gerät X"."""
    with demo_store.connect() as con:
        con.execute("UPDATE experiments SET resources = 'Projekt Raman' WHERE id = 102")
        con.execute("UPDATE experiments SET resources = 'Projekt Raman | Gerät X' WHERE id = 105")
        con.execute("UPDATE experiments SET resources = 'Projekt Raman II' WHERE id = 101")
    with demo_store.connect() as con:  # options are cached per revision
        con.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'")
    return demo_store


def ids(store, scope: Scope) -> set[int]:
    return candidate_ids(store, Filters(scope=scope))


def test_options_list_every_project_tag_category_and_status(scoped_store):
    options = scope_options(scoped_store)
    assert options["resources"] == ["Gerät X", "Projekt Raman", "Projekt Raman II"]
    assert "raman" in options["tags"] and options["categories"] == ["Biologie", "Labor", "Messung"]
    assert options["statuses"] == ["Abgeschlossen", "In Arbeit"]


def test_any_within_a_field_all_across_fields(scoped_store):
    assert ids(scoped_store, Scope.of(resources=["Projekt Raman"])) == {102, 105}  # not "Projekt Raman II"
    assert ids(scoped_store, Scope.of(resources=["Gerät X", "Projekt Raman II"])) == {101, 105}
    assert ids(scoped_store, Scope.of(tags=["raman", "pcr"])) == {102, 103, 105}
    assert ids(scoped_store, Scope.of(tags=["zellkultur"], categories=["Biologie"])) == {104}
    assert ids(scoped_store, Scope.of(categories=["Messung"], statuses=["In Arbeit"])) == {108}
    assert ids(scoped_store, Scope.of(resources=["Projekt Raman"], tags=["pcr"])) == set()


def make_engine(store, embedder, config, plan, answer="Laut Quelle [1] ist das so."):
    llm = FakeLLM(plan=plan, answer=answer)
    return Engine(store, embedder, llm, config), llm


def test_text_answers_only_use_experiments_in_scope(scoped_store, embedder, config):
    engine, _ = make_engine(scoped_store, embedder, config, plan_json("search", "Raman Messung Probe"))
    answer = engine.ask("Was wurde bei den Raman-Messungen gemessen?", scope=Scope.of(resources=["Gerät X"]))
    assert answer.sources and {s.experiment_id for s in answer.sources} == {105}


def test_counts_and_lists_respect_the_scope(scoped_store, embedder, config):
    engine, _ = make_engine(scoped_store, embedder, config, plan_json("count"))
    answer = engine.ask("Wie viele Experimente gibt es?", scope=Scope.of(tags=["raman"]))
    assert "2" in answer.text and "Eingrenzung: Tags: raman" in answer.text
    engine, _ = make_engine(scoped_store, embedder, config, plan_json("list"))
    answer = engine.ask("Liste alle Experimente", scope=Scope.of(statuses=["In Arbeit"]))
    assert sorted(row[0] for row in answer.tables[0].rows) == [103, 108]


def test_a_named_experiment_outside_the_scope_is_not_used(scoped_store, embedder, config):
    engine, _ = make_engine(scoped_store, embedder, config, plan_json("search", "Puffer"))
    answer = engine.ask("Was steht in Experiment 106?", scope=Scope.of(resources=["Projekt Raman"]))
    assert not answer.used_llm and "106" not in [str(s.experiment_id) for s in answer.sources]


def test_a_scope_without_experiments_answers_without_the_llm(scoped_store, embedder, config):
    engine, llm = make_engine(scoped_store, embedder, config, plan_json("search", "x"))
    answer = engine.ask("Irgendwas?", scope=Scope.of(resources=["Projekt Raman"], tags=["pcr"]))
    assert not answer.used_llm and "passt auf kein Experiment" in answer.text
    assert llm.complete_calls == [] and llm.stream_calls == []


def test_no_scope_changes_nothing(scoped_store, embedder, config):
    engine, _ = make_engine(scoped_store, embedder, config, plan_json("count"))
    assert "8" in engine.ask("Wie viele Experimente gibt es?", scope=Scope()).text


def test_cli_scope_arguments():
    args = SimpleNamespace(project=["P1"], tag=["a", "b", "a"], category=None, status=None)
    assert scope_from_args(args) == Scope(resources=("P1",), tags=("a", "b"))
    assert scope_from_args(SimpleNamespace(project=None, tag=None, category=None, status=None)) is None


# ---------------------------------------------------------------------- UI


def page(db_path: str):
    from lab_chatbot import ui as app_ui
    from lab_chatbot.config import ChatbotConfig
    from lab_chatbot.engine import Engine
    from lab_chatbot.store import Store
    from tests.fakes import FakeEmbedder, FakeLLM, plan_json

    config = ChatbotConfig.from_env(db_path)
    engine = Engine(Store(db_path), FakeEmbedder(), FakeLLM(plan=plan_json("count")), config)
    app_ui.render_app(lambda: engine, config)


def test_sidebar_scope_limits_the_answer_and_is_shown(scoped_store):
    at = AppTest.from_function(page, args=(str(scoped_store.path),), default_timeout=15).run()
    at.multiselect(key="chat_scope_resources").select("Projekt Raman").run()
    assert not at.exception
    assert any("Eingegrenzt auf Projekt: Projekt Raman (2 Experimente)" in i.value for i in at.info)
    at.chat_input[0].set_value("Wie viele Experimente gibt es?").run()
    reply = at.session_state["messages"][-1]["content"]
    assert "2" in reply and "8" not in reply
