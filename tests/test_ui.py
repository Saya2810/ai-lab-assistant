"""Run the real Streamlit page headlessly (Streamlit's AppTest) against fakes."""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from lab_chatbot import ui
from lab_chatbot.engine import Answer, Table
from lab_chatbot.planner import Plan
from tests.fakes import plan_json


def page(db_path: str, plan: dict | str | None, answer: str, language: str = "de"):
    """Executed by AppTest as the script body (imports must live inside)."""
    from dataclasses import replace

    from lab_chatbot import ui as app_ui
    from lab_chatbot.config import ChatbotConfig
    from lab_chatbot.engine import Engine
    from lab_chatbot.store import Store
    from tests.fakes import FakeEmbedder, FakeLLM

    config = replace(ChatbotConfig.from_env(db_path), default_language=language)
    engine = Engine(Store(db_path), FakeEmbedder(), FakeLLM(plan=plan, answer=answer), config)
    app_ui.render_app(lambda: engine, config)


@pytest.fixture
def app(demo_store):
    def make(plan=None, answer="Das Rauschen kam vom Fluoreszenzuntergrund [1].", language="de"):
        at = AppTest.from_function(page, args=(str(demo_store.path), plan, answer, language), default_timeout=15)
        return at.run()

    return make


def test_page_renders_without_error(app):
    at = app()
    assert not at.exception
    assert at.title[0].value.endswith("Lab Chatbot")
    assert any("Experimente" in m.label for m in at.metric)
    assert at.metric[0].value == "8"


def test_example_questions_are_offered_on_empty_chat(app):
    at = app()
    assert len(at.button) >= 4
    assert any("Status" in b.label for b in at.button)


def test_search_question_streams_answer_and_shows_sources(app):
    at = app(plan_json("search", "Rauschen Probe A3"))
    at.chat_input[0].set_value("Warum war das Rauschen bei Probe A3 so hoch?").run()
    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert "Fluoreszenzuntergrund" in messages[1]["content"]
    meta = messages[1]["meta"]
    assert meta["cited"] == [1] and meta["sources"][0]["experiment_id"] == 102
    assert any("Experiment 102" in e.label for e in at.expander)
    assert any("Verstanden als" in c.value for c in at.caption)


def test_sql_question_shows_a_dataframe(app):
    at = app(plan_json("count", group_by="status"))
    at.chat_input[0].set_value("Wie viele pro Status?").run()
    assert not at.exception
    frame = at.dataframe[0].value
    assert set(frame.iloc[:, 0]) == {"Abgeschlossen", "In Arbeit"}


def test_conversation_history_accumulates_and_clear_resets(app):
    at = app(plan_json("search", "Rauschen A3"))
    at.chat_input[0].set_value("Rauschen bei Probe A3?").run()
    at.chat_input[0].set_value("Und bei Probe B1?").run()
    assert len(at.session_state["messages"]) == 4
    next(b for b in at.button if b.label == "Chat leeren").click().run()
    assert at.session_state["messages"] == []


def test_example_button_asks_the_question(app):
    at = app(plan_json("count", group_by="status"))
    next(b for b in at.button if "Status" in b.label).click().run()
    assert at.session_state["messages"][0]["content"].startswith("Wie viele Experimente gibt es pro Status")


def test_backend_error_is_shown_not_raised(app):
    from lab_chatbot.errors import BackendUnavailable

    at = app(BackendUnavailable("Ollama is not reachable at http://localhost:11434"))
    at.chat_input[0].set_value("Wie viele?").run()
    assert not at.exception and any("not reachable" in e.value for e in at.error)


def test_unexpected_error_is_shown_not_raised(app):
    at = app(plan_json("search", "Rauschen A3"), answer=RuntimeError("kaputt"))
    at.chat_input[0].set_value("Rauschen bei Probe A3?").run()
    assert not at.exception and any("kaputt" in e.value for e in at.error)


def test_warnings_are_rendered(app):
    at = app(plan_json("search", "Rauschen A3"), answer="Die Leistung war 999 mW [1].")
    at.chat_input[0].set_value("Rauschen bei Probe A3?").run()
    assert any("999" in w.value for w in at.warning)


def test_sync_button_is_disabled_without_api_key(app):
    at = app()
    sync = next(b for b in at.button if "synchronisieren" in b.label)
    assert sync.disabled


def test_english_interface(app):
    at = app(language="en")
    assert any("Clear chat" == b.label for b in at.button)


def test_empty_index_shows_a_hint(tmp_path):
    at = AppTest.from_function(page, args=(str(tmp_path / "empty.db"), plan_json("search"), "x", "de"), default_timeout=15)
    at.run()
    assert not at.exception and any("Index ist leer" in i.value for i in at.info)


# ----------------------------------------------------------- pure helpers


def test_describe_plan():
    from lab_chatbot.sql_tools import Filters, MeasurementCond

    plan = Plan("list", "q", "", Filters(tags_all=["raman"]), MeasurementCond("Wellenlänge", ">", 600, None, "nm"))
    text = ui.describe_plan(plan, ui.UI["de"])
    assert "Liste (SQL)" in text and "Tags: raman" in text and "Wellenlänge > 600 nm" in text


def test_table_frame_makes_link_columns_clickable():
    frame, config = ui.table_frame(
        {"columns": ["ID", "Link"], "rows": [[1, "https://x/1"], [2, "https://x/2"]], "title": "T"}
    )
    assert isinstance(frame, pd.DataFrame) and "Link" in config and "ID" not in config


def test_answer_meta_is_json_safe():
    import json

    answer = Answer("q", "a", Plan("search", "q", "q"), tables=[Table("T", ["a"], [[1]])])
    json.dumps(ui.answer_meta(answer))
