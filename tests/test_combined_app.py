"""The combined app (``streamlit run app.py``): both pages in one navigation.

AppTest cannot switch between ``st.Page`` callables, so the real page switch is covered
by the browser test in ``tests/e2e``. Here: the pieces it relies on, headless.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lab_assistant import state as assistant_state
from lab_chatbot import state as chatbot_state
from shared.config import REPO_ROOT
from tests.fakes import plan_json

@pytest.fixture(autouse=True)
def _restore_page_config():
    """Some scripts below replace ``st.set_page_config`` (AppTest runs them in this process)."""
    import streamlit as st

    original = st.set_page_config
    yield
    st.set_page_config = original


# ------------------------------------------------------------------ pinning


class Recorder(dict):
    def __init__(self, *args):
        super().__init__(*args)
        self.assigned: list[str] = []

    def __setitem__(self, key, value):
        self.assigned.append(key)
        super().__setitem__(key, value)


def test_assistant_pins_its_form_widgets_but_not_buttons_or_the_uploader():
    state = Recorder(
        {
            "written_notes": "x", "lab_title": "t", "lab_body": "b", "lab_template": "T", "lab_tags": ["a"],
            "t5_v1_cf_Operator": "o", "t5_v1_cf_Wellenlänge_value": "5", "t5_v1_cf_Wellenlänge_unit": "nm",
            "lab_file_uploader": [object()], "generate_lab_entry": False, "create_lab_entry": False,
            "draft": object(), "lab_sources": {}, "messages": [], "chat_debug": True,
        }
    )  # fmt: skip
    assistant_state.keep_widget_state(state)
    assert sorted(state.assigned) == sorted(
        ["written_notes", "lab_title", "lab_body", "lab_template", "lab_tags",
         "t5_v1_cf_Operator", "t5_v1_cf_Wellenlänge_value", "t5_v1_cf_Wellenlänge_unit"]
    )  # fmt: skip


def test_assistant_pinning_ignores_a_state_without_its_keys():
    state = Recorder({"messages": []})
    assistant_state.keep_widget_state(state)
    assert state.assigned == []


def test_the_two_packages_pin_disjoint_keys():
    assert not set(chatbot_state.WIDGET_KEYS) & assistant_state.WIDGET_KEYS
    assert assistant_state.owned_keys(chatbot_state.WIDGET_KEYS) == []


# ----------------------------------------------------------------- launcher


def launcher_script():
    import streamlit as st

    import app as launcher

    def assistant():
        st.title("ASSISTANT PAGE")
        st.text_area("Notes", key="written_notes")
        st.text_input("Title", key="lab_title")

    def chatbot():
        st.title("CHATBOT PAGE")

    launcher.run(assistant, chatbot)


def test_launcher_starts_on_the_assistant_page_and_registers_both_pages():
    at = AppTest.from_function(launcher_script, default_timeout=15).run()
    assert not at.exception
    assert at.title[0].value == "ASSISTANT PAGE"


def test_pinning_does_not_disturb_widgets_that_are_drawn():
    at = AppTest.from_function(launcher_script, default_timeout=15).run()
    at.text_area(key="written_notes").set_value("Laser 532 nm").run()
    at.text_input(key="lab_title").set_value("Mein Titel").run()
    at.text_area(key="written_notes").set_value("Laser 532 nm, 20 mW").run()  # edits keep working after pinning
    assert not at.exception
    assert at.text_area(key="written_notes").value == "Laser 532 nm, 20 mW"
    assert at.text_input(key="lab_title").value == "Mein Titel"


def test_launcher_with_the_real_pages_and_no_api_key_explains_instead_of_crashing():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=20).run()
    assert not at.exception
    assert at.title[0].value.endswith("Lab Assistant")
    assert any("ELABFTW_API_KEY" in e.value for e in at.error)


def missing_package_script():
    import sys

    sys.modules["lab_assistant.app"] = None  # makes ``from lab_assistant.app import ...`` raise ImportError
    import app as launcher

    launcher.run()


def test_a_missing_package_only_breaks_its_own_page(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "lab_assistant.app", sys.modules.get("lab_assistant.app"))  # restored afterwards
    at = AppTest.from_function(missing_package_script, default_timeout=15).run()
    assert not at.exception
    assert any("requirements-assistant.txt" in e.value for e in at.error)


def test_app_py_exists_next_to_the_packages_and_is_documented():
    root = Path(REPO_ROOT)
    assert (root / "app.py").is_file()
    assert "streamlit run app.py" in (root / "README.md").read_text(encoding="utf-8")


# ------------------------------------------- pages inside the combined app


def assistant_embedded_script():
    import streamlit as st

    def boom(*a, **k):
        raise AssertionError("an embedded page must not call set_page_config (the launcher owns it)")

    st.set_page_config = boom
    from lab_assistant.app import render_app
    from lab_assistant.config import AssistantConfig
    from tests import fakes as f

    render_app(lambda: f.STRUCTURE, AssistantConfig.from_env(), api=f.FakeElabApi(), embedded=True)


def chatbot_embedded_script(db_path: str):
    from dataclasses import replace

    import streamlit as st

    def boom(*a, **k):
        raise AssertionError("an embedded page must not call set_page_config (the launcher owns it)")

    st.set_page_config = boom
    from lab_chatbot import ui
    from lab_chatbot.config import ChatbotConfig
    from lab_chatbot.engine import Engine
    from lab_chatbot.store import Store
    from tests.fakes import FakeEmbedder, FakeLLM

    config = replace(ChatbotConfig.from_env(db_path), default_language="de")
    engine = Engine(Store(db_path), FakeEmbedder(), FakeLLM(plan={"intent": "search", "semantic_query": "Rauschen", "filters": {}, "measurement": None, "group_by": None, "order": None}), config)
    ui.render_app(lambda: engine, config, embedded=True)


def test_embedded_assistant_does_not_set_the_page_config_and_drops_the_cli_hint():
    at = AppTest.from_function(assistant_embedded_script, default_timeout=15).run()
    assert not at.exception
    assert not any("streamlit run lab_chatbot" in c.value for c in at.sidebar.caption)


def test_standalone_assistant_still_sets_the_page_config_and_shows_the_hint():
    from tests.test_assistant_ui import run

    at = run()
    assert not at.exception
    assert any("streamlit run lab_chatbot" in c.value for c in at.sidebar.caption)


def test_embedded_chatbot_does_not_set_the_page_config(demo_store):
    at = AppTest.from_function(chatbot_embedded_script, args=(str(demo_store.path),), default_timeout=15).run()
    assert not at.exception
    assert at.title[0].value.endswith("Lab Chatbot")


def test_chatbot_sidebar_settings_have_stable_keys(demo_store):
    at = AppTest.from_function(chatbot_embedded_script, args=(str(demo_store.path),), default_timeout=15).run()
    assert at.toggle(key=chatbot_state.DEBUG_KEY).value is False
    assert at.checkbox(key=chatbot_state.FULL_SYNC_KEY) is not None


def test_example_questions_disappear_once_a_question_is_asked(demo_store):
    at = AppTest.from_function(chatbot_embedded_script, args=(str(demo_store.path),), default_timeout=15).run()
    assert any("Status" in b.label for b in at.button)
    at.chat_input[0].set_value("Warum war das Rauschen bei Probe A3 so hoch?").run()
    assert not at.exception
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
    assert not any("Status" in b.label for b in at.button)  # used to stay on screen until the next click


@pytest.mark.parametrize("plan", [plan_json("search", "Rauschen")])
def test_plan_helper_is_importable_for_the_scripts(plan):
    assert plan["intent"] == "search"
