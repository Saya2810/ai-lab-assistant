"""Lab Notebook: Lab Assistant (write entries) and Lab Chatbot (ask questions) in one app.

    streamlit run app.py

The two codebases stay separate: ``lab_assistant/`` and ``lab_chatbot/`` never import each
other. This file is the only place that knows both. It owns the page config and the
navigation and draws one of the two as a page. Both can still be started on their own:

    streamlit run lab_assistant/app.py
    streamlit run lab_chatbot/ui.py
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from lab_assistant.state import keep_widget_state as keep_assistant_state
from lab_chatbot.state import keep_widget_state as keep_chatbot_state


def _missing_package(page: str, error: ImportError, requirements: str) -> None:
    st.title(page)
    st.error(
        f"A package this page needs is not installed ({error.name or error}).\n\n"
        f"Run `pip install -r {requirements}` and reload."
    )


def assistant_page() -> None:
    try:
        from lab_assistant.app import render_app
    except ImportError as error:  # only this page is affected
        return _missing_package("🧪 Lab Assistant", error, "requirements-assistant.txt")
    render_app(embedded=True)


def chatbot_page() -> None:
    try:
        from lab_chatbot.ui import main
    except ImportError as error:
        return _missing_package("💬 Lab Chatbot", error, "requirements-chatbot.txt")
    main(embedded=True)


def run(assistant: Callable[[], None] = assistant_page, chatbot: Callable[[], None] = chatbot_page) -> None:
    """Page callables are injectable so tests can use fakes instead of Ollama and eLabFTW."""
    st.set_page_config(page_title="Lab Notebook", page_icon="🧪", layout="centered")
    # Streamlit forgets widgets of pages that are not drawn; pin them before any page runs.
    keep_assistant_state()
    keep_chatbot_state()
    navigation = st.navigation(
        [
            st.Page(assistant, title="Lab Assistant", icon="🧪", url_path="assistant", default=True),
            st.Page(chatbot, title="Lab Chatbot", icon="💬", url_path="chatbot"),
        ]
    )
    navigation.run()


if __name__ == "__main__":
    run()
