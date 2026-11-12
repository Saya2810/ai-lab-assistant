"""Keep the chatbot's sidebar settings alive while the user visits another page of a multi-page app.

Streamlit forgets the value of every widget that is not drawn during a run.
Re-assigning the value through ``st.session_state`` *before* the widget is drawn turns
it into ordinary session state, which survives. (The chat history itself is plain
session state and needs nothing.)
"""

from __future__ import annotations

import streamlit as st

DEBUG_KEY = "chat_debug"
FULL_SYNC_KEY = "chat_full_sync"
WIDGET_KEYS = (DEBUG_KEY, FULL_SYNC_KEY)


def keep_widget_state() -> None:
    """Call at the start of every run, before any page is drawn."""
    for key in WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
