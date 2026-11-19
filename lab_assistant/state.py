"""Keep the assistant's form alive while the user visits another page of a multi-page app.

Streamlit forgets the value of every widget that is not drawn during a run. In the
combined app that means: type notes, glance at the Lab Chatbot, come back -> empty
form. Re-assigning a widget's value through ``st.session_state`` *before* the widget
is drawn turns it into ordinary session state, which survives (see
``keep_widget_state``). File uploaders and buttons cannot be handled that way; the
uploaded files are mirrored into ``lab_sources`` instead (see ``components/input.py``).
"""

from __future__ import annotations

from typing import MutableMapping

import streamlit as st

# Widgets that have no ``value=`` default of their own and whose keys are fixed.
WIDGET_KEYS = frozenset(
    {
        "written_notes",
        "lab_template",
        "lab_category",
        "lab_status",
        "lab_resource",
        "lab_resource_category",
        "lab_tags",
        "lab_title",
        "lab_body",
        "allow_duplicate",
    }
)
# Custom-field widgets are keyed ``t<template>_v<draft version>_cf_<name>[_value|_unit]``.
CUSTOM_FIELD_MARKER = "_cf_"
# Not assignable through session state (Streamlit raises): buttons and the uploader.
NOT_ASSIGNABLE = frozenset({"lab_file_uploader", "generate_lab_entry", "create_lab_entry"})


def owned_keys(keys) -> list[str]:
    return [k for k in keys if k not in NOT_ASSIGNABLE and (k in WIDGET_KEYS or CUSTOM_FIELD_MARKER in k)]


def keep_widget_state(state: MutableMapping | None = None) -> None:
    """Call at the start of every run, before any page is drawn."""
    state = st.session_state if state is None else state
    for key in owned_keys(list(state.keys())):
        state[key] = state[key]
