import streamlit as st

from services.elabftw import (
    get_experiment_categories,
    get_experiment_statuses,
    get_experiment_templates,
    get_resources,
    get_tags,
)

from components.input import (
    render_input,
)

from components.lab_form import (
    render_lab_form,
)

from components.chatbot import render_chatbot

st.set_page_config(
    page_title="Lab Assistant",
    page_icon="🧪",
)

st.title("🧪 Lab Assistant")

st.write(
    "Create laboratory documentation "
    "from text and uploaded files."
)


# ---------------------------------------
# Load current eLabFTW structure
# ---------------------------------------

try:

    categories = (
        get_experiment_categories()
    )

    statuses = (
        get_experiment_statuses()
    )

    templates = (
        get_experiment_templates()
    )

    resources = (
        get_resources()
    )

    tags = (
        get_tags()
    )

except Exception as error:

    st.error(
        "Could not load eLabFTW "
        f"structure: {error}"
    )

    st.stop()


# ---------------------------------------
# Input
# ---------------------------------------

ai_result = render_input(
    templates=templates,
    categories=categories,
    statuses=statuses,
    resources=resources,
    tags=tags,
)


st.divider()


# ---------------------------------------
# Editable eLabFTW form
# ---------------------------------------

render_lab_form(
    templates=templates,
    categories=categories,
    statuses=statuses,
    resources=resources,
    tags=tags,
    ai_result=ai_result,
)

st.divider()

render_chatbot()