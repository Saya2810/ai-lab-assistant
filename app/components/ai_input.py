import streamlit as st

from services.ai import (
    generate_lab_entry_from_elab,
)


def render_ai_input(
    templates,
    categories,
    statuses,
    resources,
    tags,
):
    """
    Render the AI input section.

    The AI receives the current eLabFTW structure
    and creates a structured draft.

    The result is stored in:
        st.session_state["ai_result"]
    """

    st.subheader("🤖 AI Laboratory Assistant")

    raw_ai_notes = st.text_area(
        "Raw laboratory notes",
        height=180,
        placeholder=(
            "Example: We tested the spectrometer with "
            "a 532 nm laser. Exposure was 5 seconds. "
            "The first spectrum showed unusually high noise."
        ),
        key="raw_ai_notes",
    )

    if st.button(
        "Generate lab entry with AI",
        type="primary",
    ):

        if not raw_ai_notes.strip():

            st.warning(
                "Please enter laboratory notes first."
            )

        else:

            try:

                with st.spinner(
                    "Generating laboratory entry..."
                ):

                    ai_result = (
                        generate_lab_entry_from_elab(
                            raw_notes=raw_ai_notes,
                            templates=templates,
                            categories=categories,
                            statuses=statuses,
                            resources=resources,
                            tags=tags,
                        )
                    )

                st.session_state[
                    "ai_result"
                ] = ai_result

                st.success(
                    "AI draft generated."
                )

            except Exception as error:

                st.error(
                    f"AI generation failed: {error}"
                )

    ai_result = st.session_state.get(
        "ai_result"
    )

    if ai_result:

        with st.expander(
            "Show AI-generated draft"
        ):

            st.json(
                ai_result
            )

    return ai_result