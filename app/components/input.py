import streamlit as st

from services.ai import (
    generate_lab_entry_from_elab,
)

from services.file_processing import (
    can_use_for_ai,
    get_file_type,
    process_file_for_ai,
)


def initialize_sources():
    if "lab_sources" not in st.session_state:
        st.session_state[
            "lab_sources"
        ] = {}


def render_input(
    templates,
    categories,
    statuses,
    resources,
    tags,
):

    initialize_sources()

    st.subheader(
        "Input"
    )

    # -----------------------------------
    # Written notes
    # -----------------------------------

    written_notes = st.text_area(
        "Written notes",
        height=160,
        placeholder=(
            "Optional laboratory notes..."
        ),
        key="written_notes",
    )

    # -----------------------------------
    # ONE uploader for ALL files
    # -----------------------------------

    uploaded_files = st.file_uploader(
        "Files",
        accept_multiple_files=True,
        key="lab_file_uploader",
        help=(
            "Upload images, audio, raw data, "
            "documents or other laboratory files."
        ),
    )

    # -----------------------------------
    # Register uploaded files
    # -----------------------------------

    current_names = set()

    for uploaded_file in uploaded_files:

        name = uploaded_file.name
        current_names.add(name)

        content = (
            uploaded_file.getvalue()
        )

        file_type = get_file_type(
            name
        )

        if (
            name
            not in st.session_state[
                "lab_sources"
            ]
        ):
            st.session_state[
                "lab_sources"
            ][name] = {
                "name": name,
                "type": file_type,
                "content": content,
                "use_for_ai": (
                    can_use_for_ai(
                        name
                    )
                ),
                "attach": True,
                "extracted_text": None,
            }

    # Remove files that user removed
    # from uploader.

    stored_names = list(
        st.session_state[
            "lab_sources"
        ].keys()
    )

    for name in stored_names:

        if name not in current_names:
            del st.session_state[
                "lab_sources"
            ][name]

    # -----------------------------------
    # Display uploaded sources
    # -----------------------------------

    sources = st.session_state[
        "lab_sources"
    ]

    if sources:

        st.markdown(
            "#### Uploaded files"
        )

    for name, source in sources.items():

        file_type = source["type"]

        if file_type == "image":
            icon = "📷"

        elif file_type == "audio":
            icon = "🎙️"

        else:
            icon = "📎"

        with st.expander(
            f"{icon} {name}",
            expanded=True,
        ):

            ai_supported = (
                can_use_for_ai(name)
            )

            source[
                "attach"
            ] = st.checkbox(
                "Attach original to eLabFTW",
                value=source[
                    "attach"
                ],
                key=f"attach_{name}",
            )

            source[
                "use_for_ai"
            ] = st.checkbox(
                "Use content for AI generation",
                value=source[
                    "use_for_ai"
                ],
                disabled=not ai_supported,
                key=f"ai_{name}",
            )

            if not ai_supported:

                st.caption(
                    "AI processing for this "
                    "file type is not implemented "
                    "yet. The file can still be "
                    "attached to eLabFTW."
                )

            extracted_text = source.get(
                "extracted_text"
            )

            if extracted_text:

                edited_text = st.text_area(
                    "Extracted content",
                    value=extracted_text,
                    height=180,
                    key=f"extracted_{name}",
                )

                source[
                    "extracted_text"
                ] = edited_text

    # -----------------------------------
    # Generate
    # -----------------------------------

    if st.button(
        "Generate laboratory entry",
        type="primary",
        key="generate_lab_entry",
    ):

        ai_sections = []

        # Written text always contributes
        # when it contains something.

        if written_notes.strip():

            ai_sections.append(
                "SOURCE: WRITTEN NOTES\n\n"
                + written_notes.strip()
            )

        # Process selected files.

        for name, source in sources.items():

            if not source[
                "use_for_ai"
            ]:
                continue

            try:

                if not source.get(
                    "extracted_text"
                ):

                    with st.spinner(
                        f"Processing {name}..."
                    ):

                        source[
                            "extracted_text"
                        ] = (
                            process_file_for_ai(
                                file_name=name,
                                file_content=source[
                                    "content"
                                ],
                            )
                        )

                extracted = source.get(
                    "extracted_text"
                )

                if extracted:

                    ai_sections.append(
                        (
                            f"SOURCE: "
                            f"{source['type'].upper()}\n"
                            f"FILE: {name}\n\n"
                            f"{extracted}"
                        )
                    )

            except Exception as error:

                st.error(
                    f"Could not process "
                    f"{name}: {error}"
                )

                return None

        # No AI material

        if not ai_sections:

            st.warning(
                "There is no content selected "
                "for AI generation."
            )

            return None

        combined_input = (
            "\n\n"
            "--------------------"
            "\n\n"
        ).join(
            ai_sections
        )

        try:

            with st.spinner(
                "Generating laboratory entry..."
            ):

                ai_result = (
                    generate_lab_entry_from_elab(
                        raw_notes=combined_input,
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
                "Laboratory draft generated."
            )

        except Exception as error:

            st.error(
                f"AI generation failed: "
                f"{error}"
            )

    return st.session_state.get(
        "ai_result"
    )