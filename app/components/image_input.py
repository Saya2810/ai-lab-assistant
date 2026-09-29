import streamlit as st

from services.vision import analyze_lab_image
from services.ai import generate_lab_entry_from_elab


def render_image_input(
    templates,
    categories,
    statuses,
    resources,
    tags,
):
    st.subheader("📷 Laboratory Note Image")

    uploaded_image = st.file_uploader(
        "Upload a photo of laboratory notes",
        type=[
            "jpg",
            "jpeg",
            "png",
            "heic",
            "heif",
        ],
        key="laboratory_image",
    )

    if uploaded_image is None:
        return None

    image_content = uploaded_image.getvalue()

    # Keep ORIGINAL image for eLabFTW.
    st.session_state["source_image"] = {
        "name": uploaded_image.name,
        "content": image_content,
    }

    # HEIC preview isn't reliably supported by browsers,
    # so only preview browser-friendly formats here.
    extension = (
        uploaded_image.name
        .lower()
        .split(".")[-1]
    )

    if extension in ["jpg", "jpeg", "png"]:
        st.image(
            image_content,
            caption=uploaded_image.name,
        )
    else:
        st.info(
            f"Image loaded: {uploaded_image.name}"
        )

    if st.button(
        "Read laboratory notes",
        key="analyze_lab_image",
    ):
        try:
            with st.spinner(
                "Reading laboratory notes with Qwen3-VL..."
            ):
                transcription = analyze_lab_image(
                    image_content
                )

            st.session_state[
                "image_transcription"
            ] = transcription

            st.success(
                "Image analysis complete."
            )

        except Exception as error:
            st.error(
                f"Image analysis failed: {error}"
            )

    transcription = st.session_state.get(
        "image_transcription",
        "",
    )

    if transcription:
        transcription = st.text_area(
            "Extracted laboratory notes",
            value=transcription,
            height=250,
            key="image_transcription_editor",
        )

        st.caption(
            "Check the transcription before generating "
            "the laboratory entry."
        )

        if st.button(
            "Generate lab entry from image",
            key="generate_from_image",
            type="primary",
        ):
            if not transcription.strip():
                st.warning(
                    "The transcription is empty."
                )
                return None

            try:
                with st.spinner(
                    "Generating laboratory entry..."
                ):
                    ai_result = (
                        generate_lab_entry_from_elab(
                            raw_notes=transcription,
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
                    "AI draft generated from image."
                )

                return ai_result

            except Exception as error:
                st.error(
                    f"AI generation failed: {error}"
                )

    return st.session_state.get("ai_result")