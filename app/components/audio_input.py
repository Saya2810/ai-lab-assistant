import os
import tempfile

import streamlit as st

from services.transcription import (
    transcribe_audio,
)

from services.ai import (
    generate_lab_entry_from_elab,
)


def render_audio_input(
    templates,
    categories,
    statuses,
    resources,
    tags,
):
    """
    Upload an audio laboratory note, transcribe it
    locally with Whisper and generate an AI draft.
    """

    st.subheader("🎙️ Audio Laboratory Notes")

    uploaded_audio = st.file_uploader(
        "Upload an audio note",
        type=[
            "wav",
            "mp3",
            "m4a",
            "mp4",
            "aac",
        ],
        key="audio_note",
    )

    if uploaded_audio is None:
        return None

    st.audio(
        uploaded_audio
    )

    if st.button(
        "Transcribe audio",
        key="transcribe_audio",
    ):

        temp_path = None

        try:

            suffix = os.path.splitext(
                uploaded_audio.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_audio.getvalue()
                )

                temp_path = temp_file.name

            with st.spinner(
                "Transcribing audio locally..."
            ):

                transcript = transcribe_audio(
                    temp_path
                )

            st.session_state[
                "audio_transcript"
            ] = transcript

            st.success(
                "Audio transcription complete."
            )

        except Exception as error:

            st.error(
                f"Audio transcription failed: {error}"
            )

        finally:

            if (
                temp_path
                and os.path.exists(
                    temp_path
                )
            ):

                os.remove(
                    temp_path
                )

    transcript = st.session_state.get(
        "audio_transcript",
        "",
    )

    if transcript:

        transcript = st.text_area(
            "Transcript",
            value=transcript,
            height=180,
            key="audio_transcript_editor",
        )

        st.caption(
            "Check and correct the transcript before "
            "generating the laboratory entry."
        )

        if st.button(
            "Generate lab entry from audio",
            key="generate_from_audio",
            type="primary",
        ):

            if not transcript.strip():

                st.warning(
                    "The transcript is empty."
                )

                return None

            try:

                with st.spinner(
                    "Generating laboratory entry..."
                ):

                    ai_result = (
                        generate_lab_entry_from_elab(
                            raw_notes=transcript,
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
                    "AI draft generated from audio."
                )

                return ai_result

            except Exception as error:

                st.error(
                    f"AI generation failed: {error}"
                )

    return st.session_state.get(
        "ai_result"
    )