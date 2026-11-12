"""Step 1: collect written notes and files, run OCR / transcription, generate the draft."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from lab_assistant.ai import Draft, generate_lab_entry
from lab_assistant.config import AssistantConfig
from lab_assistant.file_processing import can_use_for_ai, get_file_type, process_file_for_ai, source_key
from lab_assistant.ollama_client import AIError

ICONS = {"image": "📷", "audio": "🎙️", "file": "📎"}
SEPARATOR = "\n\n--------------------\n\n"


def _sync_sources(uploaded_files, uploader_was_reset: bool = False) -> dict:
    """Mirror the uploader into session state; identity = content hash + name.

    Streamlit empties the uploader whenever the page is left (e.g. for the Lab
    Chatbot). ``uploader_was_reset`` marks the files that were uploaded before as
    *detached*: they are kept, with their extracted text, until removed explicitly.
    Without that, coming back would silently throw away minutes of OCR work."""
    sources = st.session_state.setdefault("lab_sources", {})
    if uploader_was_reset:
        for source in sources.values():
            source["detached"] = True
    current = set()
    for uploaded in uploaded_files or []:
        content = uploaded.getvalue()
        key = source_key(uploaded.name, content)
        current.add(key)
        if key in sources:
            sources[key]["detached"] = False  # uploaded again: the uploader owns it once more
        else:
            sources[key] = {
                "name": uploaded.name,
                "type": get_file_type(uploaded.name),
                "content": content,
                "use_for_ai": can_use_for_ai(uploaded.name),
                "attach": True,
                "extracted_text": None,
                "detached": False,
            }
    for key in [k for k, s in sources.items() if k not in current and not s.get("detached")]:
        del sources[key]  # removed from the uploader
    return sources


def _render_source(key: str, source: dict) -> None:
    name = source["name"]
    with st.expander(f"{ICONS.get(source['type'], '📎')} {name}", expanded=True):
        if source.get("detached"):
            st.caption("Kept from before you left this page (the upload box itself was reset).")
            if st.button("Remove this file", key=f"remove_{key}"):
                del st.session_state["lab_sources"][key]
                st.rerun()
        ai_supported = can_use_for_ai(name)
        source["attach"] = st.checkbox(
            "Attach original to eLabFTW", value=source["attach"], key=f"attach_{key}"
        )
        source["use_for_ai"] = st.checkbox(
            "Use content for AI generation",
            value=source["use_for_ai"],
            disabled=not ai_supported,
            key=f"ai_{key}",
        )
        if not ai_supported:
            st.caption("AI processing is available for images and audio. Other files can still be attached.")
        if source.get("extracted_text"):
            source["extracted_text"] = st.text_area(
                "Extracted content (edit to correct errors before generating)",
                value=source["extracted_text"],
                height=180,
                key=f"extracted_{key}",
            )


def build_ai_input(written_notes: str, sources: dict, process: Callable) -> str:
    """Combine notes + extracted file contents. Raises on the first file that fails:
    generating a draft that silently lacks one of the sources would be worse than an error."""
    sections = []
    if written_notes.strip():
        sections.append("SOURCE: WRITTEN NOTES\n\n" + written_notes.strip())
    for source in sources.values():
        if not source["use_for_ai"]:
            continue
        if not source.get("extracted_text"):
            with st.spinner(f"Processing {source['name']} …"):
                source["extracted_text"] = process(source["name"], source["content"])
        if source.get("extracted_text"):
            sections.append(
                f"SOURCE: {source['type'].upper()}\nFILE: {source['name']}\n\n{source['extracted_text']}"
            )
    return SEPARATOR.join(sections)


def render_input(
    config: AssistantConfig,
    generate: Callable[[str], Draft] | None = None,
    process: Callable[[str, bytes], str | None] | None = None,
) -> Draft | None:
    """``generate`` / ``process`` are injectable so the page can be tested without models."""
    generate = generate or (lambda text: generate_lab_entry(text, config))
    process = process or (lambda name, content: process_file_for_ai(name, content, config))
    st.subheader("Input")
    written_notes = st.text_area(
        "Written notes", height=160, placeholder="Optional laboratory notes …", key="written_notes"
    )
    # the uploader's key is gone from session state iff Streamlit dropped the widget
    # (page left, session restored); files uploaded before then are kept as "detached"
    uploader_was_reset = "lab_file_uploader" not in st.session_state
    uploaded = st.file_uploader(
        "Files",
        accept_multiple_files=True,
        key="lab_file_uploader",
        help="Images (handwritten notes), audio, raw data, documents or any other file.",
    )
    sources = _sync_sources(uploaded, uploader_was_reset)
    if sources:
        st.markdown("#### Uploaded files")
    for key, source in sources.items():
        _render_source(key, source)

    if st.button("Generate laboratory entry", type="primary", key="generate_lab_entry"):
        try:
            combined = build_ai_input(written_notes, sources, process)
            if not combined:
                st.warning("There is no content selected for AI generation.")
                return st.session_state.get("draft")
            with st.spinner("Generating laboratory entry …"):
                draft = generate(combined)
        except AIError as error:
            st.error(str(error))
            return st.session_state.get("draft")
        except Exception as error:  # a failing model must not take the page down
            st.error(f"Could not generate the entry: {error}")
            return st.session_state.get("draft")
        st.session_state["draft"] = draft
        st.session_state["draft_version"] = st.session_state.get("draft_version", 0) + 1
        st.success("Draft generated. Please review it below before creating the entry.")
    return st.session_state.get("draft")
