"""Lab Assistant: notes, photos and audio -> reviewed draft -> eLabFTW experiment.

    streamlit run lab_assistant/app.py
"""

from __future__ import annotations

# --- let `streamlit run lab_assistant/app.py` find the repo's packages -------
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ------------------------------------------------------------------------------

from types import ModuleType
from typing import Callable

import streamlit as st

from lab_assistant import elab_api
from lab_assistant.components.input import render_input
from lab_assistant.components.lab_form import render_lab_form
from lab_assistant.config import AssistantConfig
from shared.config import ConfigError
from shared.elabftw_client import ElabFTWError


@st.cache_data(ttl=300, show_spinner="Loading eLabFTW structure …")
def cached_structure() -> dict:
    return elab_api.load_structure()


def render_app(
    load_structure: Callable[[], dict] = cached_structure,
    config: AssistantConfig | None = None,
    api: ModuleType = elab_api,
    generate: Callable | None = None,
    process: Callable | None = None,
    embedded: bool = False,
) -> None:
    """Draw the page. ``embedded=True`` when it is one page of the combined ``app.py``
    (the launcher owns the page config and the navigation)."""
    config = config or AssistantConfig.from_env()
    if not embedded:
        st.set_page_config(page_title="Lab Assistant", page_icon="🧪")
    st.title("🧪 Lab Assistant")
    st.write("Create laboratory documentation from text and uploaded files.")

    try:
        structure = load_structure()
    except ConfigError as error:
        st.error(f"Configuration problem: {error}")
        st.stop()
    except ElabFTWError as error:
        st.error(f"Could not load the eLabFTW structure: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Unexpected error while loading the eLabFTW structure: {error}")
        st.stop()

    with st.sidebar:
        if st.button("Reload eLabFTW structure"):
            cached_structure.clear()
            st.session_state.pop("template_fields", None)
            st.rerun()
        if not embedded:
            st.caption("Questions about your notebook? Start the Lab Chatbot:\n\n`streamlit run lab_chatbot/ui.py`")

    kwargs = {k: v for k, v in (("generate", generate), ("process", process)) if v is not None}
    draft = render_input(config, **kwargs)
    st.divider()
    render_lab_form(structure, draft, api=api)


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
