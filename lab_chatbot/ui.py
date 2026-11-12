"""Streamlit chat UI for the Lab Chatbot.

    streamlit run lab_chatbot/ui.py

What the UI does on top of the engine: it streams the answer token by token,
shows *how the question was understood* (the plan), renders exact SQL results
as real tables, lists the cited sources with links back into eLabFTW, and
offers a sync button so the index can be refreshed without a terminal.
"""

from __future__ import annotations

# --- let `streamlit run lab_chatbot/ui.py` find the repo's packages ---------
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# -----------------------------------------------------------------------------

import json
import logging
from dataclasses import asdict
from typing import Callable

import pandas as pd
import streamlit as st

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.engine import Answer, Engine
from lab_chatbot.errors import BackendUnavailable, ChatbotError
from lab_chatbot.state import DEBUG_KEY, FULL_SYNC_KEY
from shared.config import ConfigError, env_str
from shared.elabftw_client import ElabFTWError

log = logging.getLogger("lab_chatbot.ui")

UI = {
    "de": {
        "title": "🧪 Lab Chatbot",
        "caption": "Fragen an dein eLabFTW-Laborbuch – Zahlen exakt per SQL, Inhalte per semantischer Suche.",
        "index": "Index",
        "experiments": "Experimente",
        "chunks": "Textabschnitte",
        "last_sync": "Letzter Sync",
        "never": "noch nie",
        "sync": "Mit eLabFTW synchronisieren",
        "sync_full": "Alles neu einlesen (--full)",
        "sync_running": "Synchronisiere …",
        "sync_no_key": "Kein ELABFTW_API_KEY gesetzt – Sync nicht möglich (siehe .env).",
        "sync_failed": "Sync fehlgeschlagen",
        "clear": "Chat leeren",
        "debug": "Plan & Zeiten anzeigen",
        "placeholder": "Frage zum Laborbuch …",
        "examples": "Beispielfragen",
        "example_list": [
            "Wie viele Experimente gibt es pro Status?",
            "Welche Experimente hatten eine Laserleistung über 50 mW?",
            "Was war die maximale Belichtungszeit?",
            "Warum war das Rauschen bei Probe A3 so hoch?",
        ],
        "empty_hint": (
            "Der Index ist leer. Entweder links **synchronisieren** oder zum Ausprobieren "
            "`python -m lab_chatbot seed-demo` ausführen und mit `LAB_DB_PATH=data/chatbot/demo.db` starten."
        ),
        "understood": "Verstanden als",
        "sources": "Quellen",
        "sources_retrieved": "Quellen (abgerufen, nicht zitiert)",
        "plan": "Plan & Zeiten",
        "intent": {
            "search": "semantische Suche",
            "count": "Zählung (SQL)",
            "aggregate": "Auswertung (SQL)",
            "list": "Liste (SQL)",
            "chitchat": "Hilfe",
        },
        "backend_error": "Ollama oder ein Modell ist nicht erreichbar",
        "link": "öffnen",
    },
    "en": {
        "title": "🧪 Lab Chatbot",
        "caption": "Ask your eLabFTW lab notebook – numbers exact via SQL, content via semantic search.",
        "index": "Index",
        "experiments": "Experiments",
        "chunks": "Text chunks",
        "last_sync": "Last sync",
        "never": "never",
        "sync": "Sync with eLabFTW",
        "sync_full": "Re-read everything (--full)",
        "sync_running": "Syncing …",
        "sync_no_key": "ELABFTW_API_KEY not set – cannot sync (see .env).",
        "sync_failed": "Sync failed",
        "clear": "Clear chat",
        "debug": "Show plan & timings",
        "placeholder": "Ask your lab notebook …",
        "examples": "Example questions",
        "example_list": [
            "How many experiments are there per status?",
            "Which experiments had a laser power above 50 mW?",
            "What was the maximum exposure time?",
            "Why was the noise so high for sample A3?",
        ],
        "empty_hint": (
            "The index is empty. Either **sync** on the left, or try `python -m lab_chatbot seed-demo` "
            "and start with `LAB_DB_PATH=data/chatbot/demo.db`."
        ),
        "understood": "Understood as",
        "sources": "Sources",
        "sources_retrieved": "Sources (retrieved, none cited)",
        "plan": "Plan & timings",
        "intent": {
            "search": "semantic search",
            "count": "count (SQL)",
            "aggregate": "aggregation (SQL)",
            "list": "list (SQL)",
            "chitchat": "help",
        },
        "backend_error": "Ollama or a model is not reachable",
        "link": "open",
    },
}


# ---------------------------------------------------------------------------
# pure helpers (importable and testable without a running Streamlit)
# ---------------------------------------------------------------------------


def describe_plan(plan, ui: dict) -> str:
    """One line: how the question was understood."""
    parts = [ui["intent"].get(plan.intent, plan.intent)]
    parts += plan.filters.describe()
    if plan.measurement and (plan.measurement.op or plan.intent == "aggregate"):
        parts.append(plan.measurement.describe())
    if plan.intent == "search" and plan.semantic_query:
        parts.append(f"„{plan.semantic_query}“")
    return " · ".join(parts)


def answer_meta(answer: Answer) -> dict:
    """Everything the UI needs to re-render an answer later (JSON-safe)."""
    plan = json.loads(json.dumps(answer.plan.to_dict(), default=str)) if answer.plan else None
    return {
        "tables": [asdict(t) for t in answer.tables],
        "sources": [asdict(s) for s in answer.sources],
        "cited": list(answer.cited),
        "warnings": list(answer.warnings),
        "plan": plan,
        "latencies": dict(answer.latencies),
        "used_llm": answer.used_llm,
    }


def table_frame(table: dict) -> tuple[pd.DataFrame, dict]:
    """DataFrame + Streamlit column_config (link columns become clickable)."""
    frame = pd.DataFrame(table["rows"], columns=table["columns"])
    config = {}
    for column in frame.columns:
        values = frame[column].astype(str)
        if len(values) and values.str.startswith(("http://", "https://")).all():
            config[column] = st.column_config.LinkColumn(column, display_text="↗")
    return frame, config


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_tables(container, tables: list[dict]) -> None:
    for table in tables:
        frame, config = table_frame(table)
        container.markdown(f"**{table['title']}**")
        container.dataframe(frame, hide_index=True, column_config=config, width="stretch")


def render_extras(container, meta: dict, ui: dict, show_debug: bool) -> None:
    sources = meta.get("sources") or []
    if sources:
        cited = set(meta.get("cited") or [])
        shown = [s for s in sources if s["n"] in cited] or sources
        label = ui["sources"] if cited else ui["sources_retrieved"]
        container.markdown(f"**{label}**")
        for s in shown:
            with container.expander(f"[{s['n']}] Experiment {s['experiment_id']} – {s['title']} ({s['date']})"):
                st.markdown(f"[eLabFTW ↗]({s['url']})" if s.get("url") else "")
                st.text(s["text"])
    for warning in meta.get("warnings") or []:
        container.warning(warning)
    if show_debug and meta.get("plan"):
        with container.expander(ui["plan"]):
            st.json(meta["plan"])
            st.caption(f"ms: {meta.get('latencies')}")


def render_message(message: dict, ui: dict, show_debug: bool) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        meta = message.get("meta")
        if meta:
            render_tables(st, meta.get("tables") or [])
            render_extras(st, meta, ui, show_debug)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


def run_turn(engine: Engine, question: str, history: list[dict], ui: dict, show_debug: bool) -> dict:
    """Stream one answer into the page; returns the message to store."""
    with st.chat_message("assistant"):
        plan_box = st.empty()
        text_box = st.empty()
        tables_box = st.container()
        extras_box = st.container()
        text = ""
        final: Answer | None = None
        try:
            for kind, payload in engine.ask_stream(question, history):
                if kind == "plan":
                    plan_box.caption(f"{ui['understood']}: {describe_plan(payload, ui)}")
                elif kind == "tables":
                    render_tables(tables_box, [asdict(t) for t in payload])
                elif kind == "token":
                    text += payload
                    text_box.markdown(text + " ▌")
                elif kind == "final":
                    final = payload
        except BackendUnavailable as exc:
            st.error(f"{ui['backend_error']}: {exc}")
            return {"role": "assistant", "content": f"⚠️ {exc}"}
        except ChatbotError as exc:
            st.error(str(exc))
            return {"role": "assistant", "content": f"⚠️ {exc}"}
        except Exception as exc:  # never leave the user with a stack trace
            log.exception("unexpected error while answering")
            st.error(f"{type(exc).__name__}: {exc}")
            return {"role": "assistant", "content": f"⚠️ {type(exc).__name__}: {exc}"}

        assert final is not None
        text_box.markdown(final.text)
        meta = answer_meta(final)
        render_extras(extras_box, meta, ui, show_debug)
        return {"role": "assistant", "content": final.text, "meta": meta}


def run_sync(engine: Engine, ui: dict, full: bool) -> None:
    """Sync from eLabFTW; the outcome is parked in session state and shown after the rerun."""
    from lab_chatbot.sync import sync_labbook
    from shared.elabftw_client import get_client

    bar = st.sidebar.progress(0.0, text=ui["sync_running"])

    def progress(done: int, total: int, message: str) -> None:
        bar.progress(min(done / total, 1.0) if total else 0.0, text=f"{ui['sync_running']} {done}/{total}")

    try:
        report = sync_labbook(
            get_client(), engine.store, engine.embedder, full=full, progress=progress
        )
    except (ChatbotError, ElabFTWError, ConfigError) as exc:
        st.session_state.sync_result = ("error", f"{ui['sync_failed']}: {exc}")
        return
    finally:
        bar.empty()
    text = report.summary()
    if report.failed:
        text += "\n\n" + "\n".join(f"✗ {i}: {e}" for i, e in report.failed[:5])
    st.session_state.sync_result = ("warning" if report.failed else "success", text)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def render_app(engine_provider: Callable[[], Engine], config: ChatbotConfig, embedded: bool = False) -> None:
    """Draw the whole page. ``engine_provider`` is injected so tests can use fakes.
    ``embedded=True`` when it is one page of the combined ``app.py`` (which owns the page config)."""
    ui = UI.get(config.default_language, UI["de"])
    if not embedded:
        st.set_page_config(page_title="Lab Chatbot", page_icon="🧪", layout="centered")
    st.session_state.setdefault("messages", [])
    engine = engine_provider()

    # ---- sidebar
    with st.sidebar:
        st.header(ui["index"])
        stats = engine.store.stats()
        st.metric(ui["experiments"], stats["experiments"])
        st.caption(
            f"{ui['chunks']}: {stats['chunks']} · {ui['last_sync']}: {stats['last_sync'] or ui['never']}\n\n"
            f"`{config.chat_model}` · `{config.embed_model}`"
        )
        has_key = bool(env_str("ELABFTW_API_KEY"))
        full = st.checkbox(ui["sync_full"], value=False, disabled=not has_key, key=FULL_SYNC_KEY)
        if st.button(ui["sync"], disabled=not has_key, width="stretch"):
            run_sync(engine, ui, full)
            st.rerun()  # refresh the index numbers; the result is shown once below
        flash = st.session_state.pop("sync_result", None)
        if flash:
            level, text = flash
            getattr(st, level)(text)
        if not has_key:
            st.caption(ui["sync_no_key"])
        show_debug = st.toggle(ui["debug"], value=False, key=DEBUG_KEY)
        if st.button(ui["clear"], width="stretch"):
            st.session_state.messages = []
            st.rerun()

    # ---- main
    st.title(ui["title"])
    st.caption(ui["caption"])
    if stats["experiments"] == 0:
        st.info(ui["empty_hint"])

    messages: list[dict] = st.session_state.messages
    for message in messages:
        render_message(message, ui, show_debug)

    examples_box = st.empty()  # emptied as soon as a question is asked (this run still draws it)
    if not messages:
        with examples_box.container():
            st.markdown(f"**{ui['examples']}**")
            for i, example in enumerate(ui["example_list"]):
                if st.button(example, key=f"example_{i}"):
                    st.session_state.pending = example
                    st.rerun()

    question = st.chat_input(ui["placeholder"]) or st.session_state.pop("pending", None)
    if question:
        examples_box.empty()
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        reply = run_turn(engine, question, history, ui, show_debug)
        messages.append({"role": "user", "content": question})
        messages.append(reply)


@st.cache_resource(show_spinner="Lade Modelle …")
def _cached_engine(db_path: str) -> Engine:
    from lab_chatbot.factory import build_engine

    return build_engine(ChatbotConfig.from_env(db_path))


def main(embedded: bool = False) -> None:
    config = ChatbotConfig.from_env()
    render_app(lambda: _cached_engine(str(config.db_path)), config, embedded=embedded)


if __name__ == "__main__":
    main()
