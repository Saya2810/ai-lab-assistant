"""``python -m lab_chatbot doctor``: tell the user exactly what is missing."""

from __future__ import annotations

import sqlite3

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.store import Store
from shared.config import ConfigError, ElabFTWConfig, env_str
from shared.diagnostics import Check, model_present, ollama_checks

_model_present = model_present  # (kept: tests and callers use this name)


def check_fts5() -> Check:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
        return Check("SQLite FTS5", True, f"available (SQLite {sqlite3.sqlite_version})")
    except sqlite3.OperationalError:
        return Check("SQLite FTS5", False, "this Python's SQLite lacks FTS5; keyword search needs it")


def check_index(config: ChatbotConfig) -> list[Check]:
    checks = []
    try:
        stats = Store(config.db_path).stats()
    except Exception as exc:
        return [Check("Index", False, f"cannot open {config.db_path}: {exc}")]
    detail = (
        f"{stats['experiments']} experiments, {stats['chunks']} chunks, "
        f"{stats['measurements']} measurements; last sync: {stats['last_sync'] or 'never'}"
    )
    checks.append(Check("Index", stats["experiments"] > 0, detail + f"  [{config.db_path}]"))
    models = stats["embed_models_in_index"]
    if models and config.embed_model not in models:
        checks.append(
            Check(
                "Embedding model",
                False,
                f"index was built with {models} but LAB_EMBED_MODEL={config.embed_model}. "
                "Run `sync --full` to re-embed.",
            )
        )
    return checks


def check_ollama(config: ChatbotConfig) -> list[Check]:
    checks, _ready = ollama_checks(
        config.ollama_host,
        [("Chat model", config.chat_model, True), ("Embedding model", config.embed_model, True)],
    )
    return checks


def check_elabftw(sample: int = 3) -> list[Check]:
    """Reach the API, then fetch and *parse* a few real experiments: this is where a wrong
    assumption about the field names of your eLabFTW version would show up."""
    if not env_str("ELABFTW_API_KEY"):
        return [Check("eLabFTW", None, "ELABFTW_API_KEY not set (only needed for `sync`)", critical=False)]
    try:
        from lab_chatbot.parse import parse_experiment
        from shared.elabftw_client import ElabFTWClient

        config = ElabFTWConfig.from_env()
        client = ElabFTWClient(config)
        overview = list(client.iter_paginated("experiments", max_items=sample))
        if not overview:
            return [Check("eLabFTW", None, f"reachable at {config.api_url}, but the experiment list is empty "
                          "(wrong team, or an API key of a user who sees nothing?)", critical=False)]
        checks = [Check("eLabFTW", True, f"reachable at {config.api_url}; sampled {len(overview)} experiment(s)", critical=False)]
        records = [parse_experiment(client.get(f"experiments/{item['id']}")) for item in overview]
    except (ConfigError, Exception) as exc:  # noqa: BLE001 - report anything
        return [Check("eLabFTW", False, str(exc)[:300], critical=False)]

    problems = []
    if any(not r.title for r in records):
        problems.append("an experiment without a title")
    if not any(r.date for r in records):
        problems.append("no experiment has a date")
    if not any(r.body.strip() for r in records):
        problems.append("no experiment has a body: wrong field name?")
    if not any(r.modified_at for r in records):
        problems.append("no modified_at: every sync re-reads everything")
    detail = (
        f"parsed {len(records)}: tags {sum(len(r.tags) for r in records)}, "
        f"measurements {sum(len(r.measurements) for r in records)}, "
        f"body {sum(len(r.body) for r in records)} chars, e.g. #{records[0].id} \"{records[0].title[:40]}\""
    )
    checks.append(Check("eLabFTW data", None if problems else True, detail + (" | " + "; ".join(problems) if problems else ""), critical=False))
    return checks


def check_live(config: ChatbotConfig) -> list[Check]:
    """Really call the two models once. Installed is not the same as working
    (out of memory, wrong version, a model that is not an embedding model)."""
    from lab_chatbot.factory import build_embedder, build_llm

    checks = []
    try:
        vector = build_embedder(config).embed_query("Raman-Spektrum bei 532 nm")
        checks.append(Check("Embedding call", True, f"{config.embed_model} returned a {len(vector)}-dimensional vector"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Embedding call", False, str(exc)[:300]))
    try:
        reply = build_llm(config).complete([{"role": "user", "content": "Antworte nur mit dem Wort: ok"}])
        checks.append(Check("Chat call", bool(reply.strip()), f"{config.chat_model} answered {reply.strip()[:40]!r}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Chat call", False, str(exc)[:300]))
    return checks


def run_doctor(config: ChatbotConfig, live: bool = False) -> list[Check]:
    """``live`` also calls the models once (slow on a cold start: the model is loaded into memory)."""
    checks = [check_fts5(), *check_index(config), *check_ollama(config), *check_elabftw()]
    ollama_ready = all(c.ok for c in checks if c.name in ("Ollama", "Chat model", "Embedding model"))
    if live and ollama_ready:
        checks += check_live(config)
    return checks
