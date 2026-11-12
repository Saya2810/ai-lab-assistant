"""``python -m lab_chatbot doctor``: tell the user exactly what is missing."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.store import Store
from shared.config import ConfigError, ElabFTWConfig, env_str


@dataclass
class Check:
    name: str
    ok: bool | None  # None = warning / not applicable
    detail: str
    critical: bool = True

    @property
    def symbol(self) -> str:
        return "✓" if self.ok else ("!" if self.ok is None else "✗")


def _model_present(wanted: str, installed: list[str]) -> bool:
    for name in installed:
        if name == wanted or name == f"{wanted}:latest":
            return True
        if ":" not in wanted and name.split(":")[0] == wanted:
            return True
    return False


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
    try:
        import ollama

        listing = ollama.Client(host=config.ollama_host).list()
        installed = [
            (m.get("model") or m.get("name") or "") for m in listing["models"]
        ]
    except Exception as exc:
        return [
            Check(
                "Ollama",
                False,
                f"not reachable at {config.ollama_host} ({type(exc).__name__}). "
                "Start it with `ollama serve`.",
            )
        ]
    checks = [Check("Ollama", True, f"reachable at {config.ollama_host}")]
    for label, model in (("Chat model", config.chat_model), ("Embedding model", config.embed_model)):
        present = _model_present(model, installed)
        checks.append(
            Check(label, present, f"{model} " + ("installed" if present else f"missing: ollama pull {model}"))
        )
    return checks


def check_elabftw() -> Check:
    if not env_str("ELABFTW_API_KEY"):
        return Check(
            "eLabFTW", None, "ELABFTW_API_KEY not set (only needed for `sync`)", critical=False
        )
    try:
        from shared.elabftw_client import ElabFTWClient

        config = ElabFTWConfig.from_env()
        ElabFTWClient(config).get("experiments", {"limit": 1})
        return Check("eLabFTW", True, f"API reachable at {config.api_url}", critical=False)
    except (ConfigError, Exception) as exc:  # noqa: BLE001 - report anything
        return Check("eLabFTW", False, str(exc)[:300], critical=False)


def run_doctor(config: ChatbotConfig) -> list[Check]:
    return [check_fts5(), *check_index(config), *check_ollama(config), check_elabftw()]
