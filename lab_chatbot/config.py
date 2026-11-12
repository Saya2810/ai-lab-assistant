"""Chatbot settings, all overridable through environment variables / .env."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.config import (
    data_dir,
    elab_site_url,
    env_float,
    env_int,
    env_str,
)


@dataclass(frozen=True)
class ChatbotConfig:
    db_path: Path
    ollama_host: str
    chat_model: str
    embed_model: str
    num_ctx: int
    top_k: int
    min_dense: float
    max_context_chars: int
    history_turns: int
    default_language: str
    site_url: str

    @classmethod
    def from_env(cls, db_path: str | Path | None = None) -> "ChatbotConfig":
        return cls(
            db_path=Path(
                db_path or env_str("LAB_DB_PATH") or data_dir() / "chatbot" / "labbook.db"
            ),
            ollama_host=env_str("OLLAMA_HOST", "http://localhost:11434"),
            chat_model=env_str("LAB_CHAT_MODEL", "qwen2.5:7b"),
            embed_model=env_str("LAB_EMBED_MODEL", "embeddinggemma"),
            num_ctx=env_int("LAB_NUM_CTX", 8192),
            top_k=env_int("LAB_TOP_K", 6),
            min_dense=env_float("LAB_MIN_DENSE", 0.30),
            max_context_chars=env_int("LAB_MAX_CONTEXT_CHARS", 9000),
            history_turns=env_int("LAB_HISTORY_TURNS", 3),
            default_language=env_str("LAB_DEFAULT_LANGUAGE", "de"),
            site_url=elab_site_url(),
        )
