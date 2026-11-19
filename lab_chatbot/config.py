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
from shared.ollama import ollama_timeout


def env_bool(name: str, default: bool) -> bool:
    value = env_str(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on", "ja")



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
    ollama_timeout: float = 300.0
    vision_model: str = "minicpm-v4.6:1b"
    index_images: bool = False
    max_image_mb: float = 20.0
    max_images_per_experiment: int = 10
    # large installations / many users
    max_concurrent_llm: int = 1
    llm_queue_timeout: float = 600.0
    query_log: bool = True
    rerank_model: str = ""
    vector_dims: int = 0
    sync_interval: int = 900

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
            min_dense=env_float("LAB_MIN_DENSE", 0.35),
            max_context_chars=env_int("LAB_MAX_CONTEXT_CHARS", 9000),
            history_turns=env_int("LAB_HISTORY_TURNS", 3),
            default_language=env_str("LAB_DEFAULT_LANGUAGE", "de"),
            site_url=elab_site_url(),
            ollama_timeout=ollama_timeout(),
            vision_model=env_str("LAB_VISION_MODEL", "minicpm-v4.6:1b"),
            index_images=env_bool("LAB_INDEX_IMAGES", False),
            max_image_mb=env_float("LAB_MAX_IMAGE_MB", 20.0),
            max_images_per_experiment=env_int("LAB_MAX_IMAGES_PER_EXPERIMENT", 10),
            max_concurrent_llm=max(1, env_int("LAB_MAX_CONCURRENT_LLM", 1)),
            llm_queue_timeout=env_float("LAB_LLM_QUEUE_TIMEOUT", 600.0),
            query_log=env_bool("LAB_QUERY_LOG", True),
            rerank_model=env_str("LAB_RERANK_MODEL", "") or "",
            vector_dims=max(0, env_int("LAB_VECTOR_DIMS", 0)),
            sync_interval=max(60, env_int("LAB_SYNC_INTERVAL", 900)),
        )
