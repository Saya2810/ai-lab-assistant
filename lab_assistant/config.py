"""Lab Assistant settings (environment variables / .env; nothing is read at import time)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.config import REPO_ROOT, env_int, env_str


@dataclass(frozen=True)
class AssistantConfig:
    ollama_host: str
    text_model: str
    vision_model: str
    num_ctx: int
    whisper_dir: Path
    whisper_language: str | None

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        return cls(
            ollama_host=env_str("OLLAMA_HOST", "http://localhost:11434"),
            text_model=env_str("LAB_ASSISTANT_MODEL", "qwen2.5:7b"),
            vision_model=env_str("LAB_VISION_MODEL", "minicpm-v4.6:1b"),
            num_ctx=env_int("LAB_ASSISTANT_NUM_CTX", 8192),
            whisper_dir=Path(env_str("LAB_WHISPER_DIR") or REPO_ROOT / "models" / "whisper" / "base"),
            whisper_language=env_str("LAB_WHISPER_LANGUAGE"),
        )
