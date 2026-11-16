"""Decide what to do with an uploaded file and run it through the right model."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from lab_assistant.config import AssistantConfig

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".opus"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def get_file_type(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    return "file"


def can_use_for_ai(file_name: str) -> bool:
    return get_file_type(file_name) in {"audio", "image"}


def source_key(file_name: str, content: bytes) -> str:
    """Stable identity of an upload: two files with the same name stay two files, and a
    re-uploaded file with new content is not mistaken for the old one."""
    return f"{hashlib.sha1(content).hexdigest()[:12]}-{file_name}"


def process_file_for_ai(file_name: str, file_content: bytes, config: AssistantConfig | None = None) -> str | None:
    """Return the text extracted from an image (OCR) or audio file (transcript)."""
    config = config or AssistantConfig.from_env()
    kind = get_file_type(file_name)
    if kind == "image":
        from lab_assistant.vision import analyze_lab_image

        return analyze_lab_image(file_content, config)
    if kind == "audio":
        from lab_assistant.transcription import transcribe_audio

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as handle:
                handle.write(file_content)
                temp_path = handle.name
            return transcribe_audio(temp_path, config)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    return None
