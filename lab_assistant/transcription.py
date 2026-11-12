"""Offline speech-to-text with a locally stored faster-whisper model."""

from __future__ import annotations

from lab_assistant.config import AssistantConfig
from lab_assistant.ollama_client import AIError

_model = None
_model_dir = None

DOWNLOAD_HINT = (
    "Download it once (needs internet), then everything runs offline:\n"
    "  python -c \"from faster_whisper import download_model; "
    "download_model('base', output_dir='{path}')\""
)


def get_whisper_model(config: AssistantConfig):
    """Load (once) the local model; never downloads anything during transcription."""
    global _model, _model_dir
    if _model is not None and _model_dir == config.whisper_dir:
        return _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AIError("Audio transcription needs faster-whisper:  pip install faster-whisper") from exc
    if not config.whisper_dir.exists():
        raise AIError(
            f"Local Whisper model not found: {config.whisper_dir}\n"
            + DOWNLOAD_HINT.format(path=config.whisper_dir)
        )
    _model = WhisperModel(str(config.whisper_dir), device="cpu", compute_type="int8", local_files_only=True)
    _model_dir = config.whisper_dir
    return _model


def transcribe_audio(audio_path: str, config: AssistantConfig | None = None, language: str | None = None) -> str:
    config = config or AssistantConfig.from_env()
    model = get_whisper_model(config)
    segments, _ = model.transcribe(
        audio_path, language=language or config.whisper_language, beam_size=1, vad_filter=True
    )
    return " ".join(part for part in (s.text.strip() for s in segments) if part)
