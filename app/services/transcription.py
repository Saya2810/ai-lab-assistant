from pathlib import Path

from faster_whisper import WhisperModel


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "whisper"
    / "base"
)

_model = None


def get_whisper_model():
    """
    Load the locally stored Whisper model.

    No model download or Hugging Face connection
    is allowed during transcription.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Local Whisper model not found: "
                f"{MODEL_PATH}"
            )

        _model = WhisperModel(
            str(MODEL_PATH),
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

    return _model


def transcribe_audio(
    audio_path: str,
    language: str | None = None,
) -> str:

    model = get_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=1,
        vad_filter=True,
    )

    text_parts = []

    for segment in segments:

        text = segment.text.strip()

        if text:
            text_parts.append(text)

    return " ".join(text_parts)