from pathlib import Path
import os
import tempfile

from services.transcription import transcribe_audio
from services.vision import analyze_lab_image


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".flac",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
}


def get_file_type(file_name: str) -> str:
    extension = Path(file_name).suffix.lower()

    if extension in AUDIO_EXTENSIONS:
        return "audio"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    return "file"


def can_use_for_ai(file_name: str) -> bool:
    return get_file_type(file_name) in {
        "audio",
        "image",
    }


def process_file_for_ai(
    file_name: str,
    file_content: bytes,
) -> str | None:

    file_type = get_file_type(file_name)

    if file_type == "image":
        return analyze_lab_image(
            file_content
        )

    if file_type == "audio":
        temp_path = None

        try:
            suffix = Path(
                file_name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    file_content
                )

                temp_path = (
                    temp_file.name
                )

            return transcribe_audio(
                temp_path
            )

        finally:
            if (
                temp_path
                and os.path.exists(
                    temp_path
                )
            ):
                os.remove(
                    temp_path
                )

    return None