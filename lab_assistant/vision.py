"""Transcribe photos of handwritten lab notes with a local vision model."""

from __future__ import annotations

import io
import logging
import time

from lab_assistant.config import AssistantConfig
from lab_assistant.ollama_client import AIError, ChatFn, chat

log = logging.getLogger(__name__)

NUMBER_OF_CHUNKS = 3
OVERLAP = 0.20

VISION_PROMPT = """
Faithfully transcribe all handwritten scientific text
visible in this image section.

STRICT RULES:

1. Transcribe only information actually visible.
2. Do not invent information.
3. Do not explain or interpret anything.
4. Do not correct calculations.
5. Preserve every number exactly.
6. Preserve physical units exactly.
7. Preserve mathematical symbols exactly.
8. Convert mathematical expressions to LaTeX.
9. Preserve Greek letters, subscripts, superscripts,
   exponents and fractions using LaTeX.
10. Pay particular attention to exponents and powers of ten.
11. If a character, number or symbol is uncertain,
    write [unclear].
12. Do not reconstruct content outside this image section.

Return only the transcription.
"""


def _register_heif() -> None:
    try:  # optional: only needed for iPhone HEIC photos
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        log.info("pillow-heif not installed; HEIC/HEIF images are not supported")


def load_image(image_content: bytes):
    """Decode an upload; applies the EXIF orientation (phone photos are often rotated)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    _register_heif()
    try:
        image = Image.open(io.BytesIO(image_content))
        image = ImageOps.exif_transpose(image)
    except UnidentifiedImageError as exc:
        raise AIError(
            "This image format cannot be read. For HEIC/HEIF photos install pillow-heif "
            "(pip install pillow-heif), or export the photo as JPEG/PNG."
        ) from exc
    return image.convert("RGB") if image.mode != "RGB" else image


def create_chunks(image, number_of_chunks: int = NUMBER_OF_CHUNKS, overlap: float = OVERLAP) -> list[dict]:
    """Full-width horizontal strips that overlap, so no line of text is cut in half."""
    width, height = image.size
    base = height / number_of_chunks
    pad = int(base * overlap)
    chunks = []
    for index in range(number_of_chunks):
        top = max(0, int(index * base) - pad)
        bottom = min(height, int((index + 1) * base) + pad)
        chunks.append(
            {"index": index + 1, "image": image.crop((0, top, width, bottom)), "top": top, "bottom": bottom}
        )
    return chunks


def _jpeg_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def transcribe_chunk(
    chunk: dict, is_last: bool, config: AssistantConfig, chat_fn: ChatFn = chat
) -> str:
    started = time.time()
    text = chat_fn(
        config,
        config.vision_model,
        [{"role": "user", "content": VISION_PROMPT, "images": [_jpeg_bytes(chunk["image"])]}],
        keep_alive=0 if is_last else "5m",  # keep loaded between chunks, free the GPU after the last
    )
    log.info("vision chunk %s (%s px) took %.1fs", chunk["index"], chunk["image"].size, time.time() - started)
    return text.strip()


def combine_transcriptions(parts: list[dict]) -> str:
    """Keep every section; overlapping text stays so conflicting readings remain visible."""
    return "\n\n".join(f"--- IMAGE SECTION {p['index']} ---\n{p['text']}" for p in parts)


def analyze_lab_image(
    image_content: bytes, config: AssistantConfig | None = None, chat_fn: ChatFn = chat
) -> str:
    """Transcribe a photo of lab notes. The result is meant to be reviewed and edited by the user."""
    config = config or AssistantConfig.from_env()
    chunks = create_chunks(load_image(image_content))
    parts = [
        {"index": c["index"], "text": transcribe_chunk(c, i == len(chunks) - 1, config, chat_fn)}
        for i, c in enumerate(chunks)
    ]
    return combine_transcriptions(parts)
