"""Transcribe photos of handwritten lab notes with a local vision model."""

from __future__ import annotations

import logging
import time

from lab_assistant.config import AssistantConfig
from lab_assistant.ollama_client import AIError, ChatFn, chat
from shared import images

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


def load_image(image_content: bytes):
    """Decode an upload; applies the EXIF orientation (phone photos are often rotated)."""
    try:
        return images.load_image(image_content)
    except images.ImageError as exc:
        raise AIError(str(exc)) from exc


def create_chunks(image, number_of_chunks: int = NUMBER_OF_CHUNKS, overlap: float = OVERLAP) -> list[dict]:
    """Full-width horizontal strips that overlap, so no line of text is cut in half."""
    return images.horizontal_strips(image, number_of_chunks, overlap)


def _jpeg_bytes(image) -> bytes:
    # Scaled to at most 1600 px: full phone resolution only costs memory and time.
    return images.jpeg_bytes(image, quality=90)


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
