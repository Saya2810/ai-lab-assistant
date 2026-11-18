"""Read images with a local vision model (Ollama).

Used in two places:

* in the chat, when someone attaches a photo or screenshot to a question
  (a plot, a gel, a spectrum, a page of handwritten notes);
* during sync, for image attachments of eLabFTW experiments when
  ``LAB_INDEX_IMAGES=true``. The text then becomes searchable like the body.

Tall images (phone photos of a notebook page) are read in overlapping strips;
small vision models lose detail when a whole page is shrunk to one tile.
"""

from __future__ import annotations

import logging
from typing import Protocol

from lab_chatbot.embeddings import translate_ollama_error
from lab_chatbot.errors import ChatbotError
from shared import images

log = logging.getLogger(__name__)

READ_PROMPT = """Describe this image from a laboratory notebook so it can be searched and quoted later.

1. Transcribe all visible text, labels, axis titles, legends and table cells exactly.
2. Keep every number and unit exactly as written. Write [unclear] for anything you cannot read.
3. For a chart or plot: name the axes with units, the visible curves/series and notable
   peaks, values or trends you can actually read off.
4. For a photo of a sample, gel, setup or device: describe what is shown in one or two sentences.
5. Do not guess, interpret or add anything that is not visible.

Answer in plain text without preamble."""


class ImageReader(Protocol):
    model: str

    def read(self, content: bytes) -> str: ...


class ImageReadError(ChatbotError):
    """An image could not be read (bad file or vision model problem)."""


class OllamaImageReader:
    def __init__(
        self,
        model: str = "minicpm-v4.6:1b",
        host: str | None = None,
        timeout: float | None = None,
        max_side: int = images.MAX_SIDE,
        num_ctx: int = 4096,
    ):
        self.model = model
        self.host = host or "http://localhost:11434"
        self.timeout = timeout
        self.max_side = max_side
        self.num_ctx = num_ctx
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from shared.ollama import get_ollama_client

            self._client = get_ollama_client(self.host, self.timeout)
        return self._client

    def _tiles(self, content: bytes) -> list[bytes]:
        try:
            image = images.load_image(content)
        except images.ImageError as exc:
            raise ImageReadError(str(exc)) from exc
        width, height = image.size
        count = 3 if height > 1.6 * width else 1
        if count == 1:
            return [images.jpeg_bytes(image, max_side=self.max_side)]
        return [
            images.jpeg_bytes(strip["image"], max_side=self.max_side)
            for strip in images.horizontal_strips(image, count)
        ]

    def read(self, content: bytes) -> str:
        tiles = self._tiles(content)
        parts = []
        for position, tile in enumerate(tiles):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": READ_PROMPT, "images": [tile]}],
                    options={"temperature": 0.0, "num_ctx": self.num_ctx},
                    keep_alive="5m" if position < len(tiles) - 1 else "2m",
                )
            except Exception as exc:
                translated = translate_ollama_error(exc, self.model, self.host)
                if translated:
                    raise translated from exc
                raise ImageReadError(f"Vision model failed: {exc}") from exc
            text = (response["message"]["content"] or "").strip()
            if text:
                parts.append(text if len(tiles) == 1 else f"[Teil {position + 1}/{len(tiles)}]\n{text}")
        return "\n\n".join(parts)
