"""Optional cross-encoder reranking of retrieval candidates.

The reranker needs ``sentence-transformers`` (and thus torch), which is NOT part of the default
install: ``pip install sentence-transformers`` (or the ``rerank`` extra). It is only imported
when ``LAB_RERANK_MODEL`` is set and the first query is scored.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, Sequence

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.errors import ChatbotError

MAX_TEXT_CHARS = 2000  # ~512 tokens; longer chunk text only costs time, the model truncates anyway


class Reranker(Protocol):
    model: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text (higher is better)."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16):
        self.model = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._encoder: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._encoder is None:
            with self._lock:
                if self._encoder is None:
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as exc:
                        raise ChatbotError(
                            f"Reranking with {self.model!r} needs the optional package "
                            "sentence-transformers: pip install sentence-transformers"
                        ) from exc
                    self._encoder = CrossEncoder(self.model, max_length=self.max_length)
        return self._encoder

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        encoder = self._load()
        pairs = [(query, text[:MAX_TEXT_CHARS]) for text in texts]
        scores = encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def build_reranker(config: ChatbotConfig) -> Reranker | None:
    if not config.rerank_model:
        return None
    return CrossEncoderReranker(config.rerank_model)
