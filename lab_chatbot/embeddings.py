"""Embedding backend (Ollama) behind a tiny interface.

Documents and queries are embedded through *separate* methods. Several model
families expect different prefixes for the two (the old code embedded both the
same way, which quietly lowers retrieval quality). Vectors are L2-normalised
so cosine similarity is a plain dot product.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from lab_chatbot.errors import BackendUnavailable

# (model-name substring, query prefix, document prefix). Check the model card of
# the model you use; set LAB_EMBED_* prefixes via the constructor if you differ.
PREFIXES: list[tuple[str, str, str]] = [
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
    ("nomic-embed", "search_query: ", "search_document: "),
    ("mxbai-embed", "Represent this sentence for searching relevant passages: ", ""),
]


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def translate_ollama_error(exc: Exception, model: str, host: str) -> Exception | None:
    """Map low-level Ollama/httpx errors to a message a user can act on."""
    status = getattr(exc, "status_code", None)
    if status == 404:
        return BackendUnavailable(
            f"Ollama has no model '{model}'. Install it with:  ollama pull {model}"
        )
    name = type(exc).__name__
    if isinstance(exc, (ConnectionError, OSError)) or name in (
        "ConnectError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError",
    ):  # fmt: skip
        return BackendUnavailable(
            f"Ollama is not reachable at {host}. Start it with `ollama serve` "
            f"(or set OLLAMA_HOST) and make sure the model '{model}' is pulled."
        )
    return None


def normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "embeddinggemma",
        host: str | None = None,
        batch_size: int = 32,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        timeout: float | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.host = host or "http://localhost:11434"
        self.batch_size = batch_size
        default_query, default_doc = next(
            ((q, d) for key, q, d in PREFIXES if key in model.lower()), ("", "")
        )
        self.query_prefix = default_query if query_prefix is None else query_prefix
        self.document_prefix = default_doc if document_prefix is None else document_prefix
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from shared.ollama import get_ollama_client

            self._client = get_ollama_client(self.host, self.timeout)
        return self._client

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            response = self.client.embed(model=self.model, input=texts)
        except Exception as exc:
            translated = translate_ollama_error(exc, self.model, self.host)
            if translated:
                raise translated from exc
            raise
        return normalise(np.array(response["embeddings"], dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, 0), dtype=np.float32)
        parts = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + t for t in texts[start : start + self.batch_size]]
            parts.append(self._embed(batch))
        return np.vstack(parts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]
