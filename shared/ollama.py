"""One place that creates Ollama clients.

Every client gets an explicit timeout (``LAB_OLLAMA_TIMEOUT``, seconds). Without
it a hung Ollama server blocks a sync or a chat answer forever. Clients are
cached per (host, timeout), so the HTTP connection pool is reused instead of
being rebuilt for every request.
"""

from __future__ import annotations

import threading

from shared.config import env_float

DEFAULT_TIMEOUT = 300.0  # generous: a cold 7B model on CPU can take minutes to load

_clients: dict[tuple, object] = {}
_lock = threading.Lock()


def ollama_timeout() -> float:
    return env_float("LAB_OLLAMA_TIMEOUT", DEFAULT_TIMEOUT)


def get_ollama_client(host: str, timeout: float | None = None):
    import ollama

    timeout = ollama_timeout() if timeout is None else timeout
    factory = ollama.Client
    key = (host, timeout, id(factory))  # id(): a test that swaps ollama.Client gets a fresh one
    with _lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = factory(host=host, timeout=timeout)
    return client
