"""Fair admission queue in front of the chat LLM.

Ollama on one GPU effectively serves one generation at a time. Without an
explicit queue, concurrent requests pile up invisibly inside Ollama and can run
into the HTTP timeout. The gate admits tickets strictly in arrival order, up to
``limit`` at a time, and reports the waiting position to an optional listener.
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Iterator

from lab_chatbot.errors import BackendBusy

# Listener signature: (position, active). position is the 1-based place in the
# queue; 0 means "admitted, stop showing the waiting state".
WaitListener = Callable[[int, int], None]

wait_listener: contextvars.ContextVar[WaitListener | None] = contextvars.ContextVar(
    "llm_wait_listener", default=None
)

BUSY_MESSAGE = (
    "Too many requests are waiting for the language model, please try again in a moment. / "
    "Zu viele Anfragen warten auf das Sprachmodell, bitte gleich nochmal versuchen."
)


@contextmanager
def listening(listener: WaitListener | None) -> Iterator[None]:
    """Set the wait listener for the current context."""
    token = wait_listener.set(listener)
    try:
        yield
    finally:
        wait_listener.reset(token)


class LLMGate:
    def __init__(self, limit: int = 1):
        self.limit = max(1, int(limit))
        self._cond = threading.Condition()
        self._queue: deque[object] = deque()
        self._active = 0

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    @property
    def waiting(self) -> int:
        with self._cond:
            return len(self._queue)

    def _notify_listener(self, listener: WaitListener | None, position: int) -> None:
        if listener is None:
            return
        try:
            listener(position, self._active)
        except Exception:  # a broken UI callback must never block the queue
            pass

    @contextmanager
    def slot(self, timeout: float | None = None) -> Iterator[None]:
        """Block until admitted (FIFO); raise BackendBusy after ``timeout`` seconds."""
        listener = wait_listener.get()
        ticket = object()
        deadline = None if timeout is None else time.monotonic() + timeout
        last_position = None
        with self._cond:
            self._queue.append(ticket)
            try:
                while True:
                    if self._queue[0] is ticket and self._active < self.limit:
                        self._queue.popleft()
                        self._active += 1
                        break
                    position = list(self._queue).index(ticket) + 1
                    if position != last_position:
                        last_position = position
                        self._notify_listener(listener, position)
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise BackendBusy(BUSY_MESSAGE)
                    self._cond.wait(remaining)
            except BaseException:
                if ticket in self._queue:
                    self._queue.remove(ticket)
                self._cond.notify_all()
                raise
            # Positions of the others shift when we leave the queue.
            self._cond.notify_all()
            if last_position is not None:
                self._notify_listener(listener, 0)
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()


@lru_cache(maxsize=None)
def get_gate(host: str, limit: int) -> LLMGate:
    """Process-wide gate per Ollama host and limit."""
    return LLMGate(limit)
