"""Chat backend (Ollama) behind a tiny interface.

``num_ctx`` is always set explicitly. Ollama's default context window is small;
when a prompt exceeds it the *beginning* is cut off, i.e. the system prompt and
the first sources silently disappear.
"""

from __future__ import annotations

from typing import Iterator, Protocol, Sequence

from lab_chatbot.embeddings import translate_ollama_error

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLM(Protocol):
    model: str

    def complete(self, messages: Sequence[Message], schema: dict | str | None = None) -> str: ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]: ...


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str | None = None,
        num_ctx: int = 8192,
        temperature: float = 0.0,
        keep_alive: str = "10m",
        timeout: float | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self.host = host or "http://localhost:11434"
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from shared.ollama import get_ollama_client

            self._client = get_ollama_client(self.host, self.timeout)
        return self._client

    def _options(self) -> dict:
        return {"temperature": self.temperature, "num_ctx": self.num_ctx}

    def _call(self, **kwargs):
        try:
            return self.client.chat(
                model=self.model, options=self._options(), keep_alive=self.keep_alive, **kwargs
            )
        except Exception as exc:
            translated = translate_ollama_error(exc, self.model, self.host)
            if translated:
                raise translated from exc
            raise

    def complete(self, messages: Sequence[Message], schema: dict | str | None = None) -> str:
        kwargs = {"messages": list(messages)}
        if schema is not None:
            kwargs["format"] = schema
        response = self._call(**kwargs)
        return response["message"]["content"]

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        for part in self._call(messages=list(messages), stream=True):
            piece = part["message"]["content"]
            if piece:
                yield piece
