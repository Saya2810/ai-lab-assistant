"""One thin wrapper around Ollama chat, shared by the text and the vision step.

It turns the usual low-level failures (server not running, model not pulled,
timeout) into one readable ``AIError``, and always sets ``num_ctx`` explicitly:
Ollama silently cuts the *beginning* of a prompt that exceeds the context window,
which would remove the system prompt's integrity rules first.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from lab_assistant.config import AssistantConfig


class AIError(RuntimeError):
    """A local-model problem the user can act on."""


ChatFn = Callable[..., str]


def explain_error(exc: Exception, model: str, host: str) -> AIError | None:
    status = getattr(exc, "status_code", None)
    if status == 404:
        return AIError(f"Ollama has no model '{model}'. Install it with:  ollama pull {model}")
    name = type(exc).__name__
    if isinstance(exc, (ConnectionError, OSError)) or name in (
        "ConnectError", "ConnectTimeout", "ReadTimeout", "RemoteProtocolError",
    ):  # fmt: skip
        return AIError(
            f"Ollama is not reachable at {host}. Start it with `ollama serve` "
            f"(or set OLLAMA_HOST) and make sure '{model}' is pulled."
        )
    return None


def chat(
    config: AssistantConfig,
    model: str,
    messages: Sequence[dict],
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    keep_alive: Any = None,
) -> str:
    """Send one chat request and return the reply text."""
    import ollama

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "options": {"temperature": temperature, "num_ctx": config.num_ctx},
    }
    if json_mode:
        kwargs["format"] = "json"
    if keep_alive is not None:
        kwargs["keep_alive"] = keep_alive
    try:
        response = ollama.Client(host=config.ollama_host).chat(**kwargs)
    except Exception as exc:
        friendly = explain_error(exc, model, config.ollama_host)
        if friendly:
            raise friendly from exc
        raise AIError(f"Ollama request failed: {exc}") from exc
    return response["message"]["content"]
