"""Exception types of the chatbot."""

from __future__ import annotations


class ChatbotError(RuntimeError):
    """Base class for expected, user-presentable failures."""


class BackendUnavailable(ChatbotError):
    """Ollama (LLM or embedding backend) cannot be reached or lacks a model."""


class SyncError(ChatbotError):
    """The eLabFTW -> index synchronisation could not run safely."""


class BackendBusy(ChatbotError):
    """The language model is busy and the request waited too long in the queue."""


class SyncBusy(SyncError):
    """Another sync (UI, CLI or the background worker, maybe in another process) is running."""
