"""Exception types of the chatbot."""

from __future__ import annotations


class ChatbotError(RuntimeError):
    """Base class for expected, user-presentable failures."""


class BackendUnavailable(ChatbotError):
    """Ollama (LLM or embedding backend) cannot be reached or lacks a model."""


class SyncError(ChatbotError):
    """The eLabFTW -> index synchronisation could not run safely."""
