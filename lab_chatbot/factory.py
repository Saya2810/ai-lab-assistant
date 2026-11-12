"""Wire the chatbot together from configuration (the only place that does)."""

from __future__ import annotations

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.embeddings import OllamaEmbedder
from lab_chatbot.engine import Engine
from lab_chatbot.llm import OllamaLLM
from lab_chatbot.store import Store


def build_store(config: ChatbotConfig) -> Store:
    return Store(config.db_path)


def build_embedder(config: ChatbotConfig) -> OllamaEmbedder:
    return OllamaEmbedder(config.embed_model, host=config.ollama_host)


def build_llm(config: ChatbotConfig) -> OllamaLLM:
    return OllamaLLM(config.chat_model, host=config.ollama_host, num_ctx=config.num_ctx)


def build_engine(config: ChatbotConfig | None = None) -> Engine:
    config = config or ChatbotConfig.from_env()
    return Engine(build_store(config), build_embedder(config), build_llm(config), config)
