"""Wire the chatbot together from configuration (the only place that does)."""

from __future__ import annotations

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.embeddings import OllamaEmbedder
from lab_chatbot.engine import Engine
from lab_chatbot.llm import OllamaLLM
from lab_chatbot.store import Store
from lab_chatbot.vision import OllamaImageReader


def build_store(config: ChatbotConfig) -> Store:
    return Store(config.db_path)


def build_embedder(config: ChatbotConfig) -> OllamaEmbedder:
    return OllamaEmbedder(
        config.embed_model, host=config.ollama_host, timeout=config.ollama_timeout
    )


def build_llm(config: ChatbotConfig) -> OllamaLLM:
    return OllamaLLM(
        config.chat_model,
        host=config.ollama_host,
        num_ctx=config.num_ctx,
        timeout=config.ollama_timeout,
    )


def build_image_reader(config: ChatbotConfig) -> OllamaImageReader:
    return OllamaImageReader(
        config.vision_model, host=config.ollama_host, timeout=config.ollama_timeout
    )


def build_engine(config: ChatbotConfig | None = None) -> Engine:
    config = config or ChatbotConfig.from_env()
    return Engine(
        build_store(config),
        build_embedder(config),
        build_llm(config),
        config,
        image_reader=build_image_reader(config),
    )
