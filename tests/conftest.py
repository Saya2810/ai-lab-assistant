from __future__ import annotations

import pytest

from lab_chatbot.config import ChatbotConfig
from lab_chatbot.demo import seed_demo
from lab_chatbot.engine import Engine
from lab_chatbot.store import Store
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """No test may read the developer's real .env or talk to a real server."""
    for key in list(__import__("os").environ):
        if key.startswith(("ELABFTW_", "LAB_", "OLLAMA_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_SKIP_DOTENV", "1")
    monkeypatch.chdir(tmp_path)  # relative paths a test might write must never land in the repo


@pytest.fixture
def config(tmp_path) -> ChatbotConfig:
    return ChatbotConfig.from_env(tmp_path / "test.db")


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "store.db")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def demo_store(store, embedder) -> Store:
    seed_demo(store, embedder)
    return store


@pytest.fixture
def make_engine(demo_store, embedder, config):
    def _make(plan=None, answer="Laut Quelle [1] ist das so.") -> tuple[Engine, FakeLLM]:
        llm = FakeLLM(plan=plan, answer=answer)
        return Engine(demo_store, embedder, llm, config), llm

    return _make
