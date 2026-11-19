"""Images in chat and sync, Ollama timeouts, and behaviour on large candidate sets."""

from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest

from lab_chatbot.demo import DEMO_EXPERIMENTS
from lab_chatbot.engine import Engine
from lab_chatbot.retrieval import hybrid_search
from lab_chatbot.sql_tools import Filters, candidate_ids, list_experiments
from lab_chatbot.sync import sync_labbook
from lab_chatbot.vision import ImageReadError, OllamaImageReader
from lab_chatbot.ui import read_images, split_submission
from tests.fakes import FakeElabClient, FakeLLM, plan_json


def png(width=40, height=30) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeReader:
    model = "fake-vision"

    def __init__(self, text="Spektrum Probe A3, Peak bei 532 nm"):
        self.text = text
        self.calls = 0

    def read(self, content: bytes) -> str:
        self.calls += 1
        return self.text


# ------------------------------------------------------------ Ollama client


def test_ollama_clients_get_a_timeout_and_are_reused(monkeypatch):
    import ollama

    from shared.ollama import get_ollama_client

    made = []

    class Client:
        def __init__(self, host=None, timeout=None):
            made.append((host, timeout))

    monkeypatch.setattr(ollama, "Client", Client)
    monkeypatch.setenv("LAB_OLLAMA_TIMEOUT", "42")
    first = get_ollama_client("http://h:1")
    assert get_ollama_client("http://h:1") is first
    assert made == [("http://h:1", 42.0)]


# ------------------------------------------------------------ vision reader


def test_reader_sends_downscaled_tiles_and_strips_tall_pages(monkeypatch):
    sent = []

    class Client:
        def chat(self, **kwargs):
            sent.append(kwargs["messages"][0]["images"][0])
            return {"message": {"content": "Text"}}

    reader = OllamaImageReader("vis")
    reader._client = Client()
    assert reader.read(png(4000, 3000)) == "Text"
    from PIL import Image

    assert max(Image.open(io.BytesIO(sent[0])).size) <= 1600
    sent.clear()
    text = reader.read(png(300, 1200))  # a notebook page: read in 3 overlapping strips
    assert len(sent) == 3 and "[Teil 3/3]" in text


def test_reader_rejects_non_images():
    with pytest.raises(ImageReadError):
        OllamaImageReader("vis").read(b"not an image")


# ------------------------------------------------------------ chat with images


def test_question_with_image_cites_image_and_notebook(demo_store, embedder, config):
    llm = FakeLLM(plan=plan_json("search", "Rauschen Probe A3"), answer="Das Bild zeigt A3 [1], vgl. [2].")
    reader = FakeReader()
    engine = Engine(demo_store, embedder, llm, config, image_reader=reader)
    evts = list(engine.ask_stream("Zu welchem Experiment gehört das?", [], [png()]))
    kinds = [k for k, _ in evts]
    assert kinds[0] == "images" and "sources" in kinds and kinds[-1] == "final"
    answer = evts[-1][1]
    assert answer.sources[0].experiment_id == 0 and "532 nm" in answer.sources[0].text
    assert any(s.experiment_id for s in answer.sources[1:])  # notebook entries too
    prompt = llm.stream_calls[0][-1]["content"]
    assert 'experiment="uploaded image"' in prompt
    assert answer.cited == [1, 2] and reader.calls == 1


def test_image_without_text_uses_default_question_and_works_on_empty_index(store, embedder, config):
    llm = FakeLLM(answer="Ein Spektrum [1].")
    engine = Engine(store, embedder, llm, config, image_reader=FakeReader())
    answer = engine.ask("", images=[png()])
    assert answer.used_llm and answer.question.startswith("Was zeigt")
    assert [s.experiment_id for s in answer.sources] == [0]


def test_unreadable_image_does_not_call_the_llm(demo_store, embedder, config):
    llm = FakeLLM(plan=plan_json())
    engine = Engine(demo_store, embedder, llm, config, image_reader=FakeReader(text="  "))
    answer = engine.ask("Was steht da?", images=[png()])
    assert not answer.used_llm and llm.stream_calls == []


def test_ui_helpers_split_submission_and_enforce_size_limit():
    upload = SimpleNamespace(name="a.png", size=10, getvalue=lambda: b"x" * 10)
    big = SimpleNamespace(name="big.png", size=5 * 1024 * 1024, getvalue=lambda: b"")
    assert split_submission("  frage ") == ("frage", [])
    assert split_submission(SimpleNamespace(text="t", files=[upload])) == ("t", [upload])
    assert split_submission({"text": None, "files": []}) == ("", [])
    assert read_images([upload, big], max_mb=1) == ([b"x" * 10], ["big.png"])


# ------------------------------------------------------------ sync with images


class ImageClient(FakeElabClient):
    def __init__(self, experiments):
        super().__init__(experiments)
        self.downloads = 0

    def request(self, method, endpoint, params=None, **kwargs):
        assert params == {"format": "binary"}
        self.downloads += 1
        return SimpleNamespace(content=png())


def test_sync_indexes_image_attachments_once(store, embedder):
    experiments = copy.deepcopy(DEMO_EXPERIMENTS)
    experiments[0]["uploads"] = [
        {"id": 7, "real_name": "gel.jpg", "hash": "abc", "filesize": 1000},
        {"id": 8, "real_name": "data.csv", "hash": "def", "filesize": 10},
        {"id": 9, "real_name": "huge.png", "hash": "ghi", "filesize": 10**9},
    ]
    client = ImageClient(experiments)
    reader = FakeReader("Gelbild mit Bande bei 40 kDa")
    report = sync_labbook(client, store, embedder, image_reader=reader)
    assert (report.images_read, client.downloads) == (1, 1)  # csv and oversized file skipped
    hits = hybrid_search(store, embedder, "Gelbild Bande kDa")
    assert hits and hits[0].experiment_id == int(experiments[0]["id"])

    # full re-sync: the reading comes from the cache, the vision model is not asked again
    report = sync_labbook(client, store, embedder, image_reader=reader, full=True)
    assert report.images_cached == 1 and reader.calls == 1 and client.downloads == 1


def test_broken_image_is_skipped_not_fatal(store, embedder):
    experiments = copy.deepcopy(DEMO_EXPERIMENTS)
    experiments[0]["uploads"] = [{"id": 7, "real_name": "x.png", "hash": "h"}]

    class Bad(FakeReader):
        def read(self, content):
            raise ImageReadError("corrupt")

    report = sync_labbook(ImageClient(experiments), store, embedder, image_reader=Bad())
    assert report.failed == [] and len(report.image_errors) == 1
    assert store.experiment_count() == len(experiments)


# ------------------------------------------------------------ large data


def test_huge_id_lists_do_not_hit_the_sqlite_parameter_limit(demo_store, embedder):
    ids = list(range(1, 60_000))  # far above SQLITE_MAX_VARIABLE_NUMBER
    result = list_experiments(demo_store, Filters(experiment_ids=ids), limit=5)
    assert result.total == demo_store.experiment_count()
    assert candidate_ids(demo_store, Filters(experiment_ids=ids))
    assert demo_store.chunks_of(ids)


def test_dense_search_ranks_the_same_with_and_without_candidates(demo_store, embedder):
    everything = set(demo_store.experiment_state())
    a = [h.cid for h in hybrid_search(demo_store, embedder, "Rauschen Probe A3")]
    b = [h.cid for h in hybrid_search(demo_store, embedder, "Rauschen Probe A3", experiment_ids=everything)]
    assert a and a == b


def test_vector_matrix_is_built_in_one_pass(demo_store, embedder):
    data = demo_store.vector_data(embedder.model)
    assert len(data.segments) == 1
    assert data.matrix.dtype == np.float32 and data.matrix.flags["C_CONTIGUOUS"]
    assert list(data.cids) == sorted(data.cids) and len(data.cids) == len(data.exp_ids)
