"""Optional reranking: ordering, cap/k after rerank, fallback, lazy import."""

from __future__ import annotations

import builtins
import dataclasses

import pytest

from lab_chatbot.errors import ChatbotError
from lab_chatbot.rerank import CrossEncoderReranker, build_reranker
from lab_chatbot.retrieval import hybrid_search

QUERY = "Probe Messung Laser Faser"
WIDE = dict(k=50, per_experiment=50, min_dense=0.0)


class FakeReranker:
    model = "fake"

    def __init__(self, scorer):
        self.scorer = scorer
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return [self.scorer(t) for t in texts]


class Boom:
    model = "boom"

    def score(self, query, texts):
        raise RuntimeError("model exploded")


def _base(demo_store, embedder, **kw):
    return hybrid_search(demo_store, embedder, QUERY, **{**WIDE, **kw})


def test_default_has_no_reranker(config):
    assert build_reranker(config) is None


def test_build_reranker_with_model(config):
    reranker = build_reranker(dataclasses.replace(config, rerank_model="some/model"))
    assert isinstance(reranker, CrossEncoderReranker) and reranker.model == "some/model"


def test_rerank_reorders(demo_store, embedder):
    base = _base(demo_store, embedder)
    assert len(base) >= 3
    target = base[-1].cid
    texts = {h.cid: h.text for h in base}
    rr = FakeReranker(lambda t: 1.0 if t == texts[target] else 0.0)
    hits = _base(demo_store, embedder, reranker=rr)
    assert hits[0].cid == target and hits[0].rerank == 1.0
    assert [h.cid for h in hits[1:]] == [h.cid for h in base if h.cid != target]  # ties keep fused order
    assert {h.cid for h in hits} == {h.cid for h in base}


def test_cap_and_k_applied_after_rerank(demo_store, embedder):
    base = _base(demo_store, embedder)
    # prefer the lowest-ranked chunks, then check the cap and k on the reranked order
    rank = {h.text: i for i, h in enumerate(base)}
    rr = FakeReranker(lambda t: float(rank[t]))
    hits = _base(demo_store, embedder, reranker=rr, per_experiment=1, k=2)
    assert len(hits) == 2
    assert len({h.experiment_id for h in hits}) == 2
    assert hits[0].rerank >= hits[1].rerank
    assert hits[0].cid == base[-1].cid or hits[0].experiment_id == base[-1].experiment_id


def test_failing_reranker_falls_back(demo_store, embedder, caplog):
    base = _base(demo_store, embedder)
    with caplog.at_level("WARNING"):
        hits = _base(demo_store, embedder, reranker=Boom())
    assert [h.cid for h in hits] == [h.cid for h in base]
    assert all(h.rerank is None for h in hits)
    assert "Reranking failed" in caplog.text


def test_wrong_score_count_falls_back(demo_store, embedder):
    class Short:
        model = "short"

        def score(self, query, texts):
            return [1.0]

    base = _base(demo_store, embedder)
    assert [h.cid for h in _base(demo_store, embedder, reranker=Short())] == [h.cid for h in base]


def test_missing_package_gives_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ChatbotError, match="pip install sentence-transformers"):
        CrossEncoderReranker("x/y").score("q", ["text"])


def test_engine_failure_of_package_does_not_break_search(demo_store, embedder):
    reranker = CrossEncoderReranker("x/y")
    reranker._load = lambda: (_ for _ in ()).throw(ChatbotError("missing"))  # type: ignore[method-assign]
    assert _base(demo_store, embedder, reranker=reranker)
