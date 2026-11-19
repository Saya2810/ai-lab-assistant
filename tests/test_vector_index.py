"""The in-memory vector index follows the database incrementally (lab_chatbot/vectors.py)."""

from __future__ import annotations

import numpy as np

from lab_chatbot.chunking import chunk_experiment
from lab_chatbot.demo import DEMO_EXPERIMENTS
from lab_chatbot.parse import parse_experiment
from lab_chatbot.retrieval import hybrid_search
from lab_chatbot.store import Store, _vector_cache
from lab_chatbot.vectors import MAX_SEGMENTS, Segment, VectorIndex


def add(store: Store, embedder, raw: dict, new_id: int, body: str | None = None) -> None:
    raw = dict(raw, id=new_id)
    if body is not None:
        raw["body"] = body
    record = parse_experiment(raw, {}, {})
    chunks = chunk_experiment(record)
    vectors = embedder.embed_documents([c.text for c in chunks])
    store.replace_experiment(record, chunks, vectors, embed_model=embedder.model, index_sig="t")


def fresh(store: Store, model: str) -> VectorIndex:
    """What a full reload from the database would give."""
    _vector_cache.clear()
    return store.vector_data(model)


def same_rows(a: VectorIndex, b: VectorIndex) -> bool:
    return np.array_equal(a.cids, b.cids) and np.allclose(a.matrix, b.matrix)


def test_changes_are_applied_as_deltas_not_full_reloads(demo_store, embedder, monkeypatch):
    first = demo_store.vector_data(embedder.model)
    loads = []
    original = Store._load_all_vectors
    monkeypatch.setattr(
        Store, "_load_all_vectors", lambda self, *a: loads.append(1) or original(self, *a)
    )
    add(demo_store, embedder, DEMO_EXPERIMENTS[0], 9001)
    demo_store.delete_experiments([int(DEMO_EXPERIMENTS[1]["id"])])
    add(demo_store, embedder, DEMO_EXPERIMENTS[2], int(DEMO_EXPERIMENTS[2]["id"]), body="Neu geschrieben.")
    updated = demo_store.vector_data(embedder.model)
    assert loads == [] and updated is not first and len(updated.segments) == 2
    assert 9001 in set(updated.exp_ids.tolist())
    assert int(DEMO_EXPERIMENTS[1]["id"]) not in set(updated.exp_ids.tolist())
    assert same_rows(updated, fresh(demo_store, embedder.model))


def test_many_small_updates_keep_the_segment_count_bounded(demo_store, embedder):
    demo_store.vector_data(embedder.model)
    for n in range(MAX_SEGMENTS + 5):
        add(demo_store, embedder, DEMO_EXPERIMENTS[0], 9100 + n)
        index = demo_store.vector_data(embedder.model)
    assert len(index.segments) <= MAX_SEGMENTS
    assert same_rows(index, fresh(demo_store, embedder.model))


def test_mass_deletion_compacts_in_memory(demo_store, embedder):
    demo_store.vector_data(embedder.model)
    ids = [int(e["id"]) for e in DEMO_EXPERIMENTS[: len(DEMO_EXPERIMENTS) // 2 + 1]]
    demo_store.delete_experiments(ids)
    index = demo_store.vector_data(embedder.model)
    assert len(index.segments) == 1 and index.segments[0].alive is None  # dead rows dropped
    assert same_rows(index, fresh(demo_store, embedder.model))


def test_reset_and_pruned_tombstones_force_a_full_reload(demo_store, embedder):
    before = demo_store.vector_data(embedder.model)
    demo_store.delete_experiments([int(DEMO_EXPERIMENTS[0]["id"])])
    assert demo_store.prune_tombstones(keep=0) > 0
    after = demo_store.vector_data(embedder.model)
    assert after.tomb_seq >= before.tomb_seq and same_rows(after, fresh(demo_store, embedder.model))
    demo_store.reset()
    assert len(demo_store.vector_data(embedder.model)) == 0


def test_a_running_refresh_does_not_block_readers(demo_store, embedder):
    from lab_chatbot import store as store_module

    old = demo_store.vector_data(embedder.model)
    add(demo_store, embedder, DEMO_EXPERIMENTS[0], 9200)
    with store_module._load_lock:  # another thread is refreshing right now
        assert demo_store.vector_data(embedder.model) is old
    assert demo_store.vector_data(embedder.model) is not old


def test_reduced_dimensions_rerank_with_full_vectors(tmp_path, embedder):
    from lab_chatbot.demo import seed_demo

    full = Store(tmp_path / "full.db")
    seed_demo(full, embedder)
    small = Store(tmp_path / "full.db", vector_dims=16)
    index = small.vector_data(embedder.model)
    assert index.reduced and index.matrix.shape[1] == 16
    norms = np.linalg.norm(index.matrix, axis=1)  # the fake embedder can leave a cut row all zero
    assert np.all(np.isclose(norms, 1.0, atol=1e-5) | (norms == 0))
    for query in ("Rauschen Probe A3", "Raman Spektrum", "Laser Wellenlänge"):
        a = hybrid_search(full, embedder, query)
        b = hybrid_search(small, embedder, query)
        assert [h.cid for h in a] == [h.cid for h in b]
        assert [h.dense for h in a] == [h.dense for h in b]  # exact scores after re-scoring


def test_segment_search_matches_brute_force():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(1000, 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    cids = np.arange(1, 1001, dtype=np.int64)
    exp_ids = cids // 10
    halves = (Segment(cids[:600], exp_ids[:600], matrix[:600]), Segment(cids[600:], exp_ids[600:], matrix[600:]))
    index = VectorIndex("m", 8, 8, halves).updated(None, np.array([5, 700]), max_cid=1000, tomb_seq=1, revision=1)
    query = matrix[4]
    got, _ = index.search(query, 10)
    alive = np.ones(1000, bool)
    alive[[4, 699]] = False
    expected = cids[alive][np.argsort(-(matrix[alive] @ query))[:10]]
    assert got.tolist() == expected.tolist() and 5 not in got
    wanted = np.array([3, 70])
    got, _ = index.search(query, 50, wanted)
    assert set((got // 10).tolist()) <= {3, 70} and 700 not in got and len(got) == 19
