"""Sync on large notebooks: one at a time, interruptible, resumable, streamed."""

from __future__ import annotations

import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest

from lab_chatbot.demo import DEMO_EXPERIMENTS
from lab_chatbot.errors import BackendUnavailable, SyncBusy, SyncError
from lab_chatbot.sync import FULL_SYNC_KEY, LEASE_TTL, SYNC_LEASE, sync_labbook, sync_status
from lab_chatbot.worker import run_worker
from shared.elabftw_client import ElabFTWError
from tests.fakes import FakeElabClient


@pytest.fixture
def client():
    return FakeElabClient(copy.deepcopy(DEMO_EXPERIMENTS))


def counting(embedder):
    calls = []
    original = embedder.embed_documents
    embedder.embed_documents = lambda texts: calls.append(len(texts)) or original(texts)
    return calls


def stop_after(n: int):
    seen = {"n": 0}

    def should_stop() -> bool:
        seen["n"] += 1
        return seen["n"] > n

    return should_stop


def test_only_one_sync_at_a_time_across_processes(store, embedder, client):
    assert store.acquire_lease(SYNC_LEASE, "other-host:42:1", LEASE_TTL)
    with pytest.raises(SyncBusy):
        sync_labbook(client, store, embedder)
    store.release_lease(SYNC_LEASE, "other-host:42:1")
    assert sync_labbook(client, store, embedder).new == len(DEMO_EXPERIMENTS)
    assert store.lease_holder(SYNC_LEASE) is None  # released afterwards


def test_a_crashed_holder_does_not_block_forever(store, embedder, client):
    assert store.acquire_lease(SYNC_LEASE, "dead-process", ttl=0.01)
    time.sleep(0.02)
    assert sync_labbook(client, store, embedder).new == len(DEMO_EXPERIMENTS)


def test_stopping_keeps_finished_work_and_deletes_nothing(store, embedder, client):
    sync_labbook(client, store, embedder)
    report = sync_labbook(client, store, embedder, full=True, should_stop=stop_after(3))
    assert report.stopped and report.updated == 3 and report.deleted == 0
    assert store.experiment_count() == len(DEMO_EXPERIMENTS)
    assert sync_status(store)["state"] == "stopped"


def test_interrupted_full_sync_resumes_instead_of_starting_over(store, embedder, client):
    sync_labbook(client, store, embedder)
    calls = counting(embedder)
    sync_labbook(client, store, embedder, full=True, should_stop=stop_after(3))
    assert len(calls) == 3 and store.get_meta(FULL_SYNC_KEY)
    calls.clear()
    report = sync_labbook(client, store, embedder, full=True)
    assert len(calls) == len(DEMO_EXPERIMENTS) - 3  # only what the first run did not reach
    assert report.unchanged == 3 and store.get_meta(FULL_SYNC_KEY) is None
    calls.clear()
    sync_labbook(client, store, embedder, full=True)  # a new full run starts from scratch
    assert len(calls) == len(DEMO_EXPERIMENTS)


class BrokenListing(FakeElabClient):
    def __init__(self, experiments, fail_after):
        super().__init__(experiments)
        self.fail_after = fail_after

    def iter_paginated(self, endpoint, params=None, page_size=None, max_items=None):
        for n, item in enumerate(super().iter_paginated(endpoint, params, page_size, max_items)):
            if n == self.fail_after:
                raise ElabFTWError("connection reset")
            yield item


def test_a_listing_that_breaks_off_keeps_the_index(store, embedder, client):
    sync_labbook(client, store, embedder)
    with pytest.raises(SyncError):
        sync_labbook(BrokenListing(copy.deepcopy(DEMO_EXPERIMENTS), 2), store, embedder)
    assert store.experiment_count() == len(DEMO_EXPERIMENTS)  # nothing deleted as "gone"
    status = sync_status(store)
    assert status["state"] == "failed" and "connection reset" in status["error"]


def test_listing_is_consumed_lazily(store, embedder):
    pulled = []

    class Lazy(FakeElabClient):
        def iter_paginated(self, *a, **k):
            for item in super().iter_paginated(*a, **k):
                pulled.append(item["id"])
                yield item

    def progress(done, total, message):
        assert len(pulled) <= done + 1  # never more than one entry ahead of the work

    sync_labbook(Lazy(copy.deepcopy(DEMO_EXPERIMENTS)), store, embedder, progress=progress)
    assert len(pulled) == len(DEMO_EXPERIMENTS)


def test_status_shows_progress_and_detects_a_dead_sync(store, embedder, client):
    seen = []
    sync_labbook(client, store, embedder, progress=lambda *a: seen.append(sync_status(store)["state"]))
    assert set(seen) == {"running"} and sync_status(store)["state"] == "done"
    store.set_meta("sync_status", json.dumps({"state": "running", "done": 3}))  # process was killed
    assert sync_status(store)["state"] == "interrupted"


# ------------------------------------------------------------------ worker


def test_worker_once_returns_the_sync_result(config):
    report = SimpleNamespace(failed=[], summary=lambda: "ok")
    calls = []
    code = run_worker(config, once=True, full_first=True,
                      sync_once=lambda **kw: calls.append(kw) or report)  # fmt: skip
    assert code == 0 and calls[0]["full"] is True


def test_worker_backs_off_and_survives_failures(config, monkeypatch):
    import lab_chatbot.worker as worker

    stop = threading.Event()
    waits, outcomes = [], [BackendUnavailable("ollama down"), SyncBusy("busy"), RuntimeError("bug")]

    def sync_once(**kw):
        if not outcomes:
            stop.set()
            return SimpleNamespace(failed=[], summary=lambda: "ok")
        raise outcomes.pop(0)

    monkeypatch.setattr(stop, "wait", lambda seconds: waits.append(seconds))
    assert run_worker(config, interval=100, stop=stop, sync_once=sync_once) == 0
    assert waits[0] == 100 and waits[1] == 100  # failure 1 -> base interval, busy -> normal wait
    assert waits[2] == 200  # second failure in a row doubles
    assert worker.MAX_BACKOFF >= max(waits)
