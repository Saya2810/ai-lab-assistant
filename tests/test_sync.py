from __future__ import annotations

import copy

import pytest

from lab_chatbot.demo import DEMO_EXPERIMENTS
from lab_chatbot.errors import BackendUnavailable, SyncError
from lab_chatbot.sync import sync_labbook
from shared.elabftw_client import ElabFTWError
from tests.fakes import FakeElabClient, FakeEmbedder


@pytest.fixture
def client():
    return FakeElabClient(copy.deepcopy(DEMO_EXPERIMENTS))


def test_first_sync_indexes_everything(store, embedder, client):
    report = sync_labbook(client, store, embedder)
    assert (report.listed, report.new, report.unchanged, report.failed) == (8, 8, 0, [])
    assert store.experiment_count() == 8
    assert store.get_meta("last_sync") and store.get_meta("embed_model") == "fake-embed"


def test_second_sync_does_no_work(store, embedder, client):
    sync_labbook(client, store, embedder)
    embedded_before = embedder.document_calls
    client.calls.clear()
    report = sync_labbook(client, store, embedder)
    assert (report.new, report.updated, report.unchanged) == (0, 0, 8)
    assert embedder.document_calls == embedded_before  # nothing re-embedded
    assert not [c for c in client.calls if c[0].startswith("experiments/")]  # no detail fetches


def test_changed_timestamp_but_same_content_skips_embedding(store, embedder, client):
    sync_labbook(client, store, embedder)
    embedded = embedder.document_calls
    client.experiments[101]["modified_at"] = "2027-01-01 00:00:00"
    report = sync_labbook(client, store, embedder)
    assert report.unchanged == 8 and embedder.document_calls == embedded
    assert store.experiment_state()[101].modified_at == "2027-01-01 00:00:00"  # remembered


def test_edited_experiment_is_replaced(store, embedder, client):
    sync_labbook(client, store, embedder)
    client.experiments[104]["body"] = "## Neu\n\nVollständig überarbeiteter Text über Mykoplasmen."
    client.experiments[104]["modified_at"] = "2027-02-02 10:00:00"
    report = sync_labbook(client, store, embedder)
    assert (report.updated, report.unchanged) == (1, 7)
    assert "Mykoplasmen" in store.get_experiment(104)["body"]
    assert store.fts_search('"mykoplasmen"') and not store.fts_search('"konfluenz"')


def test_experiment_deleted_in_elabftw_is_removed(store, embedder, client):
    sync_labbook(client, store, embedder)
    del client.experiments[103]
    report = sync_labbook(client, store, embedder)
    assert report.deleted == 1 and store.get_experiment(103) is None


def test_empty_listing_never_wipes_the_index(store, embedder, client):
    sync_labbook(client, store, embedder)
    client.experiments.clear()
    with pytest.raises(SyncError, match="Refusing"):
        sync_labbook(client, store, embedder)
    assert store.experiment_count() == 8
    report = sync_labbook(client, store, embedder, force_empty=True)
    assert report.deleted == 8 and store.experiment_count() == 0


def test_listing_failure_leaves_index_untouched(store, embedder, client):
    sync_labbook(client, store, embedder)
    client.fail_listing = ElabFTWError("HTTP 502")
    with pytest.raises(SyncError, match="Could not list"):
        sync_labbook(client, store, embedder)
    assert store.experiment_count() == 8


def test_one_broken_experiment_does_not_stop_the_rest(store, embedder, client):
    client.fail_detail = {105}
    report = sync_labbook(client, store, embedder)
    assert report.new == 7 and [f[0] for f in report.failed] == [105]
    assert "FAILED" in report.summary() and store.get_experiment(105) is None


def test_failed_update_keeps_previous_version(store, embedder, client):
    sync_labbook(client, store, embedder)
    old_body = store.get_experiment(105)["body"]
    client.experiments[105]["modified_at"] = "2027-03-03"
    client.experiments[105]["body"] = "geändert"
    client.fail_detail = {105}
    report = sync_labbook(client, store, embedder)
    assert report.failed and store.get_experiment(105)["body"] == old_body


def test_embedder_outage_aborts_instead_of_failing_every_item(store, client):
    down = FakeEmbedder(fail=BackendUnavailable("ollama down"))
    with pytest.raises(BackendUnavailable):
        sync_labbook(client, store, down)
    assert store.experiment_count() == 0


def test_changing_the_embedding_model_reindexes_everything(store, client):
    sync_labbook(client, store, FakeEmbedder("model-a"))
    other = FakeEmbedder("model-b")
    report = sync_labbook(client, store, other)
    assert report.updated == 8 and other.document_calls > 0
    assert store.stats()["embed_models_in_index"] == ["model-b"]


def test_full_flag_forces_reprocessing(store, embedder, client):
    sync_labbook(client, store, embedder)
    report = sync_labbook(client, store, embedder, full=True)
    assert report.updated == 8


def test_limit_does_not_delete_the_rest(store, embedder, client):
    sync_labbook(client, store, embedder)
    report = sync_labbook(client, store, embedder, max_items=2)
    assert report.listed == 2 and report.deleted == 0 and store.experiment_count() == 8


def test_server_with_small_page_cap_is_fully_listed(store, embedder):
    client = FakeElabClient(copy.deepcopy(DEMO_EXPERIMENTS), page_limit=3)
    assert sync_labbook(client, store, embedder).new == 8


def test_progress_callback(store, embedder, client):
    seen = []
    sync_labbook(client, store, embedder, progress=lambda done, total, msg: seen.append((done, total)))
    assert seen[0] == (1, 8) and seen[-1] == (8, 8)
