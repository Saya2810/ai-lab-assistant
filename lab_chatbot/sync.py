"""Incremental synchronisation: eLabFTW -> local index.

The old ``sync_labbook()`` was never called, re-embedded everything every time,
never removed deleted experiments, ignored pagination and swallowed errors with
``print``. This version:

* lists *all* experiments (paginated),
* skips experiments whose ``modified_at`` is unchanged, and, after fetching,
  those whose content hash is unchanged, so only edits are re-embedded,
* replaces an experiment's chunks atomically (a failed embedding keeps the
  previous version in the index),
* removes experiments that no longer exist (deleted/archived) from the index,
* refuses to wipe the index when eLabFTW suddenly returns an empty list,
* returns a report instead of printing,
* streams the listing page by page (constant memory for any number of experiments),
* runs only once at a time, also across processes (a lease in the database),
  publishes its progress for the UI, and resumes an interrupted full re-read,
* optionally reads image attachments with a vision model (``LAB_INDEX_IMAGES``)
  and indexes their text together with the body. Image readings are cached by
  file hash, so an unchanged image is never sent to the vision model twice.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from lab_chatbot.chunking import CHUNKER_VERSION, chunk_experiment
from lab_chatbot.embeddings import Embedder
from lab_chatbot.errors import BackendUnavailable, SyncBusy, SyncError
from lab_chatbot.parse import PARSER_VERSION, parse_experiment
from lab_chatbot.store import Store, StoredState, utc_now, utc_now_precise
from lab_chatbot.vision import ImageReader
from shared.images import is_image_name
from shared.elabftw_client import ElabFTWError

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]
FULL_SYNC_KEY = "full_sync_started"


@dataclass
class SyncReport:
    listed: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: list[tuple[int, str]] = field(default_factory=list)
    seconds: float = 0.0
    images_read: int = 0
    images_cached: int = 0
    image_errors: list[str] = field(default_factory=list)
    stopped: bool = False

    def summary(self) -> str:
        text = (
            f"{self.listed} in eLabFTW: {self.new} new, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.deleted} removed from index"
        )
        if self.images_read or self.images_cached:
            text += f", images: {self.images_read} read, {self.images_cached} from cache"
        if self.image_errors:
            text += f", {len(self.image_errors)} image(s) skipped"
        if self.failed:
            text += f", {len(self.failed)} FAILED"
        if self.stopped:
            text += ", stopped early (will continue next time)"
        return f"{text} ({self.seconds:.1f}s)"


def _name_map(client, endpoint: str) -> dict[int, str]:
    """id -> title for categories/statuses (best effort; entries may carry names already)."""
    try:
        items = client.get(endpoint)
    except Exception as exc:  # optional lookup, never fatal
        log.info("Could not load %s: %s", endpoint, exc)
        return {}
    names: dict[int, str] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("id") is not None:
            names[int(item["id"])] = str(item.get("title") or item.get("name") or "")
    return names


def index_signature(embedder: Embedder, image_reader: ImageReader | None = None) -> str:
    """Anything that changes this forces a re-embed of every experiment."""
    signature = f"{PARSER_VERSION}.{CHUNKER_VERSION}|{embedder.model}"
    return f"{signature}|img:{image_reader.model}" if image_reader is not None else signature


def _image_uploads(raw: dict, max_bytes: int, max_count: int) -> list[dict]:
    uploads = raw.get("uploads") if isinstance(raw, dict) else None
    chosen = []
    for upload in uploads if isinstance(uploads, list) else []:
        if not isinstance(upload, dict) or upload.get("id") is None:
            continue
        name = str(upload.get("real_name") or upload.get("long_name") or "")
        if not is_image_name(name):
            continue
        size = upload.get("filesize")
        if size is not None and str(size).isdigit() and int(size) > max_bytes:
            continue
        chosen.append(upload)
    return chosen[:max_count]


def _read_images(
    client, store: Store, reader: ImageReader, experiment_id: int, raw: dict,
    report: SyncReport, max_bytes: int, max_count: int,
) -> list[tuple[str, str]]:  # fmt: skip
    """(file name, text) for the experiment's image attachments; errors skip the image."""
    results = []
    for upload in _image_uploads(raw, max_bytes, max_count):
        name = str(upload.get("real_name") or upload.get("long_name"))
        key = str(upload.get("hash") or f"{experiment_id}:{upload['id']}:{upload.get('filesize')}")
        text = store.get_image_text(key, reader.model)
        if text is not None:
            report.images_cached += 1
        else:
            try:
                response = client.request(
                    "GET", f"experiments/{experiment_id}/uploads/{upload['id']}",
                    params={"format": "binary"},
                )  # fmt: skip
                content = response.content
                if len(content) > max_bytes:
                    continue
                text = reader.read(content).strip()
            except BackendUnavailable:
                raise
            except Exception as exc:
                log.warning("Could not read image %s of experiment %s: %s", name, experiment_id, exc)
                report.image_errors.append(f"{experiment_id}/{name}: {exc}")
                continue
            store.put_image_text(key, reader.model, text)
            report.images_read += 1
        if text:
            results.append((name, text))
    return results


SYNC_LEASE = "sync"
LEASE_TTL = 180.0  # seconds; renewed while the sync runs, so a crashed sync frees it soon
STATUS_KEY = "sync_status"


def default_owner() -> str:
    import os
    import socket
    import threading

    return f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"


def sync_status(store: Store) -> dict:
    """Progress of the running (or last) sync, as written by ``sync_labbook``."""
    value = store.get_meta(STATUS_KEY)
    try:
        status = json.loads(value) if value else {}
    except ValueError:
        status = {}
    if status.get("state") == "running" and not store.lease_holder(SYNC_LEASE):
        status["state"] = "interrupted"  # the process died without finishing
    return status


class _Heartbeat:
    """Renews the lease and publishes progress, at most every few seconds."""

    def __init__(self, store: Store, owner: str, started: str, every: float = 3.0):
        self.store, self.owner, self.started, self.every = store, owner, started, every
        self.last = 0.0

    def beat(self, done: int, total: int, current: str = "", force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last < self.every:
            return
        self.last = now
        if not self.store.acquire_lease(SYNC_LEASE, self.owner, LEASE_TTL):
            raise SyncBusy("The sync lease was taken over by another process.")
        self.publish("running", done=done, total=total, current=current)

    def publish(self, state: str, **fields) -> None:
        payload = {"state": state, "owner": self.owner, "started": self.started, "updated": utc_now()}
        self.store.set_meta(STATUS_KEY, json.dumps({**payload, **fields}))


def sync_labbook(
    client,
    store: Store,
    embedder: Embedder,
    *,
    full: bool = False,
    force_empty: bool = False,
    max_items: int | None = None,
    progress: ProgressFn | None = None,
    image_reader: ImageReader | None = None,
    max_image_bytes: int = 20 * 1024 * 1024,
    max_images_per_experiment: int = 10,
    should_stop: Callable[[], bool] | None = None,
    owner: str | None = None,
) -> SyncReport:
    """Bring the index up to date with eLabFTW.

    Safe to run from several places: a lease in the database lets only one sync
    run at a time, also across processes (UI, CLI, background worker); others
    get ``SyncBusy``. Every experiment is committed on its own, so an interrupted
    sync loses nothing; the next run skips what is already current. An
    interrupted ``full`` sync resumes where it stopped instead of starting over.
    ``should_stop`` is polled between experiments for a clean shutdown.
    """
    owner = owner or default_owner()
    if not store.acquire_lease(SYNC_LEASE, owner, LEASE_TTL):
        raise SyncBusy(f"A sync is already running ({store.lease_holder(SYNC_LEASE) or 'elsewhere'}).")
    started_at = utc_now()
    heartbeat = _Heartbeat(store, owner, started_at)
    try:
        heartbeat.publish("running", done=0, total=0, full=full)
        report = _sync(
            client, store, embedder, heartbeat,
            full=full, force_empty=force_empty, max_items=max_items, progress=progress,
            image_reader=image_reader, max_image_bytes=max_image_bytes,
            max_images_per_experiment=max_images_per_experiment, should_stop=should_stop,
        )  # fmt: skip
    except Exception as exc:
        heartbeat.publish("failed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    finally:
        store.release_lease(SYNC_LEASE, owner)
    heartbeat.publish(
        "stopped" if report.stopped else "done",
        summary=report.summary(), failed=len(report.failed), done=report.listed, total=report.listed,
    )  # fmt: skip
    return report


def _sync(
    client, store: Store, embedder: Embedder, heartbeat: _Heartbeat, *, full: bool,
    force_empty: bool, max_items: int | None, progress: ProgressFn | None,
    image_reader: ImageReader | None, max_image_bytes: int, max_images_per_experiment: int,
    should_stop: Callable[[], bool] | None,
) -> SyncReport:  # fmt: skip
    started = time.perf_counter()
    report = SyncReport()
    state = store.experiment_state()

    # The listing is streamed page by page: with hundreds of thousands of
    # experiments, holding every list entry (they include the body) costs gigabytes.
    try:
        items = iter(client.iter_paginated("experiments", max_items=max_items))
        first = next(items, None)
    except ElabFTWError as exc:
        raise SyncError(f"Could not list experiments: {exc}") from exc
    if first is None and state and not force_empty:
        raise SyncError(
            f"eLabFTW returned no experiments but the index holds {len(state)}. "
            "Refusing to delete everything (is the API key/team correct?). "
            "Re-run with force_empty=True / --force-empty if this is intended."
        )

    categories = _name_map(client, "teams/current/experiments_categories")
    statuses = _name_map(client, "teams/current/experiments_status")
    signature = index_signature(embedder, image_reader)

    # A full re-read that was interrupted continues: whatever it already rewrote
    # (synced_at after its start) is not embedded a second time.
    resume_after = ""
    if full:
        resume_after = store.get_meta(FULL_SYNC_KEY) or ""
        if resume_after:
            log.info("Resuming the full sync started at %s", resume_after)
        else:
            store.set_meta(FULL_SYNC_KEY, utc_now_precise())

    listed_ids: set[int] = set()
    position = 0
    listing_complete = False
    try:
        for item in itertools.chain([first] if first is not None else [], items):
            if should_stop is not None and should_stop():
                report.stopped = True
                break
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            position += 1
            experiment_id = int(item["id"])
            listed_ids.add(experiment_id)
            total = len(state) if position <= len(state) else 0  # 0 = not known yet
            heartbeat.beat(position, total, f"Experiment {experiment_id}")
            if progress:
                progress(position, total, f"Experiment {experiment_id}")
            _sync_one(
                client, store, embedder, report, item, state.get(experiment_id),
                categories, statuses, signature, full, resume_after, image_reader,
                max_image_bytes, max_images_per_experiment,
            )  # fmt: skip
        else:
            listing_complete = True
    except ElabFTWError as exc:  # the listing broke off mid-way
        raise SyncError(f"Listing experiments failed after {position}: {exc}") from exc
    report.listed = position

    # Only safe when the whole listing was seen; otherwise "not listed" means "not reached".
    if listing_complete and max_items is None:
        gone = sorted(set(state) - listed_ids)
        if gone:
            report.deleted = store.delete_experiments(gone)
        if full:
            store.delete_meta(FULL_SYNC_KEY)
        store.set_meta("last_sync", utc_now())
    store.set_meta("embed_model", embedder.model)

    heartbeat.beat(position, position, "Aufräumen", force=True)
    store.prune_tombstones()
    if full or report.new + report.updated + report.deleted >= 500:
        store.optimize()  # merge FTS segments, refresh SQLite statistics, trim the WAL
    report.seconds = time.perf_counter() - started
    return report


def _sync_one(
    client, store: Store, embedder: Embedder, report: SyncReport, item: dict,
    previous: StoredState | None, categories: dict, statuses: dict, signature: str,
    full: bool, resume_after: str, image_reader: ImageReader | None,
    max_image_bytes: int, max_images_per_experiment: int,
) -> None:  # fmt: skip
    experiment_id = int(item["id"])
    listed_modified = str(item.get("modified_at") or item.get("lastchange") or "")
    same_config = previous is not None and previous.index_sig == signature

    if same_config and (
        (not full and listed_modified and previous.modified_at == listed_modified)
        or (full and resume_after and previous.synced_at >= resume_after)
    ):
        report.unchanged += 1
        return

    try:
        raw = client.get(f"experiments/{experiment_id}")
        record = parse_experiment(raw, categories, statuses)
        modified = listed_modified or record.modified_at
        if image_reader is not None:
            found = _read_images(
                client, store, image_reader, experiment_id, raw, report,
                max_image_bytes, max_images_per_experiment,
            )  # fmt: skip
            if found:
                record.body = "\n\n".join(
                    [record.body, *(f"## Bild: {name}\n{text}" for name, text in found)]
                )
                record.content_hash = record.compute_hash()

        if not full and same_config and previous.content_hash == record.content_hash:
            store.touch_experiment(experiment_id, modified)
            report.unchanged += 1
            return

        chunks = chunk_experiment(record)
        vectors = embedder.embed_documents([c.text for c in chunks])
        store.replace_experiment(
            record, chunks, vectors,
            embed_model=embedder.model, index_sig=signature, modified_at=modified,
        )  # fmt: skip
        if previous is None:
            report.new += 1
        else:
            report.updated += 1
    except BackendUnavailable:
        raise  # embedding backend is down: every further item would fail too
    except Exception as exc:
        log.warning("Could not index experiment %s: %s", experiment_id, exc)
        report.failed.append((experiment_id, str(exc)))
