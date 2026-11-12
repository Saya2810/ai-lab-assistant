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
* returns a report instead of printing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from lab_chatbot.chunking import CHUNKER_VERSION, chunk_experiment
from lab_chatbot.embeddings import Embedder
from lab_chatbot.errors import BackendUnavailable, SyncError
from lab_chatbot.parse import PARSER_VERSION, parse_experiment
from lab_chatbot.store import Store, utc_now
from shared.elabftw_client import ElabFTWError

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


@dataclass
class SyncReport:
    listed: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: list[tuple[int, str]] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        text = (
            f"{self.listed} in eLabFTW: {self.new} new, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.deleted} removed from index"
        )
        if self.failed:
            text += f", {len(self.failed)} FAILED"
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


def index_signature(embedder: Embedder) -> str:
    """Anything that changes this forces a re-embed of every experiment."""
    return f"{PARSER_VERSION}.{CHUNKER_VERSION}|{embedder.model}"


def sync_labbook(
    client,
    store: Store,
    embedder: Embedder,
    *,
    full: bool = False,
    force_empty: bool = False,
    max_items: int | None = None,
    progress: ProgressFn | None = None,
) -> SyncReport:
    started = time.perf_counter()
    report = SyncReport()

    try:
        overview = list(client.iter_paginated("experiments", max_items=max_items))
    except ElabFTWError as exc:
        raise SyncError(f"Could not list experiments: {exc}") from exc

    state = store.experiment_state()
    if not overview and state and not force_empty:
        raise SyncError(
            f"eLabFTW returned no experiments but the index holds {len(state)}. "
            "Refusing to delete everything (is the API key/team correct?). "
            "Re-run with force_empty=True / --force-empty if this is intended."
        )

    categories = _name_map(client, "teams/current/experiments_categories")
    statuses = _name_map(client, "teams/current/experiments_status")
    signature = index_signature(embedder)
    report.listed = len(overview)

    listed_ids: set[int] = set()
    for position, item in enumerate(overview, start=1):
        if item.get("id") is None:
            continue
        experiment_id = int(item["id"])
        listed_ids.add(experiment_id)
        if progress:
            progress(position, len(overview), f"Experiment {experiment_id}")

        previous = state.get(experiment_id)
        listed_modified = str(item.get("modified_at") or item.get("lastchange") or "")
        same_config = previous is not None and previous.index_sig == signature

        if (
            not full
            and same_config
            and listed_modified
            and previous.modified_at == listed_modified
        ):
            report.unchanged += 1
            continue

        try:
            raw = client.get(f"experiments/{experiment_id}")
            record = parse_experiment(raw, categories, statuses)
            modified = listed_modified or record.modified_at

            if not full and same_config and previous.content_hash == record.content_hash:
                store.touch_experiment(experiment_id, modified)
                report.unchanged += 1
                continue

            chunks = chunk_experiment(record)
            vectors = embedder.embed_documents([c.text for c in chunks])
            store.replace_experiment(
                record,
                chunks,
                vectors,
                embed_model=embedder.model,
                index_sig=signature,
                modified_at=modified,
            )
            if previous is None:
                report.new += 1
            else:
                report.updated += 1
        except BackendUnavailable:
            raise  # embedding backend is down: every further item would fail too
        except Exception as exc:
            log.warning("Could not index experiment %s: %s", experiment_id, exc)
            report.failed.append((experiment_id, str(exc)))

    # Safe: the listing above completed (otherwise we raised before this point).
    gone = sorted(set(state) - listed_ids) if max_items is None else []
    if gone:
        report.deleted = store.delete_experiments(gone)

    store.set_meta("last_sync", utc_now())
    store.set_meta("embed_model", embedder.model)
    report.seconds = time.perf_counter() - started
    return report
