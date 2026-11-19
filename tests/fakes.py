"""Deterministic stand-ins for Ollama and eLabFTW, so the whole suite runs offline."""

from __future__ import annotations

import json
import re
from typing import Callable, Iterator, Sequence

import numpy as np

from lab_chatbot.demo import DEMO_EXPERIMENTS
from lab_chatbot.errors import BackendUnavailable
from shared.elabftw_client import ElabFTWError

_DIMS = 256


def _tokens(text: str) -> list[str]:
    text = text.casefold().replace("ß", "ss")
    return [w[:6] for w in re.findall(r"[a-z0-9äöüé]+", text) if len(w) > 2]


class FakeEmbedder:
    """Hashed bag-of-words: texts sharing words get high cosine similarity."""

    def __init__(self, model: str = "fake-embed", fail: Exception | None = None):
        self.model = model
        self.fail = fail
        self.document_calls = 0
        self.query_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(_DIMS, dtype=np.float32)
        for token in _tokens(text):
            v[hash_token(token) % _DIMS] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self.fail:
            raise self.fail
        self.document_calls += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, _DIMS), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        if self.fail:
            raise self.fail
        self.query_calls += 1
        return self._vec(text)


def hash_token(token: str) -> int:
    # stable across processes (Python's str hash is randomised)
    h = 2166136261
    for ch in token.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


class FakeLLM:
    """Scriptable chat model.

    ``plan``: dict / JSON string / callable(messages) -> dict|str, or an
    Exception instance to raise (to test the planner fallback).
    ``answer``: str or callable(messages) -> str, streamed word by word.
    """

    def __init__(self, plan=None, answer="Laut Quelle [1] ist das so.", model: str = "fake-chat"):
        self.model = model
        self.plan = plan
        self.answer = answer
        self.complete_calls: list[dict] = []
        self.stream_calls: list[list[dict]] = []

    def complete(self, messages, schema=None) -> str:
        self.complete_calls.append({"messages": list(messages), "schema": schema})
        plan = self.plan(messages) if callable(self.plan) else self.plan
        if isinstance(plan, Exception):
            raise plan
        if plan is None:
            raise RuntimeError("FakeLLM has no plan configured")
        return plan if isinstance(plan, str) else json.dumps(plan)

    def stream(self, messages) -> Iterator[str]:
        self.stream_calls.append(list(messages))
        text = self.answer(messages) if callable(self.answer) else self.answer
        if isinstance(text, Exception):
            raise text
        for word in re.findall(r"\S+\s*", text):
            yield word


class FakeElabClient:
    """Minimal eLabFTW API: list + detail + categories/status endpoints."""

    def __init__(self, experiments: list[dict] | None = None, page_limit: int | None = None):
        self.experiments = {int(e["id"]): dict(e) for e in (experiments or DEMO_EXPERIMENTS)}
        self.page_limit = page_limit
        self.calls: list[tuple[str, dict | None]] = []
        self.fail_listing: Exception | None = None
        self.fail_detail: set[int] = set()

    def _overview(self, e: dict) -> dict:
        return {k: e.get(k) for k in ("id", "title", "date", "modified_at", "tags")}

    def get(self, endpoint: str, params: dict | None = None):
        self.calls.append((endpoint, params))
        if endpoint == "experiments":
            if self.fail_listing:
                raise self.fail_listing
            items = [self._overview(e) for e in self.experiments.values()]
            return self._page(items, params)
        match = re.fullmatch(r"experiments/(\d+)", endpoint)
        if match:
            eid = int(match.group(1))
            if eid in self.fail_detail:
                raise ElabFTWError(f"HTTP 500 for {eid}")
            if eid not in self.experiments:
                raise ElabFTWError(f"HTTP 404 for {eid}")
            return dict(self.experiments[eid])
        if endpoint.startswith("teams/current/"):
            return []
        raise ElabFTWError(f"unexpected endpoint {endpoint}")

    def _page(self, items, params):
        params = params or {}
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 100))
        if self.page_limit:
            limit = min(limit, self.page_limit)
        return items[offset : offset + limit]

    def iter_paginated(self, endpoint, params=None, page_size=None, max_items=None):
        offset, yielded = 0, 0
        while True:
            page = self.get(endpoint, {**(params or {}), "limit": page_size or 100, "offset": offset})
            if not page:
                return
            for item in page:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            offset += len(page)


def plan_json(intent="search", semantic_query="", filters=None, measurement=None, group_by=None, order=None) -> dict:
    return {
        "intent": intent, "semantic_query": semantic_query, "filters": filters or {},
        "measurement": measurement, "group_by": group_by, "order": order,
    }  # fmt: skip


def boom(exc_type=BackendUnavailable, msg="ollama down"):
    return exc_type(msg)


class FakeElabApi:
    """Stand-in for the ``lab_assistant.elab_api`` module: records every write."""

    def __init__(self, fail: dict[str, Exception] | None = None, template_fields: dict | None = None):
        self.calls: list[tuple] = []
        self.closed: list[int] = []
        self.fail = fail or {}
        self.next_id = 500
        self.template_fields = template_fields if template_fields is not None else {}

    def _do(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name in self.fail:
            raise self.fail[name]

    def create_experiment(self, template_id=None):
        self._do("create_experiment", template_id=template_id)
        self.next_id += 1
        return self.next_id

    def update_experiment(self, experiment_id, **kwargs):
        self._do("update_experiment", experiment_id, **kwargs)

    def add_tag_to_experiment(self, experiment_id, tag):
        self._do("add_tag", experiment_id, tag)
        if f"add_tag:{tag}" in self.fail:
            raise self.fail[f"add_tag:{tag}"]

    def create_resource(self, title, category_id=None):
        self._do("create_resource", title, category_id=category_id)
        return 900

    def link_resource_to_experiment(self, experiment_id, resource_id):
        self._do("link_resource", experiment_id, resource_id)

    def update_experiment_custom_fields(self, experiment_id, custom_fields):
        self._do("custom_fields", experiment_id, custom_fields)

    def upload_file_to_experiment(self, experiment_id, file_name, file_content, comment=""):
        self._do("upload", experiment_id, file_name, file_content, comment)

    def close_entry(self, experiment_id):
        self.closed.append(experiment_id)  # not a write: kept out of ``calls``

    def get_custom_fields_from_template(self, template_id):
        self._do("get_template_fields", template_id)
        return self.template_fields.get(template_id, {})

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


STRUCTURE = {
    "templates": [{"id": 1, "title": "Spectroscopy"}, {"id": 2, "title": "Plain"}],
    "categories": [{"id": 10, "title": "Measurement"}, {"id": 11, "title": "Measurement"}],  # duplicate title
    "statuses": [{"id": 20, "title": "Running"}],
    "resources": [{"id": 30, "title": "Spectrometer"}],
    "resource_categories": [{"id": 40, "title": "Project"}, {"id": 41, "title": "Instrument"}],
    "tags": [{"tag": "raman"}, {"tag": "laser"}, {"tag": "raman"}],
}


REGISTRY: dict = {}  # AppTest runs the script in-process: tests read what the page created here
