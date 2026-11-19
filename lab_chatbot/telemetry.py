"""Per-question telemetry: one row per answered question in the index database.

Answers where time goes (planning, SQL, retrieval, generation, vision) and which
questions fail or find nothing. Writing never breaks answering: every failure is
logged and swallowed. The question text is stored in the table (it is the point
of the table) but never written to the server log.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from lab_chatbot.store import Store

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT '',
    route TEXT NOT NULL DEFAULT '',
    used_llm INTEGER NOT NULL DEFAULT 0,
    n_sources INTEGER NOT NULL DEFAULT 0,
    n_cited INTEGER NOT NULL DEFAULT 0,
    total_ms REAL NOT NULL DEFAULT 0,
    latencies_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    chat_model TEXT NOT NULL DEFAULT '',
    embed_model TEXT NOT NULL DEFAULT '',
    had_images INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(ts);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_table(con) -> None:
    con.executescript(SCHEMA)


class QueryLog:
    """Writer for the ``query_log`` table (a single INSERT per question)."""

    def __init__(self, store: Store, max_rows: int = 100_000, prune_every: int = 500):
        self.store = store
        self.max_rows = max_rows
        self.prune_every = max(1, prune_every)
        self._ready = False
        self._writes = 0
        self._lock = threading.Lock()

    def record(
        self,
        *,
        question: str,
        language: str = "",
        route: str = "",
        used_llm: bool = False,
        n_sources: int = 0,
        n_cited: int = 0,
        latencies: Mapping[str, float] | None = None,
        error: str = "",
        chat_model: str = "",
        embed_model: str = "",
        had_images: bool = False,
    ) -> None:
        latencies = dict(latencies or {})
        total = float(latencies.get("total", 0.0))
        steps = " ".join(f"{k}={v:.0f}ms" for k, v in latencies.items() if k != "total")
        # No question text here, only its length.
        log.info(
            "question len=%d route=%s sources=%d cited=%d llm=%s images=%s total=%.0fms %s%s",
            len(question), route or "-", n_sources, n_cited, used_llm, had_images, total,
            steps, f" error={error}" if error else "",
        )  # fmt: skip
        try:
            with self.store.connect() as con:
                if not self._ready:
                    ensure_table(con)
                    self._ready = True
                con.execute(
                    "INSERT INTO query_log (ts, question, language, route, used_llm, n_sources,"
                    " n_cited, total_ms, latencies_json, error, chat_model, embed_model, had_images)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        _now(), question, language, route, int(used_llm), n_sources, n_cited,
                        total, json.dumps(latencies), error, chat_model, embed_model,
                        int(had_images),
                    ),
                )  # fmt: skip
                with self._lock:
                    self._writes += 1
                    prune = self._writes % self.prune_every == 0
                if prune:
                    # ids grow monotonically (AUTOINCREMENT), so "newest max_rows" is an id cut-off
                    con.execute(
                        "DELETE FROM query_log WHERE id <= (SELECT MAX(id) FROM query_log) - ?",
                        (self.max_rows,),
                    )
        except Exception:
            log.warning("could not write query_log row", exc_info=True)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil
    return float(ordered[int(rank) - 1])


def _stats(values: list[float]) -> dict[str, float]:
    return {"p50": round(_percentile(values, 50), 1), "p95": round(_percentile(values, 95), 1)}


def summary(store: Store, since_days: float = 7) -> dict[str, Any]:
    """Counts, error rate, "nothing found" share and p50/p95 latencies (total and per step)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat(timespec="seconds")
    with store.connect() as con:
        ensure_table(con)
        rows = con.execute(
            "SELECT route, n_sources, total_ms, latencies_json, error FROM query_log WHERE ts >= ?",
            (cutoff,),
        ).fetchall()
    count = len(rows)
    errors = sum(1 for r in rows if r["error"])
    searches = [r for r in rows if r["route"] == "search" and not r["error"]]
    nothing = sum(1 for r in searches if r["n_sources"] == 0)
    steps: dict[str, list[float]] = {}
    for r in rows:
        try:
            parsed = json.loads(r["latencies_json"] or "{}")
        except ValueError:
            continue
        for key, value in parsed.items():
            if key != "total" and isinstance(value, (int, float)):
                steps.setdefault(key, []).append(float(value))
    return {
        "since_days": since_days,
        "count": count,
        "errors": errors,
        "error_rate": errors / count if count else 0.0,
        "search_count": len(searches),
        "no_sources": nothing,
        "no_sources_share": nothing / len(searches) if searches else 0.0,
        "total_ms": _stats([float(r["total_ms"]) for r in rows]),
        "steps_ms": {k: _stats(v) for k, v in sorted(steps.items())},
    }


def last_rows(store: Store, n: int = 20) -> list[dict]:
    with store.connect() as con:
        ensure_table(con)
        rows = con.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]
