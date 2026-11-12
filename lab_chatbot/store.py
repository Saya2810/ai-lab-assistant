"""One SQLite file holds everything the chatbot knows.

* ``experiments`` / ``exp_tags`` / ``measurements``: structured data for exact
  (SQL) answers: counts, date ranges, tags, numeric comparisons in SI units.
* ``chunks``: text pieces with their embedding vectors (float32 BLOBs).
* ``chunks_fts``: FTS5 index over the same chunks for keyword/BM25 search.

Why not a separate vector database? SQL filters and vector search must work
*together* ("only experiments with wavelength > 500 nm, then rank by meaning").
With everything in one database that is a plain candidate set, deletes are
transactional (no orphaned vectors for edited/deleted experiments) and there is
no client object created at import time. Vectors are ranked with numpy, which is
instant for lab-notebook sizes (tens of thousands of chunks). Beyond that,
swap ``vector_data`` for sqlite-vec or Chroma; nothing else changes.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from lab_chatbot.chunking import ChunkText
from lab_chatbot.parse import ExperimentRecord

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS experiments(
    id INTEGER PRIMARY KEY,
    title TEXT, date TEXT, body TEXT,
    category TEXT, status TEXT, author TEXT, resources TEXT,
    modified_at TEXT, content_hash TEXT, index_sig TEXT, synced_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_exp_date ON experiments(date);

CREATE TABLE IF NOT EXISTS exp_tags(
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    tag TEXT NOT NULL, tag_norm TEXT NOT NULL,
    PRIMARY KEY(experiment_id, tag_norm)
);
CREATE INDEX IF NOT EXISTS idx_tags_norm ON exp_tags(tag_norm);

CREATE TABLE IF NOT EXISTS measurements(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name TEXT, name_norm TEXT, value_text TEXT, value_num REAL,
    unit TEXT, unit_norm TEXT, dimension TEXT, value_si REAL
);
CREATE INDEX IF NOT EXISTS idx_meas_name ON measurements(name_norm);
CREATE INDEX IF NOT EXISTS idx_meas_exp ON measurements(experiment_id);

CREATE TABLE IF NOT EXISTS chunks(
    cid INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, text TEXT NOT NULL,
    embedding BLOB, embed_model TEXT, dim INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_exp ON chunks(experiment_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, tokenize='unicode61 remove_diacritics 2');
"""


def norm(text: object) -> str:
    """Case/space-insensitive key used for tags, categories, measurement names."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StoredState:
    modified_at: str
    content_hash: str
    index_sig: str


@dataclass
class ChunkRow:
    cid: int
    experiment_id: int
    idx: int
    text: str
    title: str
    date: str


@dataclass
class VectorData:
    cids: np.ndarray  # int64, shape (n,)
    exp_ids: np.ndarray  # int64, shape (n,)
    matrix: np.ndarray  # float32, shape (n, dim), L2-normalised


_EMPTY_VECTORS = VectorData(
    np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
)
_vector_cache: dict[tuple[str, str], tuple[int, VectorData]] = {}
_cache_lock = threading.Lock()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------ connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection (cheap in SQLite, and safe across threads)."""
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self.connect() as con:
            con.execute("PRAGMA journal_mode = WAL")
            con.executescript(_SCHEMA)
            con.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            con.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('revision', '0')")

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str) -> str | None:
        with self.connect() as con:
            row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def revision(self) -> int:
        return int(self.get_meta("revision") or 0)

    @staticmethod
    def _bump(con: sqlite3.Connection) -> None:
        con.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'")

    # ---------------------------------------------------------------- writes

    @staticmethod
    def _delete_rows(con: sqlite3.Connection, experiment_id: int) -> None:
        con.execute(
            "DELETE FROM chunks_fts WHERE rowid IN "
            "(SELECT cid FROM chunks WHERE experiment_id = ?)",
            (experiment_id,),
        )
        con.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))  # cascades

    def replace_experiment(
        self,
        record: ExperimentRecord,
        chunks: Sequence[ChunkText],
        vectors: np.ndarray | None,
        *,
        embed_model: str,
        index_sig: str,
        modified_at: str = "",
    ) -> None:
        """Atomically replace everything stored for one experiment.

        Old chunks/vectors are removed in the same transaction, so a failed
        write leaves the previous version intact and an edit never leaves
        stale chunks behind.
        """
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        with self.connect() as con:
            self._delete_rows(con, record.id)
            con.execute(
                "INSERT INTO experiments(id, title, date, body, category, status, author, "
                "resources, modified_at, content_hash, index_sig, synced_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id, record.title, record.date, record.body, record.category,
                    record.status, record.author, " | ".join(record.resources),
                    modified_at or record.modified_at, record.content_hash, index_sig, utc_now(),
                ),
            )  # fmt: skip
            con.executemany(
                "INSERT OR IGNORE INTO exp_tags(experiment_id, tag, tag_norm) VALUES(?,?,?)",
                [(record.id, tag, norm(tag)) for tag in record.tags],
            )
            con.executemany(
                "INSERT INTO measurements(experiment_id, name, name_norm, value_text, value_num, "
                "unit, unit_norm, dimension, value_si) VALUES(?,?,?,?,?,?,?,?,?)",
                [
                    (
                        record.id, m.name, norm(m.name), m.value_text, m.value_num,
                        m.unit, m.unit_norm, m.dimension, m.value_si,
                    )
                    for m in record.measurements
                ],
            )  # fmt: skip
            for position, chunk in enumerate(chunks):
                vector = None if vectors is None else np.asarray(vectors[position], np.float32)
                cursor = con.execute(
                    "INSERT INTO chunks(experiment_id, idx, text, embedding, embed_model, dim) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        record.id, chunk.idx, chunk.text,
                        None if vector is None else vector.tobytes(),
                        embed_model if vector is not None else None,
                        None if vector is None else int(vector.shape[0]),
                    ),
                )  # fmt: skip
                con.execute(
                    "INSERT INTO chunks_fts(rowid, text) VALUES(?, ?)",
                    (cursor.lastrowid, chunk.text),
                )
            self._bump(con)

    def touch_experiment(self, experiment_id: int, modified_at: str) -> None:
        """Content unchanged: just remember the new modification stamp."""
        with self.connect() as con:
            con.execute(
                "UPDATE experiments SET modified_at = ?, synced_at = ? WHERE id = ?",
                (modified_at, utc_now(), experiment_id),
            )

    def delete_experiments(self, ids: Sequence[int]) -> int:
        with self.connect() as con:
            for experiment_id in ids:
                self._delete_rows(con, int(experiment_id))
            if ids:
                self._bump(con)
        return len(ids)

    def reset(self) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM chunks_fts")
            con.execute("DELETE FROM experiments")  # cascades
            self._bump(con)

    # ----------------------------------------------------------------- reads

    def experiment_state(self) -> dict[int, StoredState]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT id, modified_at, content_hash, index_sig FROM experiments"
            ).fetchall()
        return {
            r["id"]: StoredState(r["modified_at"] or "", r["content_hash"] or "", r["index_sig"] or "")
            for r in rows
        }

    def stats(self) -> dict:
        with self.connect() as con:
            experiments = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded = con.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            measurements = con.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
            models = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT embed_model FROM chunks WHERE embed_model IS NOT NULL"
                )
            ]
        return {
            "experiments": experiments,
            "chunks": chunks,
            "embedded_chunks": embedded,
            "measurements": measurements,
            "embed_models_in_index": models,
            "embed_model": self.get_meta("embed_model"),
            "last_sync": self.get_meta("last_sync"),
        }

    def experiment_count(self) -> int:
        with self.connect() as con:
            return con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    def get_experiment(self, experiment_id: int) -> dict | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        return dict(row) if row else None

    def chunk_rows(self, cids: Sequence[int]) -> dict[int, ChunkRow]:
        if not cids:
            return {}
        marks = ",".join("?" * len(cids))
        with self.connect() as con:
            rows = con.execute(
                f"SELECT c.cid, c.experiment_id, c.idx, c.text, e.title, e.date "
                f"FROM chunks c JOIN experiments e ON e.id = c.experiment_id "
                f"WHERE c.cid IN ({marks})",
                list(map(int, cids)),
            ).fetchall()
        return {
            r["cid"]: ChunkRow(
                r["cid"], r["experiment_id"], r["idx"], r["text"], r["title"] or "", r["date"] or ""
            )
            for r in rows
        }

    def chunks_of(self, experiment_ids: Sequence[int]):
        """All chunks of the given experiments in reading order (as retrieval hits)."""
        from lab_chatbot.retrieval import Hit

        if not experiment_ids:
            return []
        marks = ",".join("?" * len(experiment_ids))
        with self.connect() as con:
            rows = con.execute(
                f"SELECT c.cid, c.experiment_id, c.idx, c.text, e.title, e.date FROM chunks c "
                f"JOIN experiments e ON e.id = c.experiment_id WHERE c.experiment_id IN ({marks}) "
                f"ORDER BY c.experiment_id, c.idx",
                [int(i) for i in experiment_ids],
            ).fetchall()
        return [
            Hit(r["cid"], r["experiment_id"], r["idx"], r["text"], r["title"] or "",
                r["date"] or "", None, None, 0.0)
            for r in rows
        ]  # fmt: skip

    # --------------------------------------------------------------- vectors

    def vector_data(self, embed_model: str) -> VectorData:
        """All vectors of one embedding model, cached until the index changes.

        Chunks embedded with a different model are ignored (dimensions would
        not match); ``doctor`` and ``stats`` report them.
        """
        key = (str(self.path.resolve()), embed_model)
        revision = self.revision()
        with _cache_lock:
            cached = _vector_cache.get(key)
            if cached and cached[0] == revision:
                return cached[1]
        with self.connect() as con:
            rows = con.execute(
                "SELECT cid, experiment_id, embedding, dim FROM chunks "
                "WHERE embed_model = ? AND embedding IS NOT NULL ORDER BY cid",
                (embed_model,),
            ).fetchall()
        if not rows:
            data = _EMPTY_VECTORS
        else:
            dim = rows[0]["dim"]
            rows = [r for r in rows if r["dim"] == dim]
            data = VectorData(
                np.array([r["cid"] for r in rows], dtype=np.int64),
                np.array([r["experiment_id"] for r in rows], dtype=np.int64),
                np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]),
            )
        with _cache_lock:
            _vector_cache[key] = (revision, data)
        return data

    def fts_search(
        self, match: str, limit: int = 30, experiment_ids: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """BM25 keyword search; returns ``(cid, bm25)`` best first (lower = better)."""
        if not match:
            return []
        with self.connect() as con:
            if experiment_ids is not None:
                con.execute("CREATE TEMP TABLE IF NOT EXISTS _cand(id INTEGER PRIMARY KEY)")
                con.execute("DELETE FROM _cand")
                con.executemany(
                    "INSERT INTO _cand(id) VALUES(?)", [(int(i),) for i in experiment_ids]
                )
                sql = (
                    "SELECT c.cid AS cid, bm25(chunks_fts) AS score FROM chunks_fts "
                    "JOIN chunks c ON c.cid = chunks_fts.rowid "
                    "JOIN _cand ON _cand.id = c.experiment_id "
                    "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?"
                )
            else:
                sql = (
                    "SELECT c.cid AS cid, bm25(chunks_fts) AS score FROM chunks_fts "
                    "JOIN chunks c ON c.cid = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?"
                )
            try:
                rows = con.execute(sql, (match, limit)).fetchall()
            except sqlite3.OperationalError:
                return []  # malformed MATCH expression: treat as "no keyword hits"
        return [(r["cid"], r["score"]) for r in rows]
