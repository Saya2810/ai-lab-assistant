"""One SQLite file holds everything the chatbot knows.

* ``experiments`` / ``exp_tags`` / ``measurements``: structured data for exact
  (SQL) answers: counts, date ranges, tags, numeric comparisons in SI units.
* ``chunks``: text pieces with their embedding vectors (float32 BLOBs).
* ``chunks_fts``: FTS5 index over the same chunks for keyword/BM25 search.

Why not a separate vector database? SQL filters and vector search must work
*together* ("only experiments with wavelength > 500 nm, then rank by meaning").
With everything in one database that is a plain candidate set, deletes are
transactional (no orphaned vectors for edited/deleted experiments) and there is
no client object created at import time. Vectors are ranked with numpy from an
in-memory index that follows the database incrementally (lab_chatbot/vectors.py);
exact search over a few million chunks takes well under a second.
"""

from __future__ import annotations

import logging
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
from lab_chatbot.vectors import Segment, VectorIndex

log = logging.getLogger(__name__)

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
CREATE INDEX IF NOT EXISTS idx_exp_category ON experiments(lower(category));
CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(lower(status));

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
CREATE INDEX IF NOT EXISTS idx_chunks_model ON chunks(embed_model, dim);

CREATE TABLE IF NOT EXISTS image_texts(
    key TEXT NOT NULL, model TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT,
    PRIMARY KEY(key, model)
);

-- ids of deleted chunks, so in-memory vector indexes (also in other processes)
-- can drop them without reloading everything (see lab_chatbot/vectors.py)
CREATE TABLE IF NOT EXISTS chunk_tombstones(
    seq INTEGER PRIMARY KEY AUTOINCREMENT, cid INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(text, tokenize='unicode61 remove_diacritics 2');
"""


def norm(text: object) -> str:
    """Case/space-insensitive key used for tags, categories, measurement names."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def id_list(ids) -> str:
    """Ids as one JSON parameter for ``IN (SELECT value FROM json_each(?))``.

    One bound parameter instead of one per id: SQLite limits the number of
    parameters per statement, so ``IN (?, ?, ...)`` breaks on large candidate sets.
    """
    import json

    return json.dumps([int(i) for i in ids])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_now_precise() -> str:
    """Microsecond timestamp: orders events within the same second (sync resume)."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class StoredState:
    modified_at: str
    content_hash: str
    index_sig: str
    synced_at: str = ""


@dataclass
class ChunkRow:
    cid: int
    experiment_id: int
    idx: int
    text: str
    title: str
    date: str


_vector_cache: dict[tuple[str, str, int], VectorIndex] = {}
_cache_lock = threading.Lock()
_load_lock = threading.Lock()


class Store:
    def __init__(self, path: str | Path, vector_dims: int = 0):
        self.path = Path(path)
        self.vector_dims = int(vector_dims or 0)  # 0 = full dimension
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------ connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection (cheap in SQLite, and safe across threads)."""
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        # WAL + NORMAL is crash-safe for the database file and much faster for many small
        # writes (a sync commits once per experiment). The rest keeps hot pages in memory.
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("PRAGMA busy_timeout = 30000")
        con.execute("PRAGMA temp_store = MEMORY")
        con.execute("PRAGMA cache_size = -65536")  # 64 MB
        con.execute("PRAGMA mmap_size = 268435456")  # 256 MB
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

    def delete_meta(self, key: str) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM meta WHERE key = ?", (key,))

    # ---------------------------------------------------------------- leases

    def acquire_lease(self, name: str, owner: str, ttl: float) -> bool:
        """Take or renew a named lock that also works across processes.

        The lease expires after ``ttl`` seconds unless renewed, so a crashed
        holder never blocks forever. Returns False while someone else holds it.
        """
        import json
        import time

        now = time.time()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")  # check-and-set under the write lock
            row = con.execute("SELECT value FROM meta WHERE key = ?", (f"lease:{name}",)).fetchone()
            if row:
                held = json.loads(row[0])
                if held.get("owner") != owner and float(held.get("expires", 0)) > now:
                    return False
            con.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (f"lease:{name}", json.dumps({"owner": owner, "expires": now + ttl})),
            )
        return True

    def release_lease(self, name: str, owner: str) -> None:
        import json

        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT value FROM meta WHERE key = ?", (f"lease:{name}",)).fetchone()
            if row and json.loads(row[0]).get("owner") == owner:
                con.execute("DELETE FROM meta WHERE key = ?", (f"lease:{name}",))

    def lease_holder(self, name: str) -> str | None:
        import json
        import time

        value = self.get_meta(f"lease:{name}")
        if not value:
            return None
        held = json.loads(value)
        return held.get("owner") if float(held.get("expires", 0)) > time.time() else None

    def revision(self) -> int:
        return int(self.get_meta("revision") or 0)

    @staticmethod
    def _bump(con: sqlite3.Connection) -> None:
        con.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'")

    # ---------------------------------------------------------------- writes

    @staticmethod
    def _delete_rows(con: sqlite3.Connection, experiment_id: int) -> None:
        con.execute(
            "INSERT INTO chunk_tombstones(cid) SELECT cid FROM chunks "
            "WHERE experiment_id = ? AND embedding IS NOT NULL",
            (experiment_id,),
        )
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
                    modified_at or record.modified_at, record.content_hash, index_sig, utc_now_precise(),
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
                (modified_at, utc_now_precise(), experiment_id),
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
            con.execute("DELETE FROM chunk_tombstones")
            con.execute(
                "INSERT INTO meta(key, value) VALUES('vector_epoch', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (utc_now() + f"#{id(con)}",),
            )
            self._bump(con)

    def prune_tombstones(self, keep: int = 200_000) -> int:
        """Drop old tombstones; indexes older than the floor then reload fully."""
        with self.connect() as con:
            top = con.execute("SELECT COALESCE(MAX(seq), 0) FROM chunk_tombstones").fetchone()[0]
            floor = top - keep
            if floor <= 0:
                return 0
            removed = con.execute("DELETE FROM chunk_tombstones WHERE seq <= ?", (floor,)).rowcount
            con.execute(
                "INSERT INTO meta(key, value) VALUES('tombstone_floor', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(floor),),
            )
        return removed

    def optimize(self) -> None:
        """After a large sync: merge FTS segments, refresh planner statistics, trim the WAL."""
        with self.connect() as con:
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            con.execute("PRAGMA optimize")
        with self.connect() as con:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ------------------------------------------------------- image text cache

    def get_image_text(self, key: str, model: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT text FROM image_texts WHERE key = ? AND model = ?", (key, model)
            ).fetchone()
        return row["text"] if row else None

    def put_image_text(self, key: str, model: str, text: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT INTO image_texts(key, model, text, created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(key, model) DO UPDATE SET text = excluded.text",
                (key, model, text, utc_now()),
            )

    # ----------------------------------------------------------------- reads

    def experiment_state(self) -> dict[int, StoredState]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT id, modified_at, content_hash, index_sig, synced_at FROM experiments"
            ).fetchall()
        return {
            r[0]: StoredState(r[1] or "", r[2] or "", r[3] or "", r[4] or "") for r in rows
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
        with self.connect() as con:
            rows = con.execute(
                "SELECT c.cid, c.experiment_id, c.idx, c.text, e.title, e.date "
                "FROM chunks c JOIN experiments e ON e.id = c.experiment_id "
                "WHERE c.cid IN (SELECT value FROM json_each(?))",
                (id_list(cids),),
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
        with self.connect() as con:
            rows = con.execute(
                "SELECT c.cid, c.experiment_id, c.idx, c.text, e.title, e.date FROM chunks c "
                "JOIN experiments e ON e.id = c.experiment_id "
                "WHERE c.experiment_id IN (SELECT value FROM json_each(?)) "
                "ORDER BY c.experiment_id, c.idx",
                (id_list(experiment_ids),),
            ).fetchall()
        return [
            Hit(r["cid"], r["experiment_id"], r["idx"], r["text"], r["title"] or "",
                r["date"] or "", None, None, 0.0)
            for r in rows
        ]  # fmt: skip

    # --------------------------------------------------------------- vectors

    def vector_data(self, embed_model: str) -> VectorIndex:
        """In-memory vector index of one embedding model, kept current incrementally.

        Chunks embedded with a different model are ignored (dimensions would not
        match); ``doctor`` and ``stats`` report them. While another thread is
        refreshing the index, callers get the previous snapshot instead of waiting.
        """
        key = (str(self.path.resolve()), embed_model, self.vector_dims)
        revision = self.revision()
        with _cache_lock:
            current = _vector_cache.get(key)
        if current is not None and current.revision == revision:
            return current
        if current is not None:
            if not _load_lock.acquire(blocking=False):
                return current  # being refreshed right now: slightly stale is fine
        else:
            _load_lock.acquire()  # first load: everybody waits for the one load
        try:
            with _cache_lock:
                current = _vector_cache.get(key)
            fresh = self._refresh_vectors(embed_model, current)
            with _cache_lock:
                _vector_cache[key] = fresh
            return fresh
        finally:
            _load_lock.release()

    def _refresh_vectors(self, embed_model: str, current: VectorIndex | None) -> VectorIndex:
        with self.connect() as con:
            con.execute("BEGIN")  # one consistent read snapshot (WAL) for all queries below
            meta = dict(con.execute(
                "SELECT key, value FROM meta WHERE key IN "
                "('revision', 'vector_epoch', 'tombstone_floor')"
            ).fetchall())  # fmt: skip
            revision = int(meta.get("revision") or 0)
            epoch = meta.get("vector_epoch") or ""
            floor = int(meta.get("tombstone_floor") or 0)
            if current is not None and current.revision == revision:
                return current
            if (
                current is None
                or current.epoch != epoch
                or current.tomb_seq < floor
                or not current.full_dim
            ):
                return self._load_all_vectors(con, embed_model, epoch, revision)
            dead = con.execute(
                "SELECT seq, cid FROM chunk_tombstones WHERE seq > ? ORDER BY seq",
                (current.tomb_seq,),
            ).fetchall()
            new, max_cid = self._read_segment(
                con, embed_model, current.full_dim, current.dims, after=current.max_cid
            )
        return current.updated(
            new,
            np.asarray([r[1] for r in dead], dtype=np.int64),
            max_cid=max_cid,
            tomb_seq=dead[-1][0] if dead else current.tomb_seq,
            revision=revision,
        )

    def _load_all_vectors(
        self, con: sqlite3.Connection, embed_model: str, epoch: str, revision: int
    ) -> VectorIndex:
        head = con.execute(
            "SELECT dim, COUNT(*) AS n FROM chunks WHERE embed_model = ? "
            "AND embedding IS NOT NULL GROUP BY dim ORDER BY n DESC LIMIT 1",
            (embed_model,),
        ).fetchone()
        tomb_seq = con.execute("SELECT COALESCE(MAX(seq), 0) FROM chunk_tombstones").fetchone()[0]
        if head is None:
            return VectorIndex(embed_model, 0, 0, (), 0, tomb_seq, epoch, revision)
        full_dim = int(head["dim"])
        dims = self.vector_dims if 0 < self.vector_dims < full_dim else full_dim
        segment, max_cid = self._read_segment(con, embed_model, full_dim, dims, after=0)
        index = VectorIndex(
            embed_model, full_dim, dims, (segment,) if segment is not None else (),
            max_cid, tomb_seq, epoch, revision,
        )  # fmt: skip
        log.info(
            "Vector index loaded: %d chunks x %d dims (%.0f MB)",
            len(index), dims, index.nbytes / 1e6,
        )  # fmt: skip
        return index

    @staticmethod
    def _read_segment(
        con: sqlite3.Connection, embed_model: str, full_dim: int, dims: int, after: int
    ) -> tuple[Segment | None, int]:
        """Vectors with ``cid > after`` straight into a preallocated matrix.

        Avoids a Python list of per-row arrays (twice the memory at peak), which
        matters at hundreds of thousands of chunks. Only the first ``dims``
        floats of each vector are read when the index uses reduced dimensions.
        """
        head = con.execute(
            "SELECT COUNT(*), COALESCE(MAX(cid), 0) FROM chunks WHERE embed_model = ? "
            "AND dim = ? AND embedding IS NOT NULL AND cid > ?",
            (embed_model, full_dim, after),
        ).fetchone()
        total, max_cid = int(head[0]), int(head[1])
        if not total:
            return None, after
        cids = np.empty(total, dtype=np.int64)
        exp_ids = np.empty(total, dtype=np.int64)
        matrix = np.empty((total, dims), dtype=np.float32)
        cursor = con.execute(
            "SELECT cid, experiment_id, embedding FROM chunks WHERE embed_model = ? "
            "AND dim = ? AND embedding IS NOT NULL AND cid > ? ORDER BY cid",
            (embed_model, full_dim, after),
        )
        filled = 0
        while filled < total:
            batch = cursor.fetchmany(4096)
            if not batch:
                break
            for row in batch[: total - filled]:
                cids[filled] = row[0]
                exp_ids[filled] = row[1]
                matrix[filled] = np.frombuffer(row[2], dtype=np.float32, count=dims)
                filled += 1
        matrix = matrix[:filled]
        if dims < full_dim:  # truncated vectors must be re-normalised (in place)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix /= norms
        return Segment(cids[:filled], exp_ids[:filled], matrix), max_cid

    def similarities(self, embed_model: str, cids: Sequence[int], query: np.ndarray) -> dict[int, float]:
        """Exact cosine similarity of ``query`` to the full stored vectors of ``cids``."""
        if not len(cids):
            return {}
        with self.connect() as con:
            rows = con.execute(
                "SELECT cid, embedding FROM chunks WHERE embed_model = ? AND embedding IS NOT NULL "
                "AND cid IN (SELECT value FROM json_each(?))",
                (embed_model, id_list(cids)),
            ).fetchall()
        query = np.asarray(query, dtype=np.float32)
        result = {}
        for cid, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.shape == query.shape:
                result[int(cid)] = float(vector @ query)
        return result

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
