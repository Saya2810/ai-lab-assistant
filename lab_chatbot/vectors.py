"""In-memory vector index that follows the database incrementally.

The old cache reloaded *every* vector whenever anything in the index changed.
During a long sync the index changes after every experiment, so with hundreds
of thousands of chunks each question could trigger a reload of gigabytes.

This index is built from immutable *segments* (rows sorted by ``cid``):

* the first load reads all vectors once;
* afterwards only chunks with a higher ``cid`` are read (``cid`` is
  AUTOINCREMENT, so new chunks always have larger ids) and become a new segment;
* deleted chunks are found through the ``chunk_tombstones`` table and only
  flagged as dead;
* small segments are merged, and the whole index is compacted in memory when
  many rows are dead. The database is not read again for that.

Every refresh produces a new snapshot; a search keeps working on the snapshot it
started with, so readers never see a half-updated index and never wait for a
refresh that another thread is doing.

Optional reduced dimensions (``LAB_VECTOR_DIMS``): Matryoshka-trained embedding
models (embeddinggemma, nomic-embed-text v1.5, mxbai-embed-large, qwen3-embedding)
keep most of their quality when only the first dimensions are used. The index
then holds e.g. 256 instead of 768 floats per chunk (a third of the memory and
the search time); the best candidates are re-scored with their full vectors from
the database, so the final ranking uses full precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

MAX_SEGMENTS = 8  # more small segments than this are merged into one
COMPACT_DEAD_SHARE = 0.25  # rebuild without dead rows above this share


@dataclass(frozen=True)
class Segment:
    cids: np.ndarray  # int64, ascending
    exp_ids: np.ndarray  # int64
    matrix: np.ndarray  # float32 (n, dims), rows L2-normalised
    alive: np.ndarray | None = None  # bool mask; None = all rows alive

    def __len__(self) -> int:
        return len(self.cids)

    @property
    def n_alive(self) -> int:
        return len(self.cids) if self.alive is None else int(self.alive.sum())

    def without(self, dead_cids: np.ndarray) -> "Segment":
        """Copy with ``dead_cids`` flagged dead (the vectors are shared, not copied)."""
        if not len(self.cids) or not len(dead_cids):
            return self
        pos = np.searchsorted(self.cids, dead_cids)
        inside = pos < len(self.cids)
        pos, dead = pos[inside], dead_cids[inside]
        pos = pos[self.cids[pos] == dead]
        if not len(pos):
            return self
        alive = np.ones(len(self.cids), dtype=bool) if self.alive is None else self.alive.copy()
        alive[pos] = False
        return Segment(self.cids, self.exp_ids, self.matrix, alive)

    def compacted(self) -> "Segment":
        if self.alive is None or self.alive.all():
            return Segment(self.cids, self.exp_ids, self.matrix)
        keep = self.alive
        return Segment(self.cids[keep], self.exp_ids[keep], np.ascontiguousarray(self.matrix[keep]))


def merge(segments: list[Segment]) -> Segment:
    """One segment from several (dead rows dropped); cid order is kept."""
    parts = [s.compacted() for s in segments if s.n_alive]
    if not parts:
        dims = segments[0].matrix.shape[1] if segments else 0
        return Segment(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, dims), np.float32))
    if len(parts) == 1:
        return parts[0]
    return Segment(
        np.concatenate([p.cids for p in parts]),
        np.concatenate([p.exp_ids for p in parts]),
        np.concatenate([p.matrix for p in parts]),
    )


def _top(sims: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest values, best first (argpartition is O(n))."""
    take = min(k, len(sims))
    if take <= 0:
        return np.zeros(0, dtype=np.int64)
    top = np.argpartition(-sims, take - 1)[:take]
    return top[np.argsort(-sims[top], kind="stable")]


def reduce_dims(vectors: np.ndarray, dims: int) -> np.ndarray:
    """First ``dims`` components, re-normalised (Matryoshka truncation)."""
    if not dims or dims >= vectors.shape[-1]:
        return vectors
    cut = np.ascontiguousarray(vectors[..., :dims], dtype=np.float32)
    norms = np.linalg.norm(cut, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return cut / norms


@dataclass(frozen=True)
class VectorIndex:
    """Immutable snapshot of all vectors of one embedding model."""

    model: str
    full_dim: int  # dimension stored in the database
    dims: int  # dimension held in memory (== full_dim unless reduced)
    segments: tuple[Segment, ...] = ()
    max_cid: int = 0  # highest chunk id already loaded
    tomb_seq: int = 0  # highest tombstone already applied
    epoch: str = ""  # changes on reset(): forces a full reload
    revision: int = -1
    _concat: dict = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------- size

    def __len__(self) -> int:
        return sum(s.n_alive for s in self.segments)

    @property
    def reduced(self) -> bool:
        return self.dims < self.full_dim

    @property
    def nbytes(self) -> int:
        return sum(s.matrix.nbytes + s.cids.nbytes + s.exp_ids.nbytes for s in self.segments)

    # Concatenated views of the live rows: for tests and tools, not for the search path.
    def _merged(self) -> Segment:
        if "all" not in self._concat:
            self._concat["all"] = merge(list(self.segments)) if self.segments else merge([])
        return self._concat["all"]

    @property
    def cids(self) -> np.ndarray:
        return self._merged().cids

    @property
    def exp_ids(self) -> np.ndarray:
        return self._merged().exp_ids

    @property
    def matrix(self) -> np.ndarray:
        return self._merged().matrix

    # ----------------------------------------------------------- search

    def query_vector(self, query: np.ndarray) -> np.ndarray:
        return reduce_dims(np.asarray(query, dtype=np.float32), self.dims)

    def search(
        self, query: np.ndarray, k: int, experiment_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(cids, similarities)`` of the ``k`` nearest live rows, best first.

        ``query`` must already have ``self.dims`` dimensions (see ``query_vector``).
        ``experiment_ids`` restricts the search to those experiments (SQL candidates).
        """
        found_cids, found_sims = [], []
        for segment in self.segments:
            if not len(segment):
                continue
            mask = segment.alive
            if experiment_ids is not None:
                wanted = np.isin(segment.exp_ids, experiment_ids)
                mask = wanted if mask is None else (mask & wanted)
            if mask is None:
                cids, sims = segment.cids, segment.matrix @ query
            else:
                rows = np.flatnonzero(mask)
                if not len(rows):
                    continue
                if len(rows) * 2 > len(segment):
                    # most rows wanted: one full product is cheaper than copying them
                    cids, sims = segment.cids[rows], (segment.matrix @ query)[rows]
                else:
                    cids, sims = segment.cids[rows], segment.matrix[rows] @ query
            top = _top(sims, k)
            found_cids.append(cids[top])
            found_sims.append(sims[top])
        if not found_cids:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        cids, sims = np.concatenate(found_cids), np.concatenate(found_sims)
        top = _top(sims, k)
        return cids[top], sims[top]

    # ---------------------------------------------------------- updates

    def updated(
        self,
        new: Segment | None,
        dead_cids: np.ndarray,
        *,
        max_cid: int,
        tomb_seq: int,
        revision: int,
    ) -> "VectorIndex":
        """New snapshot with ``new`` appended and ``dead_cids`` removed."""
        dead = np.unique(np.asarray(dead_cids, dtype=np.int64))
        segments = [s.without(dead) for s in self.segments]
        if new is not None and len(new):
            segments.append(new.without(dead))
        segments = self._maintain(segments)
        return VectorIndex(
            self.model, self.full_dim, self.dims, tuple(segments),
            max_cid=max(max_cid, self.max_cid), tomb_seq=max(tomb_seq, self.tomb_seq),
            epoch=self.epoch, revision=revision,
        )  # fmt: skip

    @staticmethod
    def _maintain(segments: list[Segment]) -> list[Segment]:
        total = sum(len(s) for s in segments)
        alive = sum(s.n_alive for s in segments)
        if total and (total - alive) / total > COMPACT_DEAD_SHARE:
            log.info("Vector index: compacting %d rows (%d dead)", total, total - alive)
            return [merge(segments)]
        if len(segments) > MAX_SEGMENTS:
            # keep the big base segment, merge the small delta segments behind it
            return [segments[0], merge(segments[1:])]
        return segments
