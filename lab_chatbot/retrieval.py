"""Hybrid retrieval: dense (meaning) + BM25 (exact words), fused with RRF.

Dense search alone is poor at exact tokens such as sample names, "A3" or
"532nm"; keyword search alone misses paraphrases. Both run inside the
candidate set chosen by the SQL filters, then reciprocal-rank fusion merges the
two rankings.

Relevance gating (the old code always passed 5 entries to the LLM):

* a hit found only by meaning must reach ``min_dense`` cosine similarity; a hit
  found only through the planner's synonyms needs a little less, plus that match;
* a keyword-only hit must share enough of the query's words with the chunk;
* at most ``per_experiment`` chunks per experiment, so one long entry cannot
  crowd out everything else.
If nothing passes, the caller answers "nothing found" *without* calling the LLM.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from lab_chatbot.embeddings import Embedder
from lab_chatbot.store import Store

if TYPE_CHECKING:
    from lab_chatbot.rerank import Reranker

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    """
    a an and are as at be but by can did do does for from had has have how i if in into is it its
    me my of on or our show so than that the their them then there these they this to us was we
    were what when where which who why will with would you your tell give list find
    aber alle allen als also am an auch auf aus bei bin bis bitte da dann das dass dem den der des
    die dies diese dieser dieses doch dort du durch ein eine einem einen einer eines er es für
    gab gibt hat hatte hatten hier ich ihr im in ist ja kann mal man mein mir mit nach nicht noch
    nun nur ob oder sehr sein sich sie sind so über um und uns von vom von war waren was welche
    welcher welches wenn wer wie wieder wir wird wurde wurden zu zum zur zeig zeige
    warum wieso weshalb wozu womit wofür woher wohin viele wieviele
    """.split()
)  # fmt: skip

_WORD = re.compile(r"\w+", re.UNICODE)


def fold(text: str) -> str:
    """Lower-case and strip diacritics, so 'Größe' matches 'grosse'."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def stem(token: str) -> str:
    """Crude suffix trimming so inflected forms meet: nachjustierte ~ nachjustieren, Fasern ~ Faser.

    Identifiers (anything with a digit: A3, 532nm, P7) are never touched. The result is
    used as a *prefix* in the keyword search; the relevance gate and the dense channel
    keep the occasional over-match (prob* ~ Problem) from reaching the answer.
    """
    if any(ch.isdigit() for ch in token) or len(token) < 5:
        return token
    cut = 1 if len(token) <= 6 else 2 if len(token) <= 8 else 3
    return token[:-cut]


def query_tokens(query: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(query):
        token = word.casefold()
        if token in STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        token = stem(token)
        if token not in tokens:
            tokens.append(token)
    return tokens


def fts_match_expression(tokens: list[str]) -> str:
    """OR of quoted tokens (no FTS syntax injection); words match as prefix, identifiers exactly."""
    parts = []
    for token in tokens:
        quoted = '"' + token.replace('"', '""') + '"'
        is_word = len(token) >= 4 and not any(ch.isdigit() for ch in token)
        parts.append(quoted + "*" if is_word else quoted)
    return " OR ".join(parts)


@dataclass
class Hit:
    cid: int
    experiment_id: int
    idx: int
    text: str
    title: str
    date: str
    dense: float | None
    lexical_rank: int | None
    fused: float
    rerank: float | None = None


def _lexical_ok(tokens: list[str], text: str) -> bool:
    haystack = fold(text)
    matched = sum(1 for t in tokens if fold(t) in haystack)
    return matched >= 2 or matched / max(1, len(tokens)) >= 0.5


def hybrid_search(
    store: Store,
    embedder: Embedder,
    query: str,
    *,
    experiment_ids: set[int] | None = None,
    k: int = 6,
    dense_k: int = 30,
    lexical_k: int = 30,
    min_dense: float = 0.35,
    support_margin: float = 0.10,
    alt_queries: Sequence[str] = (),
    keywords: Sequence[str] = (),
    rrf_k: int = 60,
    per_experiment: int = 2,
    reranker: Reranker | None = None,
    rerank_pool: int = 40,
) -> list[Hit]:
    """Best chunks for ``query`` (optionally only inside ``experiment_ids``).

    ``alt_queries`` are other wordings of the same question (e.g. the user's original
    question next to the planner's search phrase); a chunk counts with its best
    similarity to any of them. ``keywords`` (synonyms, translations) widen only the
    keyword search, so "Schaden" can also find "verbrannt".

    Relevance gate: a chunk found *only* by meaning needs ``min_dense``. A keyword match
    on the user's own words always counts (sample names, identifiers). A match on the
    planner's synonyms alone is weaker evidence: it needs at least
    ``min_dense - support_margin / 2`` of meaning as well.
    """
    if experiment_ids is not None and not experiment_ids:
        return []

    # ---- dense ranking (best similarity over all wordings)
    dense_scores: dict[int, float] = {}
    dense_rank: list[int] = []
    index = store.vector_data(embedder.model)
    query_vecs: list[np.ndarray] = []
    if len(index):
        wanted = None
        if experiment_ids is not None:
            wanted = np.fromiter(experiment_ids, dtype=np.int64, count=len(experiment_ids))
        wordings = list(dict.fromkeys(q.strip() for q in (query, *alt_queries) if q and q.strip()))
        best: dict[int, float] = {}
        for wording in wordings:
            query_vec = embedder.embed_query(wording)
            query_vecs.append(query_vec)
            # With reduced dimensions the in-memory search is approximate: take a wider
            # pool and re-score it with the full vectors, so the final order is exact.
            pool = dense_k * 4 if index.reduced else dense_k
            cids, sims = index.search(index.query_vector(query_vec), pool, wanted)
            if index.reduced and len(cids):
                scored = store.similarities(embedder.model, cids.tolist(), query_vec).items()
            else:
                scored = ((int(c), float(v)) for c, v in zip(cids, sims))
            for cid, sim in scored:
                best[cid] = max(best.get(cid, -1.0), sim)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:dense_k]
        dense_scores = dict(ranked)
        dense_rank = [cid for cid, sim in ranked if sim >= min_dense - support_margin]

    # ---- lexical ranking (BM25), widened by the keywords
    base_tokens = query_tokens(query)
    tokens = list(base_tokens)
    for keyword in keywords:
        tokens += [t for t in query_tokens(keyword) if t not in tokens]
    lexical = store.fts_search(fts_match_expression(tokens), lexical_k, experiment_ids)
    lexical_cids = [cid for cid, _ in lexical]
    rows = store.chunk_rows(sorted(set(dense_rank) | set(lexical_cids)))
    lexical_rank = [cid for cid in lexical_cids if cid in rows and _lexical_ok(tokens, rows[cid].text)]
    missing = [cid for cid in lexical_rank if cid not in dense_scores]
    for query_vec in query_vecs:
        for cid, sim in store.similarities(embedder.model, missing, query_vec).items():
            dense_scores[cid] = max(dense_scores.get(cid, -1.0), sim)

    # ---- relevance gate
    # strong: the user's own words match;  weak: only synonyms/translations match
    strong = {cid for cid in lexical_rank if _lexical_ok(base_tokens, rows[cid].text)}

    def relevant(cid: int) -> bool:
        dense = dense_scores.get(cid, -1.0)
        if cid in strong or dense >= min_dense:
            return True  # the user's own words (e.g. a sample name like "A3") always count
        # matched only through the planner's synonyms: weaker evidence, higher bar
        return cid in lexical_rank and dense >= min_dense - support_margin / 2

    dense_rank = [cid for cid in dense_rank if relevant(cid)]
    lexical_rank = [cid for cid in lexical_rank if relevant(cid)]

    # ---- reciprocal rank fusion
    fused: dict[int, float] = {}
    for ranking in (dense_rank, lexical_rank):
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    def make_hit(cid: int, score: float) -> Hit:
        row = rows[cid]
        return Hit(
            cid=cid,
            experiment_id=row.experiment_id,
            idx=row.idx,
            text=row.text,
            title=row.title,
            date=row.date,
            dense=dense_scores.get(cid),
            lexical_rank=(lexical_rank.index(cid) + 1) if cid in lexical_rank else None,
            fused=score,
        )

    ordered = [(cid, score) for cid, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])) if cid in rows]

    if reranker is not None and len(ordered) > 1:
        pool = [make_hit(cid, score) for cid, score in ordered[: max(rerank_pool, k)]]
        try:
            scores = reranker.score(query, [h.text for h in pool])
            if len(scores) != len(pool):
                raise ValueError(f"reranker returned {len(scores)} scores for {len(pool)} texts")
            scored = [replace(h, rerank=float(s)) for h, s in zip(pool, scores)]
            scored.sort(key=lambda h: -h.rerank)  # stable: ties keep the fused order
            return _cap_and_limit(scored, per_experiment, k)
        except Exception as exc:  # retrieval must never fail because of the reranker
            logger.warning("Reranking failed, using fused order: %s", exc)

    return _cap_and_limit([make_hit(cid, score) for cid, score in ordered], per_experiment, k)


def _cap_and_limit(hits: list[Hit], per_experiment: int, k: int) -> list[Hit]:
    out: list[Hit] = []
    per_exp: dict[int, int] = {}
    for hit in hits:
        if per_exp.get(hit.experiment_id, 0) >= per_experiment:
            continue
        per_exp[hit.experiment_id] = per_exp.get(hit.experiment_id, 0) + 1
        out.append(hit)
        if len(out) >= k:
            break
    return out
