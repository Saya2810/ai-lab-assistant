"""Hybrid retrieval: dense (meaning) + BM25 (exact words), fused with RRF.

Dense search alone is poor at exact tokens such as sample names, "A3" or
"532nm"; keyword search alone misses paraphrases. Both run inside the
candidate set chosen by the SQL filters, then reciprocal-rank fusion merges the
two rankings.

Relevance gating (the old code always passed 5 entries to the LLM):

* a dense hit must reach ``min_dense`` cosine similarity;
* a keyword-only hit must share enough of the query's words with the chunk;
* at most ``per_experiment`` chunks per experiment, so one long entry cannot
  crowd out everything else.
If nothing passes, the caller answers "nothing found" *without* calling the LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from lab_chatbot.embeddings import Embedder
from lab_chatbot.store import Store

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
    min_dense: float = 0.30,
    rrf_k: int = 60,
    per_experiment: int = 2,
) -> list[Hit]:
    """Best chunks for ``query`` (optionally only inside ``experiment_ids``)."""
    if experiment_ids is not None and not experiment_ids:
        return []

    # ---- dense ranking
    dense_scores: dict[int, float] = {}
    dense_rank: list[int] = []
    data = store.vector_data(embedder.model)
    query_vec = None
    if len(data.cids):
        if experiment_ids is None:
            rows_idx = None
        else:
            wanted = np.fromiter(experiment_ids, dtype=np.int64, count=len(experiment_ids))
            rows_idx = np.flatnonzero(np.isin(data.exp_ids, wanted))
        if rows_idx is None or len(rows_idx):
            query_vec = embedder.embed_query(query)
            if rows_idx is None:
                cids, sims = data.cids, data.matrix @ query_vec
            elif len(rows_idx) * 2 > len(data.cids):
                # large candidate set: one full product is cheaper than copying most rows
                cids, sims = data.cids[rows_idx], (data.matrix @ query_vec)[rows_idx]
            else:
                cids, sims = data.cids[rows_idx], data.matrix[rows_idx] @ query_vec
            # Only the top candidates are needed: argpartition is O(n), a full sort O(n log n).
            take = min(dense_k, len(sims))
            top = np.argpartition(-sims, take - 1)[:take]
            top = top[np.argsort(-sims[top], kind="stable")]
            dense_scores = {int(cids[i]): float(sims[i]) for i in top}
            dense_rank = [int(cids[i]) for i in top if sims[i] >= min_dense]

    # ---- lexical ranking (BM25)
    tokens = query_tokens(query)
    lexical = store.fts_search(fts_match_expression(tokens), lexical_k, experiment_ids)
    lexical_cids = [cid for cid, _ in lexical]
    rows = store.chunk_rows(sorted(set(dense_rank) | set(lexical_cids)))
    lexical_rank = [cid for cid in lexical_cids if cid in rows and _lexical_ok(tokens, rows[cid].text)]
    missing = [cid for cid in lexical_rank if cid not in dense_scores]
    if missing and query_vec is not None:
        # data.cids is sorted (ORDER BY cid), so the rows of keyword hits are found by bisection
        wanted = np.asarray(missing, dtype=np.int64)
        pos = np.searchsorted(data.cids, wanted)
        pos = np.clip(pos, 0, len(data.cids) - 1)
        found = data.cids[pos] == wanted
        if found.any():
            sims = data.matrix[pos[found]] @ query_vec
            dense_scores.update({int(c): float(v) for c, v in zip(wanted[found], sims)})

    # ---- reciprocal rank fusion
    fused: dict[int, float] = {}
    for ranking in (dense_rank, lexical_rank):
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    hits: list[Hit] = []
    per_exp: dict[int, int] = {}
    for cid, score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
        row = rows.get(cid)
        if row is None:
            continue
        if per_exp.get(row.experiment_id, 0) >= per_experiment:
            continue
        per_exp[row.experiment_id] = per_exp.get(row.experiment_id, 0) + 1
        hits.append(
            Hit(
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
        )
        if len(hits) >= k:
            break
    return hits
