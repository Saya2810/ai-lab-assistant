"""Benchmark the chatbot on question sets and keep every per-question detail.

``evaluation.py`` gives a pass/fail summary for one set. This module records what a
scientific report needs (``report.py`` draws it):

* the ranked experiments per question -> Hit@k / Recall@k curves, MRR, nDCG@10;
* the similarity of every chunk to the question, split into chunks of relevant and
  irrelevant experiments -> score distributions, how well a threshold separates them;
* the best score per question for answerable *and* unanswerable questions (sets whose
  cases expect no experiment) -> ROC curve / AUROC of "is this in the notebook at all";
* planned vs expected intent -> confusion matrix of the router;
* timings per step -> latency distributions;
* the same questions under retrieval variants (dense only, keywords only, without
  synonyms, ...) -> ablation: which component contributes how much;
* optionally the generated answers -> citation rate, answers with numbers that appear
  in no source, expected content found.

Every question is planned once with the real planner; the variants then reuse that plan,
so differences come from retrieval alone.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from lab_chatbot.engine import Engine
from lab_chatbot.evaluation import Case, load_cases
from lab_chatbot.plan import STRUCTURED_INTENTS
from lab_chatbot.retrieval import hybrid_search
from lab_chatbot.sql_tools import candidate_ids

DEMO_SETS = {
    "standard": "golden_demo.json",
    "paraphrase": "eval_demo_paraphrase.json",
    "unanswerable": "eval_demo_negative.json",
}
K_MAX = 10


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    options: dict


VARIANTS = [
    Variant("hybrid", "full system: meaning + keywords + synonyms + both wordings", {}),
    Variant("no_synonyms", "without the planner's synonyms/translations", {"use_keywords": False}),
    Variant("one_wording", "only the planner's search phrase, not the user's words", {"use_alt": False}),
    Variant("dense_only", "meaning (embeddings) only, no keyword search", {"lexical_k": 0}),
    Variant("keywords_only", "keyword search (BM25) only, no embeddings", {"dense": False}),
]


def demo_sets() -> dict[str, Path]:
    here = Path(__file__).parent
    return {name: here / file for name, file in DEMO_SETS.items()}


# ---------------------------------------------------------------------------
# metrics (binary relevance)
# ---------------------------------------------------------------------------


def first_rank(ranked: Sequence[int], relevant: set[int]) -> int | None:
    for position, experiment_id in enumerate(ranked, start=1):
        if experiment_id in relevant:
            return position
    return None


def ndcg_at(ranked: Sequence[int], relevant: set[int], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, e in enumerate(ranked[:k]) if e in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def retrieval_metrics(rows: list[dict]) -> dict:
    """Hit@k curve, MRR and nDCG@10 over answerable questions (``rows`` from ``run``)."""
    rows = [r for r in rows if r["expect_ids"]]
    if not rows:
        return {}
    ranks = [r["rank"] for r in rows]
    return {
        "n": len(rows),
        "hit_at": {k: sum(1 for x in ranks if x is not None and x <= k) / len(rows) for k in range(1, K_MAX + 1)},
        "mrr": sum(1 / x for x in ranks if x) / len(rows),
        "ndcg10": sum(ndcg_at(r["ranked"], set(r["expect_ids"])) for r in rows) / len(rows),
    }


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float] = np.mean,
                 n: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:  # fmt: skip
    """Percentile bootstrap interval; with few questions the intervals are wide, and that is the point."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    data = list(values)
    stats = sorted(stat([rng.choice(data) for _ in data]) for _ in range(n))
    return stats[int(alpha / 2 * n)], stats[int((1 - alpha / 2) * n) - 1]


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Probability that an answerable question scores higher than an unanswerable one."""
    if not positive or not negative:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in positive for q in negative)
    return wins / (len(positive) * len(negative))


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def _search(engine: Engine, plan, candidates, *, dense=True, lexical_k=30, use_keywords=True,
            use_alt=True, min_dense=None):  # fmt: skip
    cfg = engine.config
    return hybrid_search(
        engine.store, engine.embedder, plan.semantic_query or plan.question,
        experiment_ids=candidates, k=K_MAX, per_experiment=1, lexical_k=lexical_k,
        min_dense=(cfg.min_dense if min_dense is None else min_dense) if dense else 9.0,
        alt_queries=[plan.question] if use_alt else (),
        keywords=plan.keywords if use_keywords else (),
    )  # fmt: skip


def _chunk_scores(engine: Engine, plan, relevant: set[int]) -> tuple[list[float], list[float], float]:
    """Similarity of every chunk to the question (best of both wordings), split by relevance."""
    index = engine.store.vector_data(engine.embedder.model)
    if not len(index):
        return [], [], 0.0
    best: dict[int, float] = {}
    for wording in dict.fromkeys(q for q in (plan.semantic_query, plan.question) if q):
        vector = engine.embedder.embed_query(wording)
        cids, sims = index.search(index.query_vector(vector), len(index))
        for cid, sim in zip(cids.tolist(), sims.tolist()):
            best[cid] = max(best.get(cid, -1.0), sim)
    owners = dict(zip(index.cids.tolist(), index.exp_ids.tolist()))
    rel = [s for cid, s in best.items() if owners.get(cid) in relevant]
    irr = [s for cid, s in best.items() if owners.get(cid) not in relevant]
    return rel, irr, max(best.values(), default=0.0)


def run(
    engine: Engine,
    sets: dict[str, list[Case]],
    *,
    answers: bool = False,
    variants: Sequence[Variant] = VARIANTS,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Run every question of every set; returns a JSON-serialisable result."""
    questions: list[dict] = []
    for set_name, cases in sets.items():
        for case in cases:
            if progress:
                progress(f"{set_name}: {case.question}")
            questions.append(_run_one(engine, set_name, case, answers, variants))
    by_set = {}
    for set_name in sets:
        rows = [q for q in questions if q["set"] == set_name]
        by_set[set_name] = {
            "retrieval": retrieval_metrics(rows),
            "variants": {
                v.name: retrieval_metrics([{**q, **q["variants"][v.name]} for q in rows]) for v in variants
            },
            "false_answers": _false_answer_rate(rows, variants),
        }
    answerable = [q["max_score"] for q in questions if q["expect_ids"]]
    unanswerable = [q["max_score"] for q in questions if not q["expect_ids"] and q["unanswerable"]]
    return {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "config": {
            "chat_model": engine.config.chat_model, "embed_model": engine.config.embed_model,
            "min_dense": engine.config.min_dense, "top_k": engine.config.top_k,
            "experiments": engine.store.experiment_count(),
        },  # fmt: skip
        "variants": [{"name": v.name, "description": v.description} for v in variants],
        "questions": questions,
        "sets": by_set,
        "auroc": auroc(answerable, unanswerable),
    }


def _false_answer_rate(rows: list[dict], variants: Sequence[Variant]) -> dict:
    """Share of unanswerable questions that still got experiments, per variant."""
    rows = [r for r in rows if r["unanswerable"]]
    if not rows:
        return {}
    return {v.name: sum(bool(r["variants"][v.name]["ranked"]) for r in rows) / len(rows) for v in variants}


def _run_one(engine: Engine, set_name: str, case: Case, answers: bool, variants: Sequence[Variant]) -> dict:
    relevant = set(case.expect_ids)
    row = {
        "set": set_name, "question": case.question, "expect_ids": sorted(relevant),
        "unanswerable": not relevant and not case.expect_intent and not case.expect_contains,
        "expect_intent": case.expect_intent, "intent": "", "ranked": [], "rank": None,
        "latency_ms": {}, "variants": {}, "error": None,
    }  # fmt: skip
    try:
        started = time.perf_counter()
        plan = engine.planner.plan(case.question, [])
        row["latency_ms"]["plan"] = round((time.perf_counter() - started) * 1000, 1)
        row["intent"] = plan.intent
        started = time.perf_counter()
        candidates = candidate_ids(engine.store, plan.filters, plan.measurement)
        structured = engine._structured(plan) if plan.intent in STRUCTURED_INTENTS else None
        if structured is not None:
            ranked = structured.experiment_ids[:K_MAX]
        elif candidates is not None and not candidates:
            ranked = []
        else:
            ranked = list(dict.fromkeys(h.experiment_id for h in _search(engine, plan, candidates)))
        row["latency_ms"]["retrieve"] = round((time.perf_counter() - started) * 1000, 1)
        row["ranked"], row["rank"] = ranked, first_rank(ranked, relevant)

        rel, irr, row["max_score"] = _chunk_scores(engine, plan, relevant)
        row["scores_relevant"], row["scores_irrelevant"] = rel, irr
        for variant in variants:
            if structured is not None:  # SQL answers do not depend on the retrieval variant
                variant_ranked = ranked
            elif candidates is not None and not candidates:
                variant_ranked = []
            else:
                hits = _search(engine, plan, candidates, **variant.options)
                variant_ranked = list(dict.fromkeys(h.experiment_id for h in hits))
            row["variants"][variant.name] = {
                "ranked": variant_ranked, "rank": first_rank(variant_ranked, relevant)
            }

        if answers:
            answer = engine.ask(case.question)
            row["latency_ms"].update({f"answer_{k}": v for k, v in answer.latencies.items()})
            row["answer"] = {
                "used_llm": answer.used_llm,
                "n_sources": len(answer.sources),
                "cited": len(answer.cited),
                "unsupported_numbers": any("Zahlen" in w or "numbers" in w for w in answer.warnings),
                "contains_ok": (
                    all(s.lower() in answer.text.lower() for s in case.expect_contains)
                    if case.expect_contains else None
                ),  # fmt: skip
                "text": answer.text[:600],
            }
    except Exception as exc:  # one broken question must not stop the benchmark
        row["error"] = f"{type(exc).__name__}: {exc}"
        row.setdefault("max_score", 0.0)
        row.setdefault("scores_relevant", [])
        row.setdefault("scores_irrelevant", [])
        for variant in variants:
            row["variants"].setdefault(variant.name, {"ranked": [], "rank": None})
    return row


def save(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_sets(names_or_paths: Sequence[str] | None = None) -> dict[str, list[Case]]:
    """``standard``/``paraphrase``/``unanswerable`` (the demo sets) or ``name=path.json``."""
    available = demo_sets()
    chosen = names_or_paths or list(available)
    sets: dict[str, list[Case]] = {}
    for item in chosen:
        name, _, path = item.partition("=")
        sets[name] = load_cases(path or available[name])
    return sets
