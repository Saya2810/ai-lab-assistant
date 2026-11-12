from __future__ import annotations

from lab_chatbot.retrieval import (
    _lexical_ok,
    fts_match_expression,
    hybrid_search,
    query_tokens,
)


def search(store, embedder, query, **kw):
    return hybrid_search(store, embedder, query, **kw)


def test_query_tokens_drop_stopwords_and_duplicates():
    assert query_tokens("Warum war das Rauschen bei Probe A3 so hoch?") == ["rausch", "prob", "a3", "hoch"]
    assert query_tokens("laser Laser LASER") == ["lase"]
    assert query_tokens("532nm P7 1042") == ["532nm", "p7", "1042"]  # identifiers stay exact


def test_fts_expression_quotes_and_prefixes():
    expr = fts_match_expression(["rausch", "a3", 'ab"c', "532nm"])
    assert expr == '"rausch"* OR "a3" OR "ab""c"* OR "532nm"'  # words are prefixes, quotes escaped


def test_finds_the_right_experiment(demo_store, embedder):
    hits = search(demo_store, embedder, "Rauschen Probe A3 Fluoreszenzuntergrund")
    assert hits and hits[0].experiment_id == 102


def test_exact_identifier_is_found_lexically(demo_store, embedder):
    hits = search(demo_store, embedder, "P7 Primerpaar")
    assert hits[0].experiment_id == 103 and hits[0].lexical_rank == 1


def test_prefix_matching_handles_inflection(demo_store, embedder):
    hits = search(demo_store, embedder, "nachjustierte Fasern")
    assert 101 in {h.experiment_id for h in hits}


def test_gibberish_returns_nothing(demo_store, embedder):
    assert search(demo_store, embedder, "xyzzy plugh qwerty") == []


def test_one_coincidental_word_is_not_enough(demo_store, embedder):
    # "Raman" appears in the notebook, the rest of the question is unrelated
    hits = search(demo_store, embedder, "Quantenchromodynamik Gluonen Confinement Raman", min_dense=0.99)
    assert hits == []


def test_candidate_restriction(demo_store, embedder):
    hits = search(demo_store, embedder, "Rauschen", experiment_ids={105})
    assert hits and {h.experiment_id for h in hits} == {105}
    assert search(demo_store, embedder, "Rauschen", experiment_ids=set()) == []


def test_per_experiment_cap(demo_store, embedder):
    hits = search(demo_store, embedder, "Raman Messung Probe Laser Belichtung", k=20, per_experiment=1)
    exp_ids = [h.experiment_id for h in hits]
    assert len(exp_ids) == len(set(exp_ids))


def test_results_are_sorted_by_fused_score(demo_store, embedder):
    hits = search(demo_store, embedder, "Laser Leistung Kalibrierung", k=10)
    scores = [h.fused for h in hits]
    assert scores == sorted(scores, reverse=True) and len(hits) <= 10


def test_umlaut_and_eszett_folding(demo_store, embedder):
    assert _lexical_ok(["grosse", "probe"], "Die Größe der Probe")
    assert not _lexical_ok(["hund", "katze", "maus"], "Die Größe der Probe")


def test_works_on_empty_index(store, embedder):
    assert search(store, embedder, "irgendwas") == []
