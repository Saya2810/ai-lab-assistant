from __future__ import annotations

import numpy as np
import pytest

from lab_chatbot.chunking import chunk_experiment
from lab_chatbot.demo import DEMO_EXPERIMENTS
from lab_chatbot.parse import parse_experiment
from lab_chatbot.sql_tools import (
    Filters,
    MeasurementCond,
    aggregate_measurement,
    candidate_ids,
    count_experiments,
    list_experiments,
    load_catalog,
    measurement_values,
)
from lab_chatbot.sync import index_signature
from tests.fakes import FakeEmbedder


def M(name, op=None, value=None, unit=None, value2=None) -> MeasurementCond:
    """Keyword-safe constructor (the dataclass order is name, op, value, value2, unit)."""
    return MeasurementCond(name, op, value, value2, unit)


def ids(store, filters=None, meas=None):
    return sorted(candidate_ids(store, filters or Filters(), meas) or [])


# ---------------------------------------------------------------- store


def test_replace_experiment_leaves_no_stale_chunks(store, embedder):
    record = parse_experiment(DEMO_EXPERIMENTS[0])
    sig = index_signature(embedder)
    long_chunks = chunk_experiment(record, max_chars=250)
    store.replace_experiment(record, long_chunks, embedder.embed_documents([c.text for c in long_chunks]),
                             embed_model=embedder.model, index_sig=sig)  # fmt: skip
    before = store.stats()["chunks"]
    assert before > 1

    record.body = "Komplett neuer kurzer Text."
    record.content_hash = record.compute_hash()
    short = chunk_experiment(record)
    store.replace_experiment(record, short, embedder.embed_documents([c.text for c in short]),
                             embed_model=embedder.model, index_sig=sig)  # fmt: skip
    assert store.stats()["chunks"] == len(short) < before
    assert store.fts_search('"Faser*"') == [] or all(
        "Komplett" in store.chunk_rows([cid])[cid].text for cid, _ in store.fts_search('"komplett"')
    )
    assert store.fts_search('"nachjustieren"') == []  # the old text is gone from FTS too


def test_delete_removes_everything(demo_store):
    n = demo_store.experiment_count()
    demo_store.delete_experiments([101, 102])
    assert demo_store.experiment_count() == n - 2
    assert ids(demo_store, Filters(experiment_ids=[101, 102])) == []
    assert all(demo_store.chunk_rows([cid]) for cid, _ in demo_store.fts_search('"raman"'))
    for cid, _ in demo_store.fts_search('"raman"'):
        assert demo_store.chunk_rows([cid])[cid].experiment_id not in (101, 102)


def test_vector_cache_is_invalidated_by_changes(demo_store, embedder):
    first = demo_store.vector_data(embedder.model)
    assert demo_store.vector_data(embedder.model) is first  # cached
    demo_store.delete_experiments([101])
    second = demo_store.vector_data(embedder.model)
    assert second is not first and len(second.cids) < len(first.cids)
    assert 101 not in set(second.exp_ids.tolist())


def test_vectors_of_other_models_are_ignored(demo_store):
    assert len(demo_store.vector_data("some-other-model").cids) == 0


def test_stored_vectors_round_trip_as_float32(demo_store, embedder):
    data = demo_store.vector_data(embedder.model)
    assert data.matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(data.matrix, axis=1), 1.0, atol=1e-5)


def test_malformed_fts_expression_is_not_an_error(demo_store):
    assert demo_store.fts_search('"unclosed AND (') == []


def test_reset_clears_everything(demo_store):
    demo_store.reset()
    assert demo_store.experiment_count() == 0 and demo_store.stats()["chunks"] == 0


def test_experiment_state_reports_hash_and_signature(demo_store, embedder):
    state = demo_store.experiment_state()
    assert set(state) == {e["id"] for e in DEMO_EXPERIMENTS}
    assert state[101].index_sig == index_signature(embedder)


# ---------------------------------------------------------------- filters


def test_filter_by_tags_all_and_any(demo_store):
    assert ids(demo_store, Filters(tags_all=["raman"])) == [102, 105]
    assert ids(demo_store, Filters(tags_all=["raman", "laser"])) == []
    assert ids(demo_store, Filters(tags_any=["pcr", "puffer"])) == [103, 106]


def test_tag_matching_is_case_insensitive(demo_store):
    assert ids(demo_store, Filters(tags_all=["RAMAN"])) == [102, 105]


def test_filter_by_date_range_category_status(demo_store):
    assert ids(demo_store, Filters(date_from="2026-10-05", date_to="2026-10-12")) == [103, 104, 105]
    assert ids(demo_store, Filters(category="Biologie")) == [103, 104]
    assert ids(demo_store, Filters(status="In Arbeit")) == [103, 108]
    assert ids(demo_store, Filters(category="Messung", status="In Arbeit")) == [108]


def test_empty_filter_means_no_restriction(demo_store):
    assert candidate_ids(demo_store, Filters()) is None


def test_hostile_strings_are_just_values(demo_store):
    evil = "x'; DROP TABLE experiments; --"
    assert ids(demo_store, Filters(tags_all=[evil], category=evil, author=evil)) == []
    assert demo_store.experiment_count() == len(DEMO_EXPERIMENTS)


# ---------------------------------------------------------- measurements


def test_numeric_comparison_across_units(demo_store):
    # 785 nm in two entries; 0.5 µm == 500 nm so only >= 500 nm matches
    assert ids(demo_store, meas=M("Wellenlänge", ">", 600, "nm")) == [102, 105]
    assert ids(demo_store, meas=M("Wellenlänge", ">", 0.6, "µm")) == [102, 105]
    assert ids(demo_store, meas=M("Wellenlänge", "<", 500, "nm")) == [108]
    assert ids(demo_store, meas=M("Wellenlänge", "=", 532, "nm")) == [101, 107]
    assert ids(demo_store, meas=M("Wellenlänge", "between", 500, "nm", 600)) == [101, 107]


def test_milliseconds_compare_with_seconds(demo_store):
    # 200 ms (108) < 1 s < 5 s (101) ...
    assert ids(demo_store, meas=M("Belichtungszeit", "<", 1, "s")) == [108]
    assert ids(demo_store, meas=M("Belichtungszeit", ">=", 10, "s")) == [102, 105]


def test_temperature_offsets(demo_store):
    assert ids(demo_store, meas=M("Annealing Temperatur", ">", 300, "K")) == [103]  # 58 °C = 331 K
    assert ids(demo_store, meas=M("Annealing Temperatur", "<", 50, "°C")) == []


def test_unit_free_measurement_compares_raw_numbers(demo_store):
    assert ids(demo_store, meas=M("Zyklen", ">", 30)) == [103]
    assert ids(demo_store, meas=M("pH", "between", 7, None, 8)) == [106]


def test_incompatible_dimension_never_matches(demo_store):
    # asking for wavelength in seconds must not return anything
    assert ids(demo_store, meas=M("Wellenlänge", ">", 1, "s")) == []


def test_measurement_name_is_normalised(demo_store):
    assert ids(demo_store, meas=M("wellenlänge", ">", 600, "nm")) == [102, 105]


def test_invalid_conditions_raise(demo_store):
    with pytest.raises(ValueError):
        ids(demo_store, meas=M("pH", "LIKE", 7))
    with pytest.raises(ValueError):
        ids(demo_store, meas=M("pH", ">", None))
    with pytest.raises(ValueError):
        ids(demo_store, meas=M("pH", "between", 7))


def test_filters_and_measurement_combine(demo_store):
    meas = M("Laserleistung", ">", 30, "mW")
    assert ids(demo_store, Filters(tags_all=["raman"]), meas) == [102, 105]
    assert ids(demo_store, Filters(tags_all=["laser"]), meas) == []


def test_aggregate_converts_units_and_reports_experiments(demo_store):
    (group,) = aggregate_measurement(demo_store, Filters(), M("Belichtungszeit"))
    assert group["unit"] == "s"
    assert group["min"] == pytest.approx(0.2) and group["min_experiment"] == 108
    assert group["max"] == pytest.approx(30) and group["max_experiment"] == 105
    assert group["avg"] == pytest.approx((5 + 10 + 30 + 0.2) / 4)
    assert group["sum"] == pytest.approx(45.2)
    assert group["n_values"] == 4 and group["n_experiments"] == 4


def test_aggregate_respects_requested_unit(demo_store):
    (group,) = aggregate_measurement(demo_store, Filters(), M("Belichtungszeit"), unit="ms")
    assert group["unit"] == "ms" and group["max"] == pytest.approx(30000)


def test_aggregate_with_filter(demo_store):
    (group,) = aggregate_measurement(demo_store, Filters(tags_all=["raman"]), M("Laserleistung"))
    assert group["max"] == pytest.approx(100) and group["min"] == pytest.approx(50)


def test_aggregate_unknown_measurement_is_empty(demo_store):
    assert aggregate_measurement(demo_store, Filters(), M("Gibtsnicht")) == []


def test_measurement_values_rows(demo_store):
    rows = measurement_values(demo_store, Filters(), M("Laserleistung", ">", 30, "mW"))
    assert {r["experiment_id"] for r in rows} == {102, 105}


# ---------------------------------------------------------- list / count


def test_list_orders_and_limits(demo_store):
    result = list_experiments(demo_store, Filters(), None, "date_desc", 3)
    assert result.total == 8 and [r["id"] for r in result.rows] == [108, 107, 106]
    result = list_experiments(demo_store, Filters(), None, "date_asc", 2)
    assert [r["id"] for r in result.rows] == [101, 102]
    assert "raman" in list_experiments(demo_store, Filters(experiment_ids=[102])).rows[0]["tags"]


def test_count_plain_and_grouped(demo_store):
    assert count_experiments(demo_store, Filters(tags_all=["raman"]))[0]["count"] == 2
    by_status = {r["group"]: r["count"] for r in count_experiments(demo_store, Filters(), None, "status")}
    assert by_status == {"Abgeschlossen": 6, "In Arbeit": 2}
    by_tag = {r["group"]: r["count"] for r in count_experiments(demo_store, Filters(), None, "tag")}
    assert by_tag["spektroskopie"] == 3 and by_tag["zellkultur"] == 2
    by_month = {r["group"]: r["count"] for r in count_experiments(demo_store, Filters(), None, "month")}
    assert by_month == {"2026-09": 1, "2026-10": 6, "2026-11": 1}


def test_count_rejects_unknown_group(demo_store):
    with pytest.raises(ValueError):
        count_experiments(demo_store, Filters(), None, "title; DROP TABLE x")


def test_catalog_lists_what_exists(demo_store):
    catalog = load_catalog(demo_store)
    assert catalog.n_experiments == 8
    assert "raman" in catalog.tags and "Biologie" in catalog.categories
    names = {m.name for m in catalog.measurements}
    assert {"Wellenlänge", "Belichtungszeit", "Laserleistung"} <= names
    wl = catalog.measurement("wellenlänge")
    assert wl.unit == "nm" and wl.dimension == "length"
    assert catalog.date_min == "2026-09-28" and catalog.date_max == "2026-11-03"
