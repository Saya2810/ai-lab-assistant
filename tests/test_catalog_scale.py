"""The planner catalog must stay small yet contain rare names the question mentions."""

from __future__ import annotations

import time
from datetime import date

import pytest

from lab_chatbot.planner import RawFilters, RawMeasurement, RawPlan, build_planner_messages, normalize_plan
from lab_chatbot.sql_tools import load_catalog
from lab_chatbot.store import Store

N_EXP, N_TAGS, N_MEAS = 50_000, 5_000, 3_000


@pytest.fixture(scope="module")
def big_store(tmp_path_factory) -> Store:
    store = Store(tmp_path_factory.mktemp("scale") / "big.db")
    exp_rows, tag_rows, meas_rows = [], [], []
    for i in range(1, N_EXP + 1):
        exp_rows.append((i, f"Exp {i}", "2026-01-01", "", f"cat{i % 30}", "done", f"author{i % 20}"))
        # Zipf-like: most experiments use the first few tags; rare ones appear once.
        t = i % N_TAGS if i <= N_TAGS else i % 50
        tag_rows.append((i, f"tag{t:04d}x", f"tag{t:04d}x"))
        m = i % N_MEAS if i <= N_MEAS else i % 40
        meas_rows.append((i, f"Messgroesse{m:04d}", f"messgroesse{m:04d}", "mm", "length"))
    tag_rows.append((7, "Quantenpunkt", "quantenpunkt"))
    meas_rows.append((7, "Spaltbreite", "spaltbreite", "um", "length"))
    with store.connect() as con:
        con.executemany(
            "INSERT INTO experiments(id, title, date, body, category, status, author) "
            "VALUES(?,?,?,?,?,?,?)", exp_rows,
        )
        con.executemany(
            "INSERT OR IGNORE INTO exp_tags(experiment_id, tag, tag_norm) VALUES(?,?,?)", tag_rows
        )
        con.executemany(
            "INSERT INTO measurements(experiment_id, name, name_norm, unit, unit_norm, dimension) "
            "VALUES(?,?,?,?,?,?)",
            [(a, b, c, d, d, e) for a, b, c, d, e in meas_rows],
        )
        con.execute("UPDATE meta SET value = '1' WHERE key = 'revision'")
    return store


def _prompt(question: str, catalog) -> str:
    return build_planner_messages(question, catalog, date(2026, 11, 1), None)[1]["content"]


def test_rare_names_reach_prompt_and_survive_validation(big_store):
    question = "Wie viele Experimente mit Quantenpunkten und welche Spaltbreite?"
    catalog = load_catalog(big_store, question=question)
    prompt = _prompt(question, catalog)
    assert "Quantenpunkt" in prompt and "Spaltbreite" in prompt
    assert len(catalog.tags) <= 120 and len(catalog.measurements) <= 120
    # Without a question the rare names are not in the prompt part
    plain = load_catalog(big_store)
    assert "Quantenpunkt" not in plain.tags

    raw = RawPlan(
        intent="aggregate",
        filters=RawFilters(tags_all=["quantenpunkt", "tag4999x"]),
        measurement=RawMeasurement(name="spaltbreite", agg="max"),
    )
    plan = normalize_plan(raw, catalog, question)
    assert "Quantenpunkt" in plan.filters.tags_all
    assert plan.measurement and plan.measurement.name == "Spaltbreite"
    # a rare name that is not in the prompt still validates against the full set
    assert "tag4999x" in plan.filters.tags_all


def test_typo_in_question_finds_name(big_store):
    catalog = load_catalog(big_store, question="Zeige Quantenpunkte mit Spaltbreit")
    assert "Quantenpunkt" in catalog.tags
    assert any(m.name == "Spaltbreite" for m in catalog.measurements)


def test_catalog_is_fast_after_first_call(big_store):
    load_catalog(big_store, question="warmup")  # fills the per-revision cache
    start = time.perf_counter()
    for _ in range(5):
        load_catalog(big_store, question="Wie viele Experimente mit Quantenpunkten, Spaltbreite max?")
    per_call = (time.perf_counter() - start) / 5
    assert per_call < 0.1, f"{per_call * 1000:.0f} ms"
    print(f"catalog build: {per_call * 1000:.1f} ms")
