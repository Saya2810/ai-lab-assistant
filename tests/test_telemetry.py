from __future__ import annotations

import dataclasses
import json

import pytest

from lab_chatbot import cli
from lab_chatbot.engine import Engine
from lab_chatbot.telemetry import QueryLog, last_rows, summary
from tests.fakes import FakeLLM


def rows(store):
    return last_rows(store, 100)


def test_question_writes_row_with_latencies(make_engine, demo_store):
    engine, _ = make_engine(plan=RuntimeError("heuristics"))
    answer = engine.ask("Warum war das Rauschen bei Probe A3 so hoch?")
    (row,) = rows(demo_store)
    assert row["question"].startswith("Warum") and row["error"] == ""
    assert row["used_llm"] == 1 and row["n_sources"] == len(answer.sources)
    assert row["chat_model"] == "fake-chat" and row["embed_model"] == "fake-embed"
    lat = json.loads(row["latencies_json"])
    assert {"plan", "retrieve", "generate", "total"} <= set(lat)
    assert row["total_ms"] == lat["total"] and row["route"] == answer.plan.intent


def test_exception_writes_error_row_and_reraises(demo_store, embedder, config):
    llm = FakeLLM(plan=RuntimeError("heuristics"), answer=ValueError("boom"))
    engine = Engine(demo_store, embedder, llm, config)
    with pytest.raises(ValueError):
        engine.ask("Warum war das Rauschen bei Probe A3 so hoch?")
    (row,) = rows(demo_store)
    assert row["error"] == "ValueError" and row["total_ms"] > 0


def test_disabled_writes_nothing(demo_store, embedder, config):
    config = dataclasses.replace(config, query_log=False)
    engine = Engine(demo_store, embedder, FakeLLM(plan=RuntimeError("x")), config)
    assert engine.query_log is None
    engine.ask("Wie viele Experimente haben den Tag raman?")
    assert rows(demo_store) == []


def test_write_failure_never_breaks_answering(make_engine, demo_store, monkeypatch):
    engine, _ = make_engine(plan=RuntimeError("heuristics"))
    monkeypatch.setattr(demo_store, "connect", lambda: (_ for _ in ()).throw(RuntimeError("db gone")))
    engine.query_log.record(question="x")  # swallowed


def test_retention_prunes(demo_store):
    log = QueryLog(demo_store, max_rows=10, prune_every=5)
    for i in range(20):
        log.record(question=f"q{i}")
    got = rows(demo_store)
    assert len(got) <= 14 and got[0]["question"] == "q19"


def test_summary_percentiles(demo_store):
    log = QueryLog(demo_store)
    for i in range(1, 101):
        log.record(question="q", route="search", n_sources=0 if i <= 10 else 2,
                   latencies={"plan": float(i), "total": float(i * 2)}, error="Boom" if i == 1 else "")  # fmt: skip
    s = summary(demo_store, 7)
    assert s["count"] == 100 and s["errors"] == 1
    assert s["total_ms"] == {"p50": 100.0, "p95": 190.0}
    assert s["steps_ms"]["plan"] == {"p50": 50.0, "p95": 95.0}
    assert s["no_sources"] == 9 and s["search_count"] == 99


def test_cli_querylog(monkeypatch, demo_store, capsys):
    monkeypatch.setattr("lab_chatbot.factory.build_store", lambda c: demo_store)
    QueryLog(demo_store).record(question="Wie geht das?", route="list", latencies={"total": 12.0})
    assert cli.main(["--db", "ignored.db", "querylog", "--days", "1"]) == 0
    assert "1 questions" in capsys.readouterr().out
    assert cli.main(["--db", "ignored.db", "querylog", "--last", "5"]) == 0
    out = capsys.readouterr().out
    assert "Wie geht das?" in out and "list" in out
