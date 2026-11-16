from __future__ import annotations

import json

import pytest

from lab_chatbot import cli, doctor
from lab_chatbot.engine import Table
from lab_chatbot.evaluation import evaluate, load_cases
from tests.fakes import FakeElabClient, FakeEmbedder, FakeLLM, plan_json


@pytest.fixture
def patched(monkeypatch, config, demo_store, embedder):
    """Make the CLI use fakes instead of Ollama / eLabFTW."""
    llm = FakeLLM(plan=RuntimeError("force heuristics"), answer="Das Rauschen kam vom Fluoreszenzuntergrund [1].")
    monkeypatch.setattr("lab_chatbot.factory.build_embedder", lambda c: embedder)
    monkeypatch.setattr("lab_chatbot.factory.build_llm", lambda c: llm)
    monkeypatch.setattr("lab_chatbot.factory.build_store", lambda c: demo_store)
    return llm


def run(args, capsys):
    code = cli.main(["--db", "ignored.db", *args])
    out = capsys.readouterr()
    return code, out.out, out.err


# ---------------------------------------------------------------- CLI


def test_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("sync", "ask", "chat", "search", "stats", "eval", "doctor", "seed-demo"):
        assert cmd in out


def test_ask_count_prints_sql_answer(patched, capsys):
    code, out, _ = run(["ask", "Wie viele Experimente haben den Tag raman?"], capsys)
    assert code == 0 and "2" in out and "raman" in out


def test_ask_search_prints_answer_and_sources(patched, capsys):
    code, out, _ = run(["ask", "Warum war das Rauschen bei Probe A3 so hoch?"], capsys)
    assert code == 0 and "Fluoreszenzuntergrund" in out
    assert "Sources:" in out and "Experiment 102" in out and "experiments.php?mode=view&id=102" in out


def test_ask_explain_prints_plan_to_stderr(patched, capsys):
    code, out, err = run(["ask", "--explain", "Wie viele Experimente haben den Tag raman?"], capsys)
    assert '"intent": "count"' in err and "latency" in err


def test_search_command_shows_ranking(patched, capsys):
    code, out, _ = run(["search", "Rauschen Probe A3"], capsys)
    assert code == 0 and "exp 102" in out and "experiments:" in out


def test_stats_command(patched, capsys):
    code, out, _ = run(["stats"], capsys)
    assert code == 0 and "experiments" in out and "8" in out


def test_chat_loop(patched, monkeypatch, capsys):
    answers = iter(["Wie viele Experimente haben den Tag raman?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    code, out, _ = run(["chat"], capsys)
    assert code == 0 and "2" in out


def test_table_formatting_hides_link_column_by_default():
    table = Table("T", ["ID", "Titel", "Link"], [[1, "Ein sehr langer Titel " * 5, "https://x/1"]])
    text = cli.format_table(table)
    assert "Link" not in text and "https" not in text and "…" not in text and "ID" in text
    assert "https://x/1" in cli.format_table(table, show_links=True)


def test_user_errors_are_messages_not_tracebacks(monkeypatch, capsys):
    monkeypatch.setattr("lab_chatbot.factory.build_embedder", lambda c: (_ for _ in ()).throw(
        __import__("lab_chatbot.errors", fromlist=["BackendUnavailable"]).BackendUnavailable("ollama down")))
    code, _, err = run(["ask", "x"], capsys)
    assert code == 1 and "Error: ollama down" in err


def test_sync_command_reports(monkeypatch, tmp_path, capsys, embedder):
    from lab_chatbot.store import Store

    store = Store(tmp_path / "s.db")
    monkeypatch.setattr("lab_chatbot.factory.build_embedder", lambda c: embedder)
    monkeypatch.setattr("lab_chatbot.factory.build_store", lambda c: store)
    monkeypatch.setattr("shared.elabftw_client.get_client", lambda: FakeElabClient())
    code, out, _ = run(["sync"], capsys)
    assert code == 0 and "8 new" in out and store.experiment_count() == 8


def test_sync_without_api_key_is_a_clear_error(capsys):
    code, _, err = run(["sync"], capsys)
    assert code == 1 and "ELABFTW_API_KEY" in err


def test_seed_demo_uses_separate_database(monkeypatch, tmp_path, capsys, embedder):
    monkeypatch.setattr("lab_chatbot.factory.build_embedder", lambda c: embedder)
    db = tmp_path / "demo.db"
    code = cli.main(["--db", str(db), "seed-demo"])
    out = capsys.readouterr().out
    assert code == 0 and db.exists() and "8 demo experiments" in out


# ------------------------------------------------------------- doctor


def test_doctor_reports_missing_ollama(config, monkeypatch):
    import ollama

    class Boom:
        def __init__(self, host=None):
            pass

        def list(self):
            raise ConnectionError("refused")

    monkeypatch.setattr(ollama, "Client", Boom)
    checks = {c.name: c for c in doctor.run_doctor(config)}
    assert checks["Ollama"].ok is False and "ollama serve" in checks["Ollama"].detail
    assert checks["eLabFTW"].ok is None and not checks["eLabFTW"].critical  # no key: a hint, not an error


def test_doctor_checks_models_are_pulled(config, monkeypatch):
    import ollama

    class Client:
        def __init__(self, host=None):
            pass

        def list(self):
            return {"models": [{"model": "qwen2.5:7b"}]}

    monkeypatch.setattr(ollama, "Client", Client)
    checks = {c.name: c for c in doctor.run_doctor(config)}
    assert checks["Chat model"].ok is True
    assert checks["Embedding model"].ok is False and "ollama pull embeddinggemma" in checks["Embedding model"].detail


def test_model_name_matching():
    assert doctor._model_present("embeddinggemma", ["embeddinggemma:latest"])
    assert doctor._model_present("qwen2.5:7b", ["qwen2.5:7b"])
    assert not doctor._model_present("qwen2.5:7b", ["qwen2.5:14b"])


def test_doctor_detects_embedding_model_mismatch(config, demo_store, monkeypatch):
    from dataclasses import replace

    mismatched = replace(config, db_path=demo_store.path, embed_model="another-model")
    names = [c.name for c in doctor.check_index(mismatched)]
    assert "Embedding model" in names


def test_doctor_fts5():
    assert doctor.check_fts5().ok is True


# --------------------------------------------------------------- eval


def test_evaluation_metrics(make_engine, tmp_path):
    engine, _ = make_engine(RuntimeError("heuristics"))
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps([
        {"question": "Warum war das Rauschen bei Probe A3 so hoch?", "expect_ids": [102], "expect_intent": "search"},
        {"question": "Wie viele Experimente haben den Tag raman?", "expect_intent": "count"},
        {"question": "Zeige Experiment 104", "expect_ids": [104]},
        {"question": "xyzzy plugh", "expect_ids": [101]},  # deliberately failing
    ]), encoding="utf-8")  # fmt: skip
    results, summary = evaluate(engine, load_cases(golden))
    assert summary["cases"] == 4 and summary["passed"] == 3 and summary["errors"] == 0
    assert summary["hit_rate"] == pytest.approx(2 / 3)
    assert 0 < summary["mrr"] <= 1 and summary["intent_accuracy"] == 1.0
    assert results[-1].passed is False


def test_evaluation_full_mode_checks_answer_text(make_engine, tmp_path):
    engine, _ = make_engine(RuntimeError("heuristics"))
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps([
        {"question": "Wie viele Experimente haben den Tag raman?", "expect_contains": ["2"]},
        {"question": "Wie viele Experimente haben den Tag raman?", "expect_contains": ["17"]},
    ]), encoding="utf-8")  # fmt: skip
    _, summary = evaluate(engine, load_cases(golden), full=True)
    assert summary["contains_accuracy"] == 0.5


def test_one_crashing_case_does_not_stop_the_run(make_engine, tmp_path):
    from lab_chatbot.errors import BackendUnavailable

    engine, _ = make_engine(BackendUnavailable("down"))
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps([{"question": "Wie viele?"}, {"question": "Hallo"}]), encoding="utf-8")
    results, summary = evaluate(engine, load_cases(golden))
    assert summary["errors"] == 1 and summary["cases"] == 2 and results[1].error is None


def test_shipped_demo_golden_set_passes_on_the_demo_data(make_engine):
    """The golden file we ship must be consistent with the demo notebook."""
    from pathlib import Path

    engine, _ = make_engine(RuntimeError("heuristics"))
    cases = load_cases(Path(__file__).parents[1] / "lab_chatbot" / "golden_demo.json")
    results, summary = evaluate(engine, cases, full=True)
    failed = [(r.case.question, r.intent, r.ids, r.error) for r in results if not r.passed]
    assert not failed, failed
