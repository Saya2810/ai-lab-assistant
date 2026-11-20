"""Benchmark metrics, the runner and the HTML report (with fake models, no Ollama)."""

from __future__ import annotations

import json

import pytest

from lab_chatbot import benchmark
from lab_chatbot.engine import Engine
from lab_chatbot.evaluation import Case
from tests.fakes import FakeLLM, plan_json


def test_rank_metrics():
    assert benchmark.first_rank([5, 102, 7], {102}) == 2 and benchmark.first_rank([5], {102}) is None
    assert benchmark.ndcg_at([102, 5], {102}) == 1.0
    assert benchmark.ndcg_at([5, 102], {102}) == pytest.approx(1 / 1.5849625, rel=1e-6)
    metrics = benchmark.retrieval_metrics([
        {"expect_ids": [1], "rank": 1, "ranked": [1]},
        {"expect_ids": [2], "rank": 3, "ranked": [9, 8, 2]},
        {"expect_ids": [3], "rank": None, "ranked": []},
        {"expect_ids": [], "rank": None, "ranked": []},  # unanswerable: not part of the ranking metrics
    ])  # fmt: skip
    assert metrics["n"] == 3 and metrics["hit_at"][1] == pytest.approx(1 / 3)
    assert metrics["hit_at"][3] == pytest.approx(2 / 3) and metrics["mrr"] == pytest.approx((1 + 1 / 3) / 3)


def test_auroc_and_bootstrap():
    assert benchmark.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert benchmark.auroc([0.5], [0.5]) == 0.5 and benchmark.auroc([], [0.1]) is None
    low, high = benchmark.bootstrap_ci([0.0, 1.0] * 10)
    assert 0.2 < low < 0.5 < high < 0.8  # an honest, wide interval for n=20


@pytest.fixture
def result(demo_store, embedder, config):
    engine = Engine(demo_store, embedder, FakeLLM(plan=plan_json("search", "Rauschen Probe A3")), config)
    sets = {
        "standard": [Case("Warum war das Rauschen bei Probe A3 so hoch?", expect_ids=[102], expect_intent="search")],
        "unanswerable": [Case("Röntgenbeugung am Kristall?")],
    }
    return benchmark.run(engine, sets, answers=True)


def test_run_records_everything_a_report_needs(result):
    question = result["questions"][0]
    assert question["rank"] == 1 and question["intent"] == "search" and not question["error"]
    assert {"plan", "retrieve"} <= set(question["latency_ms"]) and question["answer"]["used_llm"]
    assert question["scores_relevant"] and question["scores_irrelevant"]
    assert set(question["variants"]) == {v.name for v in benchmark.VARIANTS}
    assert result["questions"][1]["unanswerable"] and "hybrid" in result["sets"]["unanswerable"]["false_answers"]
    json.dumps(result)  # must be serialisable


def test_report_is_a_self_contained_html_page(result, tmp_path):
    pytest.importorskip("matplotlib")
    from lab_chatbot.report import paired_permutation_p, write_report

    assistant = {"summary": {"micro_f1": 0.5}, "cases": [
        {"id": "a", "precision": 1.0, "recall": 0.5, "f1": 0.67, "invented_numbers": ["7"], "missing_numbers": [], "latency_s": 3.0}
    ]}  # fmt: skip
    page = write_report(result, tmp_path, assistant).read_text()
    assert page.count("data:image/png;base64,") >= 8 and "Lab Assistant" in page and "ROC" in page
    assert paired_permutation_p([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]) < 0.05
    assert paired_permutation_p([1, 0], [1, 0]) == 1.0


def test_cli_benchmark_writes_results_and_report(demo_store, embedder, config, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    from lab_chatbot import cli, factory

    engine = Engine(demo_store, embedder, FakeLLM(plan=plan_json("search", "Rauschen")), config)
    monkeypatch.setattr(factory, "build_engine", lambda config=None: engine)
    cases = tmp_path / "set.json"
    cases.write_text(json.dumps([{"question": "Rauschen Probe A3?", "expect_ids": [102]}]))
    assert cli.main(["benchmark", "--set", f"mine={cases}", "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "results.json").exists() and (tmp_path / "out" / "report.html").exists()
