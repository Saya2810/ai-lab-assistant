from __future__ import annotations

import json

import pytest

from lab_assistant import __main__ as cli
from lab_assistant.ai import Draft
from lab_assistant.checks import NumberCheck
from lab_assistant.evaluation import DEFAULT_CASES, evaluate_assistant, load_cases


def case(cid="c", measurements=(), words=()):
    return {"id": cid, "notes": cid, "expect_measurements": list(measurements), "expect_title_words": list(words)}


def draft(measurements, title="PCR run", invented=()):
    return Draft(title=title, lab_notes="", measurements=measurements, numbers=NumberCheck(invented=list(invented)))


M = {"name": "Annealing-Temperatur", "value": "58", "unit": "°C"}
V = {"name": "Template", "value": "2.5", "unit": "ng/µL"}


def run(c, d):
    return evaluate_assistant([c], lambda _: d)["cases"][0]


def test_perfect_draft():
    row = run(case(measurements=[M, V], words=["pcr"]), draft({"Annealing Temperatur": {"value": 58, "unit": "C"}, "template conc": {"value": "2.5", "unit": "ng/uL"}}))
    assert (row["precision"], row["recall"], row["f1"], row["title_word_recall"]) == (1.0, 1.0, 1.0, 1.0)


def test_missing_and_extra_measurement():
    d = draft({"annealing temperature": {"value": "58", "unit": "°C"}, "voltage": {"value": "5", "unit": "V"}})
    row = run(case(measurements=[M, V]), d)
    assert (row["tp"], row["fp"], row["fn"]) == (1, 1, 1)
    assert row["precision"] == 0.5 and row["recall"] == 0.5


def test_german_comma_and_unit_spelling():
    d = draft({"Template-Konzentration": {"value": "2,5", "unit": " ng / µl "}})
    assert run(case(measurements=[V]), d)["f1"] == 1.0
    d = draft({"Template": {"value": "2,5", "unit": "ng/μL"}})
    assert run(case(measurements=[V]), d)["f1"] == 1.0


def test_wrong_value_does_not_match_and_empty_is_perfect():
    assert run(case(measurements=[V]), draft({"Template": {"value": "2.6", "unit": "ng/µL"}}))["tp"] == 0
    assert run(case(), draft({}))["f1"] == 1.0


def test_error_is_recorded_and_run_continues():
    def gen(notes):
        if notes == "bad":
            raise RuntimeError("boom")
        return draft({})

    res = evaluate_assistant([case("bad", [M]), case("ok")], gen)
    bad, ok = res["cases"]
    assert "boom" in bad["error"] and bad["fn"] == 1 and bad["recall"] == 0.0
    assert ok["error"] is None and res["summary"]["errors"] == 1


def test_summary_aggregation():
    drafts = {"a": draft({"annealing": {"value": "58", "unit": "C"}}, invented=["9"]), "b": draft({})}
    res = evaluate_assistant([case("a", [M]), case("b", [M])], lambda n: drafts[n])
    s = res["summary"]
    assert s["n_cases"] == 2 and s["micro_recall"] == 0.5 and s["macro_recall"] == 0.5
    assert s["mean_invented_numbers"] == 0.5 and s["share_drafts_with_invented"] == 0.5
    assert {"latency_mean_s", "latency_median_s", "latency_p95_s", "micro_f1", "macro_f1"} <= set(s)
    json.dumps(res)


def test_demo_cases_load():
    cases = load_cases(DEFAULT_CASES)
    assert 8 <= len(cases) <= 10 and all(c["notes"] and c["id"] for c in cases)


def test_cli_writes_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("lab_assistant.ai.generate_lab_entry", lambda notes, config=None, **kw: draft({}))
    out = tmp_path / "r.json"
    assert cli.main(["eval", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert set(data) == {"summary", "cases"} and len(data["cases"]) == len(load_cases(DEFAULT_CASES))
    assert "macro_f1" in capsys.readouterr().out


def test_prompt_example_is_not_taken_from_the_test_cases():
    """Otherwise the evaluation would measure a memorised example, not the extraction."""
    import re

    from lab_assistant.ai import SYSTEM_PROMPT
    from lab_assistant.evaluation import load_cases

    example = SYSTEM_PROMPT.split("Measurement format")[1]
    names = re.findall(r'"([^"]+)": \{"value"', example)
    assert names
    for case in load_cases():
        for name in names:
            assert name.casefold() not in case["notes"].casefold(), (name, case["id"])


def test_values_written_in_the_notes_are_extras_not_errors():
    from lab_assistant.evaluation import _value_in_source

    notes = "Inkubation 10 min bei RT, Puffer pH 7,4, 2e5 cells/well"
    assert _value_in_source("7.4", notes) and _value_in_source("7,4", notes) and _value_in_source("2e5", notes)
    assert _value_in_source("RT", notes) and not _value_in_source("37", notes) and not _value_in_source("HCl", notes)


def test_a_correct_number_under_the_wrong_name_is_an_error():
    from lab_assistant.ai import Draft
    from lab_assistant.evaluation import evaluate_assistant

    case = {"id": "x", "notes": "10 µL Stammlösung + 90 µL H2O", "expect_title_words": [],
            "expect_measurements": [{"name": "Stammlösung Volumen", "value": "10", "unit": "µL"},
                                    {"name": "Wasser Volumen", "value": "90", "unit": "µL"}]}
    draft = Draft("t", "n", {"Verdünnung": {"value": "10", "unit": "µL"}, "Stammlösung": {"value": "90", "unit": "µL"}})
    row = evaluate_assistant([case], lambda notes: draft)["cases"][0]
    assert (row["tp"], row["misassigned"], row["extra"]) == (0, 2, 0) and row["supported_precision"] == 0.0
