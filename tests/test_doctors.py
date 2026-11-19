"""The two ``doctor`` commands: what they tell a user who is connecting real services for the first time."""

from __future__ import annotations

import pytest

from lab_assistant import doctor as assistant_doctor
from lab_assistant.config import AssistantConfig
from lab_chatbot import doctor as chatbot_doctor
from lab_chatbot.demo import DEMO_EXPERIMENTS
from shared import diagnostics
from shared.elabftw_client import ElabFTWError
from tests.fakes import FakeElabClient, FakeEmbedder, FakeLLM, plan_json


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("ELABFTW_API_KEY", "key")
    monkeypatch.setenv("ELABFTW_URL", "https://lab.example/api/v2")


@pytest.fixture
def ollama_up(monkeypatch):
    import ollama

    class Client:
        def __init__(self, host=None):
            pass

        def list(self):
            return {"models": [{"model": "qwen2.5:7b"}, {"model": "embeddinggemma:latest"}]}

    monkeypatch.setattr(ollama, "Client", Client)


def by_name(checks):
    return {c.name: c for c in checks}


# ----------------------------------------------------------------- chatbot


def test_chatbot_doctor_parses_a_real_sample_from_elabftw(api_key, monkeypatch):
    fake = FakeElabClient(list(DEMO_EXPERIMENTS))
    monkeypatch.setattr("shared.elabftw_client.ElabFTWClient", lambda config: fake)
    checks = by_name(chatbot_doctor.check_elabftw(sample=3))
    assert checks["eLabFTW"].ok is True and "sampled 3" in checks["eLabFTW"].detail
    assert checks["eLabFTW data"].ok is True and "parsed 3" in checks["eLabFTW data"].detail


def test_chatbot_doctor_flags_an_empty_experiment_list(api_key, monkeypatch):
    empty = FakeElabClient()
    empty.experiments = {}  # (FakeElabClient([]) would fall back to the demo notebook)
    monkeypatch.setattr("shared.elabftw_client.ElabFTWClient", lambda config: empty)
    check = chatbot_doctor.check_elabftw()[0]
    assert check.ok is None and "empty" in check.detail


def test_chatbot_doctor_flags_unexpected_field_names(api_key, monkeypatch):
    """A server whose entries carry no body/date/modified_at: the parser would index nothing useful."""
    odd = [{"id": 1, "title": "T", "something_else": "x"}]
    monkeypatch.setattr("shared.elabftw_client.ElabFTWClient", lambda config: FakeElabClient(odd))
    data = by_name(chatbot_doctor.check_elabftw())["eLabFTW data"]
    assert data.ok is None and "no experiment has a body" in data.detail and "no modified_at" in data.detail


def test_chatbot_doctor_reports_an_api_error_with_its_message(api_key, monkeypatch):
    class Broken:
        def __init__(self, config):
            pass

        def iter_paginated(self, *a, **k):
            raise ElabFTWError("eLabFTW rejected the API key (HTTP 401)", status_code=401)

    monkeypatch.setattr("shared.elabftw_client.ElabFTWClient", Broken)
    check = chatbot_doctor.check_elabftw()[0]
    assert check.ok is False and "401" in check.detail


def test_chatbot_live_check_calls_both_models(config, monkeypatch):
    monkeypatch.setattr("lab_chatbot.factory.build_embedder", lambda c: FakeEmbedder())
    monkeypatch.setattr("lab_chatbot.factory.build_llm", lambda c: FakeLLM(plan=plan_json(), answer="ok"))
    checks = by_name(chatbot_doctor.check_live(config))
    assert checks["Embedding call"].ok is True and "-dimensional" in checks["Embedding call"].detail
    assert checks["Chat call"].ok is True


def test_chatbot_live_check_reports_a_failing_model(config, monkeypatch):
    monkeypatch.setattr("lab_chatbot.factory.build_embedder", lambda c: FakeEmbedder(fail=RuntimeError("out of memory")))
    monkeypatch.setattr("lab_chatbot.factory.build_llm", lambda c: FakeLLM(plan=plan_json(), answer="ok"))
    checks = by_name(chatbot_doctor.check_live(config))
    assert checks["Embedding call"].ok is False and "out of memory" in checks["Embedding call"].detail


def test_run_doctor_skips_the_live_calls_when_ollama_is_not_ready(config, monkeypatch):
    called = []
    monkeypatch.setattr(chatbot_doctor, "check_live", lambda c: called.append(1) or [])
    monkeypatch.setattr(chatbot_doctor, "check_ollama", lambda c: [diagnostics.Check("Ollama", False, "down")])
    chatbot_doctor.run_doctor(config, live=True)
    assert called == []  # the failing Ollama check already says it; no second, confusing error


# --------------------------------------------------------------- assistant


def fake_structure(monkeypatch, **fail):
    data = {
        "get_experiment_templates": [{"id": 5, "title": "Raman"}],
        "get_experiment_categories": [{"id": 1, "title": "A"}],
        "get_experiment_statuses": [{"id": 1, "title": "Running"}],
        "get_resources": [],
        "get_resource_categories": [{"id": 1, "title": "Project"}],
        "get_tags": [{"id": 1, "tag": "raman"}],
    }
    loaders = []
    for label, loader, may_be_empty in assistant_doctor.STRUCTURE:
        name = loader.__name__

        def make(name=name):
            def load():
                if name in fail:
                    raise fail[name]
                return data[name]

            return load

        loaders.append((label, make(), may_be_empty))
    monkeypatch.setattr(assistant_doctor, "STRUCTURE", loaders)
    monkeypatch.setattr(assistant_doctor.elab_api, "get_custom_fields_from_template", lambda i: {"Operator": {}, "Leistung": {}})


def test_assistant_doctor_reads_every_structure_list(api_key, monkeypatch):
    fake_structure(monkeypatch)
    checks = by_name(assistant_doctor.check_elabftw())
    assert checks["eLabFTW"].ok is True
    assert checks["eLabFTW templates"].detail == "1 found" and checks["eLabFTW resources"].detail == "0 found"
    assert checks["Template fields"].ok is True and "2 custom field" in checks["Template fields"].detail


def test_assistant_doctor_names_the_endpoint_that_fails(api_key, monkeypatch):
    fake_structure(monkeypatch, get_tags=ElabFTWError("HTTP 404 for team_tags", status_code=404))
    checks = by_name(assistant_doctor.check_elabftw())
    assert checks["eLabFTW tags"].ok is False and "404" in checks["eLabFTW tags"].detail
    assert checks["eLabFTW templates"].ok is True
    assert "eLabFTW" not in checks  # the "reachable" summary line only appears when everything worked


def test_assistant_doctor_without_a_key_says_what_to_do(monkeypatch):
    check = assistant_doctor.check_elabftw()[0]
    assert check.ok is False and ".env" in check.detail


def test_assistant_doctor_models_and_optional_parts(ollama_up, tmp_path):
    config = AssistantConfig.from_env()
    checks, ready = assistant_doctor.check_ollama(config)
    by = by_name(checks)
    assert ready and by["Text model"].ok is True
    assert by["Vision model"].ok is None and "ollama pull minicpm-v4.6:1b" in by["Vision model"].detail  # optional: a hint, not an error
    optional = by_name(assistant_doctor.check_optional(config))
    assert optional["Photo support"].ok is True
    assert optional["Audio"].critical is False


def test_assistant_live_call_uses_the_json_mode_the_assistant_uses(monkeypatch):
    seen = {}

    def fake_chat(config, model, messages, **kwargs):
        seen.update(model=model, **kwargs)
        return '{"ok": true}'

    monkeypatch.setattr("lab_assistant.ollama_client.chat", fake_chat)
    check = assistant_doctor.check_live(AssistantConfig.from_env())[0]
    assert check.ok is True and seen["json_mode"] is True and seen["model"] == "qwen2.5:7b"


def test_assistant_module_entry_point_runs_doctor(api_key, ollama_up, monkeypatch, capsys):
    from lab_assistant import __main__ as entry

    fake_structure(monkeypatch)
    assert entry.main(["doctor", "--quick"]) == 0
    out = capsys.readouterr().out
    assert "eLabFTW templates" in out and "All good." in out


def test_report_does_not_say_all_good_when_a_soft_check_failed(capsys):
    checks = [diagnostics.Check("Ollama", True, "ok"), diagnostics.Check("eLabFTW", False, "401", critical=False)]
    assert diagnostics.print_report(checks) == 0  # usable ...
    out = capsys.readouterr().out
    assert "All good" not in out and "Usable, but not everything works: eLabFTW" in out  # ... but honest about it


def test_report_fails_on_a_critical_problem(capsys):
    assert diagnostics.print_report([diagnostics.Check("Text model", False, "missing")]) == 1
    assert "1 problem(s) to fix." in capsys.readouterr().out
