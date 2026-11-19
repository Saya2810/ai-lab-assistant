from __future__ import annotations

import json

import pytest

from lab_assistant import elab_api
from lab_assistant.ai import Draft, estimate_tokens, generate_lab_entry, parse_draft
from lab_assistant.checks import compare_numbers, numbers_in
from lab_assistant.components.custom_fields import (
    find_matching_option,
    parse_decimal,
    unmatched_measurements,
)
from lab_assistant.components.lab_form import make_options, tag_names
from lab_assistant.config import AssistantConfig
from lab_assistant.experiment_service import create_complete_experiment
from lab_assistant.file_processing import can_use_for_ai, get_file_type, source_key
from lab_assistant.formatting import normalize_math_for_elabftw, strip_wrapping_fence
from lab_assistant.ollama_client import AIError, explain_error
from tests.fakes import FakeElabApi
from tests.test_shared import FakeResponse, FakeSession


@pytest.fixture
def cfg():
    return AssistantConfig.from_env()


# ------------------------------------------------------------ formatting


def test_wrapper_fence_is_removed_but_inner_code_blocks_survive():
    text = "```markdown\n## Aufbau\n\n```python\nx = 1\n```\n\nEnde\n```"
    out = normalize_math_for_elabftw(text)
    assert out.startswith("## Aufbau") and "```python\nx = 1\n```" in out


def test_a_note_that_is_only_a_code_block_is_not_unwrapped():
    text = "```python\nprint(1)\n```"
    assert strip_wrapping_fence(text) == text
    assert normalize_math_for_elabftw(text) == text


def test_bare_wrapper_is_removed_only_for_markdown_content():
    assert strip_wrapping_fence("```\n## Titel\ntext\n```") == "## Titel\ntext"
    assert strip_wrapping_fence("```\nplain log line\n```") == "```\nplain log line\n```"


def test_math_delimiters_are_converted():
    assert normalize_math_for_elabftw(r"Es gilt \(E=mc^2\) und \[a+b\]") == "Es gilt $E=mc^2$ und $$a+b$$"


def test_math_conversion_leaves_code_alone():
    text = "Text `\\(x\\)` und\n```\n\\[y\\]\n```"
    assert normalize_math_for_elabftw(text) == text


def test_doubled_backslashes_are_repaired():
    assert normalize_math_for_elabftw(r"$\\frac{1}{2} \\cdot x$") == r"$\frac{1}{2} \cdot x$"
    assert normalize_math_for_elabftw(r"$\alpha \\ \beta$") == r"$\alpha \\ \beta$"  # real line break kept


def test_empty_input():
    assert normalize_math_for_elabftw("") == ""


# ---------------------------------------------------------------- checks


def test_number_check_flags_invented_and_missing():
    check = compare_numbers("P = 20 mW bei 532 nm, t = 1,5 s", "Leistung 25 mW, 532 nm, 1.5 s")
    assert check.invented == ["25"] and check.missing == ["20"] and not check.clean


def test_number_check_ignores_labels_ordinals_and_signs():
    source = "SOURCE: IMAGE\nFILE: IMG_2041.HEIC\n--- IMAGE SECTION 2 ---\nDrift −3,5 K"
    check = compare_numbers(source, "1. Schritt\nDrift -3.5 K\n2. Ende")
    assert check.clean, check


def test_number_check_scientific_notation():
    assert numbers_in("5.32e-7 m")["5.32e-07"] == "5.32e-7"
    assert compare_numbers("c = 3e8", "c = 3E8").clean


# ----------------------------------------------------------- draft parsing


def test_parse_draft_happy_path():
    content = json.dumps({
        "title": " Raman A3 ",
        "lab_notes": "## Messung\n\\(x\\) bei 785 nm",
        "measurements": {"Wellenlänge": {"value": 785, "unit": "nm"}, "pH": 7.4},
    })  # fmt: skip
    draft = parse_draft(content, "785 nm gemessen")
    assert draft.title == "Raman A3" and "$x$" in draft.lab_notes
    assert draft.measurements == {"Wellenlänge": {"value": "785", "unit": "nm"}, "pH": {"value": "7.4", "unit": None}}
    assert draft.numbers.clean


def test_parse_draft_repairs_are_reported_not_silent():
    draft = parse_draft(json.dumps({
        "title": 5, "lab_notes": ["a", "b"],
        "measurements": {"": 1, "ok": {"value": ""}, "good": {"value": "3", "unit": ""}, "x": [1]},
    }))  # fmt: skip
    assert draft.title == "Laboratory entry" and draft.lab_notes == "a\n\nb"
    assert list(draft.measurements) == ["good"] and draft.measurements["good"]["unit"] is None
    assert len(draft.notes) >= 4


def test_parse_draft_tolerates_fences_and_chatter():
    assert parse_draft('Hier:\n```json\n{"title": "T", "lab_notes": "n"}\n```').title == "T"


@pytest.mark.parametrize("bad", ["", "kein json", "[1, 2]", '{"title": '])
def test_parse_draft_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_draft(bad)


def test_non_dict_measurements_are_ignored_with_note():
    draft = parse_draft('{"title": "T", "lab_notes": "n", "measurements": [1, 2]}')
    assert draft.measurements == {} and draft.notes


# --------------------------------------------------------------- generation


def test_generate_calls_model_with_integrity_prompt_and_json_mode(cfg):
    seen = {}

    def fake_chat(config, model, messages, **kw):
        seen.update(model=model, messages=messages, kw=kw)
        return '{"title": "T", "lab_notes": "20 mW", "measurements": {}}'

    draft = generate_lab_entry("SOURCE: WRITTEN NOTES\n\n20 mW", cfg, chat_fn=fake_chat)
    assert isinstance(draft, Draft) and draft.numbers.clean
    assert seen["model"] == "qwen2.5:7b" and seen["kw"]["json_mode"] is True
    assert "Never invent" in seen["messages"][0]["content"]
    assert "--- BEGIN SOURCE ---" in seen["messages"][1]["content"]


def test_organisational_fields_are_never_taken_from_the_model(cfg):
    out = '{"title": "T", "lab_notes": "n", "template": 5, "category": 3, "tags": ["x"]}'
    draft = generate_lab_entry("src", cfg, chat_fn=lambda *a, **k: out)
    assert not hasattr(draft, "tags") and not hasattr(draft, "category")


def test_one_retry_on_broken_json_then_error(cfg):
    replies = iter(["{broken", '{"title": "T", "lab_notes": "n"}'])
    assert generate_lab_entry("src", cfg, chat_fn=lambda *a, **k: next(replies)).title == "T"
    with pytest.raises(AIError, match="valid JSON"):
        generate_lab_entry("src", cfg, chat_fn=lambda *a, **k: "{broken")


def test_overlong_source_is_refused_instead_of_silently_truncated(cfg):
    called = []
    with pytest.raises(AIError, match="too long"):
        generate_lab_entry("x" * 60000, cfg, chat_fn=lambda *a, **k: called.append(1))
    assert not called and estimate_tokens("x" * 280) == 100


def test_ollama_errors_are_explained():
    assert "ollama pull qwen" in str(explain_error(type("E", (Exception,), {"status_code": 404})(), "qwen", "h"))
    assert "ollama serve" in str(explain_error(ConnectionError("refused"), "qwen", "http://h"))
    assert explain_error(ValueError("other"), "m", "h") is None


# ------------------------------------------------------ custom field helpers


def test_parse_decimal():
    assert parse_decimal("") == "" and parse_decimal("  ") == ""
    assert parse_decimal("1,5") == "1.5" and parse_decimal("−3") == "-3"
    assert parse_decimal("5.32e-7") == "5.32e-7" and parse_decimal(".5") == ".5"
    for bad in ("abc", "1.2.3", "5 nm", "--1"):
        assert parse_decimal(bad) is None


def test_find_matching_option_and_unmatched_measurements():
    assert find_matching_option(" ramAN ", ["Raman", "IR"]) == "Raman"
    assert find_matching_option("x", ["a"]) is None and find_matching_option(None, ["a"]) is None
    assert unmatched_measurements({"Wellenlänge": {}}, {"wellenlänge": 1, "Leistung": 2}) == ["Leistung"]


def test_make_options_disambiguates_duplicate_titles():
    options = make_options([{"id": 1, "title": "A"}, {"id": 2, "title": "B"}, {"id": 3, "title": "B"}])
    assert options == {"A": 1, "B (#2)": 2, "B (#3)": 3}


def test_tag_names_deduplicated():
    assert tag_names([{"tag": "a"}, {"tag": "b"}, {"tag": "a"}, {}]) == ["a", "b"]


def test_file_types_and_identity():
    assert get_file_type("note.JPG") == "image" and get_file_type("a.m4a") == "audio" and get_file_type("d.csv") == "file"
    assert can_use_for_ai("x.heic") and not can_use_for_ai("x.csv")
    assert source_key("a.png", b"1") != source_key("a.png", b"2")
    assert source_key("a.png", b"1") == source_key("a.png", b"1")


# ------------------------------------------------------- create experiment


def test_create_complete_experiment_happy_path():
    api = FakeElabApi()
    sources = {"k": {"name": "raw.csv", "content": b"1,2", "attach": True},
               "n": {"name": "skip.csv", "content": b"x", "attach": False}}  # fmt: skip
    result = create_complete_experiment(
        "T", "B", template_id=1, category_id=10, status_id=20, resource_id=30,
        tags=["raman", "laser", "raman"], custom_fields={"f": {"value": "1"}}, sources=sources, api=api,
    )  # fmt: skip
    assert result.complete and result.experiment_id == 501
    assert api.names() == ["create_experiment", "update_experiment", "add_tag", "add_tag",
                           "link_resource", "custom_fields", "upload"]  # fmt: skip
    assert [c[1][1] for c in api.calls if c[0] == "upload"] == ["raw.csv"]  # unattached file skipped


def test_failure_after_creation_is_reported_not_hidden():
    api = FakeElabApi(fail={"add_tag:laser": RuntimeError("tag boom"), "upload": RuntimeError("disk full")})
    result = create_complete_experiment(
        "T", "B", tags=["raman", "laser"], sources={"k": {"name": "a.png", "content": b"x", "attach": True}}, api=api
    )
    assert result.experiment_id == 501 and not result.complete
    assert any("laser" in w and "tag boom" in w for w in result.warnings)
    assert any("a.png" in w for w in result.warnings)
    assert "title, notes, category, status" in result.done and "tag 'raman'" in result.done  # the rest still ran


def test_new_resource_is_created_after_the_experiment_and_linked():
    api = FakeElabApi()
    result = create_complete_experiment(
        "T", "B", new_resource="Projekt X", new_resource_category_id=40, api=api
    )
    assert result.complete and result.new_resource_id == 900
    assert api.names() == ["create_experiment", "update_experiment", "create_resource", "link_resource"]
    create = [c for c in api.calls if c[0] == "create_resource"][0]
    assert create[1] == ("Projekt X",) and create[2] == {"category_id": 40}
    assert [c[1] for c in api.calls if c[0] == "link_resource"] == [(501, 900)]


def test_failed_resource_creation_is_reported_and_nothing_is_linked():
    api = FakeElabApi(fail={"create_resource": RuntimeError("403 forbidden")})
    result = create_complete_experiment("T", "B", new_resource="Projekt X", api=api)
    assert not result.complete and result.new_resource_id is None
    assert any("Projekt X" in w and "403" in w for w in result.warnings)
    assert "link_resource" not in api.names()


def test_no_resource_is_created_when_the_experiment_cannot_be():
    api = FakeElabApi(fail={"create_experiment": RuntimeError("403")})
    with pytest.raises(RuntimeError):
        create_complete_experiment("T", "B", new_resource="Projekt X", api=api)
    assert "create_resource" not in api.names()  # no orphaned resource


def test_creation_failure_raises_because_nothing_exists():
    api = FakeElabApi(fail={"create_experiment": RuntimeError("403")})
    with pytest.raises(RuntimeError):
        create_complete_experiment("T", "B", api=api)
    assert api.names() == ["create_experiment"]


# ------------------------------------------------------------- elab_api


@pytest.fixture
def wired(monkeypatch):
    from shared.config import ElabFTWConfig
    from shared.elabftw_client import ElabFTWClient

    def make(script):
        session = FakeSession(script)
        client = ElabFTWClient(ElabFTWConfig("https://lab.example/api/v2", "KEY"), session=session, sleep=lambda s: None)
        monkeypatch.setattr(elab_api, "client", lambda: client)
        return session

    return make


@pytest.fixture
def just_created(monkeypatch):
    """Experiment 7 was just created by the assistant: it may still be filled in."""
    monkeypatch.setattr(elab_api, "_open_entries", {7})


def test_structure_endpoints_are_paginated(wired):
    session = wired([FakeResponse(200, [{"id": 1}, {"id": 2}]), FakeResponse(200, [{"id": 3}]), FakeResponse(200, [])])
    assert [r["id"] for r in elab_api.get_resources()] == [1, 2, 3]
    assert session.calls[0][1].endswith("/items") and session.calls[1][2]["params"]["offset"] == 2


def test_body_is_normalised_and_marked_markdown(wired, just_created):
    session = wired([FakeResponse(200, {})])
    elab_api.update_experiment(7, title="T", body="```markdown\nGilt \\(x\\)\n```", category_id=3)
    sent = session.calls[0][2]["json"]
    assert sent == {"title": "T", "body": "Gilt $x$", "content_type": 2, "category": 3}


def test_update_with_nothing_does_nothing(wired, just_created):
    session = wired([])
    elab_api.update_experiment(7)
    assert session.calls == []


def test_custom_fields_are_merged_not_replaced(wired, just_created):
    existing = {"metadata": json.dumps({"extra_fields": {"keep": {"value": "1"}, "f": {"value": "old"}}, "elabftw": {"x": 1}})}
    session = wired([FakeResponse(200, existing), FakeResponse(200, {})])
    elab_api.update_experiment_custom_fields(7, {"f": {"value": "new"}})
    sent = json.loads(session.calls[1][2]["json"]["metadata"])
    assert sent["extra_fields"] == {"keep": {"value": "1"}, "f": {"value": "new"}} and sent["elabftw"] == {"x": 1}


def test_broken_metadata_json_does_not_crash(wired):
    wired([FakeResponse(200, {"metadata": "{not json"})])
    assert elab_api.get_custom_fields_from_template(1) == {}


def test_template_fields_from_string_or_dict_metadata(wired):
    wired([FakeResponse(200, {"metadata": json.dumps({"extra_fields": {"a": {"type": "number"}}})})])
    assert elab_api.get_custom_fields_from_template(1) == {"a": {"type": "number"}}
    wired([FakeResponse(200, {"metadata": {"extra_fields": {"b": {}}}})])
    assert elab_api.get_custom_fields_from_template(1) == {"b": {}}


def test_create_experiment_returns_id_from_location(wired):
    session = wired([FakeResponse(201, headers={"Location": "https://x/api/v2/experiments/88"})])
    assert elab_api.create_experiment(template_id=4) == 88
    assert session.calls[0][2]["json"] == {"template": 4, "category_id": 4}  # both names: old and current API


def test_team_lists_fall_back_to_the_documented_paths(wired):
    session = wired([
        FakeResponse(404, {"description": "no such path"}),         # teams/current/experiments_categories
        FakeResponse(200, {"id": 7, "team": 3}),                    # users/me
        FakeResponse(200, [{"id": 1, "title": "Raman"}]),           # teams/3/experiments_categories
        FakeResponse(200, []),
    ])  # fmt: skip
    assert [c["title"] for c in elab_api.get_experiment_categories()] == ["Raman"]
    assert [call[1].rsplit("/api/v2/", 1)[1] for call in session.calls][:3] == [
        "teams/current/experiments_categories", "users/me", "teams/3/experiments_categories",
    ]


def test_tags_fall_back_to_team_tags(wired):
    session = wired([FakeResponse(404, {}), FakeResponse(200, [{"id": 1, "tag": "raman"}]), FakeResponse(200, [])])
    assert elab_api.get_tags() == [{"id": 1, "tag": "raman"}]
    assert session.calls[1][1].endswith("/team_tags")


def test_the_known_path_is_used_when_it_works_and_other_errors_are_not_masked(wired):
    session = wired([FakeResponse(200, [{"id": 1, "tag": "a"}]), FakeResponse(200, [])])
    assert elab_api.get_tags()[0]["tag"] == "a" and len(session.calls) == 2  # no fallback attempted

    from shared.elabftw_client import ElabFTWError

    wired([FakeResponse(401, {"description": "bad key"})])
    with pytest.raises(ElabFTWError) as error:
        elab_api.get_tags()
    assert error.value.status_code == 401  # a wrong key must not look like a missing path


def test_missing_team_in_users_me_is_a_clear_error(wired):
    from shared.elabftw_client import ElabFTWError

    wired([FakeResponse(404, {}), FakeResponse(200, {"id": 7})])
    with pytest.raises(ElabFTWError, match="team"):
        elab_api.get_experiment_statuses()


def test_create_resource_posts_title_and_category_then_sets_them_again(wired):
    session = wired([FakeResponse(201, headers={"Location": "https://h/api/v2/items/77"}), FakeResponse(200, {})])
    assert elab_api.create_resource("Projekt X", category_id=1) == 77
    (post, patch) = session.calls
    assert post[0] == "POST" and post[1].endswith("/items") and post[2]["json"] == {"title": "Projekt X", "category": 1}
    assert patch[0] == "PATCH" and patch[1].endswith("/items/77") and patch[2]["json"] == {"title": "Projekt X", "category": 1}


def test_resource_categories_come_from_the_team(wired):
    session = wired([FakeResponse(200, [{"id": 1, "title": "Project"}]), FakeResponse(200, [])])
    assert [c["title"] for c in elab_api.get_resource_categories()] == ["Project"]
    assert session.calls[0][1].endswith("teams/current/resources_categories")


def test_link_resource_sends_json_body(wired, just_created):
    session = wired([FakeResponse(201)])
    elab_api.link_resource_to_experiment(7, 30)
    assert session.calls[0][1].endswith("experiments/7/items_links/30") and session.calls[0][2]["json"] == {}


# ------------------------------------------------- existing entries stay untouched


WRITES = [
    ("update_experiment", lambda i: elab_api.update_experiment(i, title="x")),
    ("update_experiment_custom_fields", lambda i: elab_api.update_experiment_custom_fields(i, {"f": {"value": "1"}})),
    ("add_tag_to_experiment", lambda i: elab_api.add_tag_to_experiment(i, "t")),
    ("link_resource_to_experiment", lambda i: elab_api.link_resource_to_experiment(i, 30)),
    ("upload_file_to_experiment", lambda i: elab_api.upload_file_to_experiment(i, "a.txt", b"x")),
]


@pytest.mark.parametrize("name, write", WRITES, ids=[w[0] for w in WRITES])
def test_existing_entries_cannot_be_changed(wired, name, write):
    session = wired([FakeResponse(200, {})] * 3)
    with pytest.raises(elab_api.ProtectedEntryError, match="existing lab notebook entry"):
        write(42)
    assert session.calls == []  # refused before anything was sent


def test_a_new_entry_is_writable_only_until_its_creation_is_finished(wired, monkeypatch):
    monkeypatch.setattr(elab_api, "_open_entries", set())
    session = wired([
        FakeResponse(201, headers={"Location": "https://lab.example/api/v2/experiments/88"}),  # create
        FakeResponse(200, {}),                                                               # title, body
        FakeResponse(201, {}),                                                               # tag
    ])  # fmt: skip
    result = create_complete_experiment("T", "B", tags=["raman"], api=elab_api)
    assert result.experiment_id == 88 and result.complete
    assert [c[0] for c in session.calls] == ["POST", "PATCH", "POST"]
    with pytest.raises(elab_api.ProtectedEntryError):  # written: closed for the assistant
        elab_api.update_experiment(88, title="changed later")
    assert len(session.calls) == 3


def test_the_entry_is_closed_even_when_filling_it_in_fails():
    api = FakeElabApi(fail={"update_experiment": RuntimeError("boom")})
    result = create_complete_experiment("T", "B", api=api)
    assert not result.complete and api.closed == [result.experiment_id]
