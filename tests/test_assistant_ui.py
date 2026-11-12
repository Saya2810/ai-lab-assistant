"""The real Lab Assistant page, headless, with fake eLabFTW and a fake model."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from lab_assistant.ai import Draft
from lab_assistant.checks import NumberCheck
from tests import fakes

FIELDS = {
    1: {
        "Wellenlänge": {"type": "number", "value": "", "units": ["nm", "µm"], "unit": "nm"},
        "Leistung": {"type": "number", "value": ""},
        "Geprüft": {"type": "checkbox", "value": "0"},
        "Modus": {"type": "select", "options": ["CW", "pulsed"], "value": "CW"},
        "Operator": {"type": "text", "value": ""},
    }
}


def page(variant: str = "ok", fail: dict | None = None, draft_variant: str = "clean"):
    """Script body executed by AppTest."""
    from lab_assistant.ai import Draft
    from lab_assistant.app import render_app
    from lab_assistant.checks import NumberCheck
    from lab_assistant.config import AssistantConfig
    from shared.config import ConfigError
    from tests import fakes as f
    from tests.test_assistant_ui import FIELDS

    # the script re-runs on every interaction: create the fake API only once per test
    api = f.REGISTRY.setdefault("api", f.FakeElabApi(fail=fail, template_fields=FIELDS))

    def loader():
        if variant == "no_key":
            raise ConfigError("ELABFTW_API_KEY is not set.")
        if variant == "down":
            from shared.elabftw_client import ElabFTWError

            raise ElabFTWError("Cannot reach eLabFTW")
        return f.STRUCTURE

    def generate(text):
        f.REGISTRY["generate_input"] = text
        counter = f.REGISTRY["generations"] = f.REGISTRY.get("generations", 0) + 1
        numbers = NumberCheck(invented=["25"], missing=["20"]) if draft_variant == "suspicious" else NumberCheck()
        return Draft(
            title=f"Titel {counter}", lab_notes=f"Notizen bei 532 nm ({counter})",
            measurements={"Wellenlänge": {"value": "532", "unit": "nm"}, "Leistung": {"value": "20", "unit": "mW"}},
            numbers=numbers, notes=["Ignored malformed measurement 'x'."],
        )  # fmt: skip

    def process(name, content):
        f.REGISTRY.setdefault("processed", []).append(name)
        return f"OCR von {name}"

    render_app(loader, AssistantConfig.from_env(), api=api, generate=generate, process=process)


@pytest.fixture(autouse=True)
def _reset_registry():
    fakes.REGISTRY.clear()


class FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name, self._content = name, content

    def getvalue(self) -> bytes:
        return self._content


def use_uploader(monkeypatch, uploads: list) -> list:
    """Replace the file uploader (AppTest cannot upload). Like the real widget it registers its
    key in session state while it is drawn; ``del at.session_state[key]`` = Streamlit dropped it."""
    import streamlit as st

    def fake(*args, key=None, **kwargs):
        if key:
            st.session_state[key] = uploads
        return uploads

    monkeypatch.setattr(st, "file_uploader", fake)
    return uploads


def run(**kwargs) -> AppTest:
    at = AppTest.from_function(page, kwargs=kwargs, default_timeout=15)
    return at.run()


def generate(at: AppTest, text="Laser bei 532 nm") -> AppTest:
    at.text_area(key="written_notes").set_value(text).run()
    return at.button(key="generate_lab_entry").click().run()


def api():
    return fakes.REGISTRY["api"]


# ----------------------------------------------------------------- loading


def test_page_renders_and_offers_structure():
    at = run()
    assert not at.exception and at.title[0].value.endswith("Lab Assistant")
    assert at.selectbox(key="lab_template").options == ["Spectroscopy", "Plain"]
    assert at.selectbox(key="lab_category").options == ["Measurement (#10)", "Measurement (#11)"]  # duplicates kept apart
    assert at.multiselect(key="lab_tags").options == ["raman", "laser"]


def test_missing_api_key_is_a_message_not_a_traceback():
    at = run(variant="no_key")
    assert not at.exception and any("ELABFTW_API_KEY" in e.value for e in at.error)
    assert not at.button  # the page stopped before drawing the forms


def test_unreachable_elabftw_is_a_message():
    at = run(variant="down")
    assert not at.exception and any("Cannot reach eLabFTW" in e.value for e in at.error)


# -------------------------------------------------------------- generation


def test_generate_fills_the_form():
    at = generate(run())
    assert not at.exception
    assert at.text_input(key="lab_title").value == "Titel 1"
    assert "532 nm" in at.text_area(key="lab_body").value
    assert fakes.REGISTRY["generate_input"] == "SOURCE: WRITTEN NOTES\n\nLaser bei 532 nm"


def test_generate_without_content_warns():
    at = run().button(key="generate_lab_entry").click().run()
    assert any("no content" in w.value for w in at.warning)


def test_suspicious_numbers_are_flagged_before_creation():
    at = generate(run(draft_variant="suspicious"))
    assert any("`25`" in w.value for w in at.warning)
    assert any("`20`" in i.value for i in at.info)
    assert any("Ignored malformed" in c.value for c in at.caption)


def test_regenerating_keeps_the_users_structure_choices():
    at = run()
    at.selectbox(key="lab_template").select("Plain").run()
    at.multiselect(key="lab_tags").select("raman").run()
    at = generate(at)
    at = at.text_area(key="written_notes").set_value("noch eine Notiz").run()
    at = at.button(key="generate_lab_entry").click().run()
    assert at.text_input(key="lab_title").value == "Titel 2"  # the second draft replaced the first ...
    assert at.selectbox(key="lab_template").value == "Plain" and at.multiselect(key="lab_tags").value == ["raman"]


def test_uploaded_files_are_processed_once_and_labelled(monkeypatch):
    use_uploader(monkeypatch, [FakeUpload("tafel.jpg", b"eins"), FakeUpload("tafel.jpg", b"zwei"), FakeUpload("daten.csv", b"1,2")])
    at = run()
    assert len(at.session_state["lab_sources"]) == 3  # same name + different content = two files
    at.button(key="generate_lab_entry").click().run()
    assert fakes.REGISTRY["processed"] == ["tafel.jpg", "tafel.jpg"]  # the csv is not AI-capable
    text = fakes.REGISTRY["generate_input"]
    assert text.count("SOURCE: IMAGE\nFILE: tafel.jpg\n\nOCR von tafel.jpg") == 2 and "daten.csv" not in text
    at.button(key="generate_lab_entry").click().run()
    assert len(fakes.REGISTRY["processed"]) == 2  # extraction is cached, not repeated


def test_removing_a_file_from_the_uploader_removes_it(monkeypatch):
    uploads = use_uploader(monkeypatch, [FakeUpload("a.jpg", b"1"), FakeUpload("b.jpg", b"2")])
    at = run()
    assert len(at.session_state["lab_sources"]) == 2
    uploads.pop()
    at.run()
    assert [s["name"] for s in at.session_state["lab_sources"].values()] == ["a.jpg"]


def leave_page(at: AppTest, uploads: list) -> AppTest:
    """What Streamlit does when another page is shown: the uploader widget and its state are gone."""
    del at.session_state["lab_file_uploader"]
    uploads.clear()
    return at.run()


def test_files_survive_leaving_the_page_with_their_extracted_text(monkeypatch):
    uploads = use_uploader(monkeypatch, [FakeUpload("tafel.jpg", b"eins"), FakeUpload("daten.csv", b"1,2")])
    at = run()
    at.button(key="generate_lab_entry").click().run()  # runs OCR on the image
    assert fakes.REGISTRY["processed"] == ["tafel.jpg"]

    at = leave_page(at, uploads)
    sources = at.session_state["lab_sources"]
    assert sorted(s["name"] for s in sources.values()) == ["daten.csv", "tafel.jpg"]
    assert all(s["detached"] for s in sources.values())
    assert next(s for s in sources.values() if s["name"] == "tafel.jpg")["extracted_text"] == "OCR von tafel.jpg"
    assert any("Kept from before" in c.value for c in at.caption)

    at.button(key="generate_lab_entry").click().run()  # the OCR result is reused, not recomputed
    assert fakes.REGISTRY["processed"] == ["tafel.jpg"] and "OCR von tafel.jpg" in fakes.REGISTRY["generate_input"]


def test_a_detached_file_can_be_removed_explicitly(monkeypatch):
    uploads = use_uploader(monkeypatch, [FakeUpload("a.jpg", b"1"), FakeUpload("b.jpg", b"2")])
    at = leave_page(run(), uploads)
    key_a = next(k for k, s in at.session_state["lab_sources"].items() if s["name"] == "a.jpg")
    at.button(key=f"remove_{key_a}").click().run()
    assert [s["name"] for s in at.session_state["lab_sources"].values()] == ["b.jpg"]


def test_uploading_a_detached_file_again_gives_it_back_to_the_uploader(monkeypatch):
    uploads = use_uploader(monkeypatch, [FakeUpload("a.jpg", b"1")])
    at = leave_page(run(), uploads)
    assert all(s["detached"] for s in at.session_state["lab_sources"].values())
    uploads.append(FakeUpload("a.jpg", b"1"))  # same name + content: the same file
    at.run()
    assert [s["detached"] for s in at.session_state["lab_sources"].values()] == [False]
    uploads.clear()  # now removing it in the uploader removes it for real
    at.run()
    assert at.session_state["lab_sources"] == {}


def test_a_normal_rerun_never_marks_files_as_detached(monkeypatch):
    use_uploader(monkeypatch, [FakeUpload("a.jpg", b"1")])
    at = run()
    at.text_area(key="written_notes").set_value("x").run()
    assert [s["detached"] for s in at.session_state["lab_sources"].values()] == [False]
    assert not any("Kept from before" in c.value for c in at.caption)


def test_attached_files_are_uploaded_to_the_experiment(monkeypatch):
    use_uploader(monkeypatch, [FakeUpload("daten.csv", b"1,2")])
    at = run()
    at.text_input(key="lab_title").set_value("T").run()
    at.text_area(key="lab_body").set_value("B").run()
    at = fill_and_create(at)
    uploads = [c for c in api().calls if c[0] == "upload"]
    assert [u[1][1] for u in uploads] == ["daten.csv"] and uploads[0][1][2] == b"1,2"


# ----------------------------------------------------------------- creation


def fill_and_create(at: AppTest) -> AppTest:
    return at.button(key="create_lab_entry").click().run()


def test_create_entry_end_to_end():
    at = generate(run())
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    at.selectbox(key="lab_status").select("Running").run()
    at.selectbox(key="lab_resource").select("Spectrometer").run()
    at.multiselect(key="lab_tags").select("raman").run()
    at = fill_and_create(at)
    assert not at.exception and any("experiment 501" in s.value for s in at.success)
    names = api().names()
    assert names[:2] == ["get_template_fields", "create_experiment"] or "create_experiment" in names
    created = [c for c in api().calls if c[0] == "create_experiment"][0]
    assert created[2]["template_id"] == 1
    update = [c for c in api().calls if c[0] == "update_experiment"][0]
    assert update[2]["title"] == "Titel 1" and update[2]["status_id"] == 20 and update[2]["category_id"] is None
    fields = [c for c in api().calls if c[0] == "custom_fields"][0][1][1]
    assert fields["Wellenlänge"]["value"] == "532" and fields["Wellenlänge"]["unit"] == "nm"
    assert fields["Modus"]["value"] == "CW" and fields["Geprüft"]["value"] == "0"


def test_template_fields_are_fetched_once_not_per_rerun():
    at = generate(run())
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    at.text_input(key="lab_title").set_value("anderer Titel").run()
    at.text_input(key="lab_title").set_value("noch einer").run()
    assert api().names().count("get_template_fields") == 1


def test_creating_the_same_entry_twice_needs_confirmation():
    at = fill_and_create(generate(run()))
    assert any("experiment 501" in s.value for s in at.success)
    at = at.run()
    assert any("already created as experiment 501" in i.value for i in at.info)
    assert at.button(key="create_lab_entry").disabled
    at.checkbox(key="allow_duplicate").check().run()
    at = fill_and_create(at)
    assert any("experiment 502" in s.value for s in at.success)


def test_changing_the_entry_after_creation_allows_a_new_one():
    at = fill_and_create(generate(run()))
    at.text_input(key="lab_title").set_value("Zweiter Eintrag").run()
    assert not at.button(key="create_lab_entry").disabled


def test_partial_failure_is_reported_with_the_experiment_id():
    at = generate(run(fail={"add_tag:laser": RuntimeError("tag boom")}))
    at.multiselect(key="lab_tags").select("laser").run()
    at = fill_and_create(at)
    assert not at.exception and not at.success
    message = " ".join(w.value for w in at.warning)
    assert "Experiment 501 was created" in message and "tag 'laser'" in message and "tag boom" in message


def test_creation_failure_shows_error_and_creates_nothing():
    at = generate(run(fail={"create_experiment": RuntimeError("HTTP 403")}))
    at = fill_and_create(at)
    assert any("Could not create" in e.value and "403" in e.value for e in at.error)
    assert api().names().count("update_experiment") == 0


def test_title_and_body_are_required():
    at = run()
    at = fill_and_create(at)
    assert any("title" in w.value.lower() for w in at.warning)
    at.text_input(key="lab_title").set_value("T").run()
    at = fill_and_create(at)
    assert any("lab notes" in w.value.lower() for w in at.warning)
    assert "create_experiment" not in api().names()


# ------------------------------------------------------------ custom fields


def custom_key(at, name, suffix="value"):
    version = at.session_state["draft_version"] if "draft_version" in at.session_state else 0
    return f"t1_v{version}_cf_{name}_{suffix}"


def test_invalid_number_blocks_creation():
    at = generate(run())
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    at.text_input(key=custom_key(at, "Leistung")).set_value("viel").run()
    assert any("not a number" in e.value for e in at.error)
    at = fill_and_create(at)
    assert any("custom fields" in e.value.lower() for e in at.error)
    assert "create_experiment" not in api().names()


def test_german_decimal_comma_is_accepted_and_normalised():
    at = generate(run())
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    at.text_input(key=custom_key(at, "Leistung")).set_value("1,5").run()
    at = fill_and_create(at)
    fields = [c for c in api().calls if c[0] == "custom_fields"][0][1][1]
    assert fields["Leistung"]["value"] == "1.5"


def test_empty_number_field_stays_empty_not_zero():
    at = run()
    at.text_input(key="lab_title").set_value("T").run()
    at.text_area(key="lab_body").set_value("B").run()
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    at = fill_and_create(at)
    fields = [c for c in api().calls if c[0] == "custom_fields"][0][1][1]
    assert fields["Leistung"]["value"] == "" and fields["Wellenlänge"]["value"] == ""


def test_ai_unit_not_allowed_for_field_is_not_silently_replaced():
    at = run()
    at.session_state["ai_measurements"] = {"Wellenlänge": {"value": "5", "unit": "mm"}}
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    assert any("'mm' is not allowed" in w.value for w in at.warning)
    assert any("choose a unit" in e.value.lower() for e in at.error)
    assert at.selectbox(key=custom_key(at, "Wellenlänge", "unit")).value is None


def test_unmatched_measurements_are_listed():
    at = generate(run())
    at.session_state["ai_measurements"] = {"Wellenlänge": {"value": "5", "unit": "nm"}, "Druck": {"value": "1"}}
    at.selectbox(key="lab_template").select("Spectroscopy").run()
    assert any("Druck" in c.value for c in at.caption)
