from __future__ import annotations

import json

from lab_chatbot.chunking import build_header, chunk_experiment
from lab_chatbot.parse import html_to_markdown, parse_experiment


def raw(**kw):
    base = {"id": 5, "title": "Test", "date": "20260930", "body": "Hallo", "tags": "a|b", "content_type": 2}
    base.update(kw)
    return base


def test_tags_are_parsed_from_all_shapes():
    assert parse_experiment(raw(tags="a|b")).tags == ["a", "b"]
    assert parse_experiment(raw(tags="a, b")).tags == ["a", "b"]
    assert parse_experiment(raw(tags=["a", "b"])).tags == ["a", "b"]
    assert parse_experiment(raw(tags=None)).tags == []


def test_compact_and_created_at_dates():
    assert parse_experiment(raw(date="20260930")).date == "2026-09-30"
    r = parse_experiment(raw(date=None, created_at="2026-10-01 12:00:00"))
    assert r.date == "2026-10-01"


def test_html_body_is_converted_unless_markdown():
    html = "<h2>Aufbau</h2><p>Laser <strong>an</strong></p><ul><li>eins</li></ul>"
    record = parse_experiment(raw(body=html, content_type=1))
    assert "<" not in record.body and "Aufbau" in record.body and "eins" in record.body
    markdown = parse_experiment(raw(body="a < b and c > d", content_type=2))
    assert markdown.body == "a < b and c > d"


def test_html_to_markdown_keeps_structure():
    out = html_to_markdown("<h1>T</h1><p>x</p><p>y</p>")
    assert out.startswith("#") and "x" in out and "y" in out


def test_metadata_extra_fields_become_measurements():
    meta = {
        "extra_fields": {
            "Wellenlänge": {"type": "number", "value": "532", "unit": "nm"},
            "Notiz": {"type": "text", "value": "10 mW"},
            "Aktiv": {"type": "checkbox", "value": "on"},
            "Leer": {"type": "number", "value": ""},
        }
    }
    record = parse_experiment(raw(metadata=json.dumps(meta)))
    by_name = {m.name: m for m in record.measurements}
    wl = by_name["Wellenlänge"]
    assert wl.value_num == 532 and wl.unit == "nm" and wl.dimension == "length"
    assert wl.value_si == 5.32e-7
    assert by_name["Notiz"].value_num == 10 and by_name["Notiz"].unit == "mW"
    assert "Leer" not in by_name


def test_broken_metadata_is_ignored():
    assert parse_experiment(raw(metadata="{not json")).measurements == []
    assert parse_experiment(raw(metadata=None)).measurements == []


def test_german_decimal_comma_in_metadata():
    meta = {"extra_fields": {"pH": {"type": "number", "value": "7,4"}}}
    assert parse_experiment(raw(metadata=json.dumps(meta))).measurements[0].value_num == 7.4


def test_category_and_status_names_are_resolved_from_ids():
    record = parse_experiment(raw(category=3, status=2), {3: "Messung"}, {2: "In Arbeit"})
    assert record.category == "Messung" and record.status == "In Arbeit"
    record = parse_experiment(raw(category_title="Bio", status_title="Done"))
    assert (record.category, record.status) == ("Bio", "Done")


def test_content_hash_changes_with_content_only():
    a = parse_experiment(raw(body="x"))
    b = parse_experiment(raw(body="x", modified_at="2027-01-01"))
    c = parse_experiment(raw(body="y"))
    assert a.content_hash == b.content_hash != c.content_hash


# ------------------------------------------------------------------ chunking


def test_every_chunk_carries_the_experiment_header():
    record = parse_experiment(raw(id=77, title="Raman A3", body="## Teil 1\n\n" + "Satz. " * 400))
    chunks = chunk_experiment(record, max_chars=500)
    assert len(chunks) > 2
    assert all("77" in c.text.splitlines()[0] and "Raman A3" in c.text.splitlines()[0] for c in chunks)
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_size_limit():
    record = parse_experiment(raw(body="Wort " * 2000))
    chunks = chunk_experiment(record, max_chars=600)
    assert max(len(c.text) for c in chunks) <= 600 + 400  # header + overlap tolerance


def test_math_and_code_blocks_are_never_split():
    math = "$$\n" + "\\alpha + \\beta = \\gamma\n" * 30 + "$$"
    record = parse_experiment(raw(body=f"Davor.\n\n{math}\n\nDanach."))
    chunks = chunk_experiment(record, max_chars=300)
    assert any(math in c.text for c in chunks)
    code = "```python\n" + "x = 1\n" * 40 + "```"
    chunks = chunk_experiment(parse_experiment(raw(body=code)), max_chars=300)
    assert any(code in c.text for c in chunks)


def test_measurements_get_their_own_chunk():
    meta = {"extra_fields": {"Laserleistung": {"type": "number", "value": "20", "unit": "mW"}}}
    record = parse_experiment(raw(body="Kurz.", metadata=json.dumps(meta)))
    assert "Laserleistung" in chunk_experiment(record)[-1].text


def test_empty_body_still_yields_a_findable_header_chunk():
    chunks = chunk_experiment(parse_experiment(raw(body="", title="Nur Titel")))
    assert chunks and "Nur Titel" in chunks[0].text


def test_header_contains_tags_category_status():
    record = parse_experiment(raw(tags="raman|laser", category_title="Messung", status_title="Fertig"))
    header = build_header(record)
    for word in ("raman", "laser", "Messung", "Fertig", "2026-09-30"):
        assert word in header


def test_paragraph_before_a_fence_is_its_own_block():
    from lab_chatbot.chunking import _split_blocks

    blocks = _split_blocks("Erklärung vorher\n```\ncode()\n```\nDanach")
    assert [b.text for b in blocks] == ["Erklärung vorher", "```\ncode()\n```", "Danach"]
    assert [b.atomic for b in blocks] == [False, True, False]


def test_oversized_formula_is_split_on_line_boundaries_only():
    lines = [f"\\alpha_{i} + \\beta_{i} = \\gamma_{i}" for i in range(400)]
    body = "$$\n" + "\n".join(lines) + "\n$$"
    chunks = chunk_experiment(parse_experiment(raw(body=body)))
    assert len(chunks) > 1
    for line in lines:
        assert sum(line in c.text for c in chunks) >= 1  # no line cut in half
