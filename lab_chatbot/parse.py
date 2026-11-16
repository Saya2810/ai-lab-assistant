"""Turn a raw eLabFTW experiment payload into a clean, typed record.

The eLabFTW API returns tags as ``"a|b"`` strings or lists depending on the
endpoint/version, bodies as HTML or Markdown, dates in more than one shape and
custom fields as a JSON *string* under ``metadata``. All of that is normalised
here, defensively, in one place.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from lab_chatbot.units import lookup_unit, parse_number, parse_quantity, to_si, unit_norm

# Bump when parsing or chunking changes in a way that requires re-indexing.
PARSER_VERSION = "1"

_HTML_HINT = re.compile(r"<\s*(p|div|h[1-6]|ul|ol|li|br|table|span|strong|em)\b", re.I)


@dataclass
class Measurement:
    name: str
    value_text: str
    value_num: float | None = None
    unit: str | None = None
    unit_norm: str | None = None
    dimension: str | None = None
    value_si: float | None = None


@dataclass
class ExperimentRecord:
    id: int
    title: str
    date: str  # ISO yyyy-mm-dd or ""
    body: str  # Markdown/plain text
    tags: list[str] = field(default_factory=list)
    category: str = ""
    status: str = ""
    author: str = ""
    resources: list[str] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    modified_at: str = ""
    content_hash: str = ""

    def compute_hash(self) -> str:
        payload = {
            "v": PARSER_VERSION,
            "title": self.title,
            "date": self.date,
            "body": self.body,
            "tags": sorted(self.tags),
            "category": self.category,
            "status": self.status,
            "author": self.author,
            "resources": sorted(self.resources),
            "measurements": [
                (m.name, m.value_text, m.unit) for m in self.measurements
            ],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# HTML -> Markdown-ish text
# ---------------------------------------------------------------------------


class _HtmlToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("p", "div", "tr", "br", "ul", "ol"):
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif re.fullmatch(r"h[1-6]", tag) or tag in ("p", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_markdown(html: str) -> str:
    parser = _HtmlToText()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_html(body: str) -> bool:
    return bool(_HTML_HINT.search(body))


# ---------------------------------------------------------------------------
# field helpers
# ---------------------------------------------------------------------------


def _iso_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return "-".join(match.groups())
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8 and text.isdigit():  # YYYYMMDD
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return ""


def _parse_tags(raw: dict) -> list[str]:
    value = raw.get("tags")
    tags: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                item = item.get("tag") or item.get("name")
            if item:
                tags.append(str(item).strip())
    elif isinstance(value, str) and value.strip():
        parts = value.split("|") if "|" in value else value.split(",")
        tags = [p.strip() for p in parts]
    seen: set[str] = set()
    unique = []
    for tag in tags:
        key = tag.casefold()
        if tag and key not in seen:
            seen.add(key)
            unique.append(tag)
    return unique


def _name_of(value: Any, lookup: dict[int, str] | None) -> str:
    if isinstance(value, str):
        return value.strip() if not value.strip().isdigit() or not lookup else (
            lookup.get(int(value), "")
        )
    if isinstance(value, int) and lookup:
        return lookup.get(value, "")
    return ""


def _resource_names(raw: dict) -> list[str]:
    names = []
    for link in raw.get("items_links") or []:
        if isinstance(link, dict):
            title = link.get("title") or link.get("name")
            if title:
                names.append(str(title).strip())
    return names


def _load_metadata(raw: dict) -> dict:
    metadata = raw.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return {}
    return metadata if isinstance(metadata, dict) else {}


def _measurement(name: str, spec: Any) -> Measurement | None:
    if not isinstance(spec, dict):
        return None
    raw_value = spec.get("value")
    if raw_value in (None, "", []):
        return None
    if isinstance(raw_value, list):
        value_text = ", ".join(str(v) for v in raw_value)
    else:
        value_text = str(raw_value).strip()
    if not value_text:
        return None

    field_type = str(spec.get("type") or "text").lower()
    unit = spec.get("unit")
    unit = str(unit).strip() if unit else None
    if not unit:
        units = spec.get("units")
        if isinstance(units, list) and len(units) == 1:
            unit = str(units[0]).strip() or None

    number: float | None = None
    if field_type == "number":
        number = parse_number(value_text)
    elif field_type == "checkbox":
        number = 1.0 if value_text.lower() in ("1", "true", "on", "yes", "checked") else 0.0
    elif field_type == "text":
        # "5 s" typed into a text field is still a quantity.
        parsed = parse_quantity(value_text)
        if parsed and parsed[1] and lookup_unit(parsed[1]):
            number, unit = parsed

    value_si = dimension = None
    if number is not None and unit:
        converted = to_si(number, unit)
        if converted:
            value_si, dimension = converted

    return Measurement(
        name=name.strip(),
        value_text=value_text,
        value_num=number,
        unit=unit,
        unit_norm=unit_norm(unit),
        dimension=dimension,
        value_si=value_si,
    )


def parse_measurements(raw: dict) -> list[Measurement]:
    fields = _load_metadata(raw).get("extra_fields")
    if not isinstance(fields, dict):
        return []
    result = []
    for name, spec in fields.items():
        measurement = _measurement(str(name), spec)
        if measurement:
            result.append(measurement)
    return result


def parse_experiment(
    raw: dict,
    category_names: dict[int, str] | None = None,
    status_names: dict[int, str] | None = None,
) -> ExperimentRecord:
    """Normalise one eLabFTW experiment payload (``GET experiments/{id}``)."""
    body = raw.get("body") or ""
    if not isinstance(body, str):
        body = str(body)
    content_type = raw.get("content_type")
    if content_type != 2 and looks_like_html(body):
        body = html_to_markdown(body)
    body = body.replace("\r\n", "\n").strip()

    author = (raw.get("fullname") or "").strip()
    if not author:
        author = f"{raw.get('firstname', '')} {raw.get('lastname', '')}".strip()

    category = str(raw.get("category_title") or "").strip() or _name_of(
        raw.get("category"), category_names
    )
    status = str(raw.get("status_title") or "").strip() or _name_of(
        raw.get("status"), status_names
    )

    record = ExperimentRecord(
        id=int(raw["id"]),
        title=str(raw.get("title") or "").strip(),
        date=_iso_date(raw.get("date")) or _iso_date(raw.get("created_at")),
        body=body,
        tags=_parse_tags(raw),
        category=category,
        status=status,
        author=author,
        resources=_resource_names(raw),
        measurements=parse_measurements(raw),
        modified_at=str(raw.get("modified_at") or raw.get("lastchange") or ""),
    )
    record.content_hash = record.compute_hash()
    return record
