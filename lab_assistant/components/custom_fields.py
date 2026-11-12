"""Widgets for eLabFTW template custom fields (extra_fields).

Rules that the first version broke: never invent a value (an empty number field used
to become "0.0"), never silently pick the first unit/option when the AI's value does
not fit, and never show numbers rounded (st.number_input displays 2 decimals).
"""

from __future__ import annotations

import re

import streamlit as st

_DECIMAL = re.compile(r"^[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?$")


def find_matching_option(value, available_options):
    """Case-insensitive exact match, or None."""
    if value is None:
        return None
    wanted = str(value).strip().lower()
    for option in available_options:
        if str(option).strip().lower() == wanted:
            return option
    return None


def find_ai_measurement(field_name, ai_measurements):
    wanted = field_name.strip().lower()
    for name, measurement in (ai_measurements or {}).items():
        if name.strip().lower() == wanted:
            return measurement
    return None


def parse_decimal(text: str) -> str | None:
    """Validate user input for a number field; returns it normalised ("1,5" -> "1.5").

    "" -> "" (empty is allowed), garbage -> None. The digits are kept as typed, so
    5.32e-7 stays 5.32e-7 instead of being rounded by a widget.
    """
    text = (text or "").strip().replace("−", "-")
    if not text:
        return ""
    if not _DECIMAL.match(text):
        return None
    return text.replace(",", ".")


def unmatched_measurements(fields: dict, ai_measurements: dict) -> list[str]:
    """AI measurements that no custom field of the template picked up."""
    field_names = {name.strip().lower() for name in fields}
    return [name for name in (ai_measurements or {}) if name.strip().lower() not in field_names]


def _render_number(label, data, current, ai_unit, key):
    units = data.get("units") or []
    current_unit = ai_unit or data.get("unit")
    text = "" if current in (None, "") else str(current)
    error = None

    value_col, unit_col = st.columns([3, 1]) if units else (st.container(), None)
    with value_col:
        raw = st.text_input(label, value=text, key=f"{key}_value", help="Decimal point or comma; e.g. 532, 1,5, 5.32e-7")
    normalised = parse_decimal(raw)
    if normalised is None:
        error = f"'{raw}' is not a number."
        st.error(f"{label}: {error}")
        normalised = raw

    unit = current_unit
    if units:
        matched = find_matching_option(current_unit, units)
        with unit_col:
            unit = st.selectbox(
                "Unit", options=units, index=units.index(matched) if matched else None,
                placeholder="choose unit", key=f"{key}_unit",
            )  # fmt: skip
        if current_unit and not matched:
            st.warning(f"{label}: unit '{current_unit}' is not allowed for this field – please choose one.")
        if normalised and unit is None:
            error = error or "Please choose a unit."
            st.error(f"{label}: {error}")

    result = dict(data)
    result["value"] = normalised
    if unit is not None:
        result["unit"] = unit
    return result, error


def _render_checkbox(label, data, current, key):
    checked = str(current).strip().lower() in ("1", "true", "yes", "on")
    value = st.checkbox(label, value=checked, key=key)
    return {**data, "value": "1" if value else "0"}, None


def _render_select(label, data, current, key):
    options = data.get("options") or []
    if not options:
        return {**data, "value": st.text_input(label, value=str(current or ""), key=key)}, None
    if data.get("allow_multi_values"):
        wanted = current if isinstance(current, list) else [current]
        default = [m for m in (find_matching_option(w, options) for w in wanted) if m]
        return {**data, "value": st.multiselect(label, options, default=default, key=key)}, None
    matched = find_matching_option(current, options)
    value = st.selectbox(
        label, options, index=options.index(matched) if matched else None,
        placeholder="choose", key=key,
    )  # fmt: skip
    return {**data, "value": value or ""}, None


def render_custom_field(field_name, field_data, ai_measurements, key_prefix=""):
    """Draw one field; returns ``(field_dict_with_value, error_or_None)``."""
    key = f"{key_prefix}cf_{field_name}"
    field_type = field_data.get("type", "text")
    current = field_data.get("value", "")
    ai_unit = None

    measurement = find_ai_measurement(field_name, ai_measurements)
    if measurement is not None:
        if isinstance(measurement, dict):
            current = measurement.get("value", current)
            ai_unit = measurement.get("unit")
        else:
            current = measurement

    if field_type == "number":
        return _render_number(field_name, field_data, current, ai_unit, key)
    if field_type == "checkbox":
        return _render_checkbox(field_name, field_data, current, key)
    if field_type in ("select", "dropdown", "radio"):
        return _render_select(field_name, field_data, current, key)
    # text, date, time, email, url, ... : plain text, the user types what eLabFTW expects
    value = st.text_input(field_name, value="" if current is None else str(current), key=key)
    return {**field_data, "value": value}, None
