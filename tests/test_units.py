from __future__ import annotations

import pytest

from lab_chatbot.units import (
    format_number,
    format_quantity,
    from_si,
    lookup_unit,
    parse_number,
    parse_quantity,
    to_si,
)


@pytest.mark.parametrize(
    "value, unit, expected, dimension",
    [
        (532, "nm", 5.32e-7, "length"),
        (1, "µm", 1e-6, "length"),
        (20, "mW", 0.02, "power"),
        (5, "min", 300, "time"),
        (200, "ms", 0.2, "time"),
        (68, "°F", 293.15, "temperature"),
        (25, "°C", 298.15, "temperature"),
        (300, "rpm", 5, "frequency"),
    ],
)
def test_to_si(value, unit, expected, dimension):
    si, dim = to_si(value, unit)
    assert dim == dimension
    assert si == pytest.approx(expected)


def test_round_trip():
    si, _ = to_si(37, "°C")
    assert from_si(si, "°C") == pytest.approx(37)


def test_unknown_unit_is_not_converted():
    assert lookup_unit("furlong") is None
    assert to_si(3, "furlong") is None
    assert to_si(3, None) is None


def test_m_is_metre_but_mm_is_millimetre_and_M_is_molar():
    assert lookup_unit("m").dimension == "length"
    assert lookup_unit("mm").factor == pytest.approx(1e-3)
    assert lookup_unit("M").dimension == "concentration"
    assert lookup_unit("mM").factor == pytest.approx(1e-3 * lookup_unit("M").factor)


@pytest.mark.parametrize(
    "text, expected",
    [("1,5", 1.5), ("1.5", 1.5), ("1 000", None), ("-2", -2.0), ("abc", None), (None, None), (3, 3.0)],
)
def test_parse_number(text, expected):
    result = parse_number(text)
    assert (result is None) if expected is None else result == pytest.approx(expected)


def test_parse_quantity():
    assert parse_quantity("532 nm") == (532.0, "nm")
    assert parse_quantity("0,5 mM") == (0.5, "mM")
    assert parse_quantity("42") == (42.0, None)
    assert parse_quantity("kein Wert") is None


def test_formatting():
    assert format_number(30.0) == "30"
    assert format_number(0.2) == "0.2"
    assert format_quantity(18.4, "mW") == "18.4 mW"
    assert format_quantity(5, None) == "5"
