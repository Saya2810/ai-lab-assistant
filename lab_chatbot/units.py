"""Unit parsing and conversion, so numeric questions compare like with like.

"Which measurements used a wavelength above 0.5 µm?" must find an entry that
stored ``532 nm``. Every recognised quantity is therefore also stored in SI
(``value_si``) together with its physical ``dimension``; SQL compares those.

Unknown units are not an error: the value is kept as-is and can only be
compared against rows with the *same* unit text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitDef:
    dimension: str
    factor: float
    offset: float = 0.0
    symbol: str = ""

    def to_si(self, value: float) -> float:
        return value * self.factor + self.offset

    def from_si(self, si_value: float) -> float:
        return (si_value - self.offset) / self.factor


# Prefixable base units: symbol -> (dimension, factor to the SI base unit).
# SI bases used here: m, kg, s, m^3, mol, mol/m^3, Hz, V, A, W, Pa, J, N, K.
_PREFIXABLE: dict[str, tuple[str, float]] = {
    "m": ("length", 1.0),
    "g": ("mass", 1e-3),
    "s": ("time", 1.0),
    "L": ("volume", 1e-3),
    "l": ("volume", 1e-3),
    "mol": ("amount", 1.0),
    "M": ("concentration", 1000.0),
    "Hz": ("frequency", 1.0),
    "V": ("voltage", 1.0),
    "A": ("current", 1.0),
    "W": ("power", 1.0),
    "Pa": ("pressure", 1.0),
    "bar": ("pressure", 1e5),
    "J": ("energy", 1.0),
    "eV": ("energy", 1.602176634e-19),
    "N": ("force", 1.0),
    "K": ("temperature", 1.0),
}

_PREFIXES: dict[str, float] = {
    "y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9,
    "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1,
    "da": 1e1, "h": 1e2, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
}  # fmt: skip

# Units that never take a prefix.
_EXACT: dict[str, UnitDef] = {}


def _add(symbol: str, dimension: str, factor: float, offset: float = 0.0, aliases=()) -> None:
    definition = UnitDef(dimension, factor, offset, symbol)
    _EXACT[symbol] = definition
    for alias in aliases:
        _EXACT[alias] = definition


_add("min", "time", 60.0, aliases=("mins", "minute", "minutes"))
_add("h", "time", 3600.0, aliases=("hr", "hrs", "hour", "hours", "std"))
_add("d", "time", 86400.0, aliases=("day", "days"))
_add("sec", "time", 1.0, aliases=("secs", "second", "seconds", "sek"))
_add("°C", "temperature", 1.0, 273.15, aliases=("degC", "℃", "grad c"))
_add("°F", "temperature", 5.0 / 9.0, 273.15 - 160.0 / 9.0, aliases=("degF", "℉"))
_add("%", "fraction", 0.01)
_add("ppm", "fraction", 1e-6)
_add("ppb", "fraction", 1e-9)
_add("rpm", "frequency", 1.0 / 60.0)
_add("Å", "length", 1e-10, aliases=("angstrom", "Angstrom"))
_add("cm^-1", "wavenumber", 100.0, aliases=("cm-1", "1/cm", "cm⁻¹"))
_add("rad", "angle", 1.0)
_add("°", "angle", 3.141592653589793 / 180.0, aliases=("deg", "degree", "degrees"))

_CASE_FIXES = {"hz": "Hz", "ev": "eV", "pa": "Pa", "mhz": "MHz", "khz": "kHz", "ghz": "GHz"}

_NUMBER = r"[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?"
_NUMBER_FULL = re.compile(rf"^\s*({_NUMBER})\s*$")
_QUANTITY = re.compile(rf"^\s*({_NUMBER})\s*(\S.*?)?\s*$")


def _clean(unit: str) -> str:
    text = unicodedata.normalize("NFC", str(unit)).strip()
    text = text.replace("º", "°").replace(" ", " ")
    return text.replace(" ", "") if text.lower() != "grad c" else text.lower()


def lookup_unit(unit: str | None) -> UnitDef | None:
    """Return the definition of a unit symbol, or None if it is not known."""
    if not unit:
        return None
    text = _clean(unit)
    if not text:
        return None
    if text in _EXACT:
        return _EXACT[text]
    if text in _PREFIXABLE:
        dimension, factor = _PREFIXABLE[text]
        return UnitDef(dimension, factor, 0.0, text)
    for plen in (2, 1):
        prefix, base = text[:plen], text[plen:]
        if prefix in _PREFIXES and base in _PREFIXABLE:
            dimension, factor = _PREFIXABLE[base]
            return UnitDef(dimension, _PREFIXES[prefix] * factor, 0.0, text)
    fixed = _CASE_FIXES.get(text.lower())
    if fixed and fixed != text:
        return lookup_unit(fixed)
    if text.lower() in _EXACT:
        return _EXACT[text.lower()]
    return None


def unit_norm(unit: str | None) -> str | None:
    """Canonical text used to compare unknown units for equality."""
    if not unit:
        return None
    text = _clean(unit).casefold()
    return text or None


def parse_number(text: str | float | int | None) -> float | None:
    """Parse ``5``, ``5.5``, ``5,5`` (German), ``1.234,5``, ``1,234.5``, ``1e-3``."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    match = _NUMBER_FULL.match(str(text))
    if not match:
        return None
    raw = match.group(1)
    if "," in raw and "." in raw:
        if raw.rfind(".") > raw.rfind(","):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(".", "").replace(",", ".")
    elif raw.count(",") == 1:
        raw = raw.replace(",", ".")
    elif raw.count(",") > 1:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_quantity(text: str | None) -> tuple[float, str | None] | None:
    """Split ``"532 nm"`` / ``"1,5 mL"`` into ``(value, unit_text)``."""
    if not text:
        return None
    match = _QUANTITY.match(str(text))
    if not match:
        return None
    number = parse_number(match.group(1))
    if number is None:
        return None
    return number, (match.group(2) or None)


def to_si(value: float, unit: str | None) -> tuple[float, str] | None:
    """Convert to ``(si_value, dimension)``; None if the unit is unknown."""
    definition = lookup_unit(unit)
    if definition is None:
        return None
    return definition.to_si(value), definition.dimension


def from_si(si_value: float, unit: str | None) -> float | None:
    definition = lookup_unit(unit)
    if definition is None:
        return None
    return definition.from_si(si_value)


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def format_quantity(value: float | None, unit: str | None) -> str:
    text = format_number(value)
    return f"{text} {unit}".strip() if unit else text
