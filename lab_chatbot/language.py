"""Answer language from a question (German or English).

Only a heuristic for the deterministic answers (counts, lists, "nothing found")
and for the language line in the answer prompt; the LLM answer itself follows it.
"""

from __future__ import annotations

import re

_DE = set(
    "der die das und ist wie viele welche welcher welches was wann mit von im den dem ein eine nicht "
    "für wurde wurden gab hat haben bei zeige zeig alle mir ich gibt experimente experiment versuch "
    "versuche eintrag einträge letzte letzten diese dieser durchschnitt höchste niedrigste wo wer".split()
)  # fmt: skip
_EN = set(
    "the and is are how many which what when with from of in a an not for was were has have at show "
    "all me i there experiments experiment entry entries last this average highest lowest where who".split()
)  # fmt: skip

# Words valid in both languages ("in", "was", "an", ...) say nothing about the language.
_COMMON = (_DE & _EN) | {"in", "an"}
_DE, _EN = _DE - _COMMON, _EN - _COMMON


def detect_language(text: str, default: str = "de") -> str:
    tokens = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(t in _DE for t in tokens)
    en = sum(t in _EN for t in tokens)
    if de == en:
        return default
    return "de" if de > en else "en"
