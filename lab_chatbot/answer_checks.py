"""Checks on a generated answer: which sources it cites and which numbers it made up.

A cheap guard against the classic local-LLM failure of "improving" a value:
every number in the answer must appear in the evidence it was given.
"""

from __future__ import annotations

import re
from typing import Sequence

from lab_chatbot.answer import Source
from lab_chatbot.prompts import t

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def parse_citations(text: str, n_sources: int) -> list[int]:
    cited: list[int] = []
    for match in _CITATION.finditer(text):
        for number in re.findall(r"\d+", match.group(1)):
            value = int(number)
            if 1 <= value <= n_sources and value not in cited:
                cited.append(value)
    return sorted(cited)


def _numbers(text: str) -> set[str]:
    found = set()
    for token in _NUMBER.findall(text):
        value = token.replace(",", ".")
        try:
            found.add(f"{float(value):g}")
        except ValueError:
            continue
    return found


def unsupported_numbers(answer: str, evidence: Sequence[str]) -> list[str]:
    """Numbers in the answer that appear nowhere in the evidence.

    Citation markers, list numbering and small integers (0-20: counts and
    ordinals the model legitimately writes itself) are ignored.
    """
    cleaned = _CITATION.sub(" ", answer)
    cleaned = re.sub(r"(?m)^\s*\d+[.)]\s", " ", cleaned)
    allowed: set[str] = set()
    for chunk in evidence:
        allowed |= _numbers(chunk)
    missing = []
    for token in _NUMBER.findall(cleaned):
        try:
            value = float(token.replace(",", "."))
        except ValueError:
            continue
        key = f"{value:g}"
        if key in allowed:
            continue
        if value == int(value) and 0 <= value <= 20:
            continue
        if token not in missing:
            missing.append(token)
    return missing


def check_answer(
    text: str, sources: Sequence[Source], extra_evidence: Sequence[str], language: str
) -> tuple[list[int], list[str]]:
    """``(cited source numbers, warnings for the user)`` of a generated answer."""
    cited = parse_citations(text, len(sources))
    warnings = []
    missing = unsupported_numbers(text, [s.text for s in sources] + list(extra_evidence))
    if missing:
        warnings.append(t(language, "warn_numbers", numbers=", ".join(missing[:8])))
    if not cited and text:
        warnings.append(t(language, "warn_uncited"))
    return cited, warnings
