"""Compare the AI draft with its source: flag numbers it invented or dropped.

The model is told to copy every value verbatim, but prompts are not guarantees.
This check is deliberately simple and *advisory*: it shows the user what to look
at before anything is written into the lab notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?(?:[eE][-+−]?\d+)?")
# lines our own pipeline adds around the sources: they are not experimental facts
_LABEL_LINE = re.compile(r"(?mi)^\s*(?:SOURCE:|FILE:|-{3,}.*-{3,}\s*$|-{10,}\s*$).*$")
_CODE_OR_MATH_DELIM = re.compile(r"(?m)^\s*(?:\d+[.)])\s")  # "1. item" list numbering


@dataclass
class NumberCheck:
    invented: list[str] = field(default_factory=list)  # in the draft, not in the source
    missing: list[str] = field(default_factory=list)  # in the source, not in the draft

    @property
    def clean(self) -> bool:
        return not (self.invented or self.missing)


def _key(token: str) -> str | None:
    value = token.replace("−", "-").replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return None
    return f"{abs(number):g}"  # sign-insensitive on purpose: "-3" vs "−3" vs "3 (negative)"


def numbers_in(text: str, *, drop_labels: bool = False) -> dict[str, str]:
    """``{normalised: first spelling}`` for every number in ``text`` (single digits are ignored)."""
    if drop_labels:
        text = _LABEL_LINE.sub(" ", text)
    text = _CODE_OR_MATH_DELIM.sub(" ", text)
    found: dict[str, str] = {}
    for token in _NUMBER.findall(text):
        key = _key(token)
        if key is None:
            continue
        digits = re.sub(r"\D", "", token)
        if len(digits) < 2 and "." not in key:  # 0-9: counts, ordinals, list markers
            continue
        found.setdefault(key, token)
    return found


def compare_numbers(source: str, draft: str) -> NumberCheck:
    src = numbers_in(source, drop_labels=True)
    out = numbers_in(draft)
    return NumberCheck(
        invented=[out[k] for k in out if k not in src],
        missing=[src[k] for k in src if k not in out],
    )
