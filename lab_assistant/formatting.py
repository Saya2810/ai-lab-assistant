"""Make model output safe and correct for eLabFTW's Markdown + MathJax editor."""

from __future__ import annotations

import re

# split text into (is_code, segment) so we never touch code
_CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)
_WRAPPER = re.compile(r"\A\s*```(?P<lang>[A-Za-z]*)[ \t]*\n(?P<inner>.*)\n```\s*\Z", re.DOTALL)
_HEADING = re.compile(r"(?m)^#{1,6}\s")

# LaTeX commands a model sometimes emits with one backslash too many ("\\frac")
_DOUBLED = ("frac", "sqrt", "cdot", "times", "approx", "rightarrow", "Rightarrow",
            "left", "right", "text", "mathrm")  # fmt: skip
_DOUBLED_RE = re.compile(r"\\\\(" + "|".join(_DOUBLED) + r")(?![A-Za-z])")


def strip_wrapping_fence(text: str) -> str:
    """Remove ONE fence that wraps the whole answer ("```markdown ... ```").

    Code blocks *inside* the notes are legitimate and stay. A bare ``` wrapper is
    only removed when the content clearly is Markdown (has headings).
    """
    match = _WRAPPER.match(text)
    if not match:
        return text
    lang = match.group("lang").lower()
    inner = match.group("inner")
    if lang in ("markdown", "md") or (lang == "" and _HEADING.search(inner)):
        return inner
    return text


def _fix_math(segment: str) -> str:
    segment = segment.replace("\\\\[", "\\[").replace("\\\\]", "\\]")
    segment = segment.replace("\\\\(", "\\(").replace("\\\\)", "\\)")
    segment = segment.replace("\\[", "$$").replace("\\]", "$$")
    segment = segment.replace("\\(", "$").replace("\\)", "$")
    return _DOUBLED_RE.sub(r"\\\1", segment)


def normalize_math_for_elabftw(text: str) -> str:
    """\\[..\\] -> $$..$$, \\(..\\) -> $..$, undo doubled backslashes; code stays untouched."""
    if not text:
        return text
    text = strip_wrapping_fence(text)
    parts = _CODE.split(text)  # odd indices are the code spans / fences
    out = [part if index % 2 else _fix_math(part) for index, part in enumerate(parts)]
    return "".join(out).strip()
