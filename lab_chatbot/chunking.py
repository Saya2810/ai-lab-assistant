"""Markdown-aware chunking for lab notebook entries.

Why not "one experiment = one vector" (the old design)?

* long entries were truncated by the embedding model, silently losing the
  results/conclusion at the end;
* five whole entries in the prompt overflow small context windows.

Rules implemented here:

1. split on Markdown headings, keep ``$$ ... $$`` math and code fences atomic;
2. pack consecutive blocks into chunks of at most ``max_chars``;
3. hard-split (with overlap) only blocks that are too long on their own;
4. every chunk starts with a header carrying id, title, date, tags, category
   and status, so the experiment id reaches the LLM and tags are searchable;
5. custom-field measurements get their own chunk, so "532 nm" is findable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lab_chatbot.parse import ExperimentRecord
from lab_chatbot.units import format_quantity

CHUNKER_VERSION = "1"

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 120

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkText:
    idx: int
    text: str


def build_header(record: ExperimentRecord) -> str:
    first = [f"Experiment {record.id}", record.title or "Untitled"]
    if record.date:
        first.append(record.date)
    lines = ["[" + " | ".join(first) + "]"]
    extras = []
    if record.tags:
        extras.append("Tags: " + ", ".join(record.tags))
    if record.category:
        extras.append(f"Category: {record.category}")
    if record.status:
        extras.append(f"Status: {record.status}")
    if record.resources:
        extras.append("Resources: " + ", ".join(record.resources))
    if extras:
        lines.append("[" + " | ".join(extras) + "]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# block splitting
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    path: str  # heading path, e.g. "Messung 1 > Ergebnis"
    text: str
    is_heading: bool = False
    atomic: bool = False  # fenced code / display math: never cut mid-way if it can be avoided


# An atomic block up to this size stays in ONE chunk even if that exceeds max_chars
# (embedding models take far more than 1200 characters; a cut formula is useless).
ATOMIC_LIMIT = 2400


def _split_blocks(body: str) -> list[_Block]:
    """Split into paragraphs/headings; math and code blocks stay in one piece."""
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False
    in_math = False

    def current_path() -> str:
        return " > ".join(title for _, title in stack)

    def flush(atomic: bool = False) -> None:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(_Block(current_path(), text, atomic=atomic))
            buffer.clear()

    for line in body.split("\n"):
        stripped = line.strip()
        if not in_math and _FENCE.match(line):
            if not in_fence:
                flush()  # a paragraph right before the fence stays its own block
            in_fence = not in_fence
            buffer.append(line)
            if not in_fence:
                flush(atomic=True)
            continue
        if in_fence:
            buffer.append(line)
            continue
        if stripped.count("$$") % 2 == 1:
            if not in_math:
                flush()
            in_math = not in_math
            buffer.append(line)
            if not in_math:
                flush(atomic=True)
            continue
        if in_math:
            buffer.append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            blocks.append(_Block(current_path(), line.strip(), is_heading=True))
        elif not stripped:
            flush()
        else:
            buffer.append(line)
    flush()
    return blocks


def _split_lines(text: str, limit: int) -> list[str]:
    """Split an oversized atomic block (code / formula) at line boundaries."""
    pieces: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single endless line: unavoidable character cut
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = line
    if current:
        pieces.append(current)
    return pieces


def _hard_split(text: str, limit: int, overlap: int) -> list[str]:
    """Split an oversized block: sentences first, characters as a last resort."""
    if len(text) <= limit:
        return [text]
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            step = max(1, limit - overlap)
            for start in range(0, len(sentence), step):
                pieces.append(sentence[start : start + limit])
                if start + limit >= len(sentence):
                    break
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def chunk_experiment(
    record: ExperimentRecord,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkText]:
    """Return the chunks of one experiment; always at least one."""
    header = build_header(record)
    budget = max(200, max_chars - len(header) - 40)

    # 1. blocks, with oversized ones broken up
    units: list[_Block] = []
    for block in _split_blocks(record.body):
        if block.atomic:
            limit = max(budget, ATOMIC_LIMIT - len(header))
            pieces = [block.text] if len(block.text) <= limit else _split_lines(block.text, limit)
        else:
            pieces = _hard_split(block.text, budget, overlap)
        for piece in pieces:
            units.append(_Block(block.path, piece, block.is_heading, block.atomic))

    # 2. greedy packing; a heading never ends a chunk (it is glued to what follows)
    bodies: list[tuple[str, str]] = []  # (context path, text)
    current: list[_Block] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            bodies.append((current[0].path, "\n\n".join(b.text for b in current)))
        current, size = [], 0

    for index, unit in enumerate(units):
        extra = len(unit.text) + 2
        if current and size + extra > budget:
            # keep a trailing heading together with its first block
            if current[-1].is_heading:
                carried = current.pop()
                size -= len(carried.text) + 2
                flush()
                current, size = [carried], len(carried.text) + 2
            else:
                flush()
        current.append(unit)
        size += extra
    flush()

    chunks: list[ChunkText] = []
    for path, text in bodies:
        first_is_heading = text.lstrip().startswith("#")
        context = f"Section: {path}\n" if path and not first_is_heading else ""
        chunks.append(ChunkText(len(chunks), f"{header}\n{context}{text}"))

    # 3. measurements chunk
    if record.measurements:
        lines = [
            f"- {m.name}: {format_quantity(m.value_num, m.unit) if m.value_num is not None and m.unit else m.value_text}"
            for m in record.measurements
        ]
        text = f"{header}\nMeasurements (custom fields):\n" + "\n".join(lines)
        for piece in _hard_split(text, max_chars, 0):
            chunks.append(ChunkText(len(chunks), piece))

    # 4. never return nothing: title/tags alone are searchable
    if not chunks:
        chunks.append(ChunkText(0, header))
    return chunks
