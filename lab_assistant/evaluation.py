"""Repeatable evaluation of the draft generation against hand-labelled lab notes.

Run with ``python -m lab_assistant eval``. The result is a plain, JSON-serialisable
dict ``{"summary": {...}, "cases": [{...}, ...]}`` so per-case numbers can be plotted.

Precision is reported twice: against the reference only (``precision``) and counting
values that are not in the reference but stand in the notes exactly like that as
correct extras (``supported_precision``). The reference lists the essential values of
a note, not every number. Errors are a value found nowhere in the notes
(``unsupported``) and a value the reference has under another name (``misassigned``,
e.g. the water volume labelled as stock solution).

Matching rule for measurements (one-to-one, greedy): a predicted measurement
matches an expected one when
  * the values are equal after parsing (German decimal comma accepted, relative
    tolerance 1e-6; non-numeric values are compared as normalised strings),
  * the units are equivalent: case-insensitive, whitespace removed, "u" == "µ" == "μ",
    and a degree sign is optional ("°C" == "C" == "degC"),
  * the names are similar: after casefolding and accent folding (ä -> a, µ stays) one
    contains the other, or difflib's ratio is >= 0.6.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Callable

from lab_assistant.ai import Draft

DEFAULT_CASES = Path(__file__).with_name("eval_demo_assistant.json")
NAME_RATIO = 0.6
REL_TOL = 1e-6


def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    for case in cases:
        case.setdefault("expect_measurements", [])
        case.setdefault("expect_title_words", [])
    return cases


# ---------------------------------------------------------------- normalisation


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(c for c in text if not unicodedata.combining(c)).strip()


def _norm_unit(unit: object) -> str:
    text = unicodedata.normalize("NFKC", str(unit or "")).casefold()
    text = re.sub(r"\s+", "", text).replace("μ", "µ").replace("°", "").replace("deg", "")
    return re.sub(r"(^|[/·*])u(?=[a-zµ])", r"\1µ", text)


def _parse_value(value: object) -> float | str:
    text = str(value if value is not None else "").strip().replace("−", "-").replace("–", "-")
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return re.sub(r"\s+", "", text).casefold()


_NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _value_in_source(value: object, notes: str) -> bool:
    """Does every number of ``value`` stand in the notes (decimal comma or point)?"""
    tokens = _NUMBER_TOKEN.findall(str(value))
    if not tokens:  # a text value such as "RT": present as a word of the notes?
        text = str(value).strip()
        return bool(text) and re.search(rf"(?<!\w){re.escape(text)}(?!\w)", notes, re.I) is not None
    for token in tokens:
        spellings = {token, token.replace(",", "."), token.replace(".", ",")}
        if not any(re.search(rf"(?<![\d.,]){re.escape(t)}(?![\d]|[.,]\d)", notes) for t in spellings):
            return False
    return True


def _same_value(a: object, b: object) -> bool:
    pa, pb = _parse_value(a), _parse_value(b)
    if isinstance(pa, float) and isinstance(pb, float):
        return math.isclose(pa, pb, rel_tol=REL_TOL, abs_tol=1e-12)
    return pa == pb


def _same_name(a: str, b: str) -> bool:
    fa, fb = _fold(a), _fold(b)
    if not fa or not fb:
        return False
    return fa in fb or fb in fa or difflib.SequenceMatcher(None, fa, fb).ratio() >= NAME_RATIO


def _matches(pred: dict, exp: dict) -> bool:
    return (
        _same_value(pred["value"], exp["value"])
        and _norm_unit(pred["unit"]) == _norm_unit(exp["unit"])
        and _same_name(pred["name"], exp["name"])
    )


def _predicted(draft: Draft) -> list[dict]:
    out = []
    for name, entry in (draft.measurements or {}).items():
        entry = entry if isinstance(entry, dict) else {"value": entry}
        out.append({"name": name, "value": entry.get("value", ""), "unit": entry.get("unit", "")})
    return out


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp + fp + fn == 0:
        return 1.0, 1.0, 1.0  # nothing expected, nothing predicted: perfect
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- evaluation


def _score_case(case: dict, draft: Draft | None, error: str | None, latency: float) -> dict:
    expected = case["expect_measurements"]
    predicted = _predicted(draft) if draft else []
    unused = list(range(len(predicted)))
    tp = 0
    for exp in expected:
        hit = next((i for i in unused if _matches(predicted[i], exp)), None)
        if hit is not None:
            unused.remove(hit)
            tp += 1
    fp, fn = len(predicted) - tp, len(expected) - tp
    p, r, f1 = _prf(tp, fp, fn)
    # The reference lists the essential values, not every number of a note. A value the
    # model extracted that is not in the reference but stands in the note exactly like
    # that is an "extra", not an error; only a value found nowhere in the note is wrong.
    unmatched = [predicted[i] for i in unused]
    # a value the reference has under another name is mislabelled ("Stammlösung: 90 µL"
    # when the 90 µL were the water): an error even though the number is in the notes
    misassigned = sum(
        any(_same_value(m["value"], e["value"]) and _norm_unit(m["unit"]) == _norm_unit(e["unit"]) for e in expected)
        for m in unmatched
    )
    extra = sum(
        _value_in_source(m["value"], case["notes"])
        and not any(_same_value(m["value"], e["value"]) and _norm_unit(m["unit"]) == _norm_unit(e["unit"]) for e in expected)
        for m in unmatched
    )
    unsupported = len(unmatched) - extra - misassigned
    supported_precision = (tp + extra) / len(predicted) if predicted else 1.0
    words = case["expect_title_words"]
    title = _fold(draft.title) if draft else ""
    title_recall = sum(_fold(w) in title for w in words) / len(words) if words else None
    return {
        "id": case["id"],
        "latency_s": round(latency, 3),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "extra": extra, "misassigned": misassigned, "unsupported": unsupported,
        "supported_precision": supported_precision,
        "invented_numbers": len(draft.numbers.invented) if draft else 0,
        "missing_numbers": len(draft.numbers.missing) if draft else 0,
        "title_word_recall": title_recall,
        "title": draft.title if draft else "",
        "error": error,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summarise(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    micro_p, micro_r, micro_f1 = _prf(tp, fp, fn)
    lat = [r["latency_s"] for r in rows]
    title = [r["title_word_recall"] for r in rows if r["title_word_recall"] is not None]
    return {
        "n_cases": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "micro_supported_precision": (
            (tp + sum(r["extra"] for r in rows)) / (tp + fp) if tp + fp else 1.0
        ),
        "unsupported_values": sum(r["unsupported"] for r in rows),
        "misassigned_values": sum(r["misassigned"] for r in rows),
        "macro_precision": _mean([r["precision"] for r in rows]),
        "macro_recall": _mean([r["recall"] for r in rows]),
        "macro_f1": _mean([r["f1"] for r in rows]),
        "mean_invented_numbers": _mean([r["invented_numbers"] for r in rows]),
        "share_drafts_with_invented": _mean([1.0 if r["invented_numbers"] else 0.0 for r in rows]),
        "mean_title_word_recall": _mean(title),
        "latency_mean_s": _mean(lat),
        "latency_median_s": statistics.median(lat) if lat else 0.0,
        "latency_p95_s": _percentile(lat, 0.95),
    }


def evaluate_assistant(cases: list[dict], generate: Callable[[str], Draft]) -> dict:
    rows = []
    for case in cases:
        draft, error = None, None
        start = time.perf_counter()
        try:
            draft = generate(case["notes"])
        except Exception as exc:  # a failed case counts as all-missed, the run goes on
            error = f"{type(exc).__name__}: {exc}"
        rows.append(_score_case(case, draft, error, time.perf_counter() - start))
    return {"summary": _summarise(rows), "cases": rows}


def format_report(result: dict) -> str:
    lines = [f"{'id':<24}{'P':>6}{'R':>6}{'F1':>6}{'inv':>5}{'lat[s]':>8}"]
    for c in result["cases"]:
        flag = "  ERROR: " + c["error"] if c["error"] else ""
        lines.append(
            f"{c['id'][:23]:<24}{c['precision']:>6.2f}{c['recall']:>6.2f}{c['f1']:>6.2f}"
            f"{c['invented_numbers']:>5}{c['latency_s']:>8.1f}{flag}"
        )
    lines.append("")
    lines += [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in result["summary"].items()]
    return "\n".join(lines)
