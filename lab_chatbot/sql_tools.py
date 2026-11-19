"""Exact, parametrised SQL building blocks for structured questions.

This is deliberately *not* free-form text-to-SQL. A 7B local model writes
wrong or unsafe SQL too often, and a wrong number in a lab notebook answer is
worse than no answer. Instead the planner (``planner.py``) fills in a small,
validated description (``Filters`` + ``MeasurementCond``) and *these* functions
build the SQL. User text never becomes SQL; it only ever becomes parameters.
"""

from __future__ import annotations

import bisect
import difflib
import re
import statistics
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from lab_chatbot.retrieval import fold, query_tokens
from lab_chatbot.store import Store, id_list, norm
from lab_chatbot.units import from_si, lookup_unit, to_si, unit_norm

_WORDS = re.compile(r"\w+", re.UNICODE)
OPS = {">", ">=", "<", "<=", "=", "between"}
GROUP_BYS = {"tag", "category", "status", "author", "month"}
ORDERS = {
    "date_desc": "e.date DESC, e.id DESC",
    "date_asc": "e.date ASC, e.id ASC",
    "id_asc": "e.id ASC",
}


@dataclass
class Filters:
    date_from: str | None = None
    date_to: str | None = None
    tags_all: list[str] = field(default_factory=list)
    tags_any: list[str] = field(default_factory=list)
    category: str | None = None
    status: str | None = None
    author: str | None = None
    experiment_ids: list[int] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.date_from or self.date_to or self.tags_all or self.tags_any
            or self.category or self.status or self.author or self.experiment_ids
        )  # fmt: skip

    def describe(self) -> list[str]:
        parts = []
        if self.experiment_ids:
            parts.append("Experiment " + ", ".join(map(str, self.experiment_ids)))
        if self.date_from or self.date_to:
            parts.append(f"{self.date_from or '…'} – {self.date_to or '…'}")
        if self.tags_all:
            parts.append("Tags: " + " + ".join(self.tags_all))
        if self.tags_any:
            parts.append("Tags (any): " + ", ".join(self.tags_any))
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.status:
            parts.append(f"Status: {self.status}")
        if self.author:
            parts.append(f"Author: {self.author}")
        return parts


@dataclass
class MeasurementCond:
    name: str
    op: str | None = None
    value: float | None = None
    value2: float | None = None
    unit: str | None = None

    def describe(self) -> str:
        if not self.op or self.value is None:
            return f"has {self.name}"
        unit = f" {self.unit}" if self.unit else ""
        if self.op == "between":
            return f"{self.name} between {self.value:g} and {self.value2:g}{unit}"
        return f"{self.name} {self.op} {self.value:g}{unit}"


# ---------------------------------------------------------------------------
# WHERE builder
# ---------------------------------------------------------------------------


def _measurement_clause(cond: MeasurementCond) -> tuple[str, list]:
    params: list = [norm(cond.name)]
    sql = "m.experiment_id = e.id AND m.name_norm = ?"
    if cond.op:
        if cond.op not in OPS:
            raise ValueError(f"unsupported comparison operator: {cond.op!r}")
        if cond.value is None:
            raise ValueError("a comparison needs a value")
        if cond.op == "between" and cond.value2 is None:
            raise ValueError("'between' needs two values")

        low, high = cond.value, cond.value2
        converted = to_si(cond.value, cond.unit) if cond.unit else None
        if converted is not None:
            dimension = converted[1]
            low = converted[0]
            high = to_si(cond.value2, cond.unit)[0] if cond.value2 is not None else None
            column = "m.value_si"
            sql += " AND m.dimension = ?"
            params.append(dimension)
        else:
            # Unknown/missing unit: compare raw numbers, same unit text only.
            column = "m.value_num"
            sql += " AND m.value_num IS NOT NULL"
            if cond.unit:
                sql += " AND m.unit_norm = ?"
                params.append(unit_norm(cond.unit))

        if cond.op in (">", ">=", "<", "<="):
            sql += f" AND {column} {cond.op} ?"
            params.append(low)
        elif cond.op == "=":
            sql += f" AND ABS({column} - ?) <= ABS(?) * 1e-9 + 1e-300"
            params += [low, low]
        else:  # between
            sql += f" AND {column} BETWEEN ? AND ?"
            params += [min(low, high), max(low, high)]
    return f"EXISTS (SELECT 1 FROM measurements m WHERE {sql})", params


def build_where(filters: Filters, meas: MeasurementCond | None = None) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []

    if filters.experiment_ids:
        clauses.append("e.id IN (SELECT value FROM json_each(?))")
        params.append(id_list(filters.experiment_ids))
    if filters.date_from or filters.date_to:
        clauses.append("e.date != ''")
    if filters.date_from:
        clauses.append("e.date >= ?")
        params.append(filters.date_from)
    if filters.date_to:
        clauses.append("e.date <= ?")
        params.append(filters.date_to)
    for tag in filters.tags_all:
        clauses.append(
            "EXISTS (SELECT 1 FROM exp_tags t WHERE t.experiment_id = e.id AND t.tag_norm = ?)"
        )
        params.append(norm(tag))
    if filters.tags_any:
        marks = ",".join("?" * len(filters.tags_any))
        clauses.append(
            f"EXISTS (SELECT 1 FROM exp_tags t WHERE t.experiment_id = e.id "
            f"AND t.tag_norm IN ({marks}))"
        )
        params += [norm(t) for t in filters.tags_any]
    if filters.category:
        clauses.append("lower(e.category) = ?")
        params.append(norm(filters.category))
    if filters.status:
        clauses.append("lower(e.status) = ?")
        params.append(norm(filters.status))
    if filters.author:
        clauses.append("lower(e.author) LIKE ?")
        params.append(f"%{norm(filters.author)}%")
    if meas is not None:
        clause, meas_params = _measurement_clause(meas)
        clauses.append(clause)
        params += meas_params
    return (" AND ".join(clauses) or "1=1"), params


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------


def candidate_ids(
    store: Store, filters: Filters, meas: MeasurementCond | None = None
) -> set[int] | None:
    """Experiment ids matching the structured filters; None = no restriction."""
    if filters.is_empty() and meas is None:
        return None
    where, params = build_where(filters, meas)
    with store.connect() as con:
        rows = con.execute(f"SELECT e.id FROM experiments e WHERE {where}", params).fetchall()
    return {r[0] for r in rows}


@dataclass
class ListResult:
    rows: list[dict]
    total: int


def list_experiments(
    store: Store,
    filters: Filters,
    meas: MeasurementCond | None = None,
    order: str = "date_desc",
    limit: int = 20,
) -> ListResult:
    where, params = build_where(filters, meas)
    order_sql = ORDERS.get(order, ORDERS["date_desc"])
    with store.connect() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM experiments e WHERE {where}", params
        ).fetchone()[0]
        rows = con.execute(
            f"SELECT e.id, e.title, e.date, e.category, e.status, e.author, "
            f"(SELECT group_concat(t.tag, ', ') FROM exp_tags t WHERE t.experiment_id = e.id) "
            f"AS tags FROM experiments e WHERE {where} ORDER BY {order_sql} LIMIT ?",
            [*params, int(limit)],
        ).fetchall()
    return ListResult([dict(r) for r in rows], total)


def count_experiments(
    store: Store,
    filters: Filters,
    meas: MeasurementCond | None = None,
    group_by: str | None = None,
) -> list[dict]:
    """``[{"group": None, "count": n}]`` or one row per group, largest first."""
    where, params = build_where(filters, meas)
    if group_by is not None and group_by not in GROUP_BYS:
        raise ValueError(f"unsupported group_by: {group_by!r}")
    with store.connect() as con:
        if group_by is None:
            n = con.execute(
                f"SELECT COUNT(*) FROM experiments e WHERE {where}", params
            ).fetchone()[0]
            return [{"group": None, "count": n}]
        if group_by == "tag":
            sql = (
                f"SELECT MIN(t.tag) AS grp, COUNT(DISTINCT e.id) AS n FROM experiments e "
                f"JOIN exp_tags t ON t.experiment_id = e.id WHERE {where} "
                f"GROUP BY t.tag_norm ORDER BY n DESC, grp"
            )
        elif group_by == "month":
            sql = (
                f"SELECT substr(e.date, 1, 7) AS grp, COUNT(*) AS n FROM experiments e "
                f"WHERE {where} GROUP BY grp ORDER BY grp"
            )
        else:
            column = {"category": "e.category", "status": "e.status", "author": "e.author"}[group_by]
            sql = (
                f"SELECT {column} AS grp, COUNT(*) AS n FROM experiments e "
                f"WHERE {where} GROUP BY {column} ORDER BY n DESC, grp"
            )
        rows = con.execute(sql, params).fetchall()
    return [{"group": r["grp"] or "(none)", "count": r["n"]} for r in rows]


def measurement_values(
    store: Store,
    filters: Filters,
    cond: MeasurementCond,
    order: str = "date_desc",
    limit: int | None = 50,
) -> list[dict]:
    """The matching measurement values themselves, one row per experiment."""
    where, params = build_where(filters, cond)
    sql = (
        f"SELECT e.id AS experiment_id, e.title, e.date, m.name, m.value_text, m.value_num, "
        f"m.unit, m.dimension, m.value_si, m.unit_norm FROM experiments e "
        f"JOIN measurements m ON m.experiment_id = e.id AND m.name_norm = ? "
        f"WHERE {where} ORDER BY {ORDERS.get(order, ORDERS['date_desc'])}"
    )
    args = [norm(cond.name), *params]
    if limit:
        sql += " LIMIT ?"
        args.append(int(limit))
    with store.connect() as con:
        rows = con.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def aggregate_measurement(
    store: Store,
    filters: Filters,
    cond: MeasurementCond,
    unit: str | None = None,
) -> list[dict]:
    """min/max/avg/sum/count of one measurement, grouped by physical dimension.

    Values stored in different units of the same dimension (nm and µm) are
    combined through SI and reported in ``unit`` (if it fits) or in the most
    common stored unit. Values whose unit is unknown are grouped per unit text
    and never mixed with anything else.
    """
    rows = measurement_values(store, filters, cond, limit=None)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["value_num"] is None:
            continue
        if row["dimension"] and row["value_si"] is not None:
            groups[("dim", row["dimension"])].append(row)
        else:
            groups[("raw", row["unit_norm"] or "")].append(row)

    result = []
    for (kind, label), members in groups.items():
        if kind == "dim":
            wanted = unit if unit and (lookup_unit(unit) and lookup_unit(unit).dimension == label) else None
            display = wanted or Counter(m["unit"] for m in members).most_common(1)[0][0]
            values = [m["value_si"] for m in members]
            convert = lambda v: from_si(v, display)  # noqa: E731
        else:
            display = members[0]["unit"]
            values = [m["value_num"] for m in members]
            convert = lambda v: v  # noqa: E731
        low = min(range(len(values)), key=values.__getitem__)
        high = max(range(len(values)), key=values.__getitem__)
        result.append(
            {
                "name": members[0]["name"],
                "unit": display,
                "n_values": len(values),
                "n_experiments": len({m["experiment_id"] for m in members}),
                "min": convert(values[low]),
                "min_experiment": members[low]["experiment_id"],
                "max": convert(values[high]),
                "max_experiment": members[high]["experiment_id"],
                "avg": convert(statistics.fmean(values)),
                "sum": convert(sum(values)),
            }
        )
    result.sort(key=lambda g: -g["n_values"])
    return result


# ---------------------------------------------------------------------------
# catalog (what exists: given to the planner so it picks real names)
# ---------------------------------------------------------------------------


@dataclass
class CatalogMeasurement:
    name: str
    count: int
    unit: str | None
    dimension: str | None


class _NameIndex:
    """All distinct values of one kind (tags, categories, ...) with a word index.

    ``entries`` are (display, norm, count) sorted by count descending; matching works on
    accent-folded text so 'Größe' meets 'grosse'.
    """

    def __init__(self, entries: list[tuple[str, str, int]]):
        self.entries = entries
        self.folded = [fold(n) for _, n, _ in entries]
        self.by_norm = {n: i for i, (_, n, _) in enumerate(entries)}
        words: dict[str, list[int]] = defaultdict(list)
        for i, text in enumerate(self.folded):
            for w in set(_WORDS.findall(text)):
                words[w].append(i)
        self.words = words
        self.sorted_words = sorted(words)

    def frequent(self, limit: int) -> list[int]:
        return list(range(min(limit, len(self.entries))))

    def relevant(self, question: str, tokens: list[str], limit: int) -> list[int]:
        """Indexes of entries that the question names (exact, prefix/substring, or typo)."""
        folded_q = fold(question)
        hits: set[int] = set()
        # 1. the whole value occurs in the question
        padded = f" {' '.join(_WORDS.findall(folded_q))} "
        for i, text in enumerate(self.folded):
            if len(text) >= 3 and (
                f" {text} " in padded or (len(text) >= 5 and text in folded_q)
            ):
                hits.add(i)
        # 2. a question token matches a word of the value (exact, prefix, or fuzzy)
        for tok in tokens:
            if len(tok) < 3:
                continue
            if tok in self.words:
                hits.update(self.words[tok])
            if len(tok) >= 4:
                stem_ = tok[: max(4, len(tok) - 2)]
                lo = bisect.bisect_left(self.sorted_words, stem_)
                for w in self.sorted_words[lo:]:
                    if not w.startswith(stem_):
                        break
                    hits.update(self.words[w])
                for w in difflib.get_close_matches(tok, self.sorted_words, n=3, cutoff=0.84):
                    hits.update(self.words[w])
        return sorted(hits)[:limit]


class _FullCatalog:
    """Process-level snapshot of every distinct name, valid for one store revision."""

    def __init__(self, store: Store):
        with store.connect() as con:
            head = con.execute(
                "SELECT COUNT(*), MIN(NULLIF(date, '')), MAX(NULLIF(date, '')) FROM experiments"
            ).fetchone()
            self.head = (head[0], head[1] or "", head[2] or "")
            self.tags = _NameIndex(
                [
                    (r[0], r[1], r[2])
                    for r in con.execute(
                        "SELECT MIN(tag), tag_norm, COUNT(*) AS n FROM exp_tags "
                        "GROUP BY tag_norm ORDER BY n DESC, 1"
                    )
                ]
            )
            self.columns: dict[str, _NameIndex] = {}
            for column in ("category", "status", "author"):
                self.columns[column] = _NameIndex(
                    [
                        (r[0], norm(r[0]), r[1])
                        for r in con.execute(
                            f"SELECT MIN({column}), COUNT(*) AS n FROM experiments "
                            f"WHERE {column} != '' GROUP BY lower({column}) ORDER BY n DESC, 1"
                        )
                    ]
                )
            units: dict[str, tuple[int, str, str | None]] = {}
            for r in con.execute(
                "SELECT name_norm, unit, MIN(dimension), COUNT(*) AS n FROM measurements "
                "WHERE unit IS NOT NULL GROUP BY name_norm, unit"
            ):
                if r[0] not in units or r[3] > units[r[0]][0]:
                    units[r[0]] = (r[3], r[1], r[2])
            rows = con.execute(
                "SELECT MIN(name), name_norm, COUNT(*) AS n FROM measurements "
                "GROUP BY name_norm ORDER BY n DESC, 1"
            ).fetchall()
        self.measurements = _NameIndex([(r[0], r[1], r[2]) for r in rows])
        self.measurement_info = {
            n: CatalogMeasurement(name, count, units[n][1] if n in units else None,
                                  units[n][2] if n in units else None)
            for name, n, count in self.measurements.entries
        }  # fmt: skip


_full_cache: dict[str, tuple[int, _FullCatalog]] = {}
_full_lock = threading.Lock()


def _full_catalog(store: Store) -> _FullCatalog:
    key = str(store.path.resolve())
    revision = store.revision()
    cached = _full_cache.get(key)
    if cached and cached[0] == revision:
        return cached[1]
    with _full_lock:
        cached = _full_cache.get(key)
        if cached and cached[0] == revision:
            return cached[1]
        full = _FullCatalog(store)
        _full_cache[key] = (revision, full)
        return full


@dataclass
class Catalog:
    n_experiments: int = 0
    date_min: str = ""
    date_max: str = ""
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    measurements: list[CatalogMeasurement] = field(default_factory=list)
    # Every distinct name in the store (None: only the lists above are known). Used to
    # validate plan values, so a rare name that did not fit the prompt still survives.
    all_tags: list[str] | None = field(default=None, repr=False)
    all_categories: list[str] | None = field(default=None, repr=False)
    all_statuses: list[str] | None = field(default=None, repr=False)
    all_authors: list[str] | None = field(default=None, repr=False)
    all_measurements: dict[str, CatalogMeasurement] | None = field(default=None, repr=False)

    def measurement(self, name: str) -> CatalogMeasurement | None:
        key = norm(name)
        found = next((m for m in self.measurements if norm(m.name) == key), None)
        if found is None and self.all_measurements is not None:
            found = self.all_measurements.get(key)
        return found

    def tag_options(self) -> list[str]:
        return self.tags if self.all_tags is None else self.all_tags

    def category_options(self) -> list[str]:
        return self.categories if self.all_categories is None else self.all_categories

    def status_options(self) -> list[str]:
        return self.statuses if self.all_statuses is None else self.all_statuses

    def author_options(self) -> list[str]:
        return self.authors if self.all_authors is None else self.all_authors

    def measurement_names(self) -> list[str]:
        if self.all_measurements is None:
            return [m.name for m in self.measurements]
        return [m.name for m in self.all_measurements.values()]


def load_catalog(
    store: Store,
    max_tags: int = 40,
    max_measurements: int = 40,
    question: str | None = None,
    max_total_tags: int = 120,
    max_total_measurements: int = 120,
) -> Catalog:
    """Names that exist in the notebook: the most frequent ones plus those the question names.

    With ``question=None`` only the frequent part is returned. The full distinct sets are
    cached per store revision and attached for plan validation.
    """
    full = _full_catalog(store)
    tokens = query_tokens(question) if question else []

    def pick(index: _NameIndex, n_frequent: int, total: int) -> list[int]:
        chosen = index.frequent(n_frequent)
        if question:
            seen = set(chosen)
            extra = [
                i for i in index.relevant(question, tokens, total * 4) if i not in seen
            ]
            extra.sort(key=lambda i: i)  # entries are sorted by frequency
            chosen += extra[: max(0, total - len(chosen))]
        return chosen

    tags = [full.tags.entries[i][0] for i in pick(full.tags, max_tags, max_total_tags)]
    lists = {}
    for column, idx in full.columns.items():
        lists[column] = [idx.entries[i][0] for i in pick(idx, 40, 60)]
    meas_index = full.measurements
    measurements = [
        full.measurement_info[meas_index.entries[i][1]]
        for i in pick(meas_index, max_measurements, max_total_measurements)
    ]
    return Catalog(
        full.head[0], full.head[1], full.head[2], tags,
        lists["category"], lists["status"], lists["author"], measurements,
        all_tags=[e[0] for e in full.tags.entries],
        all_categories=[e[0] for e in full.columns["category"].entries],
        all_statuses=[e[0] for e in full.columns["status"].entries],
        all_authors=[e[0] for e in full.columns["author"].entries],
        all_measurements={n: full.measurement_info[n] for n in full.measurement_info},
    )  # fmt: skip
