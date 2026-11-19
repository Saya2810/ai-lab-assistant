"""Restrict the chatbot to part of the notebook (projects, tags, categories, statuses).

Without a scope every question searches the whole lab notebook, and an answer may
come from experiments that were not meant at all. A ``Scope`` is chosen by the user
(sidebar, or ``--project/--tag`` on the command line) and applies to *every* answer
route: text search, counts, lists and aggregates, because it becomes part of the
SQL ``WHERE`` clause (``sql_tools.build_where``).

Within one kind the choices are alternatives (tag A *or* tag B), between kinds they
all apply (project P *and* tag A).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from lab_chatbot.store import Store, norm


@dataclass(frozen=True)
class Scope:
    resources: tuple[str, ...] = ()  # projects / resources linked to the experiment
    tags: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()

    @classmethod
    def of(cls, resources=(), tags=(), categories=(), statuses=()) -> "Scope":
        def clean(values) -> tuple[str, ...]:
            return tuple(dict.fromkeys(str(v).strip() for v in values or () if str(v).strip()))

        return cls(clean(resources), clean(tags), clean(categories), clean(statuses))

    def is_empty(self) -> bool:
        return not (self.resources or self.tags or self.categories or self.statuses)

    def describe(self, language: str = "de") -> str:
        labels = _LABELS.get(language, _LABELS["en"])
        parts = []
        for kind, values in (
            ("resources", self.resources), ("tags", self.tags),
            ("categories", self.categories), ("statuses", self.statuses),
        ):  # fmt: skip
            if values:
                parts.append(f"{labels[kind]}: {' / '.join(values)}")
        return "; ".join(parts)


_LABELS = {
    "de": {"resources": "Projekt", "tags": "Tags", "categories": "Kategorie", "statuses": "Status", "scope": "Eingrenzung"},
    "en": {"resources": "Project", "tags": "Tags", "categories": "Category", "statuses": "Status", "scope": "Scope"},
}


def scope_label(language: str) -> str:
    return _LABELS.get(language, _LABELS["en"])["scope"]


def scope_clause(scope: Scope) -> tuple[str, list]:
    """SQL condition on ``experiments e`` (empty string for an empty scope)."""
    clauses: list[str] = []
    params: list = []
    if scope.resources:
        # experiments.resources holds the linked titles as "A | B | C"
        clauses.append(
            "(" + " OR ".join(
                "instr(' | ' || COALESCE(e.resources, '') || ' | ', ' | ' || ? || ' | ') > 0"
                for _ in scope.resources
            ) + ")"
        )  # fmt: skip
        params += list(scope.resources)
    if scope.tags:
        clauses.append(
            "EXISTS (SELECT 1 FROM exp_tags t WHERE t.experiment_id = e.id "
            "AND t.tag_norm IN (SELECT value FROM json_each(?)))"
        )
        params.append(_json([norm(tag) for tag in scope.tags]))
    if scope.categories:
        clauses.append("e.category IN (SELECT value FROM json_each(?))")
        params.append(_json(list(scope.categories)))
    if scope.statuses:
        clauses.append("e.status IN (SELECT value FROM json_each(?))")
        params.append(_json(list(scope.statuses)))
    return " AND ".join(clauses), params


def _json(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)


# ---------------------------------------------------------------------------
# what can be chosen
# ---------------------------------------------------------------------------

_options_cache: dict[str, tuple[int, dict[str, list[str]]]] = {}
_options_lock = threading.Lock()


def scope_options(store: Store) -> dict[str, list[str]]:
    """Every project, tag, category and status in the index (cached per index revision)."""
    key, revision = str(store.path.resolve()), store.revision()
    with _options_lock:
        cached = _options_cache.get(key)
        if cached and cached[0] == revision:
            return cached[1]
    with store.connect() as con:
        resources: set[str] = set()
        for (joined,) in con.execute(
            "SELECT DISTINCT resources FROM experiments WHERE resources IS NOT NULL AND resources != ''"
        ):
            resources.update(part.strip() for part in joined.split(" | ") if part.strip())

        def distinct(sql: str) -> list[str]:
            return [row[0] for row in con.execute(sql) if row[0]]

        options = {
            "resources": sorted(resources, key=str.casefold),
            "tags": distinct("SELECT MIN(tag) FROM exp_tags GROUP BY tag_norm ORDER BY lower(MIN(tag))"),
            "categories": distinct("SELECT DISTINCT category FROM experiments ORDER BY lower(category)"),
            "statuses": distinct("SELECT DISTINCT status FROM experiments ORDER BY lower(status)"),
        }
    with _options_lock:
        _options_cache[key] = (revision, options)
    return options
