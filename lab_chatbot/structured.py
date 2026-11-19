"""Exact answers from SQL, rendered by templates (no LLM can corrupt a number).

* ``count`` / ``list`` / ``aggregate`` plans are answered completely here;
* for ``search`` plans with a numeric condition the exact measurement values
  are shown as a table next to the generated answer and handed to the LLM as
  facts (``search_tables`` / ``facts_text``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from lab_chatbot.answer import Table
from lab_chatbot.plan import Plan
from lab_chatbot.prompts import t
from lab_chatbot.sql_tools import (
    Filters,
    aggregate_measurement,
    count_experiments,
    list_experiments,
    measurement_values,
)
from lab_chatbot.store import Store
from lab_chatbot.units import format_quantity
from shared.config import experiment_url


@dataclass
class StructuredResult:
    text: str
    tables: list[Table] = field(default_factory=list)
    experiment_ids: list[int] = field(default_factory=list)


class StructuredAnswers:
    def __init__(self, store: Store, site_url: str):
        self.store = store
        self.site_url = site_url

    def answer(self, plan: Plan) -> StructuredResult:
        if plan.intent == "count":
            return self._count(plan)
        if plan.intent == "aggregate":
            return self._aggregate(plan)
        return self._list(plan)

    # ---- tables next to a search answer

    def search_tables(self, plan: Plan, candidates: set[int] | None) -> list[Table]:
        """Exact measurement values behind a numeric filter (shown and given to the LLM)."""
        if not (plan.measurement and plan.measurement.op and candidates):
            return []
        rows = measurement_values(self.store, plan.filters, plan.measurement, plan.order, limit=30)
        return [self.values_table(plan, rows)] if rows else []

    def facts_text(self, plan: Plan, tables: Sequence[Table]) -> str:
        lines = []
        for table in tables:
            if table.title != t(plan.language, "table_values"):
                continue
            lines.append(f"Matching measurements ({plan.describe_filters()}):")
            for row in table.rows:
                lines.append(f"- Experiment {row[0]} ({row[2]}): {row[3]}")
        return "\n".join(lines)

    def matches_table(self, plan: Plan, candidates: set[int]) -> Table:
        result = list_experiments(
            self.store, Filters(experiment_ids=sorted(candidates)), None, plan.order, plan.limit
        )
        return self.experiments_table(plan, result.rows)

    # ---- table builders

    def values_table(self, plan: Plan, rows: list[dict]) -> Table:
        lang = plan.language
        return Table(
            t(lang, "table_values"),
            [t(lang, "col_id"), t(lang, "col_title"), t(lang, "col_date"), t(lang, "col_value")],
            [
                [
                    r["experiment_id"], r["title"], r["date"],
                    f"{r['name']}: "
                    + (
                        format_quantity(r["value_num"], r["unit"])
                        if r["value_num"] is not None
                        else r["value_text"]
                    ),
                ]
                for r in rows
            ],
        )  # fmt: skip

    def experiments_table(self, plan: Plan, rows: list[dict]) -> Table:
        lang = plan.language
        return Table(
            t(lang, "table_matches"),
            [t(lang, k) for k in ("col_id", "col_title", "col_date", "col_tags", "col_category", "col_status", "col_link")],
            [
                [
                    r["id"], r["title"], r["date"], r["tags"] or "", r["category"] or "",
                    r["status"] or "", experiment_url(r["id"], self.site_url),
                ]
                for r in rows
            ],
        )  # fmt: skip

    # ---- the three structured intents

    @staticmethod
    def _with_filters(text: str, plan: Plan) -> str:
        described = plan.describe_filters()
        if described != "–":
            text += f"\n\n_{t(plan.language, 'filters_used')}: {described}_"
        return text

    def _count(self, plan: Plan) -> StructuredResult:
        lang = plan.language
        rows = count_experiments(self.store, plan.filters, plan.measurement, plan.group_by)
        if plan.group_by:
            if not rows:
                return StructuredResult(t(lang, "count_grouped_empty"))
            label = t(lang, f"group_{plan.group_by}")
            table = Table(
                t(lang, "table_counts"), [label, t(lang, "col_count")],
                [[r["group"], r["count"]] for r in rows],
            )  # fmt: skip
            return StructuredResult(t(lang, "count_grouped", group=label), [table])
        n = rows[0]["count"]
        text = t(lang, "count_one") if n == 1 else t(lang, "count_many", n=n)
        return StructuredResult(self._with_filters(text, plan))

    def _list(self, plan: Plan) -> StructuredResult:
        lang = plan.language
        result = list_experiments(self.store, plan.filters, plan.measurement, plan.order, plan.limit)
        if not result.rows:
            return StructuredResult(t(lang, "no_experiments", filters=plan.describe_filters()))
        key = "listed" if result.total <= plan.limit else "listed_limited"
        text = self._with_filters(t(lang, key, total=result.total, limit=plan.limit), plan)
        tables = [self.experiments_table(plan, result.rows)]
        if plan.measurement:
            values = measurement_values(
                self.store, plan.filters, plan.measurement, plan.order, limit=plan.limit
            )
            if values:
                tables.append(self.values_table(plan, values))
        return StructuredResult(text, tables, [r["id"] for r in result.rows])

    def _aggregate(self, plan: Plan) -> StructuredResult:
        lang = plan.language
        cond = plan.measurement
        groups = aggregate_measurement(self.store, plan.filters, cond, unit=cond.unit)
        if not groups:
            return StructuredResult(t(lang, "no_values", name=cond.name))
        lines = []
        for group in groups:
            unit = group["unit"]
            lines.append(
                t(lang, "agg_header", name=group["name"], n_values=group["n_values"],
                  n_experiments=group["n_experiments"])
            )  # fmt: skip
            for stat in ("min", "max", "avg", "sum"):
                value = format_quantity(group[stat], unit)
                suffix = ""
                if stat in ("min", "max"):
                    suffix = f" ({t(lang, 'experiment')} {group[f'{stat}_experiment']})"
                line = f"- {t(lang, stat)}: {value}{suffix}"
                lines.append(f"**{line[2:]}**" if plan.agg == stat else line)
            if plan.agg == "count":
                lines.append(f"- **{t(lang, 'count')}: {group['n_values']}**")
            lines.append("")
        values = measurement_values(self.store, plan.filters, cond, plan.order, limit=30)
        tables = [self.values_table(plan, values)] if values else []
        ids = list(dict.fromkeys(v["experiment_id"] for v in values))
        return StructuredResult("\n".join(lines).strip(), tables, ids)
