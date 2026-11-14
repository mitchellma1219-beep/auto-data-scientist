"""A deterministic baseline translator from business questions to SQL.

It covers single-table aggregate questions (totals, averages, counts, rankings, time grains
and simple equality filters). Anything it cannot map to the schema is refused with a
``TranslationError`` instead of guessed, and every assumption it makes is returned with the
SQL. Inside the Agent Skill the model writes SQL itself; either way the statement passes
through ``sql_guard.validate_sql`` before it runs.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from auto_data_scientist.errors import TranslationError
from auto_data_scientist.ingest import quote_ident
from auto_data_scientist.profiling import TableProfile

_AGGS = [("AVG", r"\b(average|avg|mean)\b"), ("MEDIAN", r"\bmedian\b"),
         ("SUM", r"\b(total|sum)\b"), ("COUNT", r"\b(how many|number of|count)\b"),
         ("MAX", r"\b(maximum|max)\b"), ("MIN", r"\b(minimum|min)\b")]
_GRAINS = [("day", r"\b(daily|(by|per|each) day)\b"), ("week", r"\b(weekly|(by|per|each) week)\b"),
           ("month", r"\b(monthly|(by|per|each) month)\b"),
           ("quarter", r"\b(quarterly|(by|per|each) quarter)\b"),
           ("year", r"\b(yearly|annual|annually|(by|per|each) year)\b")]
_HIGH = r"\b(top|highest|most|largest|biggest|best)\b"
_LOW = r"\b(bottom|lowest|least|smallest|worst)\b"
_STOP = {"day", "week", "month", "quarter", "year", "each", "the", "a", "an"}


@dataclass(frozen=True)
class Translation:
    sql: str
    explanation: str
    assumptions: tuple[str, ...]


class Translator(Protocol):
    def translate(self, question: str, profile: TableProfile,
                  values: dict[str, list[str]] | None = None) -> Translation: ...


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class RuleBasedTranslator:
    """Keyword and schema matching; no model calls, fully reproducible."""

    def translate(self, question: str, profile: TableProfile,
                  values: dict[str, list[str]] | None = None) -> Translation:
        q = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.! ")
        if not q:
            raise TranslationError("the question is empty")
        cols = {c.name: c for c in profile.columns}
        listing = ", ".join(cols)
        mentioned = self._mentions(q, list(cols))
        numeric = [c for c in mentioned if cols[c].kind == "numeric"]
        dates = profile.by_kind("datetime")
        assumptions: list[str] = []

        high, low = re.search(_HIGH, q), re.search(_LOW, q)
        if high and low:
            raise TranslationError(
                "the question asks for both the highest and the lowest; ask one at a time")
        aggs = [name for name, pat in _AGGS if re.search(pat, q)]
        grain = next((g for g, pat in _GRAINS if re.search(pat, q)), None)

        dims: list[str] = []
        for m in re.finditer(r"\b(?:by|per|for each|across|in each)\s+([a-z0-9_ ]+)", q):
            phrase = re.sub(r"^(?:(?:the|total|sum of|sum|average|avg|mean|median|count of|"
                            r"number of|maximum|minimum|max|min|highest|lowest) )+", "",
                            m.group(1).strip())
            col = next((c for c in mentioned if phrase.startswith(self._alias(c))
                        or phrase.startswith(c.lower())), None)
            first = phrase.split(" ")[0]
            if col is not None:
                if cols[col].kind == "numeric" and (high or low):
                    continue  # "top 5 regions by total amount": a ranking metric, not a group
                if col not in dims:
                    dims.append(col)
            elif first not in _STOP:
                raise TranslationError(f"no column matches {first!r}; available columns: {listing}")
        if not dims and (high or low):
            dims = [c for c in mentioned if cols[c].kind in ("categorical", "boolean", "text")][:1]

        date_col: str | None = None
        if grain or re.search(r"\b(in|during|for) (19|20)\d{2}\b", q):
            named = [c for c in mentioned if c in dates]
            if len(named) == 1 or len(dates) == 1:
                date_col = (named or dates)[0]
            else:
                raise TranslationError(
                    f"a time period was requested but the date column is ambiguous: {dates}")

        metric_sql: list[tuple[str, str]] = []
        distinct = re.search(r"\b(distinct|unique)\b", q)
        if not aggs and (high or low) and numeric:
            aggs = ["SUM"]
            assumptions.append(f"Ranked by the total of {numeric[0]} (no aggregation was stated).")
        for agg in aggs:
            if agg == "COUNT":
                target = next((c for c in mentioned if c not in dims and distinct), None)
                if target:
                    metric_sql.append((f"COUNT(DISTINCT {quote_ident(target)})",
                                       f"distinct_{target}"))
                else:
                    metric_sql.append(("COUNT(*)", "row_count"))
                    assumptions.append("Counted rows of the table.")
            else:
                if not numeric:
                    nums = profile.by_kind("numeric")
                    raise TranslationError(
                        f"which numeric column should be aggregated? numeric columns: {nums}")
                for c in numeric:
                    metric_sql.append((f"{agg}({quote_ident(c)})", f"{agg.lower()}_{c}"))
        if not metric_sql:
            raise TranslationError(
                "could not find an aggregation (total, average, count, top N, ...) in the "
                f"question. Columns: {listing}. Write SQL for anything more specific.")

        where = self._filters(q, cols, values or {}, dims, assumptions)
        ym = re.search(r"\b(?:in|during|for) ((?:19|20)\d{2})\b", q)
        if ym and date_col:
            where.append(f"year(CAST({quote_ident(date_col)} AS TIMESTAMP)) = {int(ym.group(1))}")

        select: list[str] = []
        group: list[str] = []
        if grain and date_col:
            select.append(f"date_trunc('{grain}', CAST({quote_ident(date_col)} AS TIMESTAMP)) "
                          "AS period")
            group.append("period")
        for d in dims:
            select.append(quote_ident(d))
            group.append(quote_ident(d))
        select += [f"{expr} AS {quote_ident(alias)}" for expr, alias in metric_sql]
        sql = f"SELECT {', '.join(select)} FROM {quote_ident(profile.table)}"  # noqa: S608  # nosec B608
        if where:
            sql += " WHERE " + " AND ".join(where)
        if group:
            sql += " GROUP BY " + ", ".join(group)
        first_metric = quote_ident(metric_sql[0][1])
        if high or low:
            sql += f" ORDER BY {first_metric} {'DESC' if high else 'ASC'}"
            n = re.search(r"\b(?:top|bottom)\s+(\d{1,4})\b", q)
            limit = int(n.group(1)) if n else (1 if re.search(r"\b(which|what)\b", q) else None)
            if limit is None and group:
                limit = 10
                assumptions.append("Limited to 10 rows.")
            if limit and group:
                sql += f" LIMIT {limit}"
        elif grain:
            sql += " ORDER BY period"
        elif group:
            sql += f" ORDER BY {first_metric} DESC"
        what = ", ".join(e for e, _ in metric_sql)
        by = f" grouped by {', '.join(group)}" if group else ""
        flt = f" where {' and '.join(where)}" if where else ""
        return Translation(sql=sql, explanation=f"Computes {what} from {profile.table}{by}{flt}.",
                           assumptions=tuple(assumptions))

    @staticmethod
    def _alias(col: str) -> str:
        return col.lower().replace("_", " ")

    def _mentions(self, q: str, columns: list[str]) -> list[str]:
        found: list[tuple[int, str]] = []
        masked = q
        for col in sorted(columns, key=len, reverse=True):
            base = self._alias(col)
            forms = {base, col.lower(), base + "s", base.rstrip("s")} - {""}
            for form in sorted(forms, key=len, reverse=True):
                m = re.search(rf"(?<!\w){re.escape(form)}(?!\w)", masked)
                if m and len(form) > 1:
                    found.append((m.start(), col))
                    masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
                    break
        return [c for _, c in sorted(found)]

    @staticmethod
    def _filters(q: str, cols: Collection[str], values: dict[str, list[str]], dims: list[str],
                 assumptions: list[str]) -> list[str]:
        where: list[str] = []
        for col, vals in values.items():
            if col not in cols or col in dims:
                continue
            hits = [v for v in vals
                    if len(str(v)) > 1
                    and re.search(rf"(?<!\w){re.escape(str(v).lower())}(?!\w)", q)]
            if not hits:
                continue
            if len(hits) > 1:
                name = re.escape(col.lower().replace("_", " "))
                eq = rf"{name} (?:is|=|equals) \S+ and {name} (?:is|=|equals) "
                if re.search(eq, q):
                    raise TranslationError(
                        f"contradictory filters: no row can have {col} equal to "
                        f"{' and '.join(map(str, hits))} at the same time")
                where.append(f"{quote_ident(col)} IN ({', '.join(_lit(str(v)) for v in hits)})")
                assumptions.append(f"Included rows where {col} is any of {hits}.")
            else:
                where.append(f"{quote_ident(col)} = {_lit(str(hits[0]))}")
        return where
