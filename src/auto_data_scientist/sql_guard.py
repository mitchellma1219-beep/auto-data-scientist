"""Static validation of SQL before it reaches the database.

The guard is an allowlist: one statement, which must be a SELECT (optionally with CTEs or set
operations), reading only known tables and calling no function that touches files, the
network, configuration or dynamic SQL. It is the first of several layers; the connection
itself also has external access disabled and its configuration locked (see ``ingest``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from auto_data_scientist.errors import UnsafeQueryError

# sqlglot logs a warning when it falls back to an opaque "Command"; the guard rejects those.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_CHARS = 20_000
_TABLE_FUNCTIONS_ALLOWED = frozenset({"range", "generate_series", "unnest"})
_DENIED_PREFIXES = ("read_", "duckdb_", "pragma_", "sqlite_", "postgres_", "mysql_", "iceberg_",
                    "delta_", "parquet_", "json_execute", "arrow_scan", "pandas_scan")
_DENIED_NAMES = frozenset({
    "glob", "query", "query_table", "getenv", "current_setting", "sniff_csv", "which_secret",
    "system", "shell", "load", "install", "copy", "export", "import", "checkpoint",
    "force_checkpoint", "scan_arrow_ipc", "from_substrait", "seq_scan", "in_search_path",
    "set_seed", "getvariable", "setvariable",
})
_DENIED_SUFFIXES = ("_scan",)
_STATEMENT_TYPES_DENIED = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter, exp.Copy,
    exp.Command, exp.Set, exp.Pragma, exp.Use, exp.Attach, exp.Detach, exp.Into,
    exp.TruncateTable, exp.Describe, exp.Transaction, exp.Commit, exp.Rollback,
)


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str
    tables: tuple[str, ...]
    explanation: str


def _function_name(node: object) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower() if isinstance(node, exp.Func) else type(node).__name__.lower()


def _denied_function(name: str) -> bool:
    return (name in _DENIED_NAMES or name.startswith(_DENIED_PREFIXES)
            or name.endswith(_DENIED_SUFFIXES))


def _explain(tree: exp.Expression, tables: Iterable[str]) -> str:
    parts = [f"Reads from: {', '.join(sorted(set(tables))) or 'no table'}."]
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if isinstance(tree, exp.SetOperation):
        parts.append(f"Combines several SELECTs with {type(tree).__name__.upper()}.")
    if select is not None:
        aggs = sorted({a.sql(dialect="duckdb") for a in select.find_all(exp.AggFunc)})
        if aggs:
            parts.append(f"Aggregates: {', '.join(aggs)}.")
        where = select.args.get("where")
        if where is not None:
            parts.append(f"Filters rows where {where.this.sql(dialect='duckdb')}.")
        group = select.args.get("group")
        if group is not None:
            cols = ", ".join(g.sql(dialect="duckdb") for g in group.expressions)
            parts.append(f"Groups by: {cols}.")
        joins = select.args.get("joins") or []
        if joins:
            parts.append(f"Joins {len(joins)} additional table(s).")
    order = tree.args.get("order")
    if order is not None:
        parts.append(f"Sorted by: {', '.join(o.sql(dialect='duckdb') for o in order.expressions)}.")
    limit = tree.args.get("limit")
    if limit is not None:
        parts.append(f"Returns at most {limit.expression.sql(dialect='duckdb')} row(s).")
    return " ".join(parts)


def validate_sql(sql: str, allowed_tables: Iterable[str], max_joins: int = 8) -> ValidatedQuery:
    """Return the validated statement or raise ``UnsafeQueryError`` explaining the rejection."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeQueryError("the query is empty")
    if len(sql) > MAX_SQL_CHARS:
        raise UnsafeQueryError(f"the query is longer than {MAX_SQL_CHARS:,} characters")
    text = sql.strip().rstrip(";").strip()
    try:
        statements = sqlglot.parse(text, read="duckdb")
    except (SqlglotError, RecursionError) as exc:
        reason = str(exc).splitlines()[0]
        raise UnsafeQueryError(f"the query could not be parsed: {reason}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise UnsafeQueryError("exactly one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.SetOperation):
        raise UnsafeQueryError(
            f"only SELECT queries are allowed; got a {type(tree).__name__.upper()} statement")

    allowed = {t.lower(): t for t in allowed_tables}
    if any(isinstance(w, exp.With) and w.args.get("recursive") for w in tree.find_all(exp.With)):
        raise UnsafeQueryError("recursive CTEs are not allowed (unbounded execution)")
    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    used: list[str] = []
    for node in tree.walk():
        if isinstance(node, _STATEMENT_TYPES_DENIED):
            raise UnsafeQueryError(
                f"{type(node).__name__.upper()} is not allowed inside a read-only query")
        if isinstance(node, exp.Table):
            inner = node.this
            if not isinstance(inner, exp.Identifier):
                fname = _function_name(inner)
                if fname not in _TABLE_FUNCTIONS_ALLOWED:
                    raise UnsafeQueryError(f"table function {fname}() is not allowed")
                continue
            if node.db or node.catalog:
                raise UnsafeQueryError(
                    f"schema-qualified table {node.sql(dialect='duckdb')} is not allowed")
            name = node.name.lower()
            if name in ctes:
                continue
            if name not in allowed:
                raise UnsafeQueryError(
                    f"unknown table {node.name!r}; available tables: {sorted(allowed.values())}")
            used.append(allowed[name])
        elif isinstance(node, exp.Func):
            fname = _function_name(node)
            if _denied_function(fname):
                raise UnsafeQueryError(f"function {fname}() is not allowed")
    joins = list(tree.find_all(exp.Join))
    if len(joins) > max_joins:
        raise UnsafeQueryError(f"the query has {len(joins)} joins; the limit is {max_joins}")
    for j in joins:
        if isinstance(j.this, exp.Table) and not isinstance(j.this.this, exp.Identifier):
            continue  # unnest()/range() style lateral joins
        if j.args.get("on") is None and not j.args.get("using"):
            raise UnsafeQueryError(
                "joins must have an ON or USING condition (cartesian products are refused)")
    return ValidatedQuery(sql=text, tables=tuple(dict.fromkeys(used)),
                          explanation=_explain(tree, used))
