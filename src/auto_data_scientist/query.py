"""Guarded query execution: validate, run with a time budget, cap the rows returned."""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass

import duckdb
import pandas as pd

from auto_data_scientist.errors import AdsError, QueryTimeoutError
from auto_data_scientist.ingest import Dataset
from auto_data_scientist.sql_guard import validate_sql


@dataclass(frozen=True)
class QueryResult:
    sql: str
    explanation: str
    columns: list[str]
    rows: list[tuple[object, ...]]
    truncated: bool
    elapsed_s: float

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


def run_query(ds: Dataset, sql: str) -> QueryResult:
    """Validate ``sql`` against the dataset's tables and execute it read-only.

    Layers: static allowlist (``validate_sql``), a locked connection without file or network
    access, a transaction that is always rolled back, a watchdog that interrupts the query
    after ``limits.sql_timeout_s`` seconds and a cap of ``limits.sql_max_rows`` returned rows.
    """
    query = validate_sql(sql, ds.tables, ds.limits.sql_max_joins)
    timed_out = threading.Event()

    def _interrupt() -> None:
        timed_out.set()
        ds.con.interrupt()

    timer = threading.Timer(ds.limits.sql_timeout_s, _interrupt)
    start = time.perf_counter()
    ds.con.execute("BEGIN TRANSACTION")
    timer.start()
    try:
        cur = ds.con.execute(query.sql)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(ds.limits.sql_max_rows + 1)
    except duckdb.InterruptException as exc:
        raise QueryTimeoutError(
            f"query exceeded the {ds.limits.sql_timeout_s:g}s time limit and was stopped") from exc
    except duckdb.Error as exc:
        if timed_out.is_set():
            raise QueryTimeoutError(
                f"query exceeded the {ds.limits.sql_timeout_s:g}s time limit") from exc
        raise AdsError(f"query failed: {str(exc).splitlines()[0]}") from exc
    finally:
        timer.cancel()
        with contextlib.suppress(duckdb.Error):  # an interrupted transaction may be closed
            ds.con.execute("ROLLBACK")
    truncated = len(rows) > ds.limits.sql_max_rows
    return QueryResult(sql=query.sql, explanation=query.explanation, columns=columns,
                       rows=rows[: ds.limits.sql_max_rows], truncated=truncated,
                       elapsed_s=time.perf_counter() - start)
