from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from generate_sample_data import make_churn, make_orders


@pytest.fixture(scope="session")
def orders() -> pd.DataFrame:
    return make_orders()


@pytest.fixture(scope="session")
def churn() -> pd.DataFrame:
    return make_churn()


@pytest.fixture
def write(tmp_path):
    """Write a frame to disk in the requested format and return the path."""
    def _write(df: pd.DataFrame, fmt: str = "csv", name: str = "orders") -> Path:
        path = tmp_path / f"{name}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "tsv":
            df.to_csv(path, index=False, sep="\t")
        elif fmt == "xlsx":
            df.to_excel(path, index=False, sheet_name=name)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt in ("sqlite", "db"):
            with closing(sqlite3.connect(path)) as con, con:
                df.to_sql(name, con, index=False)
        elif fmt == "duckdb":
            con = duckdb.connect(str(path))
            con.register("frame", df)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM frame')
            con.close()
        else:
            raise ValueError(fmt)
        return path
    return _write
