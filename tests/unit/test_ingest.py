import sqlite3

import pandas as pd
import pytest

from auto_data_scientist.config import Limits
from auto_data_scientist.errors import DataError
from auto_data_scientist.ingest import load_dataset, save_upload
from auto_data_scientist.profiling import profile_frame


@pytest.mark.parametrize("fmt", ["csv", "tsv", "xlsx", "parquet", "sqlite", "db", "duckdb"])
def test_every_format_loads_the_same_table(orders, write, fmt):
    ds = load_dataset(write(orders, fmt))
    assert list(ds.tables) == ["orders"] and ds.tables["orders"].n_rows == 6000
    df = ds.frame()
    assert list(df.columns) == list(orders.columns)
    assert df["revenue"].sum() == pytest.approx(orders["revenue"].sum())
    kinds = {c.name: c.kind for c in profile_frame(df).columns}
    assert kinds["region"] == "categorical" and kinds["revenue"] == "numeric"
    assert kinds["order_date"] == "datetime" and kinds["order_id"] == "id"
    ds.close()


def test_multiple_tables_and_default_is_largest(orders, churn, tmp_path):
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as con:
        orders.to_sql("Orders 2024", con, index=False)
        churn.to_sql("customers", con, index=False)
    ds = load_dataset(path)
    assert set(ds.tables) == {"orders_2024", "customers"}
    assert ds.default_table() == "orders_2024" and len(ds.frame("customers")) == 3000
    with pytest.raises(DataError, match="unknown table"):
        ds.frame("nope")
    ds.close()
    with pd.ExcelWriter(tmp_path / "book.xlsx") as xl:
        orders.head(50).to_excel(xl, sheet_name="Q1 Sales", index=False)
        churn.head(20).to_excel(xl, sheet_name="2024", index=False)
    ds = load_dataset(tmp_path / "book.xlsx")
    assert set(ds.tables) == {"q1_sales", "t_2024"}
    ds.close()


def test_malformed_rows_are_rejected_and_counted(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,amount,region\n1,10.5,West\n2,11.0,East\n3,7.0\n4,8.0,North,EXTRA\n"
                 "5,9.5,South\n")
    ds = load_dataset(p)
    info = ds.tables["bad"]
    assert info.n_rows == 3 and info.rejected_rows == 2
    assert info.reject_reasons == ["MISSING COLUMNS", "TOO MANY COLUMNS"]
    ds.close()


@pytest.mark.parametrize("name, content, match", [
    ("missing.csv", None, "file not found"),
    ("data.txt", b"a,b\n1,2\n", "unsupported file type"),
    ("data.pkl", b"\x80\x04", "unsupported file type"),
    ("noext", b"a,b\n1,2\n", "unsupported file type"),
    ("empty.csv", b"", "is empty"),
    ("fake.parquet", b"this is not parquet", "could not load"),
    ("fake.xlsx", b"PK\x03\x04 not really a workbook", "could not load"),
    ("fake.sqlite", b"definitely not a database", "neither a SQLite nor a DuckDB"),
    ("fake.duckdb", b"x" * 64, "neither a SQLite nor a DuckDB"),
])
def test_unusable_files_raise_data_error(tmp_path, name, content, match):
    p = tmp_path / name
    if content is not None:
        p.write_bytes(content)
    with pytest.raises(DataError, match=match):
        load_dataset(p)


def test_directory_and_limits(orders, write, tmp_path):
    with pytest.raises(DataError, match="not a regular file"):
        load_dataset(tmp_path)
    path = write(orders)
    with pytest.raises(DataError, match="limit is 100"):
        load_dataset(path, Limits(max_rows=100))
    with pytest.raises(DataError, match="MB; limit"):
        load_dataset(path, Limits(max_file_mb=0.01))


def test_env_limits(monkeypatch):
    monkeypatch.setenv("ADS_SQL_MAX_ROWS", "25")
    assert Limits.from_env().sql_max_rows == 25
    for bad in ("abc", "0", "-5"):
        monkeypatch.setenv("ADS_SQL_MAX_ROWS", bad)
        with pytest.raises(ValueError, match="ADS_SQL_MAX_ROWS"):
            Limits.from_env()


@pytest.mark.parametrize("name", ["../../etc/passwd.csv", "/etc/cron.d/job.csv",
                                  "..\\..\\windows\\system32\\x.csv", "a/b/../../c.parquet",
                                  "report.csv/../../x.csv", "~/.ssh/key.csv"])
def test_upload_names_cannot_escape_the_directory(tmp_path, name):
    base = tmp_path / "uploads"
    saved = save_upload(b"a,b\n1,2\n", name, base)
    assert saved.parent == base.resolve() and saved.suffix in (".csv", ".parquet")
    assert len(saved.stem) == 32 and list(base.iterdir()) == [saved]


@pytest.mark.parametrize("name", ["evil.csv\x00.exe", "script.py", "noext", "data.csv.exe", ""])
def test_upload_rejects_bad_names(tmp_path, name):
    with pytest.raises(DataError):
        save_upload(b"x", name, tmp_path)


def test_upload_size_limit(tmp_path):
    with pytest.raises(DataError, match="exceeds"):
        save_upload(b"x" * 2048, "a.csv", tmp_path, Limits(max_file_mb=0.001))


def test_hostile_table_names_and_quoted_paths(orders, tmp_path):
    evil = 'x"; DROP TABLE orders; --'
    path = tmp_path / "o'brien.sqlite"
    with sqlite3.connect(path) as con:
        orders.head(40).to_sql(evil, con, index=False)
        orders.head(10).to_sql("orders", con, index=False)
    ds = load_dataset(path)
    assert set(ds.tables) == {"x_drop_table_orders", "orders"}
    assert ds.tables["x_drop_table_orders"].n_rows == 40 and ds.tables["orders"].n_rows == 10
    ds.close()

    import duckdb
    dpath = tmp_path / "it's.duckdb"
    con = duckdb.connect(str(dpath))
    con.register("frame", orders.head(25))
    con.execute(f'CREATE TABLE "{evil.replace(chr(34), chr(34) * 2)}" AS SELECT * FROM frame')
    con.close()
    ds = load_dataset(dpath)
    assert [t.n_rows for t in ds.tables.values()] == [25]
    ds.close()
