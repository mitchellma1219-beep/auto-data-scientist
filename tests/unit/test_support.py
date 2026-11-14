"""Sample data, friendly error text, safe file names and shared use of one dataset."""

import threading

import pandas as pd
import pytest

from auto_data_scientist import eda, explain, load_dataset, run_query, samples
from auto_data_scientist.errors import (
    AdsError,
    DataError,
    QueryTimeoutError,
    TranslationError,
    UnsafeQueryError,
)
from auto_data_scientist.ingest import display_name
from auto_data_scientist.pipeline import ask


def test_samples_are_the_fixture_data_and_load_like_any_file(orders, churn, tmp_path):
    pd.testing.assert_frame_equal(samples.make_orders(), orders)
    pd.testing.assert_frame_equal(samples.make_churn(), churn)
    pd.testing.assert_frame_equal(samples.make_orders(), samples.make_orders())   # seeded
    for key, sample in samples.SAMPLES.items():
        path = samples.write_sample(key, tmp_path / "nested")
        assert path.name == f"{key}.csv" and sample.key == key
        ds = load_dataset(path)
        frame = ds.frame()
        assert len(frame) == len(sample.make()) and ds.default_table() == key
        for col in (sample.target, sample.date_col, sample.value_col, sample.user_col):
            assert col is None or col in frame.columns
        assert run_query(ds, sample.sql).rows                       # the starter SQL is allowed
        for question in sample.questions:
            assert ask(ds, question)[1].rows                         # and the starter questions
        ds.close()
    with pytest.raises(DataError, match="unknown sample"):
        samples.write_sample("payroll", tmp_path)


@pytest.mark.parametrize("exc, title, hint", [
    (UnsafeQueryError("function read_csv() is not allowed"), "The query was refused",
     "Nothing was executed"),
    (QueryTimeoutError("query exceeded the 30s time limit and was stopped"),
     "The query was stopped", "less to do"),
    (TranslationError("no column matches 'store'; available columns: a, b"),
     "The question was not translated", "refuses instead of guessing"),
    (DataError("unknown target 'x'"), "This data cannot be used for that", "Check the file"),
    (AdsError("query failed: Binder Error"), "The request could not be completed", "Adjust"),
    (ValueError("horizon must be between 1 and 60"), "A setting is out of range", "Change"),
])
def test_every_expected_failure_has_a_title_reason_and_next_step(exc, title, hint):
    e = explain(exc)
    assert e.title == title and hint in e.hint
    assert e.reason.endswith(".") and e.reason[0].isupper()
    assert str(exc).split(";")[0].lower() in e.reason.lower()        # the reason is kept


def test_explanations_are_single_lines_and_unexpected_errors_are_not_disguised():
    e = explain(DataError("could not parse x.csv:\n  line 3\n  bad quote"))
    assert "\n" not in e.reason and e.reason == "Could not parse x.csv: line 3 bad quote."
    assert explain(DataError("")).reason == "DataError."
    with pytest.raises(ZeroDivisionError):
        explain(ZeroDivisionError("division by zero"))


@pytest.mark.parametrize("raw, shown", [
    ("sales 2024.csv", "sales 2024.csv"),
    ("../../etc/passwd", "passwd"),
    ("C:\\Users\\me\\q3 data.xlsx", "q3 data.xlsx"),
    ("x](http://e.invalid)<b>.csv", "e.invalid__b_.csv"),
    ("![a](b).csv", "__a__b_.csv"),
    ("销售额.csv", "销售额.csv"),
    ("a\x00b\nc.csv", "a_b_c.csv"),
    ("", "upload"), ("...", "upload"), ("n" * 200, "n" * 80),
])
def test_display_names_carry_no_path_and_no_markup(raw, shown):
    assert display_name(raw) == shown


def test_one_dataset_can_serve_several_threads(orders, write):
    ds = load_dataset(write(orders))
    expected = sorted(orders.groupby("region")["revenue"].sum().round(2).items())
    failures, results = [], []

    def work() -> None:
        for _ in range(12):
            try:
                res = run_query(ds, "SELECT region, sum(revenue) FROM orders GROUP BY 1")
                results.append(sorted((r[0], round(r[1], 2)) for r in res.rows))
                assert len(ds.frame()) == 6000
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert failures == [] and len(results) == 72
    assert all(r == expected for r in results)
    with pytest.raises(UnsafeQueryError):
        run_query(ds, "DROP TABLE orders")
    assert run_query(ds, "SELECT count(*) FROM orders").rows == [(6000,)]   # still usable
    ds.close()


def test_frequency_suggestion_follows_the_span_of_the_dates():
    def dates(days: int) -> pd.Series:
        return pd.Series(pd.date_range("2024-01-01", periods=days, freq="D").astype(str))

    assert eda.suggest_frequency(dates(400)) == "M"
    assert eda.suggest_frequency(dates(60)) == "W"
    assert eda.suggest_frequency(dates(10)) == "D"
    assert eda.suggest_frequency(pd.Series(["2024-01-01", "not a date", "2025-03-01"])) == "M"
    with pytest.raises(DataError, match="no valid dates"):
        eda.suggest_frequency(pd.Series(["x", None]))


@pytest.mark.parametrize("address, local", [
    ("localhost", True), ("127.0.0.1", True), ("::1", True), ("[::1]", True), (" LOCALHOST ", True),
    ("127.8.9.10", True), ("0.0.0.0", False), ("::", False), ("192.168.1.5", False),
    ("example.com", False), ("", False), (None, False),
])
def test_only_loopback_addresses_count_as_local(address, local):
    from auto_data_scientist.config import is_loopback

    assert is_loopback(address) is local


def test_tables_are_named_after_the_name_a_file_is_known_by(orders, write):
    from auto_data_scientist.query import preview_sql

    path = write(orders, name="5d41402abc4b2a76b9719d911017c592")      # a generated name
    assert load_dataset(path).default_table() == "t_5d41402abc4b2a76b9719d911017c592"
    for known_as, table in [("Q3 Sales.csv", "q3_sales"), ("../../x/orders.csv", "orders"),
                            ("2024 (final).csv", "t_2024__final"), ("销售.csv", "销售"),
                            ("...", "upload")]:
        ds = load_dataset(path, name=known_as)
        assert ds.default_table() == table
        assert preview_sql(table) == f'SELECT * FROM "{table}" LIMIT 20'
        assert len(run_query(ds, preview_sql(table, 7)).rows) == 7
        ds.close()
    assert load_dataset(path, name="").default_table() == load_dataset(path).default_table()
    parquet = load_dataset(write(orders, "parquet", name="abc"), name="sales.parquet")
    assert parquet.default_table() == "sales"
    # Workbooks and databases keep the names of their own sheets and tables.
    assert load_dataset(write(orders, "xlsx", name="sheet_a"), name="x.xlsx").default_table() == \
        "sheet_a"
    assert preview_sql('we"ird') == 'SELECT * FROM "we""ird" LIMIT 20'
