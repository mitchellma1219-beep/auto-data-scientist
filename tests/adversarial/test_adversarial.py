"""Hostile, malformed and misleading inputs. Each must be refused or flagged, never trusted."""

import os

import numpy as np
import pandas as pd
import pytest

from auto_data_scientist import analyze, load_dataset, profile_frame, render_report, run_query
from auto_data_scientist.cli import main
from auto_data_scientist.errors import AdsError, DataError, TranslationError
from auto_data_scientist.ingest import quote_ident
from auto_data_scientist.modeling import train_supervised
from auto_data_scientist.pipeline import ask


def test_corrupted_csv_is_loaded_with_rejections_reported(orders, tmp_path):
    lines = orders.to_csv(index=False).splitlines()
    for i in range(10, 310, 10):
        lines[i] = lines[i] + ",EXTRA,FIELDS"          # 30 rows with too many fields
    for i in range(1005, 1305, 10):
        lines[i] = ",".join(lines[i].split(",")[:4])   # 30 rows cut short
    path = tmp_path / "orders.csv"
    path.write_text("\n".join(lines) + "\n")
    a = analyze(path)
    assert a.profile.rejected_rows == 60 and a.profile.n_rows == 5940
    assert a.profile.quality.validity == pytest.approx(1 - 60 / 6000, abs=1e-4)
    text = render_report(a)
    assert "60 malformed row(s) rejected on load" in text
    assert "60 malformed row(s) were rejected while loading" in text


def test_header_only_and_blank_files(tmp_path, capsys):
    path = tmp_path / "empty.csv"
    path.write_text("a,b,c\n")
    with pytest.raises(DataError, match="has no rows"):
        analyze(path)
    assert main(["profile", str(path)]) == 0
    assert "Quality score: n/a" in capsys.readouterr().out
    path.write_text("\n\n\n")
    with pytest.raises(DataError):
        analyze(path)


def test_garbage_bytes_never_crash(tmp_path):
    for suffix in (".csv", ".parquet", ".xlsx", ".sqlite", ".duckdb", ".db"):
        path = tmp_path / f"junk{suffix}"
        path.write_bytes(os.urandom(4096))
        try:
            ds = load_dataset(path)
        except DataError:
            continue
        profile_frame(ds.frame())
        ds.close()


def test_mixed_types_and_bad_dates_are_flagged_and_survivable(orders, write):
    df = orders.copy()
    df["revenue"] = df["revenue"].astype(object)
    df.loc[df.index[::50], "revenue"] = "unknown"
    df.loc[df.index[::40], "order_date"] = "2023-02-31"
    a = analyze(write(df), date_col="order_date", value_col="revenue")
    assert a.profile.column("revenue").n_invalid == 120
    assert a.profile.column("order_date").n_invalid == 150
    assert a.forecast is not None and a.profile.quality.validity < 1
    assert any("do not parse as numeric" in i for i in a.profile.issues)


def test_extreme_outliers_reach_the_report(orders, write):
    df = orders.copy()
    df.loc[df.index[:20], "revenue"] = 9_999_999.0
    text = render_report(analyze(write(df)))
    assert "revenue: 20 extreme value(s) beyond 3 x IQR" in text


def test_leakage_scenario_is_called_out_in_the_report(churn, write):
    df = churn.assign(churn_date=np.where(churn.churned == 1, "2024-05-01", None),
                      refund=np.where(churn.churned == 1, 20.0, 0.0)
                      + np.random.default_rng(0).normal(0, 0.01, len(churn)))
    a = analyze(write(df, name="churn"), target="churned")
    assert {f.column for f in a.model.leakage} == {"churn_date", "refund"}
    assert a.model.test_metrics["roc_auc"] < 0.9
    text = render_report(a)
    assert "possible leakage" in text and "Confirm when `churn_date`, `refund` is recorded" in text


def test_target_with_missing_values_and_useless_features(churn):
    df = churn.copy()
    df["churned"] = df["churned"].astype(object)
    df.loc[df.index[:500], "churned"] = None
    assert train_supervised(df, "churned").n_rows == 2500
    flat = pd.DataFrame({"k": ["same"] * 400, "z": [1.0] * 400,
                         "target": np.random.default_rng(0).integers(0, 2, 400)})
    assert not train_supervised(flat, "target").improves_on_baseline


def test_contradictory_and_ambiguous_questions(orders, write, capsys):
    path = str(write(orders.assign(ship_date=orders.order_date)))
    for q in ("total revenue where region is West and region is East",
              "highest and lowest revenue by region", "average revenue per month"):
        assert main(["ask", path, q]) == 1
    err = capsys.readouterr().err
    assert "contradictory filters" in err and "both the highest and the lowest" in err
    assert "date column is ambiguous" in err


def test_sql_fragments_in_questions_and_values_are_inert(orders, write):
    df = orders.copy()
    df.loc[df.index[:25], "region"] = "O'Brien; DROP TABLE orders--"
    ds = load_dataset(write(df))
    tr, res = ask(ds, "how many orders in o'brien; drop table orders--")
    assert res.rows == [(25,)] and "'O''Brien; DROP TABLE orders--'" in tr.sql
    _, res = ask(ds, "total revenue by region; drop table orders")
    assert len(res.rows) == 5
    with pytest.raises(TranslationError):
        ask(ds, "'; DROP TABLE orders; --")
    assert run_query(ds, "SELECT count(*) FROM orders").rows == [(6000,)]
    ds.close()


def test_hostile_column_names_and_unicode(tmp_path):
    cols = {'amount"; DROP TABLE t; --': [1.0, 2.0, 3.0] * 20, "销售额": [10.0, 20.0, 30.0] * 20,
            "a|b\nc": ["x|y", "p\nq", "z"] * 20, "región": ["norte", "sur", "este"] * 20}
    path = tmp_path / "t.csv"
    pd.DataFrame(cols).to_csv(path, index=False)
    ds = load_dataset(path)
    weird, cjk = next(iter(cols)), "销售额"
    res = run_query(ds, f"SELECT sum({quote_ident(weird)}), sum({quote_ident(cjk)}) FROM t")
    assert res.rows == [(120.0, 1200.0)]
    ds.close()
    text = render_report(analyze(path))
    table_rows = [ln for ln in text.splitlines() if ln.startswith("| a\\|b c |")]
    # The column's row in the overview, and the header of its segment table. This used to
    # expect one line: the newline in the name split that header in two, so it did not match.
    assert table_rows == ["| a\\|b c | categorical | 0.0% | 3 | 0 | 0 |",
                          "| a\\|b c | rows | total | mean | median | share_of_total |"]
    assert '### `amount"; DROP TABLE t; --` by `a|b c`' in text


def test_wide_table_and_long_text(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 300)), columns=[f"c{i}" for i in range(300)])
    df["blob"] = ["x" * 20000] * 200
    path = tmp_path / "wide.parquet"
    df.to_parquet(path)
    a = analyze(path)
    assert a.profile.n_cols == 301 and len(a.correlations) > 1000


def test_requested_steps_fail_loudly(orders, write):
    path = write(orders)
    for kwargs in ({"target": "nope"}, {"sql": "DROP TABLE orders"},
                   {"question": "tell me a joke"}, {"table": "nope"},
                   {"date_col": "order_date", "value_col": "nope"}):
        with pytest.raises(AdsError):
            analyze(path, **kwargs)
