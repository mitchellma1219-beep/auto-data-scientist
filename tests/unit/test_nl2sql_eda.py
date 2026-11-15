import numpy as np
import pandas as pd
import pytest
from scipy import stats

from auto_data_scientist import eda
from auto_data_scientist.errors import DataError, TranslationError
from auto_data_scientist.ingest import load_dataset
from auto_data_scientist.pipeline import ask
from auto_data_scientist.profiling import profile_frame


@pytest.fixture
def ds(orders, write):
    d = load_dataset(write(orders))
    yield d
    d.close()


def test_total_by_dimension_matches_pandas(ds, orders):
    tr, res = ask(ds, "What is the total revenue by region?")
    expected = orders.groupby("region")["revenue"].sum().sort_values(ascending=False)
    assert [r[0] for r in res.rows] == list(expected.index)
    assert [r[1] for r in res.rows] == pytest.approx(list(expected.values))
    assert 'SUM("revenue")' in tr.sql and 'GROUP BY "region"' in tr.sql


def test_ranking_time_grain_filters_and_counts(ds, orders):
    _, res = ask(ds, "top 2 regions by total revenue")
    top = orders.groupby("region")["revenue"].sum().nlargest(2)
    assert [r[0] for r in res.rows] == list(top.index)

    _, res = ask(ds, "which channel has the lowest average revenue?")
    assert res.rows[0][0] == orders.groupby("channel")["revenue"].mean().idxmin()

    _, res = ask(ds, "average revenue per month")
    assert len(res.rows) == 48
    month = pd.to_datetime(orders["order_date"]).dt.to_period("M")
    assert [r[1] for r in res.rows] == pytest.approx(
        list(orders.groupby(month)["revenue"].mean().values))

    _, res = ask(ds, "how many orders in West?")
    assert res.rows == [(int((orders.region == "West").sum()),)]

    _, res = ask(ds, "total revenue in 2023 by quarter")
    y23 = orders[pd.to_datetime(orders.order_date).dt.year == 2023]
    assert len(res.rows) == 4 and sum(r[1] for r in res.rows) == pytest.approx(
        y23["revenue"].sum())

    tr, res = ask(ds, "number of distinct customer_id")
    assert res.rows == [(orders.customer_id.nunique(),)]

    tr, res = ask(ds, "total revenue for web and store")
    assert res.rows[0][0] == pytest.approx(
        orders[orders.channel.isin(["web", "store"])]["revenue"].sum())
    assert any("any of" in a for a in tr.assumptions)


@pytest.mark.parametrize("question, match", [
    ("", "empty"),
    ("total profit by region", "which numeric column"),
    ("total revenue by store", "no column matches 'store'"),
    ("highest and lowest revenue by region", "both the highest and the lowest"),
    ("total revenue where region is West and region is East", "contradictory filters"),
    ("show me everything", "could not find an aggregation"),
    ("tell me a joke", "could not find an aggregation"),
])
def test_untranslatable_questions_are_refused_not_guessed(ds, question, match):
    with pytest.raises(TranslationError, match=match):
        ask(ds, question)


def test_assumptions_are_reported(ds):
    tr, _ = ask(ds, "which region has the highest revenue")
    assert any("total of revenue" in a for a in tr.assumptions)


def test_correlations_match_scipy(orders):
    p = profile_frame(orders)
    out = eda.correlation_pairs(orders, p)
    row = out[(out.a == "quantity") & (out.b == "revenue")].iloc[0]
    ref = stats.pearsonr(orders.quantity, orders.revenue)
    assert row.r == pytest.approx(ref.statistic) and row.p_value == pytest.approx(ref.pvalue)
    assert out["r"].abs().is_monotonic_decreasing
    sp = eda.correlation_pairs(orders, p, "spearman")
    assert sp[(sp.a == "quantity") & (sp.b == "revenue")].iloc[0].r == pytest.approx(
        stats.spearmanr(orders.quantity, orders.revenue).statistic)
    assert "order_id" not in set(out.a) | set(out.b)
    with pytest.raises(ValueError):
        eda.correlation_pairs(orders, p, "kendall")


def test_correlations_with_missing_values_use_pairwise_complete_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400),
                       "c": rng.normal(size=400), "k": 7.0})
    df["b"] += 0.15 * df["a"]
    df.loc[:40, "c"] = np.nan
    p = profile_frame(df)
    for method, fn in (("pearson", stats.pearsonr), ("spearman", stats.spearmanr)):
        out = eda.correlation_pairs(df, p, method)
        assert len(out) == 3 and "k" not in set(out.a) | set(out.b)      # constant column skipped
        for row in out.itertuples():
            pair = df[[row.a, row.b]].dropna()
            ref = fn(pair[row.a], pair[row.b])
            assert row.n == len(pair) and row.r == pytest.approx(ref.statistic, abs=1e-12)
            assert row.p_value == pytest.approx(ref.pvalue, rel=1e-6)
    assert eda.correlation_pairs(df[["a"]], profile_frame(df[["a"]])).empty


def test_segment_summary(orders):
    seg = eda.segment_summary(orders, "region", "revenue")
    g = orders.groupby("region")["revenue"]
    assert seg["share_of_total"].sum() == pytest.approx(1.0)
    assert seg.index[0] == "West" and seg.loc["West", "total"] == pytest.approx(g.sum()["West"])
    assert seg.loc["East", "median"] == pytest.approx(g.median()["East"])
    with pytest.raises(DataError, match="unknown column"):
        eda.segment_summary(orders, "nope", "revenue")


def test_time_series_recovers_known_trend_and_season():
    n = 60
    dates = pd.date_range("2019-01-01", periods=n, freq="MS")
    values = 100 + 5 * np.arange(n) + 40 * np.sin(2 * np.pi * np.arange(n) / 12)
    ts = eda.time_series_summary(pd.DataFrame({"d": dates, "v": values}), "d", "v", "M")
    assert ts.slope_per_period == pytest.approx(5.0, abs=0.3)
    assert ts.seasonal_autocorr > 0.9 and len(ts.series) == n
    flat = eda.time_series_summary(pd.DataFrame({"d": dates[:10], "v": np.arange(10.0)}),
                                   "d", "v", "M")
    assert flat.seasonal_autocorr is None and any("seasonality" in x for x in flat.notes)
    with pytest.raises(DataError):
        eda.time_series_summary(pd.DataFrame({"d": ["x", "y"], "v": [1, 2]}), "d", "v")


def test_cohort_retention_known_answer():
    rows = [("a", "2024-01-05"), ("a", "2024-02-10"), ("a", "2024-03-01"),
            ("b", "2024-01-20"), ("b", "2024-03-15"),
            ("c", "2024-02-02"), ("c", "2024-02-20"),
            ("d", "2024-02-11"), ("d", "2024-03-30")]
    df = pd.DataFrame(rows, columns=["user", "date"])
    rates, sizes = eda.cohort_retention(df, "user", "date")
    assert sizes.to_dict() == {pd.Period("2024-01"): 2, pd.Period("2024-02"): 2}
    jan, feb = rates.loc[pd.Period("2024-01")], rates.loc[pd.Period("2024-02")]
    assert list(jan) == [1.0, 0.5, 1.0]            # a active in month 1; a and b in month 2
    assert feb[0] == 1.0 and feb[1] == 0.5         # only d returns in March
    assert np.isnan(feb[2])                        # April has not happened: unknown, not zero


def test_kpi_roles(orders):
    k = eda.identify_kpis(profile_frame(orders))
    assert k["metrics"][0] == "revenue" and set(k["dimensions"]) == {"region", "channel"}
    assert k["dates"] == ["order_date"] and set(k["ids"]) == {"order_id", "customer_id"}
