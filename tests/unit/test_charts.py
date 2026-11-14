"""The figures are checked through their data: what is drawn must equal what was computed."""

import os

import numpy as np
import pandas as pd
import pytest

if os.environ.get("ADS_REQUIRE_UI") != "1":      # CI sets it: there a missing extra must fail
    pytest.importorskip("plotly")

from auto_data_scientist import charts, eda
from auto_data_scientist.errors import DataError
from auto_data_scientist.forecasting import forecast_series
from auto_data_scientist.modeling import CandidateScore, ModelReport
from auto_data_scientist.profiling import profile_frame

MODES = ("light", "dark")


def _channel(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb(colour: str) -> list[float]:
    return [_channel(int(colour[i:i + 2], 16) / 255) for i in (1, 3, 5)]


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio."""
    la, lb = (0.2126 * r + 0.7152 * g + 0.0722 * bl for r, g, bl in (_rgb(a), _rgb(b)))
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def lightness(colour: str) -> float:
    """OKLab L, a perceptual lightness between 0 and 1."""
    r, g, b = _rgb(colour)
    lms = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
           0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
           0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    l_, m_, s_ = (v ** (1 / 3) for v in lms)
    return 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_


def strings(node: object) -> list[str]:
    """Every string anywhere in a figure."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in strings(v)]
    if isinstance(node, list | tuple):
        return [s for v in node for s in strings(v)]
    return []


def trace(fig, name: str):
    return next(t for t in fig.data if t.name == name)


def model_report(task: str = "classification", **changes) -> ModelReport:
    """A hand-written report, so chart tests do not depend on training a model."""
    if task == "classification":
        base = dict(
            task="classification", target="churned", primary_metric="roc_auc",
            higher_is_better=True, best_model="logistic_regression",
            candidates=[
                CandidateScore(model="baseline", cv_mean=0.5, cv_std=0.0, fold_scores=[0.5] * 5),
                CandidateScore(model="logistic_regression", cv_mean=0.73, cv_std=0.02,
                               fold_scores=[0.70, 0.72, 0.73, 0.74, 0.76]),
                CandidateScore(model="random_forest", cv_mean=0.67, cv_std=0.01,
                               fold_scores=[0.66, 0.66, 0.67, 0.68, 0.68]),
                CandidateScore(model="gradient_boosting", cv_mean=0.69, cv_std=0.03,
                               fold_scores=[0.65, 0.68, 0.69, 0.70, 0.73])],
            test_metrics={"accuracy": 0.72, "balanced_accuracy": 0.58, "f1_macro": 0.57,
                          "roc_auc": 0.726, "average_precision": 0.52},
            baseline_test_metrics={"accuracy": 0.71, "balanced_accuracy": 0.5, "f1_macro": 0.41,
                                   "roc_auc": 0.5, "average_precision": 0.29},
            importance=[("tenure_months", 0.133), ("support_tickets", 0.062), ("plan", 0.019),
                        ("monthly_spend", -0.0002), ("noise", -0.002)])
    else:
        base = dict(
            task="regression", target="y", primary_metric="rmse", higher_is_better=False,
            best_model="ridge_regression",
            candidates=[
                CandidateScore(model="baseline", cv_mean=3.7, cv_std=0.1, fold_scores=[3.6, 3.8]),
                CandidateScore(model="ridge_regression", cv_mean=1.0, cv_std=0.02,
                               fold_scores=[0.98, 1.02]),
                CandidateScore(model="random_forest", cv_mean=1.2, cv_std=0.03,
                               fold_scores=[1.17, 1.23])],
            test_metrics={"rmse": 1.01, "mae": 0.81, "r2": 0.93},
            baseline_test_metrics={"rmse": 3.78, "mae": 3.0, "r2": -0.001},
            importance=[("x1", 2.9), ("x2", 1.9), ("x3", 0.0)])
    base.update(n_rows=3000, n_train=2400, n_test=600, features=[f for f, _ in base["importance"]],
                improves_on_baseline=True, folds_better_than_baseline=5, n_folds=5, leakage=[],
                dropped={}, limitations=[], seed=0, n_selection=2400, n_importance=600)
    base.update(changes)
    return ModelReport(**base)


@pytest.mark.parametrize("mode", MODES)
def test_palette_is_legible_and_scales_are_ordered(mode):
    p = charts.palette(mode)
    for mark in (p.accent, p.second, p.muted):
        assert contrast(mark, p.surface) >= 3.0
    assert contrast(p.ink, p.surface) >= 7.0 and contrast(p.ink_secondary, p.surface) >= 4.5
    # Sequential: one hue whose lightness moves away from the surface as the value grows.
    seq = [lightness(c) for c in p.sequential]
    step = np.diff(seq)
    assert (step < -0.05).all() if mode == "light" else (step > 0.05).all()
    assert abs(seq[0] - lightness(p.surface)) < abs(seq[-1] - lightness(p.surface))
    # Diverging: a neutral middle, and two arms that step away from it by equal amounts.
    div = [lightness(c) for c in p.diverging]
    mid = len(div) // 2
    r, g, b = (int(p.diverging[mid][i:i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) <= 6                       # the midpoint has no hue
    negative, positive = div[:mid][::-1], div[mid + 1:]
    assert negative == pytest.approx(positive, abs=0.02)           # arms match step for step
    away = np.diff([div[mid], *positive])
    assert (away < -0.1).all() if mode == "light" else (away > 0.1).all()
    with pytest.raises(ValueError):
        charts.palette("sepia")


def test_number_and_name_helpers():
    assert [charts.compact(v) for v in (0.726, 3, 95.5, 1284, 12_940, 303_582, 4_200_000, 2.5e9)] \
        == ["0.726", "3", "95.5", "1,284", "12.9k", "304k", "4.2M", "2.5B"]
    assert charts.compact(float("nan")) == "" and charts.compact(-18_250) == "-18.2k"
    assert charts.short("revenue") == "revenue" and charts.short("x" * 60) == "x" * 39 + "…"
    assert charts.metric_label("roc_auc") == "ROC AUC"
    assert charts.metric_label("my_score") == "my score"
    assert charts.model_label("baseline") == "No-skill baseline"
    assert charts.model_label("gradient_boosting") == "Gradient boosting"


def test_histogram_counts_equal_numpy(churn):
    fig = charts.distribution_figure(churn["monthly_spend"])
    bars = fig.data[0]
    edges = np.array([*[c[0] for c in bars.customdata], bars.customdata[-1][1]])
    expected, _ = np.histogram(churn["monthly_spend"], bins=edges)
    assert list(bars.y) == list(expected) and sum(bars.y) == 3000
    assert len(bars.y) <= 40 and np.allclose(bars.x, (edges[:-1] + edges[1:]) / 2)
    assert np.allclose(bars.width, np.diff(edges))                  # bins touch, none overlap
    median = churn["monthly_spend"].median()
    assert fig.layout.shapes[0].x0 == pytest.approx(median)
    assert fig.layout.annotations[0].text == f"median {charts.compact(median)}"
    assert "3,000 values" in fig.layout.title.text and not fig.layout.showlegend
    assert len(charts.distribution_figure(churn["monthly_spend"], max_bins=10).data[0].y) <= 10


def test_whole_numbers_get_one_bar_per_value(orders):
    bars = charts.distribution_figure(orders["quantity"]).data[0]
    counts = orders["quantity"].value_counts().sort_index()
    assert list(bars.x) == [1.0, 2.0, 3.0, 4.0, 5.0] and list(bars.y) == list(counts)
    assert max(bars.width) < 1                                      # thin bars, not blocks


def test_histogram_leaves_out_unusable_values_and_says_so():
    s = pd.Series([1.5, 2.5, "3.5", "unknown", None, " 4.5 ", ""], name="amount")
    fig = charts.distribution_figure(s)
    assert sum(fig.data[0].y) == 4
    assert "4 values" in fig.layout.title.text
    assert "3 missing or non-numeric left out" in fig.layout.title.text
    with pytest.raises(DataError, match="no numeric values"):
        charts.distribution_figure(pd.Series(["a", None], name="label"))
    constant = charts.distribution_figure(pd.Series([7.0] * 30, name="k")).data[0]
    assert list(constant.y) == [30]


def test_category_bars_match_value_counts(orders):
    fig = charts.category_figure(orders["region"])
    counts = orders["region"].value_counts()
    assert list(fig.data[0].y) == list(counts.index) and list(fig.data[0].x) == list(counts)
    assert sum(fig.data[0].customdata) == pytest.approx(1.0)
    assert [a.text for a in fig.layout.annotations] == [f"{n:,}" for n in counts]
    s = pd.Series([*"aaaaabbbbcccdde", None, " ", "NA"], name="grade")
    folded = charts.category_figure(s, top=3)
    assert list(folded.data[0].y) == ["a", "b", "Other (3 categories)"]
    assert list(folded.data[0].x) == [5, 4, 6]
    assert folded.data[0].marker.color[-1] == charts.palette().muted
    assert "15 rows in 5 categories. 3 missing left out." in folded.layout.title.text
    with pytest.raises(DataError):
        charts.category_figure(pd.Series([None, ""], name="empty"))


def test_segment_bars_match_the_summary(orders):
    seg = eda.segment_summary(orders, "region", "revenue")
    fig = charts.segment_figure(seg, dimension="region", metric="revenue")
    assert list(fig.data[0].y) == list(seg.index) == ["West", "East", "North", "South"]
    assert list(fig.data[0].x) == pytest.approx(list(seg["total"]))
    assert fig.layout.annotations[0].text == charts.compact(seg["total"].iloc[0])
    assert fig.layout.title.text.startswith("<b>Total of revenue by region</b>")
    share = charts.segment_figure(seg, dimension="region", metric="revenue",
                                  stat="share_of_total")
    assert sum(share.data[0].x) == pytest.approx(1.0) and share.layout.xaxis.tickformat == ".0%"
    assert share.layout.annotations[0].text == f"{seg['share_of_total'].iloc[0]:.1%}"
    mean = charts.segment_figure(seg, dimension="region", metric="revenue", stat="mean")
    assert list(mean.data[0].x) == sorted(seg["mean"], reverse=True)
    with pytest.raises(ValueError):
        charts.segment_figure(seg, dimension="region", metric="revenue", stat="mode")
    with pytest.raises(DataError):
        charts.segment_figure(seg.iloc[:0], dimension="region", metric="revenue")


def test_long_segment_tails_are_folded_only_when_they_add_up():
    seg = pd.DataFrame({"rows": range(20, 0, -1), "total": [100.0 * i for i in range(20, 0, -1)],
                        "mean": np.linspace(9, 1, 20), "median": np.linspace(9, 1, 20)},
                       index=[f"s{i:02d}" for i in range(20)])
    seg["share_of_total"] = seg["total"] / seg["total"].sum()
    total = charts.segment_figure(seg, dimension="store", metric="sales", top=5)
    assert list(total.data[0].y) == ["s00", "s01", "s02", "s03", "Other (16 segments)"]
    assert sum(total.data[0].x) == pytest.approx(seg["total"].sum())      # nothing is lost
    mean = charts.segment_figure(seg, dimension="store", metric="sales", stat="mean", top=5)
    assert len(mean.data[0].y) == 5 and "Other" not in " ".join(mean.data[0].y)
    assert "Top 5 of 20 segments; 15 are not shown." in mean.layout.title.text


def test_time_series_and_forecast_traces(orders):
    series = eda.to_series(orders, "order_date", "revenue", "M")
    plain = charts.timeseries_figure(series, value_name="revenue")
    assert len(plain.data) == 1 and not plain.layout.showlegend
    assert list(plain.data[0].y) == pytest.approx(list(series.to_numpy()))
    assert plain.data[0].x[0] == "2021-01-01" and plain.data[0].x[-1] == "2024-12-01"
    assert plain.layout.yaxis.rangemode == "tozero" and plain.layout.hovermode == "x unified"

    report = forecast_series(orders, "order_date", "revenue", 6, "M")
    fig = charts.timeseries_figure(series, value_name="revenue", forecast=report)
    forecast = trace(fig, "Forecast")
    assert list(forecast.x) == [row[0] for row in report.forecast]
    assert list(forecast.y) == pytest.approx([row[1] for row in report.forecast])
    assert [list(c) for c in forecast.customdata] == [[row[2], row[3]] for row in report.forecast]
    assert list(trace(fig, "Range low").y) == [row[2] for row in report.forecast]
    assert list(trace(fig, "Range").y) == [row[3] for row in report.forecast]
    assert trace(fig, "Range").fill == "tonexty" and forecast.line.dash == "dash"
    assert forecast.line.color != trace(fig, "History").line.color
    assert list(trace(fig, "Join").y) == pytest.approx([series.iloc[-1], report.forecast[0][1]])
    assert fig.layout.showlegend and "not a model-based interval" in charts.note(fig)
    assert not hasattr(fig.layout, "yaxis2") or fig.layout.yaxis2.title.text is None


def test_time_series_axis_keeps_negative_values_visible():
    s = pd.Series([5.0, -3.0, 2.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    assert charts.timeseries_figure(s, value_name="net").layout.yaxis.rangemode == "normal"
    with pytest.raises(DataError):
        charts.timeseries_figure(s.iloc[:0], value_name="net")


def test_correlation_heatmap_shows_each_pair_once(churn):
    profile = profile_frame(churn)
    pairs = eda.correlation_pairs(churn, profile)
    columns = profile.by_kind("numeric")
    fig = charts.correlation_figure(pairs, columns=columns)
    heat = fig.data[0]
    assert list(heat.y) == columns[1:] and list(heat.x) == columns[:-1]
    z = np.array(heat.z, dtype=float)
    lookup = {frozenset((r.a, r.b)): (r.r, r.n) for r in pairs.itertuples()}
    for i, row in enumerate(heat.y):
        for j, col in enumerate(heat.x):
            if j <= i:
                r, n = lookup[frozenset((row, col))]
                assert z[i, j] == pytest.approx(r) and heat.customdata[i][j] == n
            else:
                assert np.isnan(z[i, j])                              # upper triangle is empty
    assert int(np.isfinite(z).sum()) == len(pairs) == 6
    assert (heat.zmin, heat.zmid, heat.zmax) == (-1, 0, 1)
    assert [c for _, c in heat.colorscale] == list(charts.palette().diverging)
    assert "not evidence of cause" in charts.note(fig)
    with pytest.raises(DataError):
        charts.correlation_figure(pairs.iloc[:0])


def test_correlation_heatmap_keeps_the_strongest_columns_when_there_are_too_many():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 8)), columns=[f"c{i}" for i in range(8)])
    df["c7"] = df["c0"] * 0.9 + rng.normal(0, 0.2, 300)
    pairs = eda.correlation_pairs(df, profile_frame(df))
    fig = charts.correlation_figure(pairs, max_columns=3)
    shown = set(fig.data[0].x) | set(fig.data[0].y)
    assert len(shown) == 3 and {"c0", "c7"} <= shown
    assert "5 with weaker correlations not shown" in fig.layout.title.text


def test_retention_heatmap_matches_the_matrix_and_keeps_the_future_empty():
    rows = [("a", "2024-01-05"), ("a", "2024-02-10"), ("a", "2024-03-01"),
            ("b", "2024-01-20"), ("b", "2024-03-15"),
            ("c", "2024-02-02"), ("c", "2024-02-20"),
            ("d", "2024-02-11"), ("d", "2024-03-30")]
    rates, sizes = eda.cohort_retention(pd.DataFrame(rows, columns=["user", "date"]),
                                        "user", "date")
    fig = charts.retention_figure(rates, sizes)
    heat = fig.data[0]
    assert list(heat.x) == [1, 2] and list(heat.y) == ["2024-01 (n = 2)", "2024-02 (n = 2)"]
    assert [list(r) for r in heat.z] == [[0.5, 1.0], [0.5, None]]     # April is unknown, not 0
    assert heat.zmin == 0 and heat.zmax == 1.0
    assert [c for _, c in heat.colorscale] == list(charts.palette().sequential)
    assert "empty, not zero" in charts.note(fig)
    only_first = rates[[0]]
    with pytest.raises(DataError, match="at least two periods"):
        charts.retention_figure(only_first, sizes)


def test_retention_heatmap_limits_cohorts_and_periods(orders):
    rates, sizes = eda.cohort_retention(orders, "customer_id", "order_date", "M")
    fig = charts.retention_figure(rates, sizes, max_cohorts=6, max_periods=5)
    heat = fig.data[0]
    assert len(heat.y) == 6 and list(heat.x) == [1, 2, 3, 4, 5]
    assert heat.y[-1].startswith(str(rates.index[-1]))                 # the latest cohorts
    assert f"6 of {len(rates)} cohorts" in fig.layout.title.text
    shown = np.array(heat.z, dtype=float)
    expected = rates.iloc[-6:][[1, 2, 3, 4, 5]].to_numpy(dtype=float)
    assert np.allclose(shown, expected, equal_nan=True)


def test_model_comparison_shows_every_fold_and_ranks_candidates():
    report = model_report()
    fig = charts.model_comparison_figure(report)
    selected = trace(fig, "Selected model")
    assert list(selected.x) == [0.73] and list(selected.y) == ["Logistic regression"]
    others = trace(fig, "Other candidates")
    assert list(others.y) == ["Gradient boosting", "Random forest"]    # best first
    assert list(others.x) == [0.69, 0.67] and list(others.customdata) == [0.03, 0.01]
    assert list(trace(fig, "No-skill baseline").x) == [0.5]
    folds = [x for t in fig.data if t.name.endswith("folds") for x in t.x]
    assert sorted(folds) == sorted(s for c in report.candidates for s in c.fold_scores)
    assert fig.layout.shapes[0].x0 == 0.5                              # baseline reference line
    assert [a.text for a in fig.layout.annotations] == ["0.730", "0.690", "0.670", "0.500"]
    assert list(fig.layout.yaxis.categoryarray)[-1] == "No-skill baseline"
    assert fig.layout.showlegend and "higher is better" in fig.layout.title.text
    assert "seeded sample" not in charts.note(fig)
    sampled = charts.model_comparison_figure(model_report(n_train=80_000, n_selection=20_000))
    assert "seeded sample of 20,000 of the 80,000 training rows" in charts.note(sampled)


def test_model_comparison_for_an_error_metric_ranks_the_lowest_first():
    fig = charts.model_comparison_figure(model_report("regression"))
    assert list(fig.layout.yaxis.categoryarray) == ["Ridge regression", "Random forest",
                                                    "No-skill baseline"]
    assert "lower is better" in fig.layout.title.text
    assert [a.text for a in fig.layout.annotations] == ["1.000", "1.200", "3.700"]


def test_holdout_bars_put_model_and_baseline_side_by_side():
    report = model_report()
    fig = charts.holdout_figure(report)
    model, base = fig.data
    assert model.name == "Logistic regression" and base.name == "No-skill baseline"
    assert list(model.y) == ["ROC AUC", "Accuracy", "Balanced accuracy", "F1 (macro)",
                             "Average precision"]                      # primary metric first
    assert list(model.x) == [0.726, 0.72, 0.58, 0.57, 0.52]
    assert list(base.x) == [0.5, 0.71, 0.5, 0.41, 0.29]
    assert list(model.text) == ["0.726", "0.720", "0.580", "0.570", "0.520"]
    assert fig.layout.barmode == "group" and fig.layout.showlegend
    assert list(fig.layout.xaxis.range) == [0, 1.1] and "600 held-out rows" in fig.layout.title.text
    regression = charts.holdout_figure(model_report("regression"))
    assert list(regression.data[0].y) == ["RMSE", "MAE"]               # R² has another scale
    assert list(regression.data[0].x) == [1.01, 0.81] and list(regression.data[1].x) == [3.78, 3.0]


def test_importance_bars_are_ranked_and_signed():
    report = model_report()
    fig = charts.importance_figure(report)
    assert list(fig.data[0].y) == [f for f, _ in report.importance]
    assert list(fig.data[0].x) == [v for _, v in report.importance]
    assert [a.text for a in fig.layout.annotations] == ["+0.133", "+0.062", "+0.019", "0.000",
                                                        "-0.002"]
    assert fig.layout.xaxis.range[0] < -0.002                          # negative bars are visible
    assert "not what causes the outcome" in fig.layout.title.text
    assert "600 hold-out rows" in charts.note(fig)
    top = charts.importance_figure(report, top=2)
    assert len(top.data[0].y) == 2 and "Top 2 of 5 columns" in top.layout.title.text
    with pytest.raises(DataError):
        charts.importance_figure(model_report(importance=[]))


def test_quality_components_add_up_to_the_score(orders):
    messy = orders.copy()
    messy.loc[messy.index[:600], "channel"] = None
    messy = pd.concat([messy, messy.iloc[:300]], ignore_index=True)
    quality = profile_frame(messy).quality
    fig = charts.quality_figure(quality)
    track, parts = fig.data
    assert list(track.x) == [1.0] * 5 and track.hoverinfo == "skip"
    assert list(parts.x) == [quality.completeness, quality.validity, quality.uniqueness,
                             quality.consistency, quality.outliers]
    assert parts.y[0] == "Completeness (weight 35%)" and parts.y[2] == "Uniqueness (weight 15%)"
    assert sum(c[1] for c in parts.customdata) == pytest.approx(quality.score, abs=0.01)
    assert fig.layout.annotations[0].text == f"{quality.completeness:.1%}"
    assert f"Score {quality.score:.1f} of 100" in fig.layout.title.text
    empty = profile_frame(orders.iloc[:0]).quality
    with pytest.raises(DataError):
        charts.quality_figure(empty)


def test_text_from_the_data_cannot_inject_markup():
    link = '<a href="http://example.invalid">click</a>'
    df = pd.DataFrame({link: [1.0, 2.0, 3.0, 4.0] * 10, "<b>bold</b>": [4.0, 1.0, 3.0, 2.0] * 10,
                       "kind": [link, "<i>x</i>", "plain", "a&b"] * 10})
    profile = profile_frame(df)
    seg = eda.segment_summary(df, "kind", link)
    figures = [
        charts.distribution_figure(df[link]),
        charts.category_figure(df["kind"]),
        charts.segment_figure(seg, dimension="kind", metric=link),
        charts.correlation_figure(eda.correlation_pairs(df, profile)),
        charts.importance_figure(model_report(importance=[(link, 0.2), ("<b>bold</b>", 0.1)],
                                              target=link)),
        charts.timeseries_figure(pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2,
                                                                         freq="MS")),
                                 value_name=link),
    ]
    for fig in figures:
        text = strings(fig.to_dict())
        assert not any("<a" in s or "<i>" in s or "<b>bold" in s for s in text)
        assert any("&lt;" in s for s in text)                       # shown as text instead
    assert "a&amp;b" in figures[1].data[0].y


@pytest.mark.parametrize("mode", MODES)
def test_every_chart_has_one_value_axis_and_a_legend_only_for_several_series(mode, orders, churn):
    p = charts.palette(mode)
    profile = profile_frame(churn)
    rates, sizes = eda.cohort_retention(orders, "customer_id", "order_date", "Q")
    series = eda.to_series(orders, "order_date", "revenue", "M")
    single = [
        charts.distribution_figure(orders["revenue"], mode=mode),
        charts.category_figure(orders["channel"], mode=mode),
        charts.segment_figure(eda.segment_summary(orders, "channel", "revenue"),
                              dimension="channel", metric="revenue", mode=mode),
        charts.timeseries_figure(series, value_name="revenue", mode=mode),
        charts.correlation_figure(eda.correlation_pairs(churn, profile), mode=mode),
        charts.retention_figure(rates, sizes, mode=mode),
        charts.importance_figure(model_report(), mode=mode),
        charts.quality_figure(profile.quality, mode=mode),
    ]
    several = [
        charts.timeseries_figure(series, value_name="revenue", mode=mode,
                                 forecast=forecast_series(orders, "order_date", "revenue", 6)),
        charts.model_comparison_figure(model_report(), mode=mode),
        charts.holdout_figure(model_report(), mode=mode),
    ]
    for fig in [*single, *several]:
        layout = fig.to_dict()["layout"]
        assert not [k for k in layout if k.startswith(("xaxis", "yaxis")) and k[-1].isdigit()]
        assert layout["paper_bgcolor"] == layout["plot_bgcolor"] == p.surface
        assert layout["title"]["font"]["color"] == p.ink and isinstance(charts.note(fig), str)
    assert not any(fig.layout.showlegend for fig in single)
    assert all(fig.layout.showlegend for fig in several)
    assert single[0].data[0].marker.color == p.accent
    assert trace(several[0], "Forecast").line.color == p.second
