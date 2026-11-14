"""Ground-truth checks: the tools must find real signal and must not invent it."""

import numpy as np
import pandas as pd
import pytest

from auto_data_scientist.errors import DataError
from auto_data_scientist.forecasting import forecast_series
from auto_data_scientist.modeling import train_supervised


def _with_leaks(churn: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return churn.assign(
        cancel_fee=np.where(churn.churned == 1, 25.0, 0.0) + rng.normal(0, 0.01, len(churn)),
        churn_reason=np.where(churn.churned == 1, "price", None),
        exit_survey=np.where(churn.churned == 1, "done", "none"))


def test_churn_model_finds_the_true_drivers(churn):
    r = train_supervised(churn, "churned")
    assert r.task == "classification" and r.primary_metric == "roc_auc"
    assert r.improves_on_baseline and r.folds_better_than_baseline == r.n_folds == 5
    assert 0.70 <= r.test_metrics["roc_auc"] <= 0.85      # Bayes-optimal AUC is about 0.77
    assert r.baseline_test_metrics["roc_auc"] == 0.5
    ranked = [f for f, _ in r.importance]
    assert set(ranked[:3]) == {"tenure_months", "support_tickets", "plan"}
    assert dict(r.importance)["noise"] < 0.01
    assert "customer_id" in r.dropped and "customer_id" not in r.features
    assert r.n_train + r.n_test == r.n_rows == 3000 and r.n_test == 600
    assert r.lift_top_decile > 1.5


def test_leaky_columns_are_detected_and_excluded(churn):
    r = train_supervised(_with_leaks(churn), "churned")
    flagged = {f.column for f in r.leakage}
    assert flagged == {"cancel_fee", "churn_reason", "exit_survey"}
    assert not flagged & set(r.features)
    assert r.test_metrics["roc_auc"] < 0.9               # evaluated without the leaks
    leaky = train_supervised(_with_leaks(churn), "churned", keep=["cancel_fee"])
    assert leaky.test_metrics["roc_auc"] > 0.99          # what the leak would have claimed
    assert any("kept on request" in x for x in leaky.limitations)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pure_noise_is_reported_as_no_signal(seed):
    rng = np.random.default_rng(seed)
    n = 1500
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "c": rng.choice(list("xyz"), n), "target": rng.integers(0, 2, n)})
    r = train_supervised(df, "target", seed=seed)
    assert not r.improves_on_baseline
    assert any("No candidate beat the no-skill baseline" in x for x in r.limitations)
    assert abs(r.test_metrics["roc_auc"] - 0.5) < 0.08


def test_duplicated_rows_do_not_fake_a_signal():
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500),
                         "target": rng.integers(0, 2, 500)})
    r = train_supervised(pd.concat([base] * 4, ignore_index=True), "target")
    assert r.n_rows == 500 and not r.improves_on_baseline
    assert any("duplicate row(s) were removed" in x for x in r.limitations)


def test_chance_repeats_in_low_cardinality_data_are_kept():
    rng = np.random.default_rng(5)
    n = 2000
    plan = rng.choice(["basic", "pro"], n)
    df = pd.DataFrame({"plan": plan, "region": rng.choice(list("NSEW"), n),
                       "target": (rng.random(n) < np.where(plan == "basic", 0.7, 0.2)).astype(int)})
    r = train_supervised(df, "target")
    assert r.n_rows == n and r.improves_on_baseline          # class rates are left intact
    assert r.class_balance["1"] == pytest.approx(df.target.mean())
    assert any("repeats are expected" in x for x in r.limitations)


def test_regression_recovers_noise_level_and_ranking():
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.normal(size=(n, 3))
    df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2],
                       "y": 3 * x[:, 0] - 2 * x[:, 1] + rng.normal(0, 1.0, n)})
    r = train_supervised(df, "y")
    assert r.task == "regression" and r.best_model == "ridge_regression"
    assert r.test_metrics["rmse"] == pytest.approx(1.0, rel=0.1)       # irreducible noise sd
    assert r.test_metrics["r2"] == pytest.approx(13 / 14, abs=0.03)    # 1 - 1 / (9 + 4 + 1)
    assert r.baseline_test_metrics["rmse"] == pytest.approx(14 ** 0.5, rel=0.1)
    assert [f for f, _ in r.importance] == ["x1", "x2", "x3"] and r.improves_on_baseline


def test_time_split_holds_out_the_latest_rows(churn):
    df = churn.assign(signup=pd.date_range("2020-01-01", periods=len(churn), freq="D"))
    r = train_supervised(df.sample(frac=1, random_state=0), "churned", time_col="signup")
    assert r.n_test == 600 and "signup" in r.dropped
    assert any("split by time" in x for x in r.limitations)


def test_multiclass_and_reproducibility(churn):
    r1 = train_supervised(churn, "plan", seed=7)
    r2 = train_supervised(churn, "plan", seed=7)
    assert r1.primary_metric == "balanced_accuracy" and not r1.improves_on_baseline
    assert r1.model_dump() == r2.model_dump()


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d.head(30), "at least 50"),
    (lambda d: d.assign(churned=1), "single class"),
    (lambda d: d[["customer_id", "churned"]], "no usable feature"),
    (lambda d: d.assign(churned=[f"c{i % 30}" for i in range(len(d))]), "classes; maximum"),
])
def test_unusable_modelling_inputs(churn, mutate, match):
    with pytest.raises(DataError, match=match):
        train_supervised(mutate(churn), "churned")
    with pytest.raises(DataError, match="unknown target"):
        train_supervised(churn, "nope")


def _season(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({"month": pd.date_range("2018-01-01", periods=n, freq="MS"),
                         "sales": 1000 + 8 * t + 150 * np.sin(2 * np.pi * t / 12)
                         + rng.normal(0, 30, n)})


def test_forecast_beats_naive_on_trend_and_season():
    full = _season(78)
    f = forecast_series(full.iloc[:72], "month", "sales", 6)
    assert f.best_model == "holt_winters" and f.improves_on_naive
    assert f.backtest_mae["holt_winters"] < 0.5 * f.backtest_mae["seasonal_naive"]
    assert f.backtest_mae["holt_winters"] == pytest.approx(30, rel=0.5)   # near the noise sd
    truth = full["sales"].to_numpy()[72:]
    point = np.array([x[1] for x in f.forecast])
    assert np.abs(point - truth).mean() < 60
    assert f.forecast[0][0] == "2024-01-01" and len(f.forecast) == 6
    inside = [(lo <= y <= hi) for (_, _, lo, hi), y in zip(f.forecast, truth, strict=True)]
    assert sum(inside) >= 3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_walk_is_not_oversold(seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"month": pd.date_range("2018-01-01", periods=72, freq="MS"),
                       "sales": 1000 + np.cumsum(rng.normal(0, 30, 72))})
    f = forecast_series(df, "month", "sales", 6)
    assert not f.improves_on_naive and f.best_model in ("naive", "seasonal_naive")
    assert any("did not beat the naive" in x for x in f.limitations)


def test_forecast_input_validation():
    with pytest.raises(DataError, match="at least"):
        forecast_series(_season(8), "month", "sales", 6)
    with pytest.raises(ValueError, match="horizon"):
        forecast_series(_season(72), "month", "sales", 0)
    with pytest.raises(DataError, match="unknown column"):
        forecast_series(_season(72), "month", "nope", 6)
    short = forecast_series(_season(20), "month", "sales", 3)
    assert any("two full seasons" in x for x in short.limitations)
