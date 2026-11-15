"""Sales forecasting with naive baselines and rolling-origin evaluation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from auto_data_scientist.eda import _FREQ, to_series
from auto_data_scientist.errors import DataError


class ForecastReport(BaseModel):
    value_col: str
    freq: str
    n_periods: int
    horizon: int
    seasonal_period: int
    backtest_mae: dict[str, float]
    n_backtests: int
    best_model: str
    improves_on_naive: bool
    forecast: list[tuple[str, float, float, float]]
    interval_note: str
    limitations: list[str]


def _predict(name: str, train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if name == "naive":
        return np.repeat(train[-1], horizon)
    if name == "seasonal_naive":
        return np.array([train[-period + (h % period)] for h in range(horizon)])
    seasonal = "add" if period > 1 and len(train) >= 2 * period else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(train, trend="add", seasonal=seasonal,
                                   seasonal_periods=period if seasonal else None,
                                   initialization_method="estimated").fit()
        return np.asarray(fit.forecast(horizon), dtype=float)


def forecast_series(df: pd.DataFrame, date_col: str, value_col: str, horizon: int = 6,
                    freq: str = "M", max_backtests: int = 5) -> ForecastReport:
    """Forecast an aggregated series and compare candidates by rolling-origin backtests.

    Candidates: last value (naive), same period last season (seasonal naive) and
    Holt-Winters exponential smoothing. The model with the lowest mean absolute error over
    the backtests is used. The band is the 80th percentile of absolute backtest errors of
    that model, applied symmetrically: an empirical range, not a model-based interval.
    """
    if not 1 <= horizon <= 60:
        raise ValueError("horizon must be between 1 and 60")
    s = to_series(df, date_col, value_col, freq)
    y = s.to_numpy(dtype=float)
    period = _FREQ[freq][1]
    need = max(2 * horizon + 4, 10)
    if len(y) < need:
        raise DataError(
            f"{len(y)} period(s) at frequency {freq}; at least {need} are needed to backtest "
            f"a {horizon}-step forecast")
    names = ["naive", "holt_winters"]
    limitations: list[str] = []
    if period > 1 and len(y) - horizon > period:
        names.insert(1, "seasonal_naive")
    if period > 1 and len(y) - horizon < 2 * period:
        limitations.append(f"Fewer than two full seasons ({2 * period} periods) were available "
                           "for training, so seasonality could not be modelled.")
    min_train = max(period + 1 if "seasonal_naive" in names else 2, 6)
    n_bt = int(min(max_backtests, (len(y) - min_train) // horizon))
    if n_bt < 1:
        raise DataError("the series is too short to hold out a single backtest window")
    errors: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for i in range(n_bt):
        end = len(y) - (n_bt - i) * horizon
        train, test = y[:end], y[end:end + horizon]
        for name in names:
            errors[name].append(np.abs(_predict(name, train, horizon, period) - test))
    mae = {n: float(np.mean(np.concatenate(e))) for n, e in errors.items()}
    best = min(mae, key=lambda n: mae[n])
    simple = min(mae[n] for n in names if n != "holt_winters")
    per_fold = np.array([e.mean() for e in errors[best]])
    naive_fold = np.minimum.reduce([[e.mean() for e in errors[n]]
                                    for n in names if n != "holt_winters"])
    improves = bool(best == "holt_winters" and mae[best] < 0.95 * simple
                    and (per_fold < naive_fold).mean() >= 0.6)
    if not improves:
        limitations.append("The statistical model did not beat the naive baselines reliably; "
                           "the simplest adequate forecast is reported.")
        best = min((n for n in names if n != "holt_winters"), key=lambda n: mae[n])
    band = float(np.quantile(np.concatenate(errors[best]), 0.8))
    point = _predict(best, y, horizon, period)
    index = pd.date_range(s.index[-1], periods=horizon + 1, freq=_FREQ[freq][0])[1:]
    if n_bt < 3:
        limitations.append(f"Only {n_bt} backtest window(s) were possible; error estimates are "
                           "rough.")
    limitations.append("Forecasts assume the historical pattern continues; promotions, price "
                       "changes and shocks are not modelled.")
    return ForecastReport(
        value_col=value_col, freq=freq, n_periods=len(y), horizon=horizon,
        seasonal_period=period, backtest_mae=mae, n_backtests=n_bt, best_model=best,
        improves_on_naive=improves,
        forecast=[(d.strftime("%Y-%m-%d"), float(p), float(p - band), float(p + band))
                  for d, p in zip(index, point, strict=True)],
        interval_note="Range = forecast +/- the 80th percentile of absolute backtest errors.",
        limitations=limitations)
