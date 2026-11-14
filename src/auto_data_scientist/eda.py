"""Exploratory analysis: descriptive statistics, correlation, time series, segments, cohorts."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from auto_data_scientist.errors import DataError
from auto_data_scientist.profiling import TableProfile, coerce_numeric, parse_dates

_FREQ = {"D": ("D", 7), "W": ("W", 52), "M": ("MS", 12), "Q": ("QS", 4), "Y": ("YS", 1)}
_MONEY_HINT = re.compile(r"revenue|sales|amount|profit|spend|gmv|income|total|mrr|arr|ltv",
                         re.IGNORECASE)
_METRIC_HINT = re.compile(
    r"price|cost|margin|qty|quantity|units|value|arpu|score|rate|duration|count", re.IGNORECASE)


def suggest_frequency(dates: pd.Series) -> str:
    """A sensible aggregation period for a date column: months from six months of data,
    weeks from four weeks, otherwise days."""
    valid = parse_dates(dates).dropna()
    if valid.empty:
        raise DataError("the column has no valid dates")
    span = (valid.max() - valid.min()).days
    return "M" if span >= 180 else "W" if span >= 28 else "D"


def numeric_summary(df: pd.DataFrame, profile: TableProfile) -> pd.DataFrame:
    """count, mean, std, min, quartiles, max and skew for every numeric column."""
    cols = profile.by_kind("numeric")
    if not cols:
        return pd.DataFrame()
    data = pd.DataFrame({c: coerce_numeric(df[c]) for c in cols})
    out = data.describe().T
    out["skew"] = data.skew()
    return out


def correlation_pairs(df: pd.DataFrame, profile: TableProfile, method: str = "pearson",
                      min_n: int = 10, max_columns: int = 300) -> pd.DataFrame:
    """Pairwise correlations between numeric columns, strongest first.

    Uses pairwise-complete observations. The p-value is the usual t approximation and is
    descriptive only: many pairs are examined and nothing is corrected for that. At most
    ``max_columns`` numeric columns are compared, to bound the quadratic cost.
    Correlation is a measure of association in this sample; it does not show causation.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    cols = profile.by_kind("numeric")[:max_columns]
    empty = pd.DataFrame(columns=["a", "b", "r", "n", "p_value"])
    if len(cols) < 2:
        return empty
    data = pd.DataFrame({c: coerce_numeric(df[c]) for c in cols})
    r_all = data.corr(method=method, min_periods=min_n).to_numpy()
    present = data.notna().to_numpy(dtype=float)
    n_all = present.T @ present
    i, j = np.triu_indices(len(cols), k=1)
    r, n = r_all[i, j], n_all[i, j]
    ok = ~np.isnan(r)
    i, j, r, n = i[ok], j[ok], r[ok], n[ok]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r**2))
    p = np.where(np.abs(r) >= 1, 0.0, 2 * stats.t.sf(np.abs(t), n - 2))
    out = pd.DataFrame({"a": [cols[k] for k in i], "b": [cols[k] for k in j], "r": r,
                        "n": n.astype(int), "p_value": p})
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


@dataclass(frozen=True)
class TimeSeriesSummary:
    series: pd.Series
    freq: str
    slope_per_period: float
    trend_r: float
    seasonal_period: int
    seasonal_autocorr: float | None
    notes: tuple[str, ...]


def to_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "M",
              agg: str = "sum") -> pd.Series:
    """Aggregate a value by calendar period. Periods without rows become 0 for sums."""
    if freq not in _FREQ:
        raise ValueError(f"freq must be one of {sorted(_FREQ)}")
    for c in (date_col, value_col):
        if c not in df.columns:
            raise DataError(f"unknown column {c!r}; available: {list(df.columns)}")
    dates = parse_dates(df[date_col])
    values = coerce_numeric(df[value_col])
    ok = dates.notna() & values.notna()
    if ok.sum() < 2:
        raise DataError(f"not enough valid ({date_col}, {value_col}) pairs for a time series")
    s = pd.Series(values[ok].to_numpy(), index=pd.DatetimeIndex(dates[ok])).sort_index()
    out = s.resample(_FREQ[freq][0]).agg(agg)
    return out.fillna(0.0) if agg in ("sum", "count") else out.interpolate()


def time_series_summary(df: pd.DataFrame, date_col: str, value_col: str,
                        freq: str = "M") -> TimeSeriesSummary:
    """Linear trend and seasonal autocorrelation of an aggregated series (descriptive only)."""
    s = to_series(df, date_col, value_col, freq)
    period = _FREQ[freq][1]
    notes = []
    y = s.to_numpy(dtype=float)
    if len(y) < 3:
        raise DataError(f"only {len(y)} period(s) at frequency {freq}; need at least 3")
    fit = stats.linregress(np.arange(len(y)), y)
    seasonal = None
    if period > 1 and len(y) >= 2 * period:
        resid = y - (fit.intercept + fit.slope * np.arange(len(y)))
        a, b = resid[:-period], resid[period:]
        if a.std() > 0 and b.std() > 0:
            seasonal = float(np.corrcoef(a, b)[0, 1])
    elif period > 1:
        notes.append(f"fewer than {2 * period} periods: seasonality was not assessed")
    notes.append("the trend line ignores autocorrelation; treat it as a description, not a test")
    return TimeSeriesSummary(series=s, freq=freq, slope_per_period=float(fit.slope),
                             trend_r=float(fit.rvalue), seasonal_period=period,
                             seasonal_autocorr=seasonal, notes=tuple(notes))


def segment_summary(df: pd.DataFrame, by: str, metric: str) -> pd.DataFrame:
    """Rows, total, mean, median and share of total of ``metric`` for each value of ``by``."""
    for c in (by, metric):
        if c not in df.columns:
            raise DataError(f"unknown column {c!r}; available: {list(df.columns)}")
    data = pd.DataFrame({by: df[by], metric: coerce_numeric(df[metric])}).dropna()
    if data.empty:
        raise DataError(f"no rows with both {by} and {metric}")
    out = data.groupby(by, observed=True)[metric].agg(
        rows="count", total="sum", mean="mean", median="median")
    total = out["total"].sum()
    out["share_of_total"] = out["total"] / total if total else np.nan
    return out.sort_values("total", ascending=False)


def cohort_retention(df: pd.DataFrame, user_col: str, date_col: str,
                     freq: str = "M") -> tuple[pd.DataFrame, pd.Series]:
    """Retention matrix: share of each first-activity cohort active N periods later.

    Returns (rates, cohort_sizes). Column 0 is always 1.0. Cells for periods that have not
    happened yet are NaN, not 0.
    """
    if freq not in ("D", "W", "M", "Q", "Y"):
        raise ValueError("freq must be one of D, W, M, Q, Y")
    for c in (user_col, date_col):
        if c not in df.columns:
            raise DataError(f"unknown column {c!r}; available: {list(df.columns)}")
    dates = parse_dates(df[date_col])
    data = pd.DataFrame({"user": df[user_col], "period": dates.dt.to_period(freq)}).dropna()
    if data.empty:
        raise DataError("no rows with both a user and a valid date")
    data = data.drop_duplicates()
    first = data.groupby("user")["period"].transform("min")
    data["cohort"] = first
    data["offset"] = [(p - c).n for p, c in zip(data["period"], data["cohort"], strict=True)]
    counts = data.pivot_table(index="cohort", columns="offset", values="user", aggfunc="nunique")
    sizes = counts[0]
    rates = counts.div(sizes, axis=0)
    last = data["period"].max()
    for cohort in rates.index:
        observable = (last - cohort).n
        for off in rates.columns:
            if off <= observable and pd.isna(rates.loc[cohort, off]):
                rates.loc[cohort, off] = 0.0
    return rates, sizes.astype(int)


def identify_kpis(profile: TableProfile) -> dict[str, list[str]]:
    """Heuristic roles for columns: likely metrics, dimensions, dates and identifiers."""
    numeric = profile.by_kind("numeric")
    hinted = [c for c in numeric if _MONEY_HINT.search(c)]
    hinted += [c for c in numeric if _METRIC_HINT.search(c) and c not in hinted]
    return {
        "metrics": hinted + [c for c in numeric if c not in hinted],
        "dimensions": profile.by_kind("categorical", "boolean"),
        "dates": profile.by_kind("datetime"),
        "ids": profile.by_kind("id"),
    }
