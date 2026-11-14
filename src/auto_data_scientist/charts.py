"""Plotly figures for the dashboard and the demo pages.

Every function here is pure: it takes analysis results and returns a
``plotly.graph_objects.Figure``. Nothing imports Streamlit, reads files or estimates
anything; the only arithmetic is binning a histogram and reshaping tables for display.
Plotly is an optional dependency (the ``ui`` extra), so the rest of the package never
imports this module.

Rules that every chart follows, so the set reads as one system:

* The job picks the form: ranked bars for magnitude, a line for change over time, dots for
  scores that do not start at zero, a grid for pairs.
* One value axis per chart. A single series has no legend (the title names it); two or more
  series always have one, and the forecast is also dashed so it does not rely on colour.
* Colour does one job at a time: one hue for identity or magnitude, blue and red around a
  neutral grey for signed values, grey for context such as the no-skill baseline.
* Marks are thin, grids are hairlines and labels are sparing: the value at the end of a bar,
  everything else in the hover text and in the table shown next to the chart.
* Text taken from the data (column names, category values) is escaped, because Plotly
  renders a subset of HTML in labels and a file must not be able to plant a link.

Colours come from one fixed palette with a light and a dark variant (``PALETTES``). The
tests check that every series colour keeps at least 3:1 contrast against its surface and
that the sequential and diverging scales change lightness monotonically.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from auto_data_scientist.errors import DataError
from auto_data_scientist.forecasting import ForecastReport
from auto_data_scientist.modeling import ModelReport
from auto_data_scientist.profiling import QualityScore, coerce_numeric, missing_mask

Mode = Literal["light", "dark"]
SegmentStat = Literal["total", "mean", "median", "rows", "share_of_total"]

FONT = ('"Source Sans", "Source Sans 3", "Source Sans Pro", system-ui, -apple-system, '
        '"Segoe UI", sans-serif')
_ADDITIVE = ("total", "rows", "share_of_total")
_STAT_LABEL = {"total": "Total", "mean": "Mean", "median": "Median", "rows": "Rows",
               "share_of_total": "Share of total"}
_METRIC_LABEL = {
    "roc_auc": "ROC AUC", "average_precision": "Average precision", "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy", "f1_macro": "F1 (macro)", "rmse": "RMSE",
    "mae": "MAE", "r2": "R²",
}
_QUALITY_PARTS = ("completeness", "validity", "uniqueness", "consistency", "outliers")


@dataclass(frozen=True)
class Palette:
    """Colour roles for one mode. ``sequential`` runs low to high, ``diverging`` -1 to +1."""

    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    accent: str
    accent_soft: str
    second: str
    sequential: tuple[str, ...]
    diverging: tuple[str, ...]


PALETTES: dict[str, Palette] = {
    "light": Palette(
        surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", accent="#2a78d6", accent_soft="#86b6ef",
        second="#eb6834",
        sequential=("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"),
        diverging=("#762221", "#c74845", "#ea9a93", "#f0efec", "#86b6ef", "#2a78d6", "#104281"),
    ),
    "dark": Palette(
        surface="#1a1a19", ink="#f0efec", ink_secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", accent="#3987e5", accent_soft="#1c5cab",
        second="#d95926",
        sequential=("#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"),
        diverging=("#ea9a93", "#d75853", "#9e3432", "#383835", "#1c5cab", "#3987e5", "#86b6ef"),
    ),
}


def palette(mode: Mode = "light") -> Palette:
    """The colour roles for ``mode``."""
    if mode not in PALETTES:
        raise ValueError(f"mode must be one of {sorted(PALETTES)}")
    return PALETTES[mode]


def safe_text(value: object) -> str:
    """Escape text that comes from the data before Plotly renders it as a label."""
    return html.escape(str(value), quote=False).replace("\n", " ")


def compact(value: float) -> str:
    """Short number for a label, in the style of the axis ticks: 1,284 / 12.9k / 4.2M / 0.726."""
    if value != value:  # NaN
        return ""
    size = abs(value)
    for limit, unit, suffix in ((1e9, 1e9, "B"), (1e6, 1e6, "M"), (1e4, 1e3, "k")):
        if size >= limit:
            return f"{value / unit:.3g}{suffix}"
    if size >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:.3g}"


def short(name: object, limit: int = 40) -> str:
    """Shorten a long column name for a title; the full name stays on the axis."""
    text = str(name)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def note(fig: go.Figure) -> str:
    """The explanatory note stored with a figure, for display under the chart."""
    meta = fig.layout.meta
    return str(meta.get("note", "")) if isinstance(meta, dict) else ""


def metric_label(name: str) -> str:
    """Display name of an evaluation metric."""
    return _METRIC_LABEL.get(name, name.replace("_", " "))


def model_label(name: str) -> str:
    """Display name of a candidate model."""
    return "No-skill baseline" if name == "baseline" else name.replace("_", " ").capitalize()


def _signed(value: float) -> str:
    """Three decimals with a sign, and a plain 0.000 for anything that rounds to zero."""
    return "0.000" if round(value, 3) == 0 else f"{value:+.3f}"


def _hover_format(values: Sequence[float]) -> str:
    """A d3 number format that suits the size of the values."""
    finite = [abs(v) for v in values if v == v]
    top = max(finite, default=0.0)
    return ",.0f" if top >= 1000 else ",.1f" if top >= 10 else ",.3f"


def _rgba(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _axis(p: Palette, *, grid: bool = False, title: str = "", line: bool = False,
          **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "showgrid": grid, "gridcolor": p.grid, "gridwidth": 1, "zeroline": False,
        "showline": line, "linecolor": p.axis, "linewidth": 1, "ticks": "",
        "tickfont": {"size": 12, "color": p.ink_secondary}, "automargin": True,
        "exponentformat": "B",
        "title": {"text": safe_text(title), "font": {"size": 12, "color": p.ink_secondary},
                  "standoff": 10},
    }
    out.update(extra)
    return out


def _figure(p: Palette, title: str, subtitle: str = "", *, height: int = 360,
            legend: bool = False, text: str = "") -> go.Figure:
    """An empty figure with the shared chrome. ``text`` is kept as the figure's note."""
    heading = f"<b>{safe_text(title)}</b>"
    if subtitle:
        heading += (f"<br><span style='font-size:12px;color:{p.ink_secondary}'>"
                    f"{safe_text(subtitle)}</span>")
    top = (72 if subtitle else 48) + (28 if legend else 0)
    fig = go.Figure()
    fig.update_layout(
        title={"text": heading, "x": 0, "xref": "container", "xanchor": "left", "y": 1,
               "yref": "container", "yanchor": "top", "pad": {"t": 14, "l": 8},
               "font": {"size": 15, "color": p.ink}},
        font={"family": FONT, "size": 13, "color": p.ink_secondary},
        paper_bgcolor=p.surface, plot_bgcolor=p.surface, height=height,
        margin={"l": 8, "r": 16, "t": top, "b": 8},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0,
                "bgcolor": "rgba(0,0,0,0)", "traceorder": "normal",
                "font": {"size": 12, "color": p.ink_secondary},
                "itemclick": False, "itemdoubleclick": False},
        hoverlabel={"bgcolor": p.surface, "bordercolor": p.axis,
                    "font": {"family": FONT, "size": 13, "color": p.ink}},
        bargap=0.38, dragmode=False, meta={"note": text},
    )
    return fig


def _ranked_bars(p: Palette, labels: Sequence[str], values: Sequence[float], texts: Sequence[str],
                 colours: Sequence[str], title: str, subtitle: str, value_title: str,
                 hover: str, *, percent: bool = False, text: str = "") -> go.Figure:
    """Horizontal bars, largest at the top, each with its value just past the bar."""
    fig = _figure(p, title, subtitle, height=max(200, 34 * len(labels) + 110), text=text)
    names = [safe_text(v) for v in labels]
    fig.add_bar(
        x=list(values), y=names, orientation="h", text=list(texts), textposition="none",
        marker={"color": list(colours), "cornerradius": 4, "line": {"width": 0}},
        hovertemplate=hover, name="")
    for name, value, label in zip(names, values, texts, strict=True):
        fig.add_annotation(x=max(value, 0.0), y=name, text=label, showarrow=False,
                           xanchor="left", xshift=5, font={"size": 12, "color": p.ink})
    low, high = min(0.0, *values), max(0.0, *values)
    span = (high - low) or 1.0
    fig.update_layout(
        hovermode="closest",
        xaxis=_axis(p, grid=True, title=value_title, tickformat=".0%" if percent else None,
                    zeroline=True, zerolinecolor=p.axis, zerolinewidth=1,
                    range=[low - (0.02 * span if low < 0 else 0), high + 0.13 * span]),
        yaxis=_axis(p, autorange="reversed", type="category",
                    tickfont={"size": 13, "color": p.ink}))
    return fig


def distribution_figure(values: pd.Series, *, max_bins: int = 40,
                        mode: Mode = "light") -> go.Figure:
    """Histogram of a numeric column.

    Missing and non-numeric values are left out and counted in the subtitle. Whole numbers
    with a short range get one bar per value; otherwise numpy chooses the bins, capped at
    ``max_bins``. The bar heights are row counts and always add up to the plotted rows.
    """
    p = palette(mode)
    name = str(values.name) if values.name is not None else "value"
    v = coerce_numeric(values).dropna().to_numpy(dtype=float)
    if v.size == 0:
        raise DataError(f"column {name!r} has no numeric values to plot")
    lo, hi = float(v.min()), float(v.max())
    discrete = bool(np.all(np.mod(v, 1) == 0)) and hi - lo + 1 <= max_bins
    if discrete:
        edges = np.arange(lo - 0.5, hi + 1.0, 1.0)
    else:
        edges = np.histogram_bin_edges(v, bins="auto")
        if len(edges) - 1 > max_bins:
            edges = np.histogram_bin_edges(v, bins=max_bins)
    counts, edges = np.histogram(v, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    # One bar per whole number is drawn thin, with air between bars; bins of a continuous
    # variable touch, separated only by a hairline in the surface colour.
    widths = np.full(len(counts), min(0.7, 0.04 * len(counts))) if discrete else np.diff(edges)
    left_out = int(len(values) - v.size)
    median = float(np.median(v))
    subtitle = f"{v.size:,} values, median {compact(median)}."
    if left_out:
        subtitle += f" {left_out:,} missing or non-numeric left out."
    fig = _figure(p, f"Distribution of {short(name)}", subtitle)
    fig.add_bar(
        x=centres.tolist(), y=counts.tolist(), width=widths.tolist(),
        customdata=np.column_stack([edges[:-1], edges[1:]]).tolist(),
        marker={"color": p.accent, "cornerradius": 4 if discrete else 0,
                "line": {"color": p.surface, "width": 1}},
        hovertemplate=("%{x:,.4~g}: %{y:,} rows<extra></extra>" if discrete else
                       "%{customdata[0]:,.4~g} to %{customdata[1]:,.4~g}: %{y:,} rows"
                       "<extra></extra>"),
        name="")
    fig.add_vline(x=median, line={"color": p.ink_secondary, "width": 1})
    fig.add_annotation(x=median, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"median {compact(median)}", xanchor="left", xshift=4,
                       font={"size": 12, "color": p.ink_secondary})
    fig.update_layout(bargap=0, hovermode="closest",
                      xaxis=_axis(p, title=name, line=True, dtick=1 if discrete else None),
                      yaxis=_axis(p, grid=True, title="Rows", rangemode="tozero"))
    return fig


def category_figure(values: pd.Series, *, top: int = 12, mode: Mode = "light") -> go.Figure:
    """Rows per category, largest first. Categories beyond ``top`` are folded into "Other"."""
    p = palette(mode)
    name = str(values.name) if values.name is not None else "value"
    present = values[~missing_mask(values)].astype("string").str.strip()
    if present.empty:
        raise DataError(f"column {name!r} has no values to plot")
    counts = present.value_counts()
    labels, numbers = [str(k) for k in counts.index], [int(c) for c in counts.to_numpy()]
    colours = [p.accent] * len(labels)
    if len(labels) > top:
        tail = len(labels) - (top - 1)
        labels = [*labels[: top - 1], f"Other ({tail} categories)"]
        numbers = [*numbers[: top - 1], int(sum(numbers[top - 1:]))]
        colours = [*colours[: top - 1], p.muted]
    total = int(sum(numbers))
    left_out = int(len(values) - total)
    subtitle = f"{total:,} rows in {len(counts):,} categories."
    if left_out:
        subtitle += f" {left_out:,} missing left out."
    fig = _ranked_bars(p, labels, numbers, [f"{n:,}" for n in numbers], colours,
                       f"Rows by {short(name)}", subtitle, "Rows",
                       "%{y}: %{x:,} rows (%{customdata:.1%})<extra></extra>")
    fig.update_traces(customdata=[n / total for n in numbers])
    return fig


def segment_figure(segments: pd.DataFrame, *, dimension: str, metric: str,
                   stat: SegmentStat = "total", top: int = 12, mode: Mode = "light") -> go.Figure:
    """Ranked bars of one statistic from ``eda.segment_summary``.

    Totals, row counts and shares add up, so segments beyond ``top`` are folded into one
    "Other" bar. Means and medians do not, so the tail is cut and the subtitle says so.
    """
    p = palette(mode)
    if stat not in _STAT_LABEL:
        raise ValueError(f"stat must be one of {sorted(_STAT_LABEL)}")
    if segments.empty or stat not in segments.columns:
        raise DataError(f"no segment data for {dimension!r}")
    ranked = segments.sort_values(stat, ascending=False, kind="stable")
    labels = [str(k) for k in ranked.index]
    numbers = [float(x) for x in ranked[stat].to_numpy()]
    colours = [p.accent] * len(labels)
    subtitle = f"{len(labels):,} segments, largest first."
    if len(labels) > top:
        if stat in _ADDITIVE:
            tail = len(labels) - (top - 1)
            labels = [*labels[: top - 1], f"Other ({tail} segments)"]
            numbers = [*numbers[: top - 1], float(sum(numbers[top - 1:]))]
            colours = [*colours[: top - 1], p.muted]
            subtitle = f"{len(ranked):,} segments; the smallest {tail} are combined."
        else:
            hidden = len(labels) - top
            labels, numbers, colours = labels[:top], numbers[:top], colours[:top]
            subtitle = f"Top {top} of {len(ranked):,} segments; {hidden} are not shown."
    percent = stat == "share_of_total"
    texts = [f"{x:.1%}" if percent else compact(x) for x in numbers]
    what = "Rows" if stat == "rows" else f"{_STAT_LABEL[stat]} of {short(metric, 28)}"
    fmt = ".1%" if percent else _hover_format(numbers)
    return _ranked_bars(p, labels, numbers, texts, colours, f"{what} by {short(dimension, 28)}",
                        subtitle, what, f"%{{y}}: %{{x:{fmt}}}<extra></extra>", percent=percent,
                        text="Differences between segments can reflect who is in each segment "
                             "rather than an effect of the attribute.")


def timeseries_figure(history: pd.Series, *, value_name: str,
                      forecast: ForecastReport | None = None, mode: Mode = "light") -> go.Figure:
    """Line of an aggregated series, optionally with a forecast and its range band.

    The forecast is drawn dashed in a second colour, with the band behind it. The band is the
    empirical backtest range reported by ``forecast_series``, not a model-based interval.
    """
    p = palette(mode)
    if history.empty:
        raise DataError("the series has no periods to plot")
    y = [float(v) for v in history.to_numpy(dtype=float)]
    x = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in history.index]
    points = [row[1] for row in forecast.forecast] if forecast else []
    fmt = _hover_format([*y, *points])
    title = f"{short(value_name)} over time"
    subtitle, text = f"{len(y):,} periods.", ""
    if forecast is not None:
        title = f"{short(value_name)}: history and {forecast.horizon}-period forecast"
        subtitle = (f"{len(y):,} periods of history; forecast by "
                    f"{forecast.best_model.replace('_', ' ')}.")
        text = ("The shaded range is the forecast plus or minus the 80th percentile of the "
                "absolute errors in the backtests: an empirical range, not a model-based "
                "interval.")
    fig = _figure(p, title, subtitle, height=400, legend=forecast is not None, text=text)
    fig.add_scatter(x=x, y=y, mode="lines", name="History",
                    line={"color": p.accent, "width": 2},
                    hovertemplate=f"%{{y:{fmt}}}<extra>History</extra>")
    if forecast is not None:
        fx = [row[0] for row in forecast.forecast]
        low = [row[2] for row in forecast.forecast]
        high = [row[3] for row in forecast.forecast]
        fig.add_scatter(
            x=fx, y=points, mode="lines+markers" if len(points) <= 24 else "lines",
            name="Forecast", customdata=[[lo, hi] for lo, hi in zip(low, high, strict=True)],
            line={"color": p.second, "width": 2, "dash": "dash"},
            marker={"size": 8, "color": p.second, "line": {"color": p.surface, "width": 2}},
            hovertemplate=(f"%{{y:{fmt}}} (range %{{customdata[0]:{fmt}}} to "
                           f"%{{customdata[1]:{fmt}}})<extra>Forecast</extra>"))
        fig.add_scatter(x=fx, y=low, mode="lines", line={"width": 0}, hoverinfo="skip",
                        showlegend=False, name="Range low")
        fig.add_scatter(x=fx, y=high, mode="lines", line={"width": 0}, fill="tonexty",
                        fillcolor=_rgba(p.second, 0.16), hoverinfo="skip", name="Range")
        fig.add_scatter(x=[x[-1], fx[0]], y=[y[-1], points[0]], mode="lines", hoverinfo="skip",
                        showlegend=False, name="Join",
                        line={"color": p.second, "width": 2, "dash": "dash"})
    floor = min([*y, *(row[2] for row in forecast.forecast)] if forecast else y)
    fig.update_layout(
        hovermode="x unified",
        xaxis=_axis(p, line=True, type="date", showspikes=True, spikemode="across",
                    spikethickness=1, spikedash="solid", spikecolor=p.axis),
        yaxis=_axis(p, grid=True, title=value_name,
                    rangemode="tozero" if floor >= 0 else "normal"))
    return fig


def _colorbar(p: Palette, title: str, x: float, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": {"text": title, "side": "right", "font": {"size": 12, "color": p.ink_secondary}},
        "thickness": 12, "len": 0.8, "outlinewidth": 0, "x": x, "xanchor": "left", "xpad": 12,
        "tickfont": {"size": 12, "color": p.ink_secondary}}
    out.update(extra)
    return out


def correlation_figure(pairs: pd.DataFrame, *, columns: Sequence[str] | None = None,
                       max_columns: int = 20, mode: Mode = "light") -> go.Figure:
    """Lower-triangle heatmap of the pairs returned by ``eda.correlation_pairs``.

    Blue is positive, red is negative and the neutral midpoint is zero. Each pair appears
    once. With more than ``max_columns`` columns, the ones with the strongest correlations
    are kept and the subtitle says how many were dropped.
    """
    p = palette(mode)
    if pairs.empty:
        raise DataError("correlations need at least two numeric columns with enough rows")
    found = list(dict.fromkeys([*pairs["a"], *pairs["b"]]))
    order = [c for c in columns if c in found] if columns is not None else found
    order += [c for c in found if c not in order]
    dropped = 0
    if len(order) > max_columns:
        strength = pd.concat([pairs.set_index("a")["r"], pairs.set_index("b")["r"]]).abs()
        keep = set(strength.groupby(level=0).max().nlargest(max_columns).index)
        dropped = len(order) - len(keep)
        order = [c for c in order if c in keep]
    position = {c: i for i, c in enumerate(order)}
    size = len(order)
    z = np.full((size, size), np.nan)
    n = np.zeros((size, size), dtype=int)
    for a, b, r, count in zip(pairs["a"], pairs["b"], pairs["r"], pairs["n"], strict=True):
        if a in position and b in position:
            i, j = sorted((position[a], position[b]), reverse=True)
            z[i, j], n[i, j] = float(r), int(count)
    rows, cols = order[1:], order[:-1]
    zz, nn = z[1:, :-1], n[1:, :-1]
    subtitle = f"{int(np.isfinite(zz).sum()):,} pairs of {size} numeric columns."
    if dropped:
        subtitle += f" {dropped} with weaker correlations not shown."
    fig = _figure(p, "Correlation between numeric columns", subtitle,
                  height=max(240, 44 * len(rows) + 150),
                  text="Blue is positive, red is negative. A correlation is an association in "
                       "this sample, not evidence of cause; many pairs are compared and "
                       "nothing is corrected for that.")
    scale = [[i / (len(p.diverging) - 1), c] for i, c in enumerate(p.diverging)]
    width = min(0.9, 0.14 * len(cols))    # few columns: keep the cells compact
    fig.add_heatmap(
        z=[[None if v != v else float(v) for v in row] for row in zz],
        x=[safe_text(c) for c in cols], y=[safe_text(c) for c in rows],
        customdata=nn.tolist(), zmin=-1, zmax=1, zmid=0, colorscale=scale, xgap=2, ygap=2,
        hoverongaps=False, texttemplate="%{z:+.2f}" if size <= 10 else None,
        hovertemplate="%{y} and %{x}: r = %{z:+.3f} (n = %{customdata:,})<extra></extra>",
        colorbar=_colorbar(p, "Correlation (r)", width, tickvals=[-1, -0.5, 0, 0.5, 1]),
        name="")
    fig.update_layout(
        hovermode="closest",
        xaxis=_axis(p, type="category", domain=[0, width],
                    tickangle=-35 if max(len(str(c)) for c in cols) > 9 else 0,
                    tickfont={"size": 12, "color": p.ink}),
        yaxis=_axis(p, type="category", autorange="reversed",
                    tickfont={"size": 12, "color": p.ink}))
    return fig


def retention_figure(rates: pd.DataFrame, sizes: pd.Series, *, max_cohorts: int = 24,
                     max_periods: int = 24, mode: Mode = "light") -> go.Figure:
    """Heatmap of the matrix returned by ``eda.cohort_retention``.

    Period 0 is left out because it is 100% by definition. Periods that have not happened
    yet stay empty, as in the matrix. At most the latest ``max_cohorts`` cohorts and the
    first ``max_periods`` periods are drawn.
    """
    p = palette(mode)
    later = [c for c in rates.columns if c != 0]
    if rates.empty or not later:
        raise DataError("retention needs activity in at least two periods")
    shown = rates.loc[rates.index[-max_cohorts:], later[:max_periods]]
    labels = [f"{cohort} (n = {int(sizes.loc[cohort]):,})" for cohort in shown.index]
    values = shown.to_numpy(dtype=float)
    subtitle = f"Share of each cohort active again in later periods; {len(shown):,} of " \
               f"{len(rates):,} cohorts."
    fig = _figure(p, "Cohort retention", subtitle, height=max(240, 30 * len(shown) + 150),
                  text="A cohort is everyone whose first activity falls in the same period. "
                       "Period 0 is 100% by definition and is left out. Periods that have not "
                       "happened yet are empty, not zero. Small cohorts (see n) are noisy.")
    scale = [[i / (len(p.sequential) - 1), c] for i, c in enumerate(p.sequential)]
    small = shown.shape[0] <= 16 and shown.shape[1] <= 16
    width = min(0.9, 0.08 * shown.shape[1] + 0.1)
    fig.add_heatmap(
        z=[[None if v != v else float(v) for v in row] for row in values],
        x=[int(c) for c in shown.columns], y=[safe_text(v) for v in labels],
        zmin=0, zmax=max(float(np.nanmax(values)) if np.isfinite(values).any() else 0.0, 0.01),
        colorscale=scale, xgap=2, ygap=2, hoverongaps=False,
        texttemplate="%{z:.0%}" if small else None,
        hovertemplate="Cohort %{y}, period %{x}: %{z:.1%} active<extra></extra>",
        colorbar=_colorbar(p, "Share active", width, tickformat=".0%"), name="")
    fig.update_layout(
        hovermode="closest",
        xaxis=_axis(p, title="Periods since first activity", type="category",
                    domain=[0, width]),
        yaxis=_axis(p, type="category", autorange="reversed",
                    tickfont={"size": 12, "color": p.ink}))
    return fig


def model_comparison_figure(report: ModelReport, *, mode: Mode = "light") -> go.Figure:
    """Cross-validation scores of every candidate next to the no-skill baseline.

    Small dots are the individual folds, the large marker is their mean and the vertical
    line is the baseline's mean. Dots are used instead of bars because scores such as AUC
    do not start at zero.
    """
    p = palette(mode)
    metric = metric_label(report.primary_metric)
    sign = 1 if report.higher_is_better else -1
    ranked = sorted((c for c in report.candidates if c.model != "baseline"),
                    key=lambda c: -sign * c.cv_mean)
    ranked += [c for c in report.candidates if c.model == "baseline"]
    roles = {"Selected model": (p.accent, [c for c in ranked if c.model == report.best_model]),
             "Other candidates": (p.accent_soft, [c for c in ranked if c.model not in (
                 report.best_model, "baseline")]),
             "No-skill baseline": (p.muted, [c for c in ranked if c.model == "baseline"])}
    better = "higher" if report.higher_is_better else "lower"
    text = (f"Each small dot is one of {report.n_folds} cross-validation folds; the large "
            "marker is their mean and the vertical line is the no-skill baseline.")
    if report.n_selection < report.n_train:
        text += (f" Candidates were compared on a seeded sample of {report.n_selection:,} of "
                 f"the {report.n_train:,} training rows.")
    fig = _figure(p, f"Cross-validated {metric} by candidate",
                  f"Small dots are folds, large markers their mean; {better} is better.",
                  height=max(240, 44 * len(ranked) + 150), legend=True, text=text)
    for role, (colour, members) in roles.items():
        if not members:
            continue
        fig.add_scatter(
            x=[s for c in members for s in c.fold_scores],
            y=[model_label(c.model) for c in members for _ in c.fold_scores],
            mode="markers", name=f"{role} folds", showlegend=False, hoverinfo="skip",
            marker={"size": 7, "color": _rgba(colour, 0.45)})
        fig.add_scatter(
            x=[c.cv_mean for c in members], y=[model_label(c.model) for c in members],
            mode="markers", name=role, customdata=[c.cv_std for c in members],
            marker={"size": 12, "color": colour, "line": {"color": p.surface, "width": 2}},
            hovertemplate=f"%{{y}}: mean {metric} %{{x:.3f}} (SD %{{customdata:.3f}} across "
                          "folds)<extra></extra>")
    base = next((c for c in ranked if c.model == "baseline"), None)
    if base is not None:
        fig.add_vline(x=base.cv_mean, line={"color": p.muted, "width": 1})
    fmt = ".3f" if report.task == "classification" else _hover_format(
        [c.cv_mean for c in ranked])[1:]
    for c in ranked:
        fig.add_annotation(x=1, xref="paper", xanchor="right", y=model_label(c.model),
                           text=format(c.cv_mean, fmt), showarrow=False,
                           font={"size": 13, "color": p.ink})
    fig.update_layout(
        hovermode="closest",
        xaxis=_axis(p, grid=True, title=f"{metric} ({better} is better)", domain=[0, 0.88]),
        yaxis=_axis(p, type="category", autorange="reversed",
                    categoryarray=[model_label(c.model) for c in ranked],
                    categoryorder="array", tickfont={"size": 13, "color": p.ink}))
    return fig


def holdout_figure(report: ModelReport, *, mode: Mode = "light") -> go.Figure:
    """Hold-out metrics of the selected model next to the no-skill baseline.

    Classification metrics share the 0 to 1 scale. For regression only the two error
    metrics (same unit as the target) are drawn; R² stays in the table.
    """
    p = palette(mode)
    names: list[str]
    if report.task == "regression":
        names = [m for m in ("rmse", "mae") if m in report.test_metrics]
        title = "Hold-out error: selected model and baseline"
        value_title = "Error (lower is better)"
    else:
        names = [m for m in report.test_metrics if m == report.primary_metric]
        names += [m for m in report.test_metrics if m != report.primary_metric]
        title, value_title = "Hold-out scores: selected model and baseline", "Score (0 to 1)"
    labels = [metric_label(m) for m in names]
    model = [report.test_metrics[m] for m in names]
    base = [report.baseline_test_metrics[m] for m in names]
    regression = report.task == "regression"
    fmt = _hover_format([*model, *base]) if regression else ".3f"
    top = max([*model, *base, 0.0])
    fig = _figure(p, title, f"Measured once on {report.n_test:,} held-out rows.",
                  height=max(240, 58 * len(names) + 150), legend=True,
                  text="The hold-out rows were set aside before any model was fitted and were "
                       "used once, for these numbers.")
    for name, numbers, colour in ((model_label(report.best_model), model, p.accent),
                                  ("No-skill baseline", base, p.muted)):
        fig.add_bar(
            x=numbers, y=labels, orientation="h", name=name,
            text=[compact(v) if regression else f"{v:.3f}" for v in numbers],
            textposition="outside", cliponaxis=False, textfont={"size": 12, "color": p.ink},
            marker={"color": colour, "cornerradius": 4, "line": {"color": p.surface, "width": 1}},
            hovertemplate=f"%{{y}}: %{{x:{fmt}}}<extra>{safe_text(name)}</extra>")
    fig.update_layout(
        barmode="group", bargap=0.34, bargroupgap=0.0, hovermode="closest",
        xaxis=_axis(p, grid=True, title=value_title, zeroline=True, zerolinecolor=p.axis,
                    zerolinewidth=1, range=[0, 1.12 * top] if regression else [0, 1.1]),
        yaxis=_axis(p, type="category", autorange="reversed",
                    tickfont={"size": 13, "color": p.ink}))
    return fig


def importance_figure(report: ModelReport, *, top: int = 15, mode: Mode = "light") -> go.Figure:
    """Permutation importance of the selected model, largest first.

    Each bar is the mean drop in the primary metric on the hold-out when that column is
    shuffled. It shows what the model relies on, not what causes the outcome.
    """
    p = palette(mode)
    if not report.importance:
        raise DataError("the model report has no feature importance")
    rows = report.importance[:top]
    labels = [name for name, _ in rows]
    numbers = [float(v) for _, v in rows]
    metric = metric_label(report.primary_metric)
    subtitle = "What the model relies on, not what causes the outcome."
    if len(report.importance) > top:
        subtitle = f"Top {top} of {len(report.importance)} columns. " + subtitle
    text = (f"Each bar is the mean change in {metric} on {report.n_importance:,} hold-out rows "
            "when that column is shuffled, over 5 shuffles. Values near zero or below mean the "
            "model does not rely on the column.")
    return _ranked_bars(
        p, labels, numbers, [_signed(v) for v in numbers], [p.accent] * len(rows),
        f"Permutation importance for {short(report.target, 28)}", subtitle,
        f"Change in {metric} when the column is shuffled",
        "%{y}: %{x:+.4f}<extra></extra>", text=text)


def quality_figure(quality: QualityScore, *, mode: Mode = "light") -> go.Figure:
    """The five components of the data-quality score, each as a share of its maximum."""
    p = palette(mode)
    if quality.score is None:
        raise DataError("the table has no rows, so there is no quality score")
    parts = [(name, float(getattr(quality, name) or 0.0), quality.weights[name])
             for name in _QUALITY_PARTS]
    labels = [f"{name.capitalize()} (weight {weight:.0%})" for name, _, weight in parts]
    values = [value for _, value, _ in parts]
    fig = _figure(p, "Quality score components",
                  f"Score {quality.score:.1f} of 100 is the weighted sum of these shares.",
                  height=300,
                  text="Completeness: cells that are not missing. Validity: values that parse "
                       "as their column's type. Uniqueness: rows that are not exact "
                       "duplicates. Consistency: columns without constant values, padded "
                       "text or categories that differ only by case. Outliers: numeric "
                       "values within 3 x IQR of the quartiles. A screening aid, not a "
                       "statistical test.")
    fig.add_bar(x=[1.0] * len(parts), y=labels, orientation="h", hoverinfo="skip", name="Maximum",
                marker={"color": _rgba(p.accent, 0.16), "cornerradius": 4, "line": {"width": 0}})
    fig.add_bar(
        x=values, y=labels, orientation="h", name="Component",
        customdata=[[weight, 100 * weight * value] for _, value, weight in parts],
        marker={"color": p.accent, "cornerradius": 4, "line": {"width": 0}},
        hovertemplate="%{y}: %{x:.2%}, adding %{customdata[1]:.1f} points to the score"
                      "<extra></extra>")
    for label, value in zip(labels, values, strict=True):
        fig.add_annotation(x=1, xref="paper", xanchor="right", y=label, showarrow=False,
                           text=f"{value:.1%}", font={"size": 13, "color": p.ink})
    fig.update_layout(
        barmode="overlay", bargap=0.5, hovermode="closest",
        xaxis=_axis(p, range=[0, 1], showticklabels=False, domain=[0, 0.88]),
        yaxis=_axis(p, type="category", autorange="reversed",
                    tickfont={"size": 13, "color": p.ink}))
    return fig
