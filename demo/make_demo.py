#!/usr/bin/env python3
"""Regenerate the demo: two sample reports and standalone charts, all from synthetic data.

    python demo/make_demo.py              # writes into demo/
    python demo/make_demo.py --out DIR    # writes somewhere else (the tests do this)

What it writes:

* ``churn/`` and ``orders/``: report.md, analysis.json, queries.sql and reproduce.sh, exactly
  as ``ads report`` produces them. The command in reproduce.sh runs from the repository root
  after ``python examples/generate_sample_data.py``.
* ``charts/``: one HTML page per chart plus ``index.html`` with all of them. Each page shows
  the chart, its note and the numbers behind it as a table. ``plotly.min.js`` is Plotly's
  own library, written next to the pages so they open without a network connection.

Everything comes from the seeded generators in ``auto_data_scientist.samples``; nothing here
is real data, and nothing is fetched.
"""

from __future__ import annotations

import argparse
import html
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

try:
    import auto_data_scientist  # noqa: F401
except ModuleNotFoundError:  # running from a repository checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_data_scientist import analyze, charts, eda
from auto_data_scientist.pipeline import Analysis, artifact_texts
from auto_data_scientist.samples import SAMPLES, write_sample

HERE = Path(__file__).resolve().parent
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
STYLE = """
:root { color-scheme: light; }
body { margin: 0; background: #fcfcfb; color: #0b0b0b;
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 40px 0 0; }
p.lead, figcaption, .source { color: #52514e; }
p.lead { margin: 0 0 8px; max-width: 70ch; }
figure { margin: 8px 0 0; }
figcaption { max-width: 78ch; font-size: 0.9rem; }
details { margin-top: 8px; }
summary { cursor: pointer; color: #1c5cab; width: fit-content; }
summary:focus-visible, a:focus-visible { outline: 2px solid #1c5cab; outline-offset: 2px; }
table { border-collapse: collapse; margin-top: 8px; font-size: 0.85rem;
        font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 4px 12px 4px 0; text-align: left; }
th { color: #52514e; font-weight: 600; }
a { color: #1c5cab; }
nav { margin: 12px 0 0; font-size: 0.9rem; }
"""


def table_html(frame: pd.DataFrame, rows: int = 40) -> str:
    """The numbers behind a chart as an HTML table; every cell is escaped."""
    shown = frame.head(rows)

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if value != value else f"{value:,.4g}"
        return html.escape(str(value))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>"
                   for row in shown.itertuples(index=False, name=None))
    more = (f"<p class='source'>{len(frame) - rows:,} more rows not shown.</p>"
            if len(frame) > rows else "")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}"


def block(name: str, fig: go.Figure, frame: pd.DataFrame) -> str:
    """One chart with its note and its data table."""
    plot = fig.to_html(
        full_html=False, include_plotlyjs=False, div_id=f"chart-{name}", config=PLOT_CONFIG,
        default_height=f"{fig.layout.height}px")
    note = html.escape(charts.note(fig))
    caption = f"<figcaption>{note}</figcaption>" if note else ""
    return (f"<figure>{plot}{caption}</figure>"
            f"<details><summary>Data table</summary>{table_html(frame)}</details>")


def page(title: str, lead: str, body: str) -> str:
    return (f"<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style>"
            f"<script src='plotly.min.js'></script></head><body><main>"
            f"<h1>{html.escape(title)}</h1><p class='lead'>{html.escape(lead)}</p>"
            f"{body}</main></body></html>\n")


def figures(orders: Analysis, churn: Analysis, orders_frame: pd.DataFrame,
            ) -> dict[str, tuple[str, go.Figure, pd.DataFrame]]:
    """Every demo chart: a heading, the figure and the table of numbers it shows."""
    assert orders.trend is not None and orders.forecast is not None and churn.model is not None
    model, forecast = churn.model, orders.forecast
    segments = orders.segments["region"]
    rates, sizes = eda.cohort_retention(orders_frame, "customer_id", "order_date", "Q")
    retention = rates.copy()
    retention.insert(0, "cohort size", sizes)
    retention.index = retention.index.astype(str)
    retention.columns = [str(c) for c in retention.columns]
    quality = orders.profile.quality
    return {
        "quality": ("Data quality score", charts.quality_figure(quality), pd.DataFrame(
            {"component": list(quality.weights), "weight": list(quality.weights.values()),
             "share": [getattr(quality, k) for k in quality.weights]})),
        "distribution": ("Distribution of a numeric column",
                         charts.distribution_figure(orders_frame["revenue"]),
                         orders_frame["revenue"].describe().rename("revenue").reset_index()
                         .rename(columns={"index": "statistic"})),
        "segments": ("Segments", charts.segment_figure(
            segments, dimension="region", metric="revenue"), segments.reset_index()),
        "correlation": ("Correlations", charts.correlation_figure(
            orders.correlations, columns=orders.profile.by_kind("numeric")),
            orders.correlations),
        "forecast": ("Time series with forecast and range", charts.timeseries_figure(
            orders.trend.series, value_name="revenue", forecast=forecast), pd.DataFrame(
            forecast.forecast, columns=["period", "forecast", "low", "high"])),
        "retention": ("Cohort retention", charts.retention_figure(rates, sizes),
                      retention.reset_index(names="cohort")),
        "model_comparison": ("Candidates against the no-skill baseline",
                             charts.model_comparison_figure(model), pd.DataFrame(
                                 [{"model": c.model, "cv mean": c.cv_mean, "cv sd": c.cv_std,
                                   "folds": ", ".join(f"{s:.3f}" for s in c.fold_scores)}
                                  for c in model.candidates])),
        "holdout": ("Hold-out scores", charts.holdout_figure(model), pd.DataFrame(
            {"metric": list(model.test_metrics), "selected model": list(
                model.test_metrics.values()), "no-skill baseline": [
                model.baseline_test_metrics[k] for k in model.test_metrics]})),
        "importance": ("Permutation importance", charts.importance_figure(model), pd.DataFrame(
            model.importance, columns=["column", "importance"])),
    }


def build(out: Path) -> list[Path]:
    """Write the demo into ``out`` and return the files written."""
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        paths = {key: write_sample(key, tmp) for key in SAMPLES}
        orders = analyze(
            paths["orders"], question="top 2 regions by total revenue", date_col="order_date",
            value_col="revenue",
            business_problem="Where does revenue come from, and what range should we plan "
                             "for over the next six months?")
        churn = analyze(
            paths["churn"], target="churned", sql=SAMPLES["churn"].sql,
            business_problem="Which customers are most at risk of churning?")
        orders_frame = SAMPLES["orders"].make()
    for key, analysis in (("orders", orders), ("churn", churn)):
        folder = out / key
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in artifact_texts(analysis, f"demo/{key}", f"examples/{key}.csv").items():
            (folder / name).write_text(text, encoding="utf-8")
            written.append(folder / name)

    gallery = out / "charts"
    gallery.mkdir(parents=True, exist_ok=True)
    (gallery / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    written.append(gallery / "plotly.min.js")
    lead = ("Drawn by auto_data_scientist.charts from the built-in synthetic samples. Hover a "
            "mark for its value, or open the data table under a chart.")
    sections, links = [], []
    for name, (heading, fig, frame) in figures(orders, churn, orders_frame).items():
        body = block(name, fig, frame)
        nav = "<nav><a href='index.html'>All charts</a></nav>"
        (gallery / f"{name}.html").write_text(page(heading, lead, body + nav), encoding="utf-8")
        written.append(gallery / f"{name}.html")
        sections.append(f"<h2 id='{name}'>{html.escape(heading)}</h2>{body}")
        links.append(f"<a href='{name}.html'>{html.escape(heading)}</a>")
    index = page("Auto Data Scientist: chart gallery", lead,
                 f"<nav>One page per chart: {', '.join(links)}.</nav>" + "".join(sections))
    (gallery / "index.html").write_text(index, encoding="utf-8")
    written.append(gallery / "index.html")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE, help="folder to write into")
    for path in build(parser.parse_args().out):
        print(f"wrote {path}")
