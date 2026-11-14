"""Dashboard for auto_data_scientist: load a table, look at it, question it, model it.

Run it from a checkout with::

    pip install -e ".[ui]"
    streamlit run app/streamlit_app.py

This file is a thin user interface. Every number, chart, refusal and report comes from the
``auto_data_scientist`` package; the code here collects choices, calls the package and lays
out what it returns. Results are cached per file, table and setting, so moving between
sections does not recompute them.

Security notes (see SECURITY.md):

* There is no sign-in. ``.streamlit/config.toml`` next to this file binds the server to
  localhost, and the sidebar warns if it is started on any other address.
* An upload is stored by ``ingest.save_upload`` under a generated name in a private
  temporary folder that is removed when the server stops.
* SQL typed here runs only through ``query.run_query``: allowlist, locked connection,
  rollback, time and row budget. Nothing on this page executes generated Python.
* Text that comes from the data is shown in tables and code blocks, or escaped before it
  reaches Markdown, so a file cannot plant links or images in the page.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import auto_data_scientist  # noqa: F401
except ModuleNotFoundError:  # running from a repository checkout without installation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_data_scientist import __version__, charts, eda, samples
from auto_data_scientist.config import is_loopback
from auto_data_scientist.errors import AdsError, TranslationError, UnsafeQueryError, explain
from auto_data_scientist.forecasting import ForecastReport, forecast_series
from auto_data_scientist.ingest import (
    SUPPORTED_SUFFIXES,
    Dataset,
    display_name,
    load_dataset,
    save_upload,
)
from auto_data_scientist.modeling import ModelReport, train_supervised
from auto_data_scientist.nl2sql import RuleBasedTranslator
from auto_data_scientist.pipeline import (
    ARTIFACTS,
    Analysis,
    analyze,
    artifact_texts,
    bundle_artifacts,
    categorical_values,
)
from auto_data_scientist.profiling import TableProfile, profile_frame
from auto_data_scientist.query import QueryResult, preview_sql, run_query
from auto_data_scientist.report import code_span, escape_md

SECTIONS = ("Data", "Explore", "Ask", "Predict", "Forecast", "Report")
NONE = "(none)"
FREQUENCIES = {"D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}
STATS = {"total": "Total", "mean": "Mean", "median": "Median", "rows": "Rows",
         "share_of_total": "Share of total"}
PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}
EXPECTED = (AdsError, ValueError)
RESULT_KEYS = ("ask", "predict", "forecast", "report", "last_sql", "done_predict",
               "done_forecast")
CHOICE_PREFIXES = ("dist_", "corr_", "seg_", "ts_", "cohort_", "ask_", "predict_", "forecast_",
                   "report_")
Ref = tuple[str, str]    # (stored path, name the person knows the file by)


@dataclass(frozen=True)
class Source:
    """A file to analyse: where it is stored, what to call it, and how to name it in the
    command that reproduces a report."""

    path: str
    name: str
    command_path: str
    sample: samples.Sample | None = None

    @property
    def ref(self) -> Ref:
        return (self.path, self.name)


@dataclass(frozen=True)
class Ctx:
    """Everything a section needs about the table on screen."""

    source: Source
    table: str
    ds: Dataset
    df: pd.DataFrame
    profile: TableProfile
    kpis: dict[str, list[str]]
    mode: charts.Mode

    @property
    def ref(self) -> Ref:
        return self.source.ref


# --------------------------------------------------------------------------- cached work

@st.cache_resource(show_spinner=False)
def workspace() -> tempfile.TemporaryDirectory[str]:
    """One private temporary folder per server process, removed when the process exits."""
    return tempfile.TemporaryDirectory(prefix="ads-dashboard-")


@st.cache_data(show_spinner=False)
def sample_path(key: str) -> str:
    return str(samples.write_sample(key, Path(workspace().name) / "samples"))


@st.cache_data(show_spinner=False, max_entries=8)
def stored_upload(file_id: str, name: str, _data: bytes) -> str:
    """Save an upload once per uploaded file; ``file_id`` identifies the bytes."""
    return str(save_upload(_data, name, Path(workspace().name) / "uploads"))


# A file is identified by ``Ref``: where it is stored and what the person calls it. The name
# matters because a CSV or Parquet table is named after it, not after the stored file.

@st.cache_resource(show_spinner="Loading the data…", max_entries=4)
def open_dataset(ref: Ref) -> Dataset:
    return load_dataset(ref[0], name=ref[1])


@st.cache_resource(show_spinner=False, max_entries=8)
def table_frame(ref: Ref, table: str) -> pd.DataFrame:
    """The table as a DataFrame. Shared and never modified: the package copies what it changes."""
    return open_dataset(ref).frame(table)


@st.cache_data(show_spinner="Profiling the columns…", max_entries=8)
def table_profile(ref: Ref, table: str) -> TableProfile:
    rejected = open_dataset(ref).tables[table].rejected_rows
    return profile_frame(table_frame(ref, table), table, rejected)


@st.cache_data(show_spinner=False, max_entries=8)
def filter_values(ref: Ref, table: str) -> dict[str, list[str]]:
    return categorical_values(table_frame(ref, table), table_profile(ref, table))


@st.cache_data(show_spinner=False)
def correlation_table(ref: Ref, table: str, method: str) -> pd.DataFrame:
    return eda.correlation_pairs(table_frame(ref, table), table_profile(ref, table), method)


@st.cache_data(show_spinner=False)
def segment_table(ref: Ref, table: str, by: str, metric: str) -> pd.DataFrame:
    return eda.segment_summary(table_frame(ref, table), by, metric)


@st.cache_data(show_spinner=False)
def trend_summary(ref: Ref, table: str, date_col: str, value_col: str,
                  freq: str) -> eda.TimeSeriesSummary:
    return eda.time_series_summary(table_frame(ref, table), date_col, value_col, freq)


@st.cache_data(show_spinner=False)
def retention_matrix(ref: Ref, table: str, user_col: str, date_col: str,
                     freq: str) -> tuple[pd.DataFrame, pd.Series]:
    return eda.cohort_retention(table_frame(ref, table), user_col, date_col, freq)


@st.cache_data(show_spinner="Running the query…", max_entries=64)
def query_result(ref: Ref, sql: str) -> QueryResult:
    return run_query(open_dataset(ref), sql)


@st.cache_data(show_spinner="Comparing models with cross-validation…", max_entries=16)
def model_report(ref: Ref, table: str, target: str, time_col: str | None,
                 keep: tuple[str, ...], seed: int) -> ModelReport:
    return train_supervised(table_frame(ref, table), target, time_col=time_col,
                            keep=list(keep), seed=seed)


@st.cache_data(show_spinner="Backtesting the forecasts…", max_entries=16)
def forecast_report(ref: Ref, table: str, date_col: str, value_col: str, horizon: int,
                    freq: str) -> ForecastReport:
    return forecast_series(table_frame(ref, table), date_col, value_col, horizon, freq)


@st.cache_data(show_spinner="Building the report…", max_entries=8)
def report_analysis(ref: Ref, table: str, problem: str | None, sql: str | None,
                    target: str | None, time_col: str | None, keep: tuple[str, ...],
                    date_col: str | None, value_col: str | None, horizon: int,
                    freq: str | None, seed: int) -> Analysis:
    return analyze(ref[0], table=table, source_name=ref[1], business_problem=problem, sql=sql,
                   target=target, time_col=time_col, keep=list(keep), date_col=date_col,
                   value_col=value_col, horizon=horizon, freq=freq, seed=seed)


# --------------------------------------------------------------------------- small helpers

def theme_mode() -> charts.Mode:
    """Light unless the browser reports the dark theme."""
    return "dark" if st.context.theme.get("type") == "dark" else "light"


def show_error(exc: Exception, extra: str = "") -> None:
    """A failure in words. A refusal by a guard is a warning: the guard did its job."""
    told = explain(exc)
    body = f"**{told.title}.** {escape_md(told.reason)}"
    if isinstance(exc, UnsafeQueryError | TranslationError):
        st.warning(body, icon=":material/block:")
    else:
        st.error(body, icon=":material/error:")
    st.caption(" ".join(part for part in (told.hint, extra) if part))


def show(fig: object, key: str) -> None:
    """A chart with its explanatory note underneath."""
    st.plotly_chart(fig, key=key, theme=None, config=PLOT_CONFIG)
    text = charts.note(fig)
    if text:
        st.caption(text)


def suggested_period(ctx: Ctx, date_col: str) -> str:
    """The period that suits the span of a date column, or months if it has no valid dates."""
    try:
        return eda.suggest_frequency(ctx.df[date_col])
    except EXPECTED:
        return "M"


def pick(options: list[str], preferred: str | None) -> int:
    """Index of ``preferred`` in ``options``, or the first option."""
    return options.index(preferred) if preferred in options else 0


def bullets(lines: list[str]) -> None:
    st.markdown("\n".join(f"- {escape_md(line)}" for line in lines))


# --------------------------------------------------------------------------- sidebar

def choose_source() -> tuple[Source | None, Exception | None]:
    """The data picker. Returns the chosen source, or the reason an upload was refused."""
    st.title("Auto Data Scientist")
    st.caption("From a raw table to findings you can defend: a baseline beside every model, "
               "limits beside every number, a refusal instead of a guess.")
    origin = st.radio("Data", ["Built-in sample", "Upload a file"], key="origin")
    if origin == "Built-in sample":
        key = st.selectbox("Sample", list(samples.SAMPLES), key="sample",
                           format_func=lambda k: samples.SAMPLES[k].title)
        sample = samples.SAMPLES[key]
        st.caption(f"{sample.description} Synthetic: no real people or companies.")
        return Source(sample_path(key), f"{key}.csv", f"examples/{key}.csv", sample), None
    upload = st.file_uploader(
        "Data file", type=sorted(s.lstrip(".") for s in SUPPORTED_SUFFIXES), key="upload")
    st.caption("CSV, TSV, XLSX, Parquet, SQLite or DuckDB, up to 200 MB. The file is kept "
               "under a generated name in a private temporary folder and removed when the "
               "app stops.")
    if upload is None:
        return None, None
    try:
        path = stored_upload(upload.file_id, upload.name, upload.getvalue())
    except EXPECTED as exc:
        return None, exc
    name = display_name(upload.name)
    return Source(path, name, name), None


def choose_table(ds: Dataset) -> str:
    names = sorted(ds.tables)
    if len(names) == 1:
        return names[0]
    return str(st.selectbox(
        "Table", names, index=names.index(ds.default_table()), key="table",
        format_func=lambda t: f"{t} ({ds.tables[t].n_rows:,} rows)"))


def footer() -> None:
    st.divider()
    if is_loopback(st.get_option("server.address")):
        st.caption("Runs on this computer only: the server is bound to localhost. There is "
                   "no sign-in, and your data is not sent anywhere.")
    else:
        st.warning("This dashboard has no sign-in and is not bound to localhost, so other "
                   "computers may be able to reach it. Restart it with "
                   "`--server.address localhost`.", icon=":material/warning:")
    st.caption(f"auto-data-scientist {__version__}")


# --------------------------------------------------------------------------- Data

def column_table(profile: TableProfile) -> pd.DataFrame:
    """One row per column of the profile, for display."""
    blank = float("nan")
    return pd.DataFrame([{
        "Column": c.name, "Kind": c.kind, "Missing": c.pct_missing, "Distinct": c.n_unique,
        "Invalid": c.n_invalid, "Extreme": c.n_extreme, "Min": c.stats.get("min", blank),
        "Median": c.stats.get("median", blank), "Mean": c.stats.get("mean", blank),
        "Max": c.stats.get("max", blank),
        "Most common": ", ".join(f"{v} ({n:,})" for v, n in c.top_values[:3]),
    } for c in profile.columns])


def section_data(ctx: Ctx) -> None:
    p, q = ctx.profile, ctx.profile.quality
    tiles = st.columns(5)
    tiles[0].metric("Rows", f"{p.n_rows:,}", border=True)
    tiles[1].metric("Columns", f"{p.n_cols:,}", border=True)
    tiles[2].metric("Quality score", "n/a" if q.score is None else f"{q.score:.1f} / 100",
                    border=True, help="A weighted sum of the five components below. A "
                                      "screening aid, not a statistical test.")
    tiles[3].metric("Duplicate rows", f"{p.n_duplicate_rows:,}", border=True)
    tiles[4].metric("Rows rejected on load", f"{p.rejected_rows:,}", border=True,
                    help="Rows with the wrong number of fields are left out and counted, "
                         "never repaired silently.")
    if q.score is None:
        st.info("This table has no rows, so there is nothing to profile. Choose another "
                "table or file.", icon=":material/info:")
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        show(charts.quality_figure(q, mode=ctx.mode), "quality_chart")
    with right:
        st.subheader("Issues")
        if p.issues:
            st.dataframe(pd.DataFrame({"Issue": p.issues}), hide_index=True, key="issues_table")
            st.caption("Fix these at the source before relying on numbers further down.")
        else:
            st.success("No issues detected.", icon=":material/check_circle:")
        roles = {"Likely metrics": ctx.kpis["metrics"], "Dimensions": ctx.kpis["dimensions"],
                 "Dates": ctx.kpis["dates"], "Identifiers": ctx.kpis["ids"]}
        st.dataframe(pd.DataFrame({"Role": list(roles),
                                   "Columns": [", ".join(v) or "none" for v in roles.values()]}),
                     hide_index=True, key="roles_table")

    st.subheader("Columns")
    st.dataframe(
        column_table(p), hide_index=True, key="columns_table",
        column_config={
            "Missing": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                       max_value=1),
            "Invalid": st.column_config.NumberColumn(
                help="Values that do not parse as the column's kind."),
            "Extreme": st.column_config.NumberColumn(
                help="Numeric values beyond 3 x IQR from the quartiles."),
        })
    with st.expander("First 100 rows"):
        st.dataframe(ctx.df.head(100), hide_index=True, key="preview_table")


# --------------------------------------------------------------------------- Explore

def explore_distribution(ctx: Ctx) -> None:
    columns = ctx.kpis["metrics"] + ctx.kpis["dimensions"]
    if not columns:
        st.info("No numeric or categorical column to plot.", icon=":material/info:")
        return
    left, right = st.columns([1, 3], gap="large")
    with left:
        column = st.selectbox("Column", columns, key="dist_column")
        info = ctx.profile.column(column)
        facts = {"Kind": info.kind, "Missing": f"{info.pct_missing:.1%}",
                 "Distinct": f"{info.n_unique:,}", "Invalid": f"{info.n_invalid:,}"}
        facts.update({k.capitalize(): charts.compact(v) for k, v in info.stats.items()})
        st.dataframe(pd.DataFrame({"Statistic": list(facts), "Value": list(facts.values())}),
                     hide_index=True, key="dist_table")
    with right:
        try:
            fig = (charts.distribution_figure(ctx.df[column], mode=ctx.mode)
                   if info.kind == "numeric"
                   else charts.category_figure(ctx.df[column], mode=ctx.mode))
        except EXPECTED as exc:
            show_error(exc)
            return
        show(fig, "dist_chart")


def explore_correlation(ctx: Ctx) -> None:
    method = st.segmented_control("Method", ["pearson", "spearman"], default="pearson",
                                  key="corr_method", required=True,
                                  format_func=str.capitalize)
    pairs = correlation_table(ctx.ref, ctx.table, method)
    if pairs.empty:
        st.info("Correlations need at least two numeric columns with enough rows.",
                icon=":material/info:")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        show(charts.correlation_figure(pairs, columns=ctx.profile.by_kind("numeric"),
                                       mode=ctx.mode), "corr_chart")
    with right:
        shown = pairs.rename(columns={"a": "Column A", "b": "Column B", "n": "Rows"})
        shown["p-value"] = [f"{v:.3f}" if v >= 0.001 else "< 0.001" for v in shown.pop("p_value")]
        st.dataframe(shown, hide_index=True, key="corr_table",
                     column_config={"r": st.column_config.NumberColumn(format="%.3f"),
                                    "Rows": st.column_config.NumberColumn(format="localized")})
        st.caption("Strongest pairs first. The p-values are descriptive: they are not "
                   "corrected for the number of pairs, and no significance is claimed.")


def explore_segments(ctx: Ctx) -> None:
    dims, metrics = ctx.kpis["dimensions"], ctx.kpis["metrics"]
    if not dims or not metrics:
        st.info("Segments need one categorical column and one numeric column.",
                icon=":material/info:")
        return
    a, b, c = st.columns(3)
    by = a.selectbox("Segment by", dims, key="seg_by")
    metric = b.selectbox("Metric", metrics, key="seg_metric")
    stat = c.selectbox("Statistic", list(STATS), key="seg_stat", format_func=STATS.__getitem__)
    try:
        table = segment_table(ctx.ref, ctx.table, by, metric)
        fig = charts.segment_figure(table, dimension=by, metric=metric, stat=stat,
                                    mode=ctx.mode)
    except EXPECTED as exc:
        show_error(exc)
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        show(fig, "seg_chart")
    with right:
        st.dataframe(
            table.reset_index().rename(columns={
                by: "Segment", "rows": "Rows", "total": "Total", "mean": "Mean",
                "median": "Median", "share_of_total": "Share of total"}),
            hide_index=True, key="seg_table",
            column_config={"Share of total": st.column_config.NumberColumn(format="percent")})


def explore_time(ctx: Ctx) -> None:
    dates, metrics = ctx.kpis["dates"], ctx.kpis["metrics"]
    if not dates or not metrics:
        st.info("A time series needs a date column and a numeric column. This table has "
                f"{len(dates)} date column(s) and {len(metrics)} numeric column(s).",
                icon=":material/info:")
        return
    sample = ctx.source.sample
    a, b, c = st.columns(3)
    date_col = a.selectbox("Date", dates, key="ts_date",
                           index=pick(dates, sample.date_col if sample else None))
    value_col = b.selectbox("Value (summed per period)", metrics, key="ts_value",
                            index=pick(metrics, sample.value_col if sample else None))
    try:
        suggested = eda.suggest_frequency(ctx.df[date_col])
        freq = c.selectbox("Period", list(FREQUENCIES), key="ts_freq",
                           index=pick(list(FREQUENCIES), suggested),
                           format_func=FREQUENCIES.__getitem__)
        summary = trend_summary(ctx.ref, ctx.table, date_col, value_col, freq)
    except EXPECTED as exc:
        show_error(exc)
        return
    tiles = st.columns(4)
    tiles[0].metric("Periods", f"{len(summary.series):,}", border=True)
    tiles[1].metric(f"Change per {FREQUENCIES[freq].lower()}",
                    f"{summary.slope_per_period:+,.4g}", border=True,
                    help="Slope of a straight line fitted to the series.")
    tiles[2].metric("Fit of the straight line (r)", f"{summary.trend_r:+.2f}", border=True)
    tiles[3].metric(
        f"Seasonal autocorrelation, lag {summary.seasonal_period}",
        "not assessed" if summary.seasonal_autocorr is None
        else f"{summary.seasonal_autocorr:+.2f}", border=True,
        help="Correlation of the detrended series with itself one season earlier.")
    show(charts.timeseries_figure(summary.series, value_name=value_col, mode=ctx.mode),
         "ts_chart")
    bullets([note[0].upper() + note[1:] + "." for note in summary.notes])


def explore_cohorts(ctx: Ctx) -> None:
    ids, dates = ctx.kpis["ids"], ctx.kpis["dates"]
    if not ids or not dates:
        st.info("Cohort retention needs an identifier column (who) and a date column "
                "(when). This table has "
                f"{len(ids)} identifier column(s) and {len(dates)} date column(s).",
                icon=":material/info:")
        return
    sample = ctx.source.sample
    a, b, c = st.columns(3)
    user_col = a.selectbox("Who (identifier)", ids, key="cohort_user",
                           index=pick(ids, sample.user_col if sample else None))
    date_col = b.selectbox("When (date)", dates, key="cohort_date",
                           index=pick(dates, sample.date_col if sample else None))
    periods = ["M", "Q", "Y"]
    freq = c.selectbox("Cohort period", periods, index=1, key="cohort_freq",
                       format_func=FREQUENCIES.__getitem__)
    try:
        rates, sizes = retention_matrix(ctx.ref, ctx.table, user_col, date_col, freq)
        fig = charts.retention_figure(rates, sizes, mode=ctx.mode)
    except EXPECTED as exc:
        show_error(exc)
        return
    show(fig, "cohort_chart")
    with st.expander("Retention table"):
        table = rates.copy()
        table.insert(0, "Cohort size", sizes)
        table.index = table.index.astype(str)
        table.columns = [str(col) for col in table.columns]
        st.dataframe(table.reset_index(names="Cohort"), hide_index=True, key="cohort_table")


def section_explore(ctx: Ctx) -> None:
    tabs = st.tabs(["Distributions", "Correlations", "Segments", "Time series",
                    "Cohort retention"])
    for tab, body in zip(tabs, (explore_distribution, explore_correlation, explore_segments,
                                explore_time, explore_cohorts), strict=True):
        with tab:
            body(ctx)


# --------------------------------------------------------------------------- Ask

def show_answer(ctx: Ctx, kind: str, text: str) -> None:
    """Translate if needed, show the SQL, run it through the guard, show the outcome."""
    sql, assumptions = text, []
    if kind == "question":
        st.markdown(f"**Question.** {escape_md(text)}")
        try:
            translation = RuleBasedTranslator().translate(
                text, ctx.profile, filter_values(ctx.ref, ctx.table))
        except EXPECTED as exc:
            show_error(exc, "No SQL was generated, so nothing ran.")
            return
        sql, assumptions = translation.sql, list(translation.assumptions)
    st.markdown("**SQL**")
    st.code(sql or " ", language="sql")
    try:
        result = query_result(ctx.ref, sql)
    except EXPECTED as exc:
        show_error(exc)
        return
    st.session_state["last_sql"] = result.sql
    st.markdown(f"**What it does.** {escape_md(result.explanation)}")
    if assumptions:
        st.markdown("**Assumptions made by the translator**")
        bullets(assumptions)
    st.dataframe(result.frame(), hide_index=True, key="ask_result")
    note = f"{len(result.rows):,} row(s) in {result.elapsed_s * 1000:,.0f} ms."
    if result.truncated:
        note += f" Only the first {ctx.ds.limits.sql_max_rows:,} rows are shown."
    st.caption(note)


def section_ask(ctx: Ctx) -> None:
    st.caption("Every query is read-only. It must be a single SELECT on the loaded tables, "
               "it runs inside a transaction that is rolled back, and it is stopped after "
               f"{ctx.ds.limits.sql_timeout_s:g} seconds or {ctx.ds.limits.sql_max_rows:,} rows.")
    sample = ctx.source.sample
    how = st.segmented_control("Ask with", ["A question", "SQL"], default="A question",
                               key="ask_how", required=True)
    if how == "SQL":
        starter = sample.sql if sample else preview_sql(ctx.table)
        with st.form("ask_sql_form"):
            sql = st.text_area("SQL (one SELECT statement, DuckDB dialect)", value=starter,
                               height=120, key="ask_sql")
            if st.form_submit_button("Run query", type="primary"):
                st.session_state["ask"] = ("sql", sql)
    else:
        with st.form("ask_question_form"):
            question = st.text_input(
                "Question about one table", key="ask_question",
                placeholder=sample.questions[0] if sample else "total amount by region")
            if st.form_submit_button("Translate and run", type="primary"):
                st.session_state["ask"] = ("question", question)
        st.caption("A rule-based translator handles totals, averages, counts, top N, time "
                   "grains and simple filters. It refuses anything else rather than guess."
                   + (" Try: " + "; ".join(sample.questions) + "." if sample else ""))
    asked = st.session_state.get("ask")
    if asked:
        show_answer(ctx, *asked)
    with st.expander("Tables and columns"):
        st.dataframe(pd.DataFrame(
            [{"Table": t.name, "Column": name, "Type": dtype}
             for t in ctx.ds.tables.values() for name, dtype in t.columns]),
            hide_index=True, key="ask_schema")


# --------------------------------------------------------------------------- Predict

def section_predict(ctx: Ctx) -> None:
    targets = ctx.kpis["dimensions"] + ctx.kpis["metrics"]
    if not targets:
        st.info("No column can serve as a target: a target is a category or a number.",
                icon=":material/info:")
        return
    sample = ctx.source.sample
    with st.form("predict_form"):
        a, b, c = st.columns([2, 2, 1])
        target = a.selectbox("Column to predict", targets, key="predict_target",
                             index=pick(targets, sample.target if sample else None))
        when = b.selectbox("Hold out the latest rows by (optional)",
                           [NONE, *ctx.kpis["dates"]], key="predict_time",
                           help="With a date column, the model is tested on the most recent "
                                "20% of rows instead of a random 20%.")
        seed = c.number_input("Seed", min_value=0, max_value=99_999, value=0, step=1,
                              key="predict_seed")
        if st.form_submit_button("Train and evaluate", type="primary"):
            st.session_state["predict"] = (target, None if when == NONE else when, int(seed))
            st.session_state.pop("predict_keep", None)
    chosen = st.session_state.get("predict")
    if not chosen:
        st.info("Choose the column to predict, then select Train and evaluate. Candidates "
                "are compared with cross-validation and the winner is scored once on rows "
                "it has never seen, next to a no-skill baseline.", icon=":material/info:")
        return
    target, time_col, seed = chosen
    keep = tuple(st.session_state.get("predict_keep", ()))
    try:
        m = model_report(ctx.ref, ctx.table, target, time_col, keep, seed)
    except EXPECTED as exc:
        st.session_state.pop("done_predict", None)
        show_error(exc)
        return
    st.session_state["done_predict"] = (target, time_col, seed, keep)   # what Report may use

    metric = charts.metric_label(m.primary_metric)
    score, base = m.test_metrics[m.primary_metric], m.baseline_test_metrics[m.primary_metric]
    name = charts.model_label(m.best_model)
    if m.improves_on_baseline:
        st.success(f"**Signal found.** {name} predicts {code_span(m.target)} with hold-out "
                   f"{metric} {score:.3f} against {base:.3f} for a no-skill baseline, and "
                   f"beat it in {m.folds_better_than_baseline} of {m.n_folds} folds.",
                   icon=":material/check_circle:")
    else:
        st.warning(f"**No usable signal.** No candidate beat the no-skill baseline in every "
                   f"fold and by a margin (best hold-out {metric} {score:.3f} against "
                   f"{base:.3f}). Do not deploy a model for {code_span(m.target)} on these "
                   "columns.", icon=":material/warning:")
    tiles = st.columns(4)
    tiles[0].metric(f"Hold-out {metric}", f"{score:.3f}", delta=f"{score - base:+.3f} vs baseline",
                    delta_color="normal" if m.higher_is_better else "inverse", border=True)
    tiles[1].metric("Folds better than baseline", f"{m.folds_better_than_baseline} of {m.n_folds}",
                    border=True)
    tiles[2].metric("Training / hold-out rows", f"{m.n_train:,} / {m.n_test:,}", border=True)
    tiles[3].metric("Lift in the top 10%",
                    "n/a" if m.lift_top_decile is None else f"{m.lift_top_decile:.2f}x",
                    border=True,
                    help="Rate of the positive class among the 10% of hold-out rows the "
                         "model ranks highest, divided by the overall rate.")
    left, right = st.columns(2, gap="large")
    with left:
        show(charts.model_comparison_figure(m, mode=ctx.mode), "predict_cv_chart")
    with right:
        show(charts.holdout_figure(m, mode=ctx.mode), "predict_holdout_chart")
    left, right = st.columns(2, gap="large")
    with left:
        show(charts.importance_figure(m, mode=ctx.mode), "predict_importance_chart")
    with right:
        st.subheader("Leakage screen")
        if m.leakage:
            st.dataframe(pd.DataFrame(
                [{"Column": f.column, "Why it was flagged": f.reason,
                  "Used anyway": "no" if f.excluded else "yes"} for f in m.leakage]),
                hide_index=True, key="predict_leakage")
            st.multiselect(
                "Keep a flagged column", [f.column for f in m.leakage], key="predict_keep",
                help="Only keep a column you know is recorded before the outcome. The "
                     "report will say that it was kept despite the warning.")
        else:
            st.success("No column was flagged as recorded with or after the outcome.",
                       icon=":material/check_circle:")
        if m.dropped:
            st.dataframe(pd.DataFrame({"Column left out": list(m.dropped),
                                       "Reason": list(m.dropped.values())}),
                         hide_index=True, key="predict_dropped")
    st.subheader("Limits of this result")
    bullets(m.limitations)


# --------------------------------------------------------------------------- Forecast

def section_forecast(ctx: Ctx) -> None:
    dates, metrics = ctx.kpis["dates"], ctx.kpis["metrics"]
    if not dates or not metrics:
        st.info("A forecast needs a date column and a numeric column. This table has "
                f"{len(dates)} date column(s) and {len(metrics)} numeric column(s).",
                icon=":material/info:")
        return
    sample = ctx.source.sample
    with st.form("forecast_form"):
        a, b, c, d = st.columns([2, 2, 1, 1])
        date_col = a.selectbox("Date", dates, key="forecast_date",
                               index=pick(dates, sample.date_col if sample else None))
        value_col = b.selectbox("Value (summed per period)", metrics, key="forecast_value",
                                index=pick(metrics, sample.value_col if sample else None))
        periods = list(FREQUENCIES)[:4]
        freq = c.selectbox("Period", periods, key="forecast_freq",
                           index=pick(periods, suggested_period(ctx, date_col)),
                           format_func=FREQUENCIES.__getitem__)
        horizon = d.number_input("Periods ahead", min_value=1, max_value=60, value=6, step=1,
                                 key="forecast_horizon")
        if st.form_submit_button("Forecast", type="primary"):
            st.session_state["forecast"] = (date_col, value_col, int(horizon), freq)
    chosen = st.session_state.get("forecast")
    if not chosen:
        st.info("Choose a date and a value, then select Forecast. Three methods are scored "
                "on past periods they did not see; the simplest one that holds up is used.",
                icon=":material/info:")
        return
    date_col, value_col, horizon, freq = chosen
    try:
        f = forecast_report(ctx.ref, ctx.table, date_col, value_col, horizon, freq)
        history = trend_summary(ctx.ref, ctx.table, date_col, value_col, freq).series
    except EXPECTED as exc:
        st.session_state.pop("done_forecast", None)
        show_error(exc)
        return
    st.session_state["done_forecast"] = chosen                          # what Report may use
    name = f.best_model.replace("_", " ")
    best = f.backtest_mae[f.best_model]
    if f.improves_on_naive:
        st.success(f"**The statistical model earned its place.** {name.capitalize()} had the "
                   f"lowest backtest error ({charts.compact(best)} per period) and beat the "
                   "naive forecasts in most windows.", icon=":material/check_circle:")
    else:
        st.warning(f"**A naive forecast is used.** The statistical model did not beat the "
                   f"simple baselines reliably, so the {name} forecast is reported "
                   f"(backtest error {charts.compact(best)} per period).",
                   icon=":material/warning:")
    tiles = st.columns(4)
    tiles[0].metric(f"Next {f.horizon} periods, total",
                    charts.compact(sum(row[1] for row in f.forecast)), border=True)
    tiles[1].metric("Planning range", f"{charts.compact(sum(r[2] for r in f.forecast))} to "
                                      f"{charts.compact(sum(r[3] for r in f.forecast))}",
                    border=True, help=f.interval_note)
    tiles[2].metric("Backtest error per period", charts.compact(best), border=True,
                    help="Mean absolute error over the rolling backtests.")
    tiles[3].metric("Backtest windows", f"{f.n_backtests} x {f.horizon} periods", border=True)
    show(charts.timeseries_figure(history, value_name=value_col, forecast=f, mode=ctx.mode),
         "forecast_chart")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.dataframe(pd.DataFrame(f.forecast, columns=["Period", "Forecast", "Low", "High"]),
                     hide_index=True, key="forecast_table")
    with right:
        st.dataframe(pd.DataFrame(
            {"Method": [k.replace("_", " ") for k in f.backtest_mae],
             "Backtest error (MAE)": list(f.backtest_mae.values())}),
            hide_index=True, key="forecast_backtests")
    st.subheader("Limits of this result")
    bullets(f.limitations)


# --------------------------------------------------------------------------- Report

def section_report(ctx: Ctx) -> None:
    predict = st.session_state.get("done_predict")
    forecast = st.session_state.get("done_forecast")
    last_sql = st.session_state.get("last_sql")
    with st.form("report_form"):
        problem = st.text_area("Business problem, in your words (optional)", height=80,
                               key="report_problem")
        a, b, c = st.columns(3)
        use_model = a.checkbox(
            f"Model for {predict[0]}" if predict else "Model (run Predict first)",
            value=bool(predict), disabled=not predict, key="report_model")
        use_forecast = b.checkbox(
            f"Forecast of {forecast[1]}" if forecast else "Forecast (run Forecast first)",
            value=bool(forecast), disabled=not forecast, key="report_forecast")
        use_query = c.checkbox("Last query" if last_sql else "Query (run one in Ask first)",
                               value=bool(last_sql), disabled=not last_sql, key="report_query")
        if st.form_submit_button("Build report", type="primary"):
            st.session_state["report"] = (
                problem.strip() or None, last_sql if use_query else None,
                predict if use_model else None, forecast if use_forecast else None)
    chosen = st.session_state.get("report")
    if not chosen:
        st.info("Select Build report for an executive summary, findings, evidence, limits "
                "and actions, with the files needed to reproduce it. Steps you ran in "
                "Predict, Forecast and Ask can be included.", icon=":material/info:")
        return
    problem, sql, model, forecast = chosen
    target, time_col, seed, keep = model or (None, None, 0, ())
    date_col, value_col, horizon, freq = forecast or (None, None, 6, None)
    try:
        analysis = report_analysis(ctx.ref, ctx.table, problem, sql, target, time_col, keep,
                                   date_col, value_col, horizon, freq, seed)
        texts = artifact_texts(analysis, "report", ctx.source.command_path)
        bundle = bundle_artifacts(analysis, ctx.source.command_path)
    except EXPECTED as exc:
        show_error(exc)
        return
    st.caption("Downloads: everything as one archive, or each file on its own. "
               "`reproduce.sh` holds the command that regenerates the report from the same "
               "file.")
    buttons = st.columns(5)
    buttons[0].download_button("All files (.zip)", bundle, file_name="report.zip",
                               mime="application/zip", type="primary", on_click="ignore",
                               key="download_zip", icon=":material/download:")
    mimes = {"report.md": "text/markdown", "analysis.json": "application/json",
             "queries.sql": "text/plain", "reproduce.sh": "text/plain"}
    for slot, name in zip(buttons[1:], ARTIFACTS.values(), strict=True):
        slot.download_button(name, texts[name], file_name=name, mime=mimes[name],
                             on_click="ignore", key=f"download_{name}")
    with st.container(border=True):
        st.markdown(texts["report.md"])


# --------------------------------------------------------------------------- page

def main() -> None:
    # The icon is a local file: a ``:material/...:`` page icon would be fetched from a font
    # CDN, and this page is meant to work, and stay, offline.
    st.set_page_config(
        page_title="Auto Data Scientist", page_icon=str(Path(__file__).with_name("favicon.svg")),
        layout="wide",
        menu_items={"Get help": None, "Report a bug": None,
                    "About": f"auto-data-scientist {__version__}. Runs on this computer; "
                             "no sign-in and no network calls."})
    with st.sidebar:
        source, problem = choose_source()
    if problem is not None or source is None:
        if problem is not None:
            show_error(problem)
        else:
            st.title("Choose a file to begin")
            st.info("Upload a data file in the sidebar, or switch to a built-in sample to "
                    "see every section working without a file.", icon=":material/upload_file:")
        with st.sidebar:
            footer()
        return
    try:
        ds = open_dataset(source.ref)
        with st.sidebar:
            table = choose_table(ds)
        profile = table_profile(source.ref, table)
        ctx = Ctx(source, table, ds, table_frame(source.ref, table), profile,
                  eda.identify_kpis(profile), theme_mode())
    except EXPECTED as exc:
        show_error(exc)
        with st.sidebar:
            footer()
        return
    with st.sidebar:
        footer()

    scope = f"{source.path}|{table}"
    if st.session_state.get("scope") != scope:      # a new table: forget choices and results
        for key in list(st.session_state.keys()):
            if key in RESULT_KEYS or str(key).startswith(CHOICE_PREFIXES):
                del st.session_state[key]
        st.session_state["scope"] = scope

    st.title(escape_md(source.name))
    st.caption(f"Table {code_span(table)} with {profile.n_rows:,} rows and "
               f"{profile.n_cols:,} columns.")
    section = st.segmented_control("Section", SECTIONS, default="Data", key="section",
                                   required=True, label_visibility="collapsed")
    if section != "Data" and profile.n_rows == 0:
        st.info("This table has no rows, so there is nothing to analyse.",
                icon=":material/info:")
        return
    {"Data": section_data, "Explore": section_explore, "Ask": section_ask,
     "Predict": section_predict, "Forecast": section_forecast,
     "Report": section_report}[section](ctx)


main()
