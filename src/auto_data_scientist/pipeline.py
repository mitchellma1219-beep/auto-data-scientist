"""End-to-end analysis: load, profile, explore, optionally query, model and forecast."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import pandas as pd

from auto_data_scientist import eda
from auto_data_scientist.config import Limits
from auto_data_scientist.errors import AdsError, DataError
from auto_data_scientist.forecasting import ForecastReport, forecast_series
from auto_data_scientist.ingest import Dataset, load_dataset
from auto_data_scientist.modeling import SELECTION_ROWS, ModelReport, train_supervised
from auto_data_scientist.nl2sql import RuleBasedTranslator, Translation, Translator
from auto_data_scientist.profiling import (
    TableProfile,
    missing_mask,
    parse_dates,
    profile_frame,
)
from auto_data_scientist.query import QueryResult, run_query


@dataclass
class Analysis:
    source: str
    sha256: str
    table: str
    profile: TableProfile
    kpis: dict[str, list[str]]
    numeric_summary: pd.DataFrame
    correlations: pd.DataFrame
    segments: dict[str, pd.DataFrame] = field(default_factory=dict)
    primary_metric: str | None = None
    trend: eda.TimeSeriesSummary | None = None
    trend_cols: tuple[str, str] | None = None
    question: str | None = None
    translation: Translation | None = None
    query: QueryResult | None = None
    model: ModelReport | None = None
    forecast: ForecastReport | None = None
    business_problem: str | None = None
    notes: list[str] = field(default_factory=list)
    params: dict[str, object] = field(default_factory=dict)


def categorical_values(df: pd.DataFrame, profile: TableProfile,
                       limit: int = 50) -> dict[str, list[str]]:
    """Distinct values of low-cardinality columns, used to recognise filters in questions."""
    out: dict[str, list[str]] = {}
    for col in profile.by_kind("categorical"):
        vals = df.loc[~missing_mask(df[col]), col].astype("string").str.strip().unique()
        if len(vals) <= limit:
            out[col] = sorted(map(str, vals))
    return out


def ask(ds: Dataset, question: str, table: str | None = None,
        translator: Translator | None = None) -> tuple[Translation, QueryResult]:
    """Translate a question, validate the SQL and run it."""
    name = ds.resolve(table)
    df = ds.frame(name)
    profile = profile_frame(df, name, ds.tables[name].rejected_rows)
    translation = (translator or RuleBasedTranslator()).translate(
        question, profile, categorical_values(df, profile))
    return translation, run_query(ds, translation.sql)


def _freq_for(dates: pd.Series) -> str:
    span = (dates.max() - dates.min()).days
    return "M" if span >= 180 else "W" if span >= 28 else "D"


def analyze(
    path: str | Path, *, table: str | None = None, question: str | None = None,
    sql: str | None = None, target: str | None = None, time_col: str | None = None,
    date_col: str | None = None, value_col: str | None = None, horizon: int = 6,
    business_problem: str | None = None, seed: int = 0, limits: Limits | None = None,
    selection_rows: int | None = None,
) -> Analysis:
    """Run the full analysis. Optional steps that cannot run are recorded in ``notes``;
    explicitly requested steps (question, sql, target, forecast columns) raise instead.

    ``selection_rows`` overrides the number of training rows used to compare candidate
    models on large tables (default ``modeling.SELECTION_ROWS``).
    """
    ds = load_dataset(path, limits)
    try:
        name = ds.resolve(table)
        df = ds.frame(name)
        profile = profile_frame(df, name, ds.tables[name].rejected_rows)
        if profile.n_rows == 0:
            raise DataError(f"table {name!r} has no rows; nothing to analyse")
        if all(c.kind == "empty" for c in profile.columns):
            raise DataError(f"table {name!r} has no usable values; every column is blank")
        kpis = eda.identify_kpis(profile)
        a = Analysis(
            source=ds.source.name, sha256=hashlib.sha256(ds.source.read_bytes()).hexdigest(),
            table=name, profile=profile, kpis=kpis,
            numeric_summary=eda.numeric_summary(df, profile),
            correlations=eda.correlation_pairs(df, profile), business_problem=business_problem,
            params={"table": table, "question": question, "sql": sql, "target": target,
                    "time_col": time_col, "date_col": date_col, "value_col": value_col,
                    "horizon": horizon, "seed": seed, "selection_rows": selection_rows})
        if len(profile.by_kind("numeric")) > 300:
            a.notes.append("correlations: only the first 300 numeric columns were compared")
        metric = value_col or (kpis["metrics"][0] if kpis["metrics"] else None)
        a.primary_metric = metric
        if metric:
            for dim in kpis["dimensions"][:3]:
                try:
                    a.segments[dim] = eda.segment_summary(df, dim, metric)
                except AdsError as exc:
                    a.notes.append(f"segment by {dim}: {exc}")
        dcol = date_col or (kpis["dates"][0] if kpis["dates"] else None)
        if metric and dcol:
            try:
                dates = parse_dates(df[dcol]).dropna()
                a.trend = eda.time_series_summary(df, dcol, metric, _freq_for(dates))
                a.trend_cols = (dcol, metric)
            except AdsError as exc:
                a.notes.append(f"time series: {exc}")
        if sql:
            a.query = run_query(ds, sql)
        elif question:
            a.question = question
            a.translation, a.query = ask(ds, question, name)
        if target:
            a.model = train_supervised(
                df, target, time_col=time_col, seed=seed,
                max_selection_rows=SELECTION_ROWS if selection_rows is None else selection_rows)
        if date_col and value_col:
            a.forecast = forecast_series(df, date_col, value_col, horizon,
                                         _freq_for(parse_dates(df[date_col]).dropna()))
        return a
    finally:
        ds.close()


def write_artifacts(a: Analysis, out_dir: str | Path, source_path: str | Path) -> dict[str, Path]:
    """Write report.md, analysis.json, queries.sql and reproduce.sh. Returns their paths."""
    from auto_data_scientist.report import render_report

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v for k, v in {"report": "report.md", "data": "analysis.json",
                                     "sql": "queries.sql", "reproduce": "reproduce.sh"}.items()}
    paths["report"].write_text(render_report(a), encoding="utf-8")
    payload = {
        "source": a.source, "sha256": a.sha256, "table": a.table, "params": a.params,
        "versions": {p: metadata.version(p) for p in
                     ("auto-data-scientist", "duckdb", "pandas", "scikit-learn", "statsmodels")
                     if _installed(p)},
        "profile": a.profile.model_dump(), "kpis": a.kpis,
        "correlations": a.correlations.to_dict("records"),
        "segments": {k: v.reset_index().to_dict("records") for k, v in a.segments.items()},
        "query": None if a.query is None else {
            "sql": a.query.sql, "explanation": a.query.explanation, "columns": a.query.columns,
            "rows": [list(r) for r in a.query.rows[:200]], "truncated": a.query.truncated},
        "model": None if a.model is None else a.model.model_dump(),
        "forecast": None if a.forecast is None else a.forecast.model_dump(),
        "notes": a.notes,
    }
    paths["data"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths["sql"].write_text((a.query.sql + ";\n") if a.query else "-- no query was run\n",
                            encoding="utf-8")
    args = ["ads", "report", str(source_path), "--out", str(out)]
    for flag, key in (("--table", "table"), ("--sql", "sql"), ("--question", "question"),
                      ("--target", "target"), ("--time-col", "time_col"),
                      ("--date-col", "date_col"), ("--value-col", "value_col"),
                      ("--selection-rows", "selection_rows")):
        if a.params.get(key):
            args += [flag, str(a.params[key])]
    args += ["--horizon", str(a.params["horizon"]), "--seed", str(a.params["seed"])]
    paths["reproduce"].write_text(
        "#!/bin/sh\n# Regenerates this report. Input file sha256: " + a.sha256 + "\n"
        + " ".join(shlex.quote(x) for x in args) + "\n", encoding="utf-8")
    return paths


def _installed(package: str) -> bool:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        return False
    return True
