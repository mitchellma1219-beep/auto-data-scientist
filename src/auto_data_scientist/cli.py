"""Command-line interface: ``ads profile | sql | ask | model | forecast | report``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from auto_data_scientist.errors import AdsError
from auto_data_scientist.forecasting import forecast_series
from auto_data_scientist.ingest import load_dataset
from auto_data_scientist.modeling import SELECTION_ROWS, train_supervised
from auto_data_scientist.pipeline import analyze, ask, write_artifacts
from auto_data_scientist.profiling import profile_frame
from auto_data_scientist.query import QueryResult, run_query
from auto_data_scientist.report import code_block, escape_md, md_table, render_report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ads", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("file", type=Path, help="CSV, TSV, XLSX, Parquet, SQLite or DuckDB file")
        sp.add_argument("--table", help="table or sheet name (default: the largest)")
        return sp

    add("profile", "schema, data quality score and issues").add_argument(
        "--format", choices=["markdown", "json"], default="markdown")
    s = add("sql", "run one read-only SELECT through the guard")
    s.add_argument("--query", required=True)
    s.add_argument("--format", choices=["markdown", "json"], default="markdown")
    a = add("ask", "answer a simple aggregate question with the rule-based translator")
    a.add_argument("question")
    a.add_argument("--format", choices=["markdown", "json"], default="markdown")
    rows_help = ("compare candidate models on a seeded sample of at most this many training "
                 f"rows (default {SELECTION_ROWS:,}); the selected model is refitted on all rows")
    m = add("model", "train and evaluate a classifier or regressor against a baseline")
    m.add_argument("--target", required=True)
    m.add_argument("--task", choices=["classification", "regression"])
    m.add_argument("--time-col", help="split chronologically on this column")
    m.add_argument("--keep", action="append", default=[],
                   help="keep a column the leakage screen would drop")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--selection-rows", type=int, default=SELECTION_ROWS, help=rows_help)
    f = add("forecast", "forecast an aggregated series with backtests")
    f.add_argument("--date-col", required=True)
    f.add_argument("--value-col", required=True)
    f.add_argument("--horizon", type=int, default=6)
    f.add_argument("--freq", choices=["D", "W", "M", "Q"], default="M")
    r = add("report", "full analysis and executive report")
    r.add_argument("--out", type=Path, help="directory for report.md and reproducible artifacts")
    r.add_argument("--question")
    r.add_argument("--sql")
    r.add_argument("--target")
    r.add_argument("--time-col")
    r.add_argument("--date-col")
    r.add_argument("--value-col")
    r.add_argument("--horizon", type=int, default=6)
    r.add_argument("--problem", help="the business problem, in the user's words")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--selection-rows", type=int, help=rows_help)
    return p


def _show_query(res: QueryResult, fmt: str, question: str | None = None,
                assumptions: Sequence[str] = ()) -> str:
    if fmt == "json":
        return json.dumps({"question": question, "assumptions": list(assumptions),
                           "sql": res.sql, "explanation": res.explanation,
                           "columns": res.columns, "rows": [list(r) for r in res.rows],
                           "truncated": res.truncated}, indent=2, default=str)
    lines = [*code_block(res.sql, "sql"), "", escape_md(res.explanation), ""]
    lines += [f"- Assumption: {escape_md(x)}" for x in assumptions]
    lines += ["", md_table(res.frame(), 50, index=False)]
    if res.truncated:
        lines.append("\n_Result truncated at the row limit._")
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> str:
    if args.command == "report":
        a = analyze(args.file, table=args.table, question=args.question, sql=args.sql,
                    target=args.target, time_col=args.time_col, date_col=args.date_col,
                    value_col=args.value_col, horizon=args.horizon,
                    business_problem=args.problem, seed=args.seed,
                    selection_rows=args.selection_rows)
        if args.out:
            paths = write_artifacts(a, args.out, args.file)
            return "\n".join(f"wrote {p}" for p in paths.values())
        return render_report(a)
    ds = load_dataset(args.file)
    try:
        name = ds.resolve(args.table)
        if args.command == "profile":
            prof = profile_frame(ds.frame(name), name, ds.tables[name].rejected_rows)
            if args.format == "json":
                return prof.model_dump_json(indent=2)
            score = "n/a" if prof.quality.score is None else f"{prof.quality.score:.1f}/100"
            rows = [f"| {escape_md(c.name)} | {c.kind} | {c.pct_missing:.1%} | {c.n_unique} | "
                    f"{c.n_invalid} | {c.n_extreme} |" for c in prof.columns]
            return "\n".join([
                f"# Profile: {escape_md(name)}", "",
                f"{prof.n_rows:,} rows, {prof.n_cols} columns, {prof.n_duplicate_rows:,} "
                f"duplicate rows. Quality score: {score}", "",
                "| column | kind | missing | unique | invalid | extreme |",
                "|---|---|---|---|---|---|",
                *rows, "", *[f"- {escape_md(i)}" for i in prof.issues]])
        if args.command == "sql":
            return _show_query(run_query(ds, args.query), args.format)
        if args.command == "ask":
            tr, res = ask(ds, args.question, name)
            return _show_query(res, args.format, args.question, tr.assumptions)
        if args.command == "model":
            return train_supervised(ds.frame(name), args.target, task=args.task,
                                    time_col=args.time_col, keep=args.keep, seed=args.seed,
                                    max_selection_rows=args.selection_rows
                                    ).model_dump_json(indent=2)
        return forecast_series(ds.frame(name), args.date_col, args.value_col, args.horizon,
                               args.freq).model_dump_json(indent=2)
    finally:
        ds.close()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("ADS_LOG_LEVEL", "WARNING"), stream=sys.stderr,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
    args = _parser().parse_args(argv)
    try:
        print(_run(args))
    except (AdsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
