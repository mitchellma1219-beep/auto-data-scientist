"""Executive report in Markdown. Every number is computed from the data; nothing is assumed.

The report is read in renderers (the dashboard, a repository page, a chat window), so text
that comes from the data is never trusted as markup: names go into code spans whose fence
cannot be closed from inside, and everything else is escaped. A column called
``![x](http://...)`` is printed, not fetched.
"""

from __future__ import annotations

import re

import pandas as pd

from auto_data_scientist.pipeline import Analysis

# Characters that can start emphasis, code, links, images, HTML, tables, strike-through,
# math, entities or e-mail links. An underscore only matters at the edge of a word, so
# `order_date` stays readable. The last two alternatives stop bare URLs from being
# turned into links by renderers that auto-link them.
_MD_SPECIAL = re.compile(
    r"[\\`*\[\]<>|~$&@]|(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])|/(?=/)|(?<=[wW]{3})\.")


def escape_md(value: object) -> str:
    """Make text that comes from the data inert in Markdown.

    Column names, category values and file names are data, not markup: after escaping they
    cannot open emphasis, a link, an image, a code span, a table cell or HTML, and a URL in
    a value stays plain text. Line breaks become spaces so a value cannot start a new block.
    """
    return _MD_SPECIAL.sub(lambda m: "\\" + m.group(0), " ".join(str(value).split()))


def code_span(value: object) -> str:
    """An inline code span that backticks inside the text cannot close early."""
    text = " ".join(str(value).split())
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") or not text else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def code_block(text: str, language: str = "") -> list[str]:
    """Lines of a fenced block whose fence is longer than any run of backticks inside."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence + language, text, fence]


def _names(values: list[str]) -> str:
    return ", ".join(code_span(v) for v in values) or "none"


def md_table(df: pd.DataFrame, max_rows: int = 12, index: bool = True) -> str:
    """Render a small DataFrame as a Markdown table."""
    if df.empty:
        return "_no rows_"
    view = df.head(max_rows).reset_index() if index else df.head(max_rows)

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}"
        return escape_md(v)

    head = "| " + " | ".join(escape_md(c) for c in view.columns) + " |"
    rule = "|" + "---|" * len(view.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in view.itertuples(index=False, name=None)]
    more = [f"\n_{len(df) - max_rows} more row(s) not shown._"] if len(df) > max_rows else []
    return "\n".join([head, rule, *body, *more])


def _num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"


def _findings(a: Analysis) -> list[str]:
    p, out = a.profile, []
    q = p.quality
    out.append(f"The table {code_span(a.table)} has {p.n_rows:,} rows and {p.n_cols} columns; "
               f"data quality score {q.score:.1f}/100 (completeness {q.completeness:.1%}, validity "
               f"{q.validity:.1%}, {p.n_duplicate_rows:,} duplicate rows).")
    for dim, seg in list(a.segments.items())[:2]:
        top = seg.iloc[0]
        out.append(f"By {code_span(dim)}, {code_span(seg.index[0])} accounts for "
                   f"{top['share_of_total']:.1%} of total {code_span(a.primary_metric)} "
                   f"({_num(top['total'])} across {int(top['rows']):,} rows).")
    strong = a.correlations[a.correlations["r"].abs() >= 0.3]
    if len(strong):
        r = strong.iloc[0]
        out.append(f"{code_span(r['a'])} and {code_span(r['b'])} move together "
                   f"(r = {r['r']:+.2f}, n = {int(r['n']):,}). This is an association, not "
                   "evidence of cause.")
    if a.trend is not None and a.trend_cols:
        t = a.trend
        direction = "rose" if t.slope_per_period > 0 else "fell"
        out.append(f"{code_span(a.trend_cols[1])} {direction} by about "
                   f"{_num(abs(t.slope_per_period))} per "
                   f"{t.freq} period on a straight-line fit over {len(t.series)} periods "
                   f"(r = {t.trend_r:+.2f})."
                   + (f" Seasonal autocorrelation at lag {t.seasonal_period}: "
                      f"{t.seasonal_autocorr:+.2f}." if t.seasonal_autocorr is not None else ""))
    if a.model is not None:
        m = a.model
        k, kb = m.primary_metric, m.baseline_test_metrics[m.primary_metric]
        if m.improves_on_baseline:
            out.append(f"A {m.best_model.replace('_', ' ')} model predicts "
                       f"{code_span(m.target)} with hold-out {k} {m.test_metrics[k]:.3f} versus "
                       f"{kb:.3f} for a no-skill baseline ({m.n_test:,} held-out rows).")
        else:
            out.append(f"No model predicted {code_span(m.target)} better than a no-skill baseline "
                       f"(best hold-out {k} {m.test_metrics[k]:.3f} versus {kb:.3f}).")
    if a.forecast is not None:
        f = a.forecast
        total = sum(x[1] for x in f.forecast)
        out.append(f"Forecast of {code_span(f.value_col)} for the next {f.horizon} periods totals "
                   f"{_num(total)} using {f.best_model.replace('_', ' ')} (backtest mean absolute "
                   f"error {_num(f.backtest_mae[f.best_model])} per period).")
    return out


def _actions(a: Analysis) -> list[str]:
    out = []
    if a.profile.issues:
        out.append(f"Fix the {len(a.profile.issues)} data issue(s) listed under Supporting "
                   "evidence at the source before relying on downstream numbers.")
    if a.model is not None and a.model.improves_on_baseline:
        out.append(f"Pilot the {code_span(a.model.target)} model on a small group and measure "
                   "the outcome against a randomly selected control group before any wider "
                   "roll-out.")
    elif a.model is not None:
        out.append(f"Do not deploy a {code_span(a.model.target)} model on these features; collect "
                   "variables that are known before the outcome and plausibly related to it.")
    if a.model is not None and a.model.leakage:
        cols = ", ".join(code_span(f.column) for f in a.model.leakage)
        out.append(f"Confirm when {cols} is recorded. If it is only known after the outcome it "
                   "must stay out of any model.")
    if a.forecast is not None and not a.forecast.improves_on_naive:
        out.append("Use the simple baseline forecast for planning and revisit once more history "
                   "or explanatory drivers are available.")
    if a.segments:
        out.append("Review the largest segments with the business owner to check that the "
                   "concentration is expected, then decide where a deeper analysis is worth it.")
    out.append("Treat every relationship in this report as a hypothesis; confirm the ones that "
               "matter with a controlled experiment.")
    return out


def _impact(a: Analysis) -> list[str]:
    out = []
    if a.model is not None and a.model.lift_top_decile is not None:
        out.append(f"Ranking by the model, the top 10% of held-out rows contain "
                   f"{a.model.lift_top_decile:.2f} times the average rate of the positive "
                   f"{code_span(a.model.target)} class (measured on {a.model.n_test:,} rows).")
    if a.forecast is not None:
        lo = sum(x[2] for x in a.forecast.forecast)
        hi = sum(x[3] for x in a.forecast.forecast)
        out.append(f"Planning range for the next {a.forecast.horizon} periods of "
                   f"{code_span(a.forecast.value_col)}: {_num(lo)} to {_num(hi)} "
                   f"({a.forecast.interval_note.lower()})")
    out.append("**No monetary impact is estimated.** The data holds no costs, margins or "
               "measured effect of any intervention. To quantify impact, supply the value of the "
               "outcome per unit, the cost of acting and an effect size from an experiment.")
    return out


def _limits(a: Analysis) -> list[str]:
    out = ["The data is observational: differences between groups can reflect who is in each "
           "group rather than the effect of any attribute.",
           f"{len(a.correlations)} correlation pair(s) were examined without correction for "
           "multiple comparisons; p-values are descriptive and no significance is claimed."]
    if a.trend is not None:
        out += [n[0].upper() + n[1:] + "." for n in a.trend.notes]
    if a.model is not None:
        out += [escape_md(x) for x in a.model.limitations]
    if a.forecast is not None:
        out += [escape_md(x) for x in a.forecast.limitations]
    out += [f"Step skipped: {escape_md(n)}" for n in a.notes]
    return out


def render_report(a: Analysis) -> str:
    """Build the report with the eight required sections."""
    p = a.profile
    findings = _findings(a)
    cols = pd.DataFrame([{"column": c.name, "kind": c.kind, "missing": f"{c.pct_missing:.1%}",
                          "unique": c.n_unique, "invalid": c.n_invalid,
                          "extreme": c.n_extreme} for c in p.columns])
    parts = [
        f"# Analysis report: {escape_md(a.source)}", "",
        "## Executive summary", "", *[f"- {x}" for x in findings[:4]], "",
        "## Business problem", "",
        escape_md(a.business_problem) if a.business_problem else
        "_Not stated. This report describes the data and what it can and cannot support; add "
        "a question to focus it._", "",
        "## Data overview", "",
        f"Source {code_span(a.source)} (sha256 `{a.sha256[:16]}…`), table {code_span(a.table)}: "
        f"{p.n_rows:,} rows, "
        f"{p.n_cols} columns, {p.rejected_rows:,} malformed row(s) rejected on load.", "",
        md_table(cols, 40, index=False), "",
        f"Likely metrics: {_names(a.kpis['metrics'])}. Dimensions: "
        f"{_names(a.kpis['dimensions'])}. Dates: {_names(a.kpis['dates'])}.", "",
        "## Key findings", "", *[f"{i}. {x}" for i, x in enumerate(findings, 1)], "",
        "## Supporting evidence", "",
        "### Data quality", "",
        f"Score {p.quality.score:.1f}/100 = 35% completeness ({p.quality.completeness:.4f}) + 25% "
        f"validity ({p.quality.validity:.4f}) + 15% uniqueness ({p.quality.uniqueness:.4f}) + 10% "
        f"consistency ({p.quality.consistency:.4f}) + 15% outliers ({p.quality.outliers:.4f}).", "",
        *([*[f"- {escape_md(i)}" for i in p.issues[:25]], ""] if p.issues
          else ["No issues detected.", ""]),
    ]
    if not a.numeric_summary.empty:
        parts += ["### Numeric columns", "",
                  md_table(a.numeric_summary.rename_axis("column")), ""]
    for dim, seg in a.segments.items():
        parts += [f"### {code_span(a.primary_metric)} by {code_span(dim)}", "", md_table(seg), ""]
    if len(a.correlations):
        parts += ["### Correlations (strongest first)", "", md_table(a.correlations, 10, False), ""]
    if a.query is not None:
        if a.question:
            parts += [f"### Question: {escape_md(a.question)}", ""]
        else:
            parts += ["### Query", ""]
        parts += [*code_block(a.query.sql, "sql"), "", escape_md(a.query.explanation), ""]
        if a.translation and a.translation.assumptions:
            parts += [*[f"- Assumption: {escape_md(x)}" for x in a.translation.assumptions], ""]
        parts += [md_table(a.query.frame(), 20, index=False), ""]
        if a.query.truncated:
            parts += ["_Result truncated at the row limit._", ""]
    if a.model is not None:
        m = a.model
        cand = pd.DataFrame([{"model": c.model, f"cv {m.primary_metric}": c.cv_mean,
                              "cv std": c.cv_std} for c in m.candidates])
        test = pd.DataFrame({"selected model": m.test_metrics,
                             "no-skill baseline": m.baseline_test_metrics})
        imp = pd.DataFrame(m.importance[:10], columns=["feature", "permutation importance"])
        sampled = (f" Candidates were compared on a seeded sample of {m.n_selection:,} training "
                   "rows; the selected model was refitted on all of them."
                   if m.n_selection < m.n_train else "")
        parts += [
            f"### Model for {code_span(m.target)} ({m.task})", "",
            f"{m.n_rows:,} rows; {m.n_train:,} for training with {m.n_folds}-fold "
            f"cross-validation, {m.n_test:,} held out. Selected: **{m.best_model}** (better than "
            f"the baseline in {m.folds_better_than_baseline} of {m.n_folds} folds). Seed {m.seed}."
            + sampled,
            "", md_table(cand, index=False), "", md_table(test), "", md_table(imp, index=False), ""]
        if m.dropped:
            parts += ["Columns left out:", "",
                      *[f"- {code_span(k)}: {escape_md(v)}" for k, v in m.dropped.items()], ""]
    if a.forecast is not None:
        f = a.forecast
        fc = pd.DataFrame(f.forecast, columns=["period", "forecast", "low", "high"])
        bt = pd.DataFrame({"backtest MAE": f.backtest_mae})
        parts += [f"### Forecast of {code_span(f.value_col)}", "",
                  f"{f.n_periods} historical periods, {f.n_backtests} rolling backtest(s) of "
                  f"{f.horizon} steps. Selected: **{f.best_model}**. {f.interval_note}", "",
                  md_table(bt), "", md_table(fc, 60, index=False), ""]
    parts += ["## Statistical limitations", "", *[f"- {x}" for x in _limits(a)], "",
              "## Recommended actions", "", *[f"{i}. {x}" for i, x in enumerate(_actions(a), 1)],
              "", "## Potential business impact", "", *[f"- {x}" for x in _impact(a)], ""]
    return "\n".join(parts)
