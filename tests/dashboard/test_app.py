"""The real dashboard script, driven through its main flows with Streamlit's AppTest.

Every number asserted here is recomputed with the package or with pandas, so the tests
check that the page shows what was computed, not merely that it renders.
"""

import io
import json
import os
import shlex
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if os.environ.get("ADS_REQUIRE_UI") != "1":      # CI sets it: there a missing extra must fail
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")

from markdown_it import MarkdownIt
from streamlit import config
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from auto_data_scientist import analyze, bundle_artifacts, eda, load_dataset, profile_frame
from auto_data_scientist.forecasting import forecast_series
from auto_data_scientist.modeling import train_supervised
from auto_data_scientist.samples import write_sample

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
PARSER = MarkdownIt("gfm-like")


@pytest.fixture(autouse=True)
def bound_to_localhost():
    """AppTest does not read app/.streamlit/config.toml, so give it the address set there."""
    before = config.get_option("server.address")
    config.set_option("server.address", "localhost")
    yield
    config.set_option("server.address", before)


def start() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception
    return at


def go(at: AppTest, section: str) -> AppTest:
    at.segmented_control(key="section").set_value(section).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def press(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def upload(at: AppTest, name: str, data: bytes) -> AppTest:
    at.radio(key="origin").set_value("Upload a file").run()
    at.file_uploader(key="upload").upload(name, data, "application/octet-stream").run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def figures(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def figure(at: AppTest, title: str) -> dict:
    return next(f for f in figures(at) if title in f["layout"]["title"]["text"])


def table(at: AppTest, column: str) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if column in d.value.columns)


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def said(at: AppTest) -> str:
    """Everything the page says in Markdown-capable elements, as one string."""
    parts = [*at.title, *at.header, *at.subheader, *at.markdown, *at.caption, *at.info,
             *at.success, *at.warning, *at.error]
    return "\n\n".join(str(p.value) for p in parts)


# ----------------------------------------------------------------------------- Data

def test_data_section_profiles_the_built_in_sample(orders):
    at = start()
    assert at.title[0].value == "orders.csv"
    assert metrics(at) == {"Rows": "6,000", "Columns": "8", "Quality score": "100.0 / 100",
                           "Duplicate rows": "0", "Rows rejected on load": "0"}
    parts = figure(at, "Quality score components")["data"][1]
    assert parts["x"] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert [y.split(" (")[0] for y in parts["y"]] == [
        "Completeness", "Validity", "Uniqueness", "Consistency", "Outliers"]
    assert [s.value for s in at.success] == ["No issues detected."]
    columns = table(at, "Kind")
    assert list(columns["Column"]) == list(orders.columns)
    assert dict(zip(columns["Column"], columns["Kind"], strict=True)) == {
        "order_id": "id", "customer_id": "id", "order_date": "datetime", "region": "categorical",
        "channel": "categorical", "quantity": "numeric", "unit_price": "numeric",
        "revenue": "numeric"}
    assert columns.set_index("Column").loc["revenue", "Median"] == orders["revenue"].median()
    roles = table(at, "Role").set_index("Role")["Columns"]
    assert roles["Likely metrics"].startswith("revenue") and roles["Dates"] == "order_date"
    assert len(table(at, "order_id")) == 100                      # the preview


def messy_orders(orders: pd.DataFrame) -> bytes:
    df = orders.head(400).copy()
    df["channel"] = df["channel"].astype(object)
    df.loc[df.index[:40], "channel"] = None
    df["country"] = "US"
    lines = pd.concat([df, df.head(20)]).to_csv(index=False).splitlines()
    for i in range(5, 125, 10):
        lines[i] += ",EXTRA,FIELDS"                              # 12 rows with too many fields
    return ("\n".join(lines) + "\n").encode()


def test_upload_shows_rejections_issues_and_the_five_components(orders, tmp_path):
    at = start()
    at.radio(key="origin").set_value("Upload a file").run()
    assert at.title[0].value == "Choose a file to begin" and not at.metric
    data = messy_orders(orders)
    upload(at, "../../Q3 sales.csv", data)
    assert at.title[0].value == "Q3 sales.csv"                    # a display name, not a path

    path = tmp_path / "check.csv"
    path.write_bytes(data)
    ds = load_dataset(path)
    expected = profile_frame(ds.frame(), "check", ds.tables["check"].rejected_rows)
    ds.close()
    q = expected.quality
    assert expected.rejected_rows == 12 and q.score < 100
    shown = metrics(at)
    assert shown["Rows"] == f"{expected.n_rows:,}"
    assert shown["Rows rejected on load"] == "12"
    assert shown["Duplicate rows"] == f"{expected.n_duplicate_rows:,}"
    assert shown["Quality score"] == f"{q.score:.1f} / 100"
    parts = figure(at, "Quality score components")["data"][1]
    assert parts["x"] == pytest.approx([q.completeness, q.validity, q.uniqueness,
                                        q.consistency, q.outliers])
    issues = list(table(at, "Issue")["Issue"])
    assert issues == [i.replace("check", "check") for i in expected.issues]
    assert "12 malformed row(s) were rejected while loading" in issues
    assert "country: constant column" in issues and any("duplicate" in i for i in issues)
    assert not at.success


def test_unreadable_uploads_get_a_reason_not_a_stack_trace(monkeypatch):
    at = start()
    monkeypatch.setenv("ADS_MAX_FILE_MB", "0.001")
    upload(at, "big.csv", b"a,b\n" + b"1,2\n" * 2000)             # 8 kB against a 1 kB limit
    assert "Upload exceeds the" in at.error[0].value and not at.metric
    monkeypatch.delenv("ADS_MAX_FILE_MB")
    upload(at, "budget.xlsx", os.urandom(2048))
    assert not at.metric and len(at.error) == 1
    assert at.error[0].value.startswith("**This data cannot be used for that.**")
    assert "as an Excel workbook" in at.error[0].value
    assert any("Check the file" in c.value for c in at.caption)

    upload(at, "empty.csv", b"")
    assert "is empty" in at.error[0].value

    upload(at, "header_only.csv", b"a,b,c\n")
    assert metrics(at)["Rows"] == "0" and metrics(at)["Quality score"] == "n/a"
    assert "no rows" in at.info[0].value
    for section in ("Explore", "Ask", "Predict", "Forecast", "Report"):
        go(at, section)
        assert "no rows" in at.info[0].value and not at.get("plotly_chart")


# ----------------------------------------------------------------------------- Explore

def test_explore_charts_carry_the_computed_numbers(orders):
    at = go(start(), "Explore")
    assert [t.label for t in at.tabs] == ["Distributions", "Correlations", "Segments",
                                          "Time series", "Cohort retention"]
    assert len(figures(at)) == 5

    histogram = figure(at, "Distribution of revenue")["data"][0]
    assert sum(histogram["y"]) == 6000
    at.selectbox(key="dist_column").set_value("region").run()
    bars = figure(at, "Rows by region")["data"][0]
    counts = orders["region"].value_counts()
    assert bars["y"] == list(counts.index) and bars["x"] == list(counts)

    by_region = orders.groupby("region")["revenue"].sum().sort_values(ascending=False)
    segments = figure(at, "Total of revenue by region")["data"][0]
    assert segments["y"] == list(by_region.index)
    assert segments["x"] == pytest.approx(list(by_region))
    assert table(at, "Share of total")["Total"].sum() == pytest.approx(orders["revenue"].sum())
    at.selectbox(key="seg_by").set_value("channel")
    at.selectbox(key="seg_stat").set_value("share_of_total").run()
    share = figure(at, "Share of total of revenue by channel")["data"][0]
    assert sum(share["x"]) == pytest.approx(1.0)

    pairs = eda.correlation_pairs(orders, profile_frame(orders))
    heat = figure(at, "Correlation between numeric columns")["data"][0]
    cells = [v for row in heat["z"] for v in row if v is not None]
    assert sorted(cells) == pytest.approx(sorted(pairs["r"]))
    shown = table(at, "Column A")
    assert list(shown["r"]) == pytest.approx(list(pairs["r"]))
    assert "< 0.001" in set(shown["p-value"])
    at.segmented_control(key="corr_method").set_value("spearman").run()
    spearman = eda.correlation_pairs(orders, profile_frame(orders), "spearman")
    assert list(table(at, "Column A")["r"]) == pytest.approx(list(spearman["r"]))

    monthly = eda.to_series(orders, "order_date", "revenue", "M")
    line = figure(at, "revenue over time")["data"][0]
    assert line["y"] == pytest.approx(list(monthly)) and len(line["x"]) == 48
    assert metrics(at)["Periods"] == "48"
    assert metrics(at)["Seasonal autocorrelation, lag 12"].startswith("+0.")

    rates, sizes = eda.cohort_retention(orders, "customer_id", "order_date", "Q")
    cohort = figure(at, "Cohort retention")["data"][0]
    assert cohort["y"][0] == f"{rates.index[0]} (n = {sizes.iloc[0]:,})"
    assert cohort["z"][0][0] == pytest.approx(rates.iloc[0, 1])
    assert cohort["z"][-1] == [None] * (rates.shape[1] - 1)      # the last cohort has no future yet
    at.selectbox(key="cohort_freq").set_value("Y").run()
    assert len(figure(at, "Cohort retention")["data"][0]["y"]) == 4
    assert not at.exception


def test_explore_says_what_a_table_lacks_instead_of_failing():
    at = start()
    at.selectbox(key="sample").set_value("churn").run()
    go(at, "Explore")
    notes = [i.value for i in at.info]
    assert any("A time series needs a date column" in n and "0 date column(s)" in n for n in notes)
    assert any("Cohort retention needs an identifier column" in n for n in notes)
    assert len(figures(at)) == 3 and at.title[0].value == "churn.csv"


# ----------------------------------------------------------------------------- Ask

def run_sql(at: AppTest, sql: str) -> AppTest:
    at.text_area(key="ask_sql").set_value(sql)
    return press(at, "Run query")


def test_sql_goes_through_the_guard_and_refusals_give_their_reason(orders):
    at = go(start(), "Ask")
    at.segmented_control(key="ask_how").set_value("SQL").run()
    press(at, "Run query")                                        # the starter query
    assert at.code[0].language == "sql" and at.code[0].value.startswith("SELECT channel")
    result = table(at, "avg_revenue")
    expected = orders.groupby("channel")["revenue"].agg(["count", "mean"]).sort_values(
        "mean", ascending=False)
    assert list(result["channel"]) == list(expected.index)
    assert list(result["avg_revenue"]) == pytest.approx(list(expected["mean"]))
    assert "**What it does.** Reads from: orders." in said(at)
    assert any(c.value.startswith("3 row(s) in ") for c in at.caption)

    for sql, title, reason in [
        ("DROP TABLE orders", "The query was refused", "only SELECT queries are allowed"),
        ("SELECT * FROM read_csv('/etc/passwd')", "The query was refused",
         "table function read_csv() is not allowed"),
        ("SELECT 1; SELECT 2", "The query was refused", "exactly one SQL statement"),
        ("", "The query was refused", "the query is empty"),
    ]:
        run_sql(at, sql)
        assert at.warning[-1].value.lower().startswith(f"**{title}.**".lower())
        assert reason.lower() in at.warning[-1].value.lower()
        assert (at.code[0].value or "").strip() == sql             # the SQL is always shown
        assert any("Nothing was executed" in c.value for c in at.caption)
        assert not [d for d in at.dataframe if "avg_revenue" in d.value.columns]
    run_sql(at, "SELECT nope FROM orders")
    assert at.error[0].value.startswith("**The request could not be completed.** Query failed")
    run_sql(at, "SELECT count(*) AS n FROM orders")
    assert table(at, "n")["n"].tolist() == [6000]                 # the table is untouched
    assert "Tables and columns" in [e.label for e in at.expander]
    assert set(table(at, "Type")["Column"]) == set(orders.columns)


def test_long_results_are_cut_at_the_row_limit_and_say_so(orders, monkeypatch):
    monkeypatch.setenv("ADS_SQL_MAX_ROWS", "50")
    at = upload(start(), "limited.csv", csv(orders.head(300)))
    go(at, "Ask")
    at.segmented_control(key="ask_how").set_value("SQL").run()
    assert at.text_area(key="ask_sql").value == 'SELECT * FROM "limited" LIMIT 20'
    run_sql(at, "SELECT order_id, revenue FROM limited ORDER BY order_id")
    assert len(table(at, "revenue")) == 50
    assert any("Only the first 50 rows are shown." in c.value for c in at.caption)


def ask(at: AppTest, question: str) -> AppTest:
    at.text_input(key="ask_question").set_value(question)
    return press(at, "Translate and run")


def test_questions_show_their_sql_assumptions_and_refusals(orders):
    at = go(start(), "Ask")
    ask(at, "which region has the highest revenue")
    assert at.code[0].value == ('SELECT "region", SUM("revenue") AS "sum_revenue" FROM "orders" '
                                'GROUP BY "region" ORDER BY "sum_revenue" DESC LIMIT 1')
    assert "Ranked by the total of revenue" in said(at)
    answer = table(at, "sum_revenue")
    best = orders.groupby("region")["revenue"].sum().idxmax()
    assert answer["region"].tolist() == [best] == ["West"]

    ask(at, "total revenue by region; drop table orders")
    assert len(table(at, "sum_revenue")) == 4                     # the fragment is inert

    for question, reason in [("tell me a joke", "could not find an aggregation"),
                             ("total profit by region", "which numeric column"),
                             ("highest and lowest revenue by region", "both the highest and"),
                             ("", "the question is empty")]:
        ask(at, question)
        assert at.warning[-1].value.startswith("**The question was not translated.**")
        assert reason in at.warning[-1].value.lower()
        assert not at.code                                         # no SQL was produced
        assert any("No SQL was generated, so nothing ran." in c.value for c in at.caption)
        assert not [d for d in at.dataframe if "sum_revenue" in d.value.columns]


# ----------------------------------------------------------------------------- Predict

@pytest.fixture(scope="module")
def churn_model(churn):
    return train_supervised(churn, "churned")


def test_predict_shows_the_model_next_to_its_baseline(churn_model):
    m = churn_model
    at = start()
    at.selectbox(key="sample").set_value("churn").run()
    go(at, "Predict")
    assert "Choose the column to predict" in at.info[0].value and not figures(at)
    assert at.selectbox(key="predict_target").value == "churned"
    press(at, "Train and evaluate")

    auc, base = m.test_metrics["roc_auc"], m.baseline_test_metrics["roc_auc"]
    verdict = at.success[0].value
    assert verdict.startswith("**Signal found.** Logistic regression predicts `churned`")
    assert f"hold-out ROC AUC {auc:.3f} against {base:.3f} for a no-skill baseline" in verdict
    shown = {x.label: (x.value, x.delta) for x in at.metric}
    assert shown["Hold-out ROC AUC"] == (f"{auc:.3f}", f"{auc - base:+.3f} vs baseline")
    assert shown["Folds better than baseline"][0] == "5 of 5"
    assert shown["Training / hold-out rows"][0] == "2,400 / 600"
    assert shown["Lift in the top 10%"][0] == f"{m.lift_top_decile:.2f}x"

    cv = figure(at, "Cross-validated ROC AUC by candidate")
    selected = next(t for t in cv["data"] if t["name"] == "Selected")
    means = {c.model: c.cv_mean for c in m.candidates}
    assert selected["x"] == [pytest.approx(means["logistic_regression"])]
    assert selected["y"] == ["Logistic regression (selected)"]
    holdout = figure(at, "Hold-out scores")["data"]
    assert holdout[0]["x"][0] == pytest.approx(auc) and holdout[1]["x"][0] == pytest.approx(base)
    importance = figure(at, "Permutation importance for churned")["data"][0]
    assert importance["y"] == [f for f, _ in m.importance]
    assert importance["x"] == pytest.approx([v for _, v in m.importance])

    assert "No column was flagged" in at.success[1].value
    left_out = table(at, "Column left out")
    assert left_out["Column left out"].tolist() == ["customer_id"]
    assert "Feature importance shows what the model relies on" in said(at)


def test_predict_flags_leakage_and_keeping_a_column_is_explicit(churn):
    leaky = churn.assign(cancel_fee=np.where(churn.churned == 1, 25.0, 0.0)
                         + np.random.default_rng(1).normal(0, 0.01, len(churn)))
    at = upload(start(), "leaky.csv", csv(leaky))
    go(at, "Predict")
    at.selectbox(key="predict_target").set_value("churned")
    press(at, "Train and evaluate")
    flagged = table(at, "Why it was flagged")
    assert flagged["Column"].tolist() == ["cancel_fee"]
    assert flagged["Used anyway"].tolist() == ["no"]
    assert "near-perfect predictor" in flagged["Why it was flagged"].iloc[0]
    honest = float(next(x for x in at.metric if x.label == "Hold-out ROC AUC").value)
    assert honest < 0.9

    at.multiselect(key="predict_keep").set_value(["cancel_fee"]).run()
    assert not at.exception
    assert float(next(x for x in at.metric if x.label == "Hold-out ROC AUC").value) > 0.99
    assert table(at, "Why it was flagged")["Used anyway"].tolist() == ["yes"]
    assert "cancel_fee was kept on request despite a leakage warning" in said(at)

    go(at, "Report")                                               # the report keeps it too
    press(at, "Build report")
    report = at.markdown[-1].value
    assert "cancel_fee was kept on request despite a leakage warning" in report
    assert "hold-out roc_auc 1.000" in report or "hold-out roc_auc 0.99" in report


def test_predict_says_when_there_is_nothing_to_find_and_when_it_cannot_run(orders):
    rng = np.random.default_rng(0)
    noise = pd.DataFrame({"a": rng.normal(size=1500), "b": rng.normal(size=1500),
                          "c": rng.choice(list("xyz"), 1500), "flag": rng.integers(0, 2, 1500),
                          "country": "US"})
    at = upload(start(), "noise.csv", csv(noise))
    go(at, "Predict")
    at.selectbox(key="predict_target").set_value("flag")
    press(at, "Train and evaluate")
    assert at.warning[-1].value.startswith("**No usable signal.**")
    assert not any("Signal found" in x.value for x in at.success)
    assert "Do not deploy a model for `flag`" in at.warning[-1].value
    assert "No candidate beat the no-skill baseline consistently" in said(at)

    at.selectbox(key="predict_target").set_value("country")       # a single class
    press(at, "Train and evaluate")
    assert at.error[0].value.startswith("**This data cannot be used for that.**")
    assert "single class" in at.error[0].value and not figures(at)


# ----------------------------------------------------------------------------- Forecast

def test_forecast_reports_the_chosen_method_range_and_backtests(orders):
    f = forecast_series(orders, "order_date", "revenue", 6, "M")
    at = go(start(), "Forecast")
    assert "Choose a date and a value" in at.info[0].value
    press(at, "Forecast")
    verdict = (at.success or at.warning)[0].value
    assert ("earned its place" in verdict) == f.improves_on_naive
    assert f.best_model.replace("_", " ") in verdict.lower()
    assert metrics(at)["Backtest windows"] == f"{f.n_backtests} x 6 periods"
    shown = table(at, "Forecast")
    assert shown["Period"].tolist() == [row[0] for row in f.forecast]
    assert shown["Forecast"].tolist() == pytest.approx([row[1] for row in f.forecast])
    assert shown["Low"].tolist() == pytest.approx([row[2] for row in f.forecast])
    backtests = table(at, "Backtest error (MAE)")
    assert backtests["Backtest error (MAE)"].tolist() == pytest.approx(
        list(f.backtest_mae.values()))
    chart = figure(at, "revenue: history and 6-period forecast")
    drawn = next(t for t in chart["data"] if t["name"] == "Forecast")
    assert drawn["y"] == pytest.approx([row[1] for row in f.forecast])
    assert len(next(t for t in chart["data"] if t["name"] == "History")["y"]) == 48
    assert "Forecasts assume the historical pattern continues" in said(at)

    at.number_input(key="forecast_horizon").set_value(40)
    press(at, "Forecast")
    assert at.error[0].value.startswith("**This data cannot be used for that.**")
    assert "are needed to backtest" in at.error[0].value and not figures(at)
    go(at, "Report")                                               # a failed run is not offered
    assert [(c.label, c.disabled) for c in at.checkbox][1] == ("Forecast (run Forecast first)",
                                                               True)


def test_the_report_uses_the_period_chosen_for_the_forecast(orders):
    quarterly = forecast_series(orders, "order_date", "revenue", 2, "Q")
    at = go(start(), "Forecast")
    assert at.selectbox(key="forecast_freq").value == "M"          # suggested from the date span
    at.selectbox(key="forecast_freq").set_value("Q")
    at.number_input(key="forecast_horizon").set_value(2)
    press(at, "Forecast")
    assert table(at, "Forecast")["Forecast"].tolist() == pytest.approx(
        [row[1] for row in quarterly.forecast])
    go(at, "Report")
    assert [(c.label, c.value) for c in at.checkbox][1] == ("Forecast of revenue", True)
    press(at, "Build report")
    report = at.markdown[-1].value
    assert "16 historical periods, " in report and "2 steps" in report
    assert f"| {quarterly.forecast[0][0]} |" in report


# ----------------------------------------------------------------------------- Report

def test_report_is_rendered_and_downloads_hold_the_same_files(monkeypatch, tmp_path):
    sent = {}
    original = DeltaGenerator.download_button

    def record(self, label, data, *args, **kwargs):
        sent[label] = (data, kwargs.get("file_name"), kwargs.get("mime"))
        return original(self, label, data, *args, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "download_button", record)
    sql = "SELECT plan, avg(churned) AS churn_rate FROM churn GROUP BY plan ORDER BY 2 DESC"
    at = start()
    at.selectbox(key="sample").set_value("churn").run()
    go(at, "Report")
    assert "Select Build report" in at.info[0].value and not at.download_button
    assert [c.disabled for c in at.checkbox] == [True, True, True]

    go(at, "Ask")
    at.segmented_control(key="ask_how").set_value("SQL").run()
    run_sql(at, sql)
    go(at, "Predict")
    press(at, "Train and evaluate")
    go(at, "Report")
    assert [(c.label, c.value) for c in at.checkbox] == [
        ("Model for churned", True), ("Forecast (run Forecast first)", False),
        ("Last query", True)]
    at.text_area(key="report_problem").set_value("Which customers are likely to churn?")
    press(at, "Build report")

    path = write_sample("churn", tmp_path)
    expected = analyze(path, table="churn", source_name="churn.csv", sql=sql, target="churned",
                       business_problem="Which customers are likely to churn?")
    bundle = bundle_artifacts(expected, "examples/churn.csv")
    assert [b.label for b in at.download_button] == [
        "All files (.zip)", "report.md", "analysis.json", "queries.sql", "reproduce.sh"]
    assert sent["All files (.zip)"] == (bundle, "report.zip", "application/zip")
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        files = {n.split("/", 1)[1]: z.read(n).decode() for n in z.namelist()}
    for name, text in files.items():
        assert sent[name][0] == text and sent[name][1] == name
    assert at.markdown[-1].value == files["report.md"].strip()    # the page shows the same report
    report = files["report.md"]
    assert report.startswith("# Analysis report: churn.csv")
    assert "Which customers are likely to churn?" in report
    assert "### Model for `churned` (classification)" in report and "### Query" in report
    assert files["reproduce.sh"].splitlines()[-1] == (
        f"ads report examples/churn.csv --out report --table churn --sql {shlex.quote(sql)} "
        "--target churned --horizon 6 --seed 0 --problem 'Which customers are likely to churn?'")
    assert json.loads(files["analysis.json"])["model"]["best_model"] == "logistic_regression"


# ----------------------------------------------------------------------------- other tables

def test_workbook_with_two_sheets_offers_a_table_choice(orders, churn):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        orders.head(200).to_excel(writer, sheet_name="Orders", index=False)
        churn.head(50).to_excel(writer, sheet_name="Churn sample", index=False)
    at = upload(start(), "book.xlsx", buffer.getvalue())
    picker = at.sidebar.selectbox(key="table")
    assert picker.options == ["churn_sample (50 rows)", "orders (200 rows)"]
    assert picker.value == "orders" and metrics(at)["Rows"] == "200"     # the largest first
    go(at, "Forecast")
    press(at, "Forecast")
    assert "forecast" in at.session_state
    at.sidebar.selectbox(key="table").set_value("churn_sample").run()
    assert not at.exception and "forecast" not in at.session_state       # a new table, new slate
    assert "A forecast needs a date column" in at.info[0].value
    go(at, "Data")
    assert metrics(at)["Rows"] == "50" and "churn_sample" in at.caption[0].value

    # Another workbook with other sheet names: the earlier table choice must not linger.
    second = io.BytesIO()
    with pd.ExcelWriter(second, engine="openpyxl") as writer:
        orders.head(30).to_excel(writer, sheet_name="North", index=False)
        orders.head(80).to_excel(writer, sheet_name="South", index=False)
    at.file_uploader(key="upload").upload("regions.xlsx", second.getvalue()).run()
    assert not at.exception
    assert at.sidebar.selectbox(key="table").value == "south" and metrics(at)["Rows"] == "80"


def test_tables_with_little_to_analyse_explain_what_is_missing():
    notes = pd.DataFrame({"note_id": range(1, 61),
                          "comment": [f"free text number {i} with no pattern" for i in range(60)]})
    at = upload(start(), "notes.csv", csv(notes))
    assert metrics(at)["Rows"] == "60"
    go(at, "Explore")
    shown = " ".join(i.value for i in at.info)
    for expected in ("No numeric or categorical column to plot",
                     "Correlations need at least two numeric columns",
                     "Segments need one categorical column and one numeric column",
                     "A time series needs a date column", "Cohort retention needs"):
        assert expected in shown
    assert not figures(at)
    go(at, "Predict")
    assert "No column can serve as a target" in at.info[0].value
    go(at, "Forecast")
    assert "A forecast needs a date column" in at.info[0].value
    go(at, "Report")
    press(at, "Build report")                                       # still a valid, short report
    assert at.markdown[-1].value.startswith("# Analysis report: notes.csv")

    brief = pd.DataFrame({"user_id": [1, 2, 3, 4] * 15, "day": ["2024-03-04", "2024-03-05"] * 30,
                          "amount": np.random.default_rng(0).gamma(2, 30, 60).round(2),
                          "tier": ["a", "b", "c"] * 20})
    at = go(upload(start(), "brief.csv", csv(brief)), "Explore")
    reasons = " ".join(e.value for e in at.error)
    assert "Only 2 period(s) at frequency D; need at least 3" in reasons      # time series
    assert "Retention needs activity in at least two periods" in reasons      # cohorts
    assert "Correlations need at least two numeric columns" in at.info[0].value
    assert len(figures(at)) == 2 and not at.exception               # distribution and segments


# ----------------------------------------------------------------------------- page-wide

def test_a_new_table_starts_from_a_clean_slate():
    at = go(start(), "Forecast")
    press(at, "Forecast")
    assert figures(at) and "forecast" in at.session_state
    at.selectbox(key="sample").set_value("churn").run()
    assert not at.exception and "forecast" not in at.session_state
    assert "A forecast needs a date column" in at.info[0].value
    at.selectbox(key="sample").set_value("orders").run()
    assert "Choose a date and a value" in at.info[0].value and not figures(at)


def test_sidebar_says_where_the_server_listens():
    local = start()
    assert not local.sidebar.warning
    assert any("bound to localhost" in c.value and "no sign-in" in c.value
               for c in local.sidebar.caption)
    for address in ("0.0.0.0", "192.168.1.20", ""):    # every interface, a LAN address, unset
        config.set_option("server.address", address)
        exposed = start()
        assert any("not bound to localhost" in w.value for w in exposed.sidebar.warning)
        assert not any("bound to localhost. There" in c.value for c in exposed.sidebar.caption)


def test_hostile_names_and_values_never_become_markup_on_the_page():
    image, link = "![logo](http://attacker.invalid/p.png)", "[click](http://attacker.invalid)"
    script = "<img src=x onerror=alert(1)>"
    rng = np.random.default_rng(0)
    n = 300
    outcome = rng.integers(0, 2, n)
    df = pd.DataFrame({
        image: rng.normal(100, 10, n), link: rng.normal(size=n),
        script: rng.choice([link, image, "*bold* `tick`"], n),
        "when": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        "leak " + link: outcome * 3.0 + rng.normal(0, 0.001, n), "outcome": outcome,
    })
    at = upload(start(), f"{link}.csv", csv(df))
    pages = [said(at)]
    go(at, "Explore")
    pages.append(said(at))
    go(at, "Ask")
    ask(at, f"total {image} by {script}")
    pages.append(said(at))
    at.segmented_control(key="ask_how").set_value("SQL").run()
    name = table(at, "Type")["Table"].iloc[0]
    assert name == "attacker_invalid"                              # from the sanitised file name
    run_sql(at, f'SELECT "{script}", count(*) AS n FROM {name} GROUP BY 1')
    assert len(table(at, "n")) == 3
    pages.append(said(at))
    go(at, "Predict")
    at.selectbox(key="predict_target").set_value("outcome")
    press(at, "Train and evaluate")
    assert table(at, "Why it was flagged")["Column"].tolist() == ["leak " + link]
    pages.append(said(at))
    go(at, "Report")
    press(at, "Build report")
    pages.append(said(at))
    for page in pages:
        tokens = [t for block in PARSER.parse(page) for t in (block, *(block.children or []))]
        active = [(t.type, t.content[:80], dict(t.attrs or {})) for t in tokens
                  if t.type in ("image", "link_open", "html_inline", "html_block")]
        assert not active, (active, pages.index(page))
    assert "attacker.invalid" in pages[-1]                         # shown, as text
    for fig in figures(go(at, "Explore")):
        text = json.dumps(fig)
        assert "<a " not in text and "<img" not in text
