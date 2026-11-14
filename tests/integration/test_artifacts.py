"""Artefacts: one zip, a reproduce command that really regenerates the report, safe names."""

import io
import json
import shlex
import zipfile

import pytest

from auto_data_scientist import analyze, bundle_artifacts, render_report, write_artifacts
from auto_data_scientist.cli import main
from auto_data_scientist.errors import DataError
from auto_data_scientist.pipeline import ARTIFACTS, artifact_texts


@pytest.fixture(scope="module")
def analysis(orders, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "orders.csv"
    orders.to_csv(path, index=False)
    a = analyze(path, question="total revenue by region", date_col="order_date",
                value_col="revenue", business_problem="Where does revenue come from?")
    return a, path


def test_bundle_holds_the_four_artefacts(analysis):
    a, _ = analysis
    blob = bundle_artifacts(a, "orders.csv")
    assert blob == bundle_artifacts(a, "orders.csv")                 # fixed timestamps
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert sorted(names) == sorted(f"report/{n}" for n in ARTIFACTS.values())
        texts = {n.split("/", 1)[1]: z.read(n).decode("utf-8") for n in names}
        modes = {i.filename: i.external_attr >> 16 for i in z.infolist()}
    assert texts == artifact_texts(a, "report", "orders.csv")
    assert texts["report.md"] == render_report(a)
    assert json.loads(texts["analysis.json"])["sha256"] == a.sha256
    assert texts["queries.sql"].startswith('SELECT "region"')
    assert texts["reproduce.sh"].splitlines()[-1].startswith("ads report orders.csv --out report ")
    assert modes["report/reproduce.sh"] == 0o755 and modes["report/report.md"] == 0o644


def test_files_on_disk_equal_the_bundle(analysis, tmp_path):
    a, path = analysis
    paths = write_artifacts(a, tmp_path / "out", path)
    texts = artifact_texts(a, tmp_path / "out", path)
    assert {p.name: p.read_text() for p in paths.values()} == texts
    assert set(paths) == set(ARTIFACTS)


def test_the_reproduce_command_regenerates_the_same_report(analysis, tmp_path, capsys):
    a, path = analysis
    first = write_artifacts(a, tmp_path / "first", path)
    command = shlex.split(first["reproduce"].read_text().splitlines()[-1])
    assert command[:2] == ["ads", "report"]
    assert "--problem" in command and "Where does revenue come from?" in command
    again = tmp_path / "again"
    command[command.index("--out") + 1] = str(again)
    assert main(command[1:]) == 0
    capsys.readouterr()
    assert (again / "report.md").read_text() == first["report"].read_text()
    assert (again / "queries.sql").read_text() == first["sql"].read_text()
    assert json.loads((again / "analysis.json").read_text())["forecast"] == \
        json.loads(first["data"].read_text())["forecast"]


def test_uploads_are_reported_under_a_safe_display_name(orders, write):
    path = write(orders, name="3f2a9c")
    a = analyze(path, source_name="../Q3 sales](http://e.invalid).csv")
    assert a.source == "e.invalid_.csv" and a.table == "e_invalid"   # named after the upload
    # The name is sanitised first, then escaped like any other text from outside.
    assert render_report(a).startswith("# Analysis report: e.invalid\\_.csv\n")
    assert "Source `e.invalid_.csv` (sha256" in render_report(a)
    plain = analyze(path)
    assert plain.source == "3f2a9c.csv" and plain.table == "t_3f2a9c"


@pytest.mark.parametrize("kwargs", [
    {"date_col": "nope", "value_col": "revenue"}, {"date_col": "nope"},
    {"target": "region", "time_col": "nope"},
])
def test_unknown_requested_columns_are_refused_not_crashed_on(orders, write, kwargs, capsys):
    path = write(orders)
    with pytest.raises(DataError, match="unknown column 'nope'"):
        analyze(path, **kwargs)
    assert main(["report", str(path), "--date-col", "nope", "--value-col", "revenue"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: unknown column 'nope'") and "Traceback" not in err


def test_kept_columns_and_the_period_are_part_of_the_reproducible_request(churn, orders, write,
                                                                          tmp_path, capsys):
    import numpy as np

    leaky = churn.assign(cancel_fee=np.where(churn.churned == 1, 25.0, 0.0)
                         + np.random.default_rng(1).normal(0, 0.01, len(churn)))
    path = write(leaky, name="churn")
    honest = analyze(path, target="churned")
    kept = analyze(path, target="churned", keep=["cancel_fee"])
    assert honest.model.test_metrics["roc_auc"] < 0.9 < kept.model.test_metrics["roc_auc"]
    text = render_report(kept)
    assert "cancel_fee was kept on request despite a leakage warning" in text
    command = artifact_texts(kept, "report", path)["reproduce.sh"].splitlines()[-1]
    assert "--keep cancel_fee" in command
    assert "--keep" not in artifact_texts(honest, "report", path)["reproduce.sh"]
    assert main(["report", str(path), "--target", "churned", "--keep", "cancel_fee"]) == 0
    assert capsys.readouterr().out == text + "\n"

    o = write(orders)
    monthly = analyze(o, date_col="order_date", value_col="revenue")
    quarterly = analyze(o, date_col="order_date", value_col="revenue", freq="Q")
    assert (monthly.forecast.freq, monthly.trend.freq) == ("M", "M")
    assert (quarterly.forecast.freq, quarterly.trend.freq) == ("Q", "Q")
    assert len(quarterly.trend.series) == 16 and quarterly.forecast.n_periods == 16
    command = artifact_texts(quarterly, "report", o)["reproduce.sh"].splitlines()[-1]
    assert "--freq Q" in command and "--freq" not in artifact_texts(monthly, "r", o)["reproduce.sh"]
    assert main(["report", str(o), "--date-col", "order_date", "--value-col", "revenue",
                 "--freq", "Q"]) == 0
    assert capsys.readouterr().out == render_report(quarterly) + "\n"
    with pytest.raises(ValueError, match="freq must be one of"):
        analyze(o, date_col="order_date", value_col="revenue", freq="hourly")
