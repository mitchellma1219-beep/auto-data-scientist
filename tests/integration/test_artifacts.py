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
    assert a.source == "e.invalid_.csv"
    # The name is sanitised first, then escaped like any other text from outside.
    assert render_report(a).startswith("# Analysis report: e.invalid\\_.csv\n")
    assert "Source `e.invalid_.csv` (sha256" in render_report(a)
    assert analyze(path).source == "3f2a9c.csv"


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
