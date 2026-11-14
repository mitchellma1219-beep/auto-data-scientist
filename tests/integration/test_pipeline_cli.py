import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from auto_data_scientist import analyze, render_report, write_artifacts
from auto_data_scientist.cli import main

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ["## Executive summary", "## Business problem", "## Data overview", "## Key findings",
            "## Supporting evidence", "## Statistical limitations", "## Recommended actions",
            "## Potential business impact"]


def test_full_report_on_orders(orders, write, tmp_path):
    path = write(orders)
    a = analyze(path, question="total revenue by region", date_col="order_date",
                value_col="revenue", business_problem="Where does revenue come from?")
    text = render_report(a)
    assert [s for s in SECTIONS if s in text] == SECTIONS
    assert text.index(SECTIONS[0]) < text.index(SECTIONS[3]) < text.index(SECTIONS[7])
    west = orders[orders.region == "West"]["revenue"].sum() / orders["revenue"].sum()
    assert f"`West` accounts for {west:.1%} of total `revenue`" in text
    assert "Where does revenue come from?" in text and 'SUM("revenue")' in text
    assert "No monetary impact is estimated" in text and "not evidence of cause" in text
    f = a.forecast
    assert f is not None and f.best_model in f.backtest_mae and len(f.forecast) == 6
    assert f.improves_on_naive == (f.best_model == "holt_winters")
    assert a.trend is not None and a.trend.seasonal_autocorr > 0.3
    assert "$" not in text and "statistically significant" not in text.lower()

    paths = write_artifacts(a, tmp_path / "out", path)
    data = json.loads(paths["data"].read_text())
    assert data["sha256"] == a.sha256 and data["profile"]["n_rows"] == 6000
    assert data["forecast"]["horizon"] == 6 and data["query"]["columns"][0] == "region"
    assert paths["sql"].read_text().startswith('SELECT "region"')
    cmd = paths["reproduce"].read_text().splitlines()[-1]
    assert cmd.startswith("ads report ") and "--question 'total revenue by region'" in cmd
    assert paths["report"].read_text() == text


def test_churn_report_and_determinism(churn, write):
    path = write(churn, name="churn")
    a = analyze(path, target="churned", seed=3)
    b = analyze(path, target="churned", seed=3)
    text = render_report(a)
    assert text == render_report(b)
    assert "### Model for `churned` (classification)" in text and "no-skill baseline" in text
    assert "times the average rate of the positive `churned` class" in text
    assert "Pilot the `churned` model" in text and "controlled experiment" in text


def test_sql_takes_precedence_and_optional_steps_are_noted(orders, write):
    a = analyze(write(orders.drop(columns=["order_date"])), sql="SELECT count(*) AS n FROM orders")
    assert a.query.rows == [(6000,)] and a.trend is None and a.translation is None
    assert "### Query" in render_report(a)


def test_cli_commands(orders, churn, write, tmp_path, capsys):
    o, c = str(write(orders)), str(write(churn, name="churn"))
    assert main(["profile", o]) == 0
    assert "Quality score: 100.0/100" in capsys.readouterr().out
    assert main(["profile", o, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_rows"] == 6000
    assert main(["sql", o, "--query", "SELECT region, count(*) n FROM orders GROUP BY 1",
                 "--format", "json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["rows"]) == 4
    assert main(["ask", o, "top 2 regions by total revenue"]) == 0
    assert "LIMIT 2" in capsys.readouterr().out
    assert main(["model", c, "--target", "churned"]) == 0
    assert json.loads(capsys.readouterr().out)["improves_on_baseline"] is True
    assert main(["forecast", o, "--date-col", "order_date", "--value-col", "revenue"]) == 0
    assert len(json.loads(capsys.readouterr().out)["forecast"]) == 6
    out = tmp_path / "rep"
    assert main(["report", o, "--out", str(out), "--question", "average revenue by channel"]) == 0
    assert {p.name for p in out.iterdir()} == {"report.md", "analysis.json", "queries.sql",
                                               "reproduce.sh"}
    assert main(["report", c, "--target", "churned"]) == 0
    assert "## Executive summary" in capsys.readouterr().out


@pytest.mark.parametrize("args, message", [
    (["profile", "missing.csv"], "file not found"),
    (["sql", "{o}", "--query", "DROP TABLE orders"], "only SELECT queries are allowed"),
    (["sql", "{o}", "--query", "SELECT * FROM read_csv('/etc/passwd')"], "not allowed"),
    (["sql", "{o}", "--query", "SELECT nope FROM orders"], "query failed"),
    (["ask", "{o}", "show me everything"], "could not find an aggregation"),
    (["model", "{o}", "--target", "nope"], "unknown target"),
    (["forecast", "{o}", "--date-col", "order_date", "--value-col", "nope"], "unknown column"),
    (["forecast", "{o}", "--date-col", "order_date", "--value-col", "revenue", "--horizon",
      "500"], "horizon"),
    (["profile", "{o}", "--table", "nope"], "unknown table"),
])
def test_cli_errors_exit_1_with_reason(orders, write, capsys, args, message):
    o = str(write(orders))
    assert main([a.replace("{o}", o) for a in args]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err and "Traceback" not in err


def test_skill_script_runs_from_checkout_without_install(orders, write, tmp_path):
    script = ROOT / "skills" / "auto-data-scientist" / "scripts" / "ads.py"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": ""}
    path = str(write(orders))
    res = subprocess.run([sys.executable, str(script), "profile", path], capture_output=True,
                         text=True, env=env, cwd=tmp_path, check=False)
    assert res.returncode == 0, res.stderr
    assert "6,000 rows, 8 columns" in res.stdout
    res = subprocess.run([sys.executable, str(script), "sql", path, "--query",
                          "DELETE FROM orders"], capture_output=True, text=True, env=env,
                         cwd=tmp_path, check=False)
    assert res.returncode == 1 and "only SELECT queries are allowed" in res.stderr


def test_skill_frontmatter_is_valid():
    folder = ROOT / "skills" / "auto-data-scientist"
    _, fm, body = (folder / "SKILL.md").read_text().split("---", 2)
    meta = yaml.safe_load(fm)
    assert meta["name"] == folder.name and 0 < len(meta["description"]) <= 1024
    assert len(body.splitlines()) < 500 and (folder / "references" / "methodology.md").exists()
