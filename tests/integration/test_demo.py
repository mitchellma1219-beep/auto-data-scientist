"""The demo generator: it runs offline, and the committed demo matches what it produces."""

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

if os.environ.get("ADS_REQUIRE_UI") != "1":      # CI sets it: there a missing extra must fail
    pytest.importorskip("plotly")

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
CHARTS = ["quality", "distribution", "segments", "correlation", "forecast", "retention",
          "model_comparison", "holdout", "importance"]
REPORT_FILES = ["analysis.json", "queries.sql", "report.md", "reproduce.sh"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_demo", DEMO / "make_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("demo")
    return out, module.build(out)


def headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("#")]


def test_demo_writes_two_reports_and_one_page_per_chart(built):
    out, written = built
    expected = {f"{key}/{name}" for key in ("orders", "churn") for name in REPORT_FILES}
    expected |= {f"charts/{name}.html" for name in [*CHARTS, "index"]} | {"charts/plotly.min.js"}
    assert {str(p.relative_to(out)) for p in written} == expected
    assert all(p.stat().st_size > 0 for p in written)
    churn = (out / "churn" / "report.md").read_text()
    orders = (out / "orders" / "report.md").read_text()
    assert "### Model for `churned` (classification)" in churn and "no-skill baseline" in churn
    assert "### Forecast of `revenue`" in orders and "### Question: top 2 regions" in orders
    assert (out / "churn" / "reproduce.sh").read_text().splitlines()[2].startswith(
        "ads report examples/churn.csv --out demo/churn --sql ")
    assert "ads report examples/orders.csv --out demo/orders --question" in (
        out / "orders" / "reproduce.sh").read_text()


def test_demo_pages_work_offline_and_carry_their_numbers(built):
    out, _ = built
    library = (out / "charts" / "plotly.min.js").read_text()
    assert library.lstrip().startswith("/**") and "Licensed under the MIT license" in library[:300]
    index = (out / "charts" / "index.html").read_text()
    for name in CHARTS:
        page = (out / "charts" / f"{name}.html").read_text()
        assert page.startswith("<!DOCTYPE html>")
        assert "<script src='plotly.min.js'></script>" in page
        assert f'id="chart-{name}"' in page and f'id="chart-{name}"' in index
        assert "<details><summary>Data table</summary><table>" in page      # the table view
        assert page.count("<html") == 1 and "<a href='index.html'>All charts</a>" in page
        # Nothing is loaded from another host: no CDN, no remote script, style, image or font.
        assert not re.search(r"""(?:src|href)\s*=\s*['"](?:https?:)?//""", page)
        assert "cdn." not in page and "@import" not in page
    assert index.count("class=\"plotly-graph-div\"") == len(CHARTS)
    assert len(set(re.findall(r"id=['\"]([^'\"]+)['\"]", index))) == len(
        re.findall(r"id=['\"]([^'\"]+)['\"]", index))                       # ids are unique
    segments = (out / "charts" / "segments.html").read_text()
    assert "<td>West</td>" in segments and "Total of revenue by region" in segments


def test_committed_demo_is_what_the_generator_writes(built):
    out, written = built
    committed = sorted(str(p.relative_to(DEMO)) for p in DEMO.rglob("*") if p.is_file()
                       and p.parent.name in ("orders", "churn", "charts"))
    assert committed == sorted(str(p.relative_to(out)) for p in written)
    same_versions = True
    for key in ("orders", "churn"):
        old = json.loads((DEMO / key / "analysis.json").read_text())
        new = json.loads((out / key / "analysis.json").read_text())
        assert old["sha256"] == new["sha256"] and old["params"] == new["params"]
        assert headings((DEMO / key / "report.md").read_text()) == headings(
            (out / key / "report.md").read_text())
        same_versions &= old["versions"] == new["versions"]
    if same_versions:      # same library versions as when the demo was committed: same bytes
        for path in written:
            name = path.relative_to(out)
            assert (DEMO / name).read_bytes() == path.read_bytes(), name
