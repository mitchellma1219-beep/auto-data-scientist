"""Text from the data must stay text, whatever a Markdown renderer would make of it.

The report is parsed with a CommonMark parser (GitHub-style tables, strike-through and
automatic links switched on) and the result is searched for links, images and HTML.
"""

import html

import numpy as np
import pandas as pd
import pytest
from markdown_it import MarkdownIt

from auto_data_scientist import analyze, render_report
from auto_data_scientist.cli import main
from auto_data_scientist.ingest import quote_ident
from auto_data_scientist.report import code_block, code_span, escape_md, md_table

IMAGE = "![logo](http://attacker.invalid/pixel.png)"
LINK = "[click here](http://attacker.invalid/login)"
HTML = "<img src=x onerror=alert(1)>"
STYLED = "`rm -rf` *bold* _em_ __strong__ ~~gone~~"
URLS = "http://attacker.invalid/auto www.attacker.invalid user@attacker.invalid"
FENCE = "```\n# Fake heading\n```"
PIPES = "a | b | c"
MATH = "$x^2$ &amp; <b>x</b> <http://attacker.invalid>"
HOSTILE = [IMAGE, LINK, HTML, STYLED, URLS, FENCE, PIPES, MATH, "[ref]: http://attacker.invalid",
           "back\\slash \\[x\\]", "` leading tick", "trailing tick `", "", "   "]
ACTIVE = {"image", "link_open", "html_inline", "html_block"}

PARSER = MarkdownIt("gfm-like")


def flat(text: str) -> list:
    out = []
    for token in PARSER.parse(text):
        out.append(token)
        out.extend(token.children or [])
    return out


def kinds(text: str) -> set[str]:
    return {t.type for t in flat(text)}


@pytest.mark.parametrize("value", HOSTILE)
def test_escaped_text_and_code_spans_are_inert(value):
    plain = " ".join(value.split())
    escaped = flat(f"Value {escape_md(value)} end")
    assert not {t.type for t in escaped} & (ACTIVE | {"em_open", "strong_open", "s_open",
                                                     "code_inline", "heading_open", "fence"})
    shown = "".join(t.content for t in escaped if t.type == "text")
    assert shown.split() == f"Value {plain} end".split()       # nothing lost, nothing added
    spanned = flat(f"Value {code_span(value)} end")
    assert not {t.type for t in spanned} & (ACTIVE | {"em_open", "strong_open", "s_open"})
    code = [t.content for t in spanned if t.type == "code_inline"]
    assert code == [plain] if plain else len(code) <= 1


def test_code_blocks_cannot_be_closed_from_inside():
    sql = "SELECT 1 /*\n```\n![x](http://attacker.invalid/q.png)\n````\n*/"
    tokens = flat("\n".join(["Before", "", *code_block(sql, "sql"), "", "After"]))
    fences = [t for t in tokens if t.type == "fence"]
    assert len(fences) == 1 and fences[0].content == sql + "\n" and fences[0].info == "sql"
    assert not {t.type for t in tokens} & ACTIVE
    assert code_block("SELECT 1", "sql") == ["```sql", "SELECT 1", "```"]


def test_tables_keep_their_shape_with_hostile_cells():
    df = pd.DataFrame({PIPES: [IMAGE, "x\ny", HTML], LINK: [1.5, 2.0, float("nan")]},
                      index=pd.Index([FENCE, "b", "c"], name=STYLED))
    tokens = flat(md_table(df))
    assert not {t.type for t in tokens} & ACTIVE
    cells = [t.type for t in tokens if t.type in ("th_open", "td_open")]
    assert cells.count("th_open") == 3 and cells.count("td_open") == 9      # 3 x 3, intact


@pytest.fixture(scope="module")
def hostile(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 400
    outcome = (rng.random(n) < 0.4).astype(int)
    df = pd.DataFrame({
        IMAGE: rng.normal(100, 10, n),
        LINK: rng.normal(size=n),
        HTML: rng.choice([LINK, IMAGE, STYLED, PIPES], n),
        FENCE: rng.choice([URLS, MATH], n),
        STYLED: pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        URLS: outcome,
        "leak " + LINK: outcome * 5.0 + rng.normal(0, 0.001, n),
        "constant " + PIPES: "same",
    })
    path = tmp_path_factory.mktemp("hostile") / "t.parquet"
    df.to_parquet(path, index=False)
    # The upload's name is hostile too. It is sanitised to "pixel.png_.parquet", and the table
    # is named after it.
    sql = (f"SELECT {quote_ident(HTML)} AS seg, count(*) AS n FROM pixel_png "
           f"/* ``` {IMAGE} ``` */ GROUP BY 1 ORDER BY 2 DESC")
    analysis = analyze(path, sql=sql, target=URLS, date_col=STYLED, value_col=IMAGE, horizon=3,
                       business_problem=f"Why? {LINK} {IMAGE}",
                       source_name=f"../{IMAGE}.parquet")
    assert analysis.source == "pixel.png_.parquet" and analysis.table == "pixel_png"
    return path, df, analysis


def test_report_from_hostile_data_has_no_links_images_or_html(hostile):
    _, df, analysis = hostile
    assert analysis.model is not None and analysis.forecast is not None
    assert {f.column for f in analysis.model.leakage} == {"leak " + LINK}
    text = render_report(analysis)
    tokens = flat(text)
    assert not {t.type for t in tokens} & ACTIVE
    page = PARSER.render(text)
    assert "href=" not in page and "<img" not in page and "onerror=alert(1)>" not in page
    # The planted heading stayed inside its table cell or code span.
    headings = [tokens[i + 1].content for i, t in enumerate(tokens)
                if t.type == "heading_open" and t.tag == "h2"]
    assert headings == ["Executive summary", "Business problem", "Data overview", "Key findings",
                        "Supporting evidence", "Statistical limitations", "Recommended actions",
                        "Potential business impact"]
    for i, t in enumerate(tokens):          # where it appears in a heading, it is quoted code
        if t.type == "heading_open" and "Fake heading" in tokens[i + 1].content:
            assert any(c.type == "code_inline" and "Fake heading" in c.content
                       for c in tokens[i + 1].children)
            assert not any(c.type == "text" and "Fake heading" in c.content
                           for c in tokens[i + 1].children)
    # Every table is still a table: each body row has as many cells as its header.
    widths, rows = [], []
    for t in tokens:
        if t.type == "table_open":
            widths.append(0)
        elif t.type == "th_open":
            widths[-1] += 1
        elif t.type == "tr_open":
            rows.append([len(widths) - 1, 0])
        elif t.type == "td_open":
            rows[-1][1] += 1
    assert len(widths) >= 8
    assert all(cells in (0, widths[table]) for table, cells in rows)
    # The names are still there for the reader, as text.
    for name in (IMAGE, LINK, HTML):
        assert html.escape(name, quote=False) in page.replace("&quot;", '"')
    assert len(df.columns) == 8


def test_cli_output_for_hostile_data_is_inert(hostile, capsys):
    path, _, _ = hostile
    assert main(["profile", str(path)]) == 0
    out = capsys.readouterr().out
    assert not kinds(out) & ACTIVE and "constant column" in out
    assert main(["sql", str(path), "--query",
                 f"SELECT {quote_ident(HTML)} AS seg, count(*) AS n FROM t GROUP BY 1"]) == 0
    assert not kinds(capsys.readouterr().out) & ACTIVE
