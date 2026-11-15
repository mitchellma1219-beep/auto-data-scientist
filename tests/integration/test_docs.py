"""The documents must agree with each other and with what the test run can see.

Numbers in the portfolio come from QUALITY_REPORT.md only, the stated test count is the
number of tests actually collected, and paths mentioned in the documents exist.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUALITY = (ROOT / "QUALITY_REPORT.md").read_text(encoding="utf-8")
PORTFOLIO = (ROOT / "PORTFOLIO.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?(?![\w])")


def numbers(text: str) -> set[str]:
    return {n.rstrip(",") for n in NUMBER.findall(text)}


def section(document: str, heading: str) -> str:
    body = document.split(f"\n## {heading}\n", 1)
    assert len(body) == 2, f"missing section {heading!r}"
    return body[1].split("\n## ", 1)[0]


def test_portfolio_has_every_required_section_and_says_how_it_was_built():
    for heading in ["Problem", "Business relevance", "Architecture", "Key algorithms",
                    "Technical trade-offs", "Statistical methodology",
                    "Reliability and security", "Demonstrated capabilities",
                    "Verified outcomes", "Future improvements", "How this was built",
                    "Resume bullet candidates", "60-second elevator pitch",
                    "2-minute technical walkthrough", "Interview questions",
                    "LinkedIn showcase post"]:
        assert section(PORTFOLIO, heading).strip(), heading
    built = " ".join(section(PORTFOLIO, "How this was built").split())
    assert "AI coding agent" in built and "commit history" in built
    assert len(re.findall(r"(?m)^### \d+\. ", section(PORTFOLIO, "Interview questions"))) == 10
    assert len(re.findall(r"(?m)^\d\. ", section(PORTFOLIO, "Resume bullet candidates"))) == 3


def test_portfolio_outcomes_and_bullets_quote_only_numbers_from_the_quality_report():
    measured = numbers(QUALITY)
    for heading in ("Verified outcomes", "Resume bullet candidates"):
        quoted = numbers(section(PORTFOLIO, heading))
        assert quoted, heading
        unknown = sorted(quoted - measured)
        assert not unknown, f"{heading}: not in QUALITY_REPORT.md: {unknown}"
    lowered = " ".join(PORTFOLIO.lower().split())
    for claim in ("statistically significant", "100% reliab", "fully secure", "completely secure",
                  "users rely", "adopted by", "github stars", "revenue impact", "saved $",
                  "hand-written", "hand-coded", "i wrote every"):
        assert claim not in lowered, claim


def test_stated_test_count_is_the_number_of_tests_collected(request):
    stated = re.search(r"\*\*Total\*\* \| \*\*(\d+)\*\*", QUALITY)
    assert stated, "QUALITY_REPORT.md must state the total number of tests"
    assert f"{stated.group(1)} tests" in README
    config = request.config
    whole_suite = (not config.getoption("keyword") and not config.getoption("markexpr")
                   and [Path(a).resolve() for a in config.args] == [ROOT / "tests"])
    if not whole_suite:
        pytest.skip("only checked when the whole suite runs")
    assert request.session.testscollected == int(stated.group(1))


def test_paths_named_in_the_documents_exist():
    for name in ("README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "PORTFOLIO.md",
                 "SECURITY.md", "QUALITY_REPORT.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?://|#)([^)#\s]+)", text):
            assert (ROOT / target).exists(), f"{name} links to missing {target}"
        for path in re.findall(r"`((?:src|app|demo|tests|skills|benchmarks|examples|\.claude)"
                               r"/[\w./\-]+)`", text):
            if "*" not in path and not path.endswith(".csv"):
                assert (ROOT / path.rstrip("/.")).exists(), f"{name} names missing {path}"
