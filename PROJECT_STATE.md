# Project state — three AI Skills (updated 2026-11-13, 22:20 PT)

## GitHub

- Account: `mitchellma1219-beep` (personal). Claude GitHub App installed on exactly three
  repositories and linked to Claude (installation 181814682).
- All three repositories are **private** until their release gates are met.
- Commit identity: `Mitchell Ma <325205137+mitchellma1219-beep@users.noreply.github.com>`.
- Open decision for the owner: the auto-generated initial commit in each repository carries
  the owner's personal email as author. Before a repository is made public, decide whether to
  replace that commit (force-push, needs explicit approval) or keep it.

## Environment

Package installs work in the build sandbox since the owner enabled network access
(2026-11-13, about 21:35 PT). DuckDB, Streamlit, Plotly, statsmodels, pytest-cov, bandit and
pip-audit are installed there.

## Milestones

| Project | State |
|---|---|
| [auto-data-scientist](https://github.com/mitchellma1219-beep/auto-data-scientist) | Core engine, CLI, skill on `feat/core-engine`, draft pull request #1. 194 tests green in CI on Python 3.11 / 3.12 / 3.13, coverage 96.84%, bandit and pip-audit clean. **Open: dashboard and charts, independent review, ARCHITECTURE / CONTRIBUTING / PORTFOLIO, release audit.** Not released. |
| [ab-testing-copilot](https://github.com/mitchellma1219-beep/ab-testing-copilot) | Core engine, CLI, skill on `feat/core-engine`, draft pull request #1. 99 tests green in CI on Python 3.11 / 3.12 / 3.13, coverage 97.74%. **Open: dashboard, group-sequential testing, novelty / HTE, security scans, independent review, docs, release audit.** Not released. |
| [agent-reliability-lab](https://github.com/mitchellma1219-beep/agent-reliability-lab) | Not started. `main` holds the auto-generated README. |

Milestone A was blocked at first, so Milestone B's core was built first; Milestone A's core
followed once installs worked. Neither milestone is complete.

## Evidence

- auto-data-scientist: CI run 43565100645 (first run, passed). Details in its QUALITY_REPORT.md.
- ab-testing-copilot: CI runs 43561092763 (failed: PyYAML missing from dev dependencies,
  fixed), 43561369486 and 43561578608 (passed). Details in its QUALITY_REPORT.md.

## Next actions, in order

1. auto-data-scientist: Streamlit dashboard with Plotly charts and downloads, tested with
   Streamlit's app testing; six-role review with separate reviewer agents and fixes;
   ARCHITECTURE.md, CONTRIBUTING.md, PORTFOLIO.md; release audit; merge; owner makes it public.
2. ab-testing-copilot: the same list, plus group-sequential boundaries and novelty / HTE checks.
3. agent-reliability-lab: full build.

## Continue with

```bash
git clone https://github.com/mitchellma1219-beep/auto-data-scientist && cd auto-data-scientist
git checkout feat/core-engine
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,ui]"
pytest --cov=auto_data_scientist && ruff check . && mypy
python examples/generate_sample_data.py && ads report examples/churn.csv --target churned
```
