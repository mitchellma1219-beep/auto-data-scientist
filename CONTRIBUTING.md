# Contributing

Thanks for looking at the code. This project is small on purpose; changes are welcome when
they keep it honest (every number traceable to a command) and safe (data and SQL treated as
untrusted). [ARCHITECTURE.md](ARCHITECTURE.md) explains how the pieces fit.

## Set up

```bash
git clone <repo-url> && cd auto-data-scientist
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"          # Python 3.11 or newer
python examples/generate_sample_data.py
```

`dev` brings the test and lint tools; `ui` brings Streamlit and Plotly for the dashboard and
the charts. Without `ui`, the chart and dashboard tests are skipped instead of failing.
CI sets `ADS_REQUIRE_UI=1` so that a missing extra is an error there.

## Checks to run before a pull request

```bash
ruff check .                                             # style, imports, flake8-bandit rules
mypy                                                     # strict, src/ only
pytest --cov=auto_data_scientist --cov=app --cov-fail-under=85
bandit -q -r src
pip-audit --skip-editable                                # in an environment with ".[ui]"
```

CI runs the first three on Python 3.11, 3.12 and 3.13 and the last two in a separate job.
The whole suite takes a few minutes; while working, run the folder you are touching:

```bash
pytest tests/unit -q                 # functions against trusted implementations
pytest tests/statistical -q          # ground-truth models and forecasts (slowest)
pytest tests/dashboard -q            # the real Streamlit script and a headless server
```

Do not add `pytest-xdist` settings to the repository: several tests start servers or share
Streamlit's process-wide caches.

## How tests are written here

- **Compare with something trusted.** A statistic is checked against numpy, pandas or
  scipy; a query result against the equivalent pandas group-by; a dashboard number against
  the package function that produced it. "It renders" is not a test.
- **Use ground truth for models.** Synthetic data with planted structure
  (`auto_data_scientist.samples`) says what must be found. Pure noise says what must not.
- **Attack it.** New input paths get a case in `tests/adversarial`: malformed, oversized,
  hostile names, injection attempts.
- **Deterministic and offline.** Fixed seeds, no network, no dependence on wall-clock time.
- **Never weaken a test to make it pass.** Fix the code. If an expectation was wrong, say
  so in the commit message and make the corrected test stricter, not looser.
- **Keep the suite quiet.** A new warning is fixed at its source. The only filter in
  `pyproject.toml` is for `FutureWarning` raised inside dependencies.

## Rules for the code

- **The UI stays thin.** `app/streamlit_app.py` may lay out widgets, call cached wrappers
  and format a table. Anything that computes belongs in the package, with a unit test.
- **Charts are pure.** A function in `charts.py` takes results and returns a figure. It does
  not read files, import Streamlit or estimate anything.
- **Data is never markup.** Text that comes from a file goes through `escape_md` or
  `code_span` before it reaches Markdown, and through `charts.safe_text` before it reaches
  a Plotly label. Use tables and code blocks for raw values in the dashboard, and never
  `unsafe_allow_html`.
- **SQL is built in one way.** Identifiers through `quote_ident`, values escaped or bound,
  and every statement through `validate_sql` before `run_query`. A new `# noqa: S608` needs
  a reason in the review.
- **No generated code is executed.** No `eval`, `exec`, `pickle` or `subprocess` in `src/`
  or `app/`; a test enforces it.
- **Expected failures are `AdsError`s** with a message that says what to change. Unexpected
  exceptions are bugs and must stay visible.
- **Randomness is seeded and recorded.** A new sampling step needs a seed parameter, a field
  in the result that says what was sampled, and a line in the report's limitations.
- **Dependencies are declared** in `pyproject.toml`, in the smallest extra that needs them.

## Data and documentation

- Synthetic data only. No real customer data, personal data, credentials or e-mail
  addresses in the repository, in tests or in examples.
- `QUALITY_REPORT.md` holds only numbers produced by a command that was run; state the
  command and the environment. If something was not verified, list it under
  "Not verified / not done" instead of leaving it out.
- No claims of statistical significance, complete security or financial impact.
- After changing anything that affects output, regenerate the demo and commit the result:

  ```bash
  python demo/make_demo.py
  python demo/capture_screenshots.py     # optional: needs ".[demo]" and Chrome or Chromium
  ```

## Commits and pull requests

- Small commits with a subject that says what changed and a body that says why. Mention a
  defect that a test found, and how it is now covered.
- Work on a branch and open a pull request against `main`; do not push to `main` or rewrite
  published history.
- A pull request is ready when CI is green on all jobs and the documents that state numbers
  (`README.md`, `QUALITY_REPORT.md`, `CHANGELOG.md`) match what the commands print.

## Reporting a security problem

A way around the SQL guard, the upload handling or the escaping is the most useful report
this project can get. Open a GitHub issue with the smallest input that shows it. See
[SECURITY.md](SECURITY.md) for the threat model and what is out of scope.
