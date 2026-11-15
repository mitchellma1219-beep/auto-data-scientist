# Portfolio notes: Auto Data Scientist

Talking points for applications and interviews (data analyst, business analyst, product
analyst, AI consultant, forward-deployed engineer). Every figure in "Verified outcomes" and
in the resume bullets is taken from [QUALITY_REPORT.md](QUALITY_REPORT.md); a test fails if
one of them is not found there.

## Problem

An analyst who receives a spreadsheet and a question usually has an afternoon, not a
quarter. The fast path is well known and risky: skip the data checks, fit a model, report
the best-looking number, and write "drives" where the data only shows "is associated with".
AI assistants make the fast path faster. They will happily write SQL that reads the wrong
thing, a model that scores 1.000 because a column was recorded after the outcome, or a
confident paragraph about revenue that nothing in the data supports.

Auto Data Scientist is a small toolkit that makes the careful path the default. Given one
table, it profiles the data and says what is wrong with it, answers questions through SQL
that can only read, judges every model against a no-skill baseline and every forecast
against naive methods, and writes a report in which each number can be regenerated and each
limit is stated. When it cannot do something honestly, it refuses and says why.

## Business relevance

- **Decisions get the caveats they need.** A churn model that "works" because of leakage
  leads to a retention budget spent on the wrong customers. The tool flags and excludes such
  columns, and reports what the score would otherwise have claimed.
- **"No" is a deliverable.** For a business owner, "these columns do not predict the
  outcome" or "a naive forecast is as good as the statistical one" saves a project that
  would have failed later. The tool states both plainly.
- **It is safe to put in front of an assistant.** The SQL guard, the locked database and the
  output escaping mean an agent can explore a client's file without being able to change
  it, read other files or pass on markup hidden in the data.
- **Work can be handed over.** Each report comes with the query that ran, a machine-readable
  copy of the analysis and the command that reproduces it from the same file.
- **It does not promise money.** The report says that no monetary impact is estimated and
  lists what would be needed to estimate one: the value of the outcome, the cost of acting
  and an effect size from an experiment.

## Architecture

One engine, four thin entry points. The package `auto_data_scientist` holds all analysis;
the `ads` command line, the Agent Skill script, the Streamlit dashboard and the demo script
only collect inputs and present outputs.

```
file or upload -> ingest (validate, copy into in-memory DuckDB, lock)
                    -> profiling (column kinds, quality score)
                         -> eda / modeling / forecasting
SQL or question -> nl2sql (rules, or refusal) -> sql_guard (allowlist) -> query (lock,
                   transaction, timeout, row cap)
all results     -> Analysis -> report (data escaped) + charts (labels escaped)
                            -> report.md, analysis.json, queries.sql, reproduce.sh
```

Module responsibilities, the data-flow diagram, the security layers and the trade-offs are
in [ARCHITECTURE.md](ARCHITECTURE.md).

## Key algorithms

| Area | What it does |
|---|---|
| Type inference | Decides per column whether it is numeric, categorical, a date, free text or an identifier, from parse rates, cardinality and naming; identifiers and free text are kept out of models |
| Quality score | A documented weighted sum of completeness, validity, uniqueness, consistency and absence of extreme values; the components are always shown next to the score |
| SQL guard | Parses the statement into a syntax tree and accepts exactly one `SELECT` over known tables, with table functions and scalar functions checked against allow and deny lists, no recursive CTEs and no joins without a condition |
| Question translation | Keyword and schema matching for totals, averages, counts, rankings, time grains and filters; returns its assumptions with the SQL, or a refusal |
| Correlations | Pairwise-complete Pearson or Spearman for all numeric pairs in one vectorised pass, with the t approximation for descriptive p-values |
| Leakage screen | Flags a column that separates the classes almost perfectly on its own (AUC, rank correlation or purity at or above 0.995), is filled in for one outcome only, or is named after the target |
| Model comparison | A no-skill baseline, a linear model, a random forest and gradient boosting as full pipelines, so imputation, scaling and encoding are fitted inside each cross-validation fold |
| Selection sample | On large tables, a seeded, class-proportional sample (rare classes keep a minimum number of rows) is used to choose the model; the winner is refitted on all training rows |
| Importance | Permutation importance on hold-out rows: how much the score drops when a column is shuffled |
| Forecasting | Naive, seasonal-naive and Holt-Winters forecasts scored by rolling-origin backtests; the range is the 80th percentile of absolute backtest errors |
| Cohort retention | Share of each first-activity cohort active N periods later, with periods that have not happened yet left empty rather than zero |
| Output escaping | Names go into code spans whose fence is longer than any run of backticks inside; other text has Markdown control characters escaped; chart labels are HTML-escaped |

## Technical trade-offs

| Decision | Why | Cost |
|---|---|---|
| Copy the file into an in-memory DuckDB | The connection can then be locked: no file access, no network, no configuration changes | Memory, and a cap on table size |
| Allowlist on the syntax tree, plus the locked connection | A denylist only stops what someone thought of; two independent layers cover each other | Some harmless queries are refused, for example schema-qualified names |
| A narrow rule-based translator that refuses | Deterministic, offline and testable against pandas; in the skill, the agent writes SQL and the guard checks it | It answers only simple aggregate questions |
| Four fixed model candidates, no tuning | The question is "is there signal, and what does it rely on", not "what is the best score"; fewer choices mean less selection bias | Scores are below what tuning could reach |
| Sample for model selection only | Selection is the slow step and tolerates sampling noise; the reported score keeps the whole hold-out | Close candidates could be ranked differently by another sample |
| An empirical forecast range | No distributional assumption; it reports what the backtests saw | The range does not widen with the horizon |
| Streamlit with pure chart functions | The UI stays in Python and is tested in the same suite; chart data is unit-tested | Less control over layout than a custom front end |
| One table at a time | Joining tables correctly needs keys and grain that cannot be guessed | Multi-table work goes through SQL |

## Statistical methodology

1. **Split first.** Rows without a target are dropped, repeated records are removed, and 20%
   of the rows are held out (the latest 20% when a time column is given). The hold-out is
   used once.
2. **Screen for leakage** before any model sees the data, and list what was dropped and why.
3. **Compare candidates by cross-validation** on the training rows, with all preprocessing
   inside the pipeline, next to a no-skill baseline scored the same way.
4. **Decide conservatively.** A model "improves on the baseline" only if it wins in every
   fold and by a margin. Otherwise the report says that the data does not support a model.
5. **Report on the hold-out**: the selected model and the baseline side by side, with the
   number of rows, plus the lift in the top decile for binary targets.
6. **Explain without overclaiming.** Permutation importance is described as what the model
   relies on. Correlations, segment differences and importance are called associations;
   p-values are labelled descriptive because many pairs are compared.
7. **Forecast against naive methods.** Holt-Winters is used only if it beats the best naive
   method by a margin in most backtest windows; otherwise the simple forecast is reported.
8. **State every limit** that applies to the result at hand, in the report itself.

Full detail: [methodology.md](skills/auto-data-scientist/references/methodology.md).

## Reliability and security

- **Tests by layer**: unit tests against numpy, pandas and scipy; integration tests of the
  whole pipeline and of the command that reproduces a report; adversarial tests with
  malformed files and injection attempts; statistical tests on synthetic ground truth; and
  dashboard tests that drive the real Streamlit script and start it as a server.
- **Continuous integration** on three Python versions with lint, strict typing, a coverage
  gate and a dependency audit.
- **Untrusted SQL** passes an allowlist, runs on a connection without file or network
  access, inside a transaction that is rolled back, with a time and row budget.
- **Untrusted data** cannot inject markup into the report, the charts or the dashboard, and
  an agent using the skill is told to treat it as data, not as instructions.
- **The dashboard** has no sign-in, so it is bound to localhost, warns if it is not, sends
  no usage statistics and loads nothing from other hosts.
- **Known gaps** are written down: no independent review yet, one browser tested by hand,
  database and workbook files parsed by their native libraries, no authentication. See
  [SECURITY.md](SECURITY.md) and the "Not verified" section of the quality report.

## Demonstrated capabilities

| Capability | Where to see it | Relevant for |
|---|---|---|
| Framing an analysis so it cannot oversell | Report sections for limitations and impact; the rule that a model must beat its baseline | Data, business and product analyst |
| SQL and data modelling judgement | `sql_guard.py`, `nl2sql.py`, tests that compare query results with pandas | Data analyst, forward-deployed engineer |
| Statistical reasoning | Leakage screen, cross-validation protocol, the sample-size argument in `modeling.py`, backtests | Data and product analyst |
| Communicating to decision makers | The executive report, the verdict line at the top of each dashboard section | Business analyst, AI consultant |
| Working with AI agents responsibly | A written specification, acceptance tests, a quality report that separates measured from unverified | AI consultant, forward-deployed engineer |
| Security thinking for AI-facing tools | Layered query guard, output escaping, localhost binding, the findings list in the quality report | AI consultant, forward-deployed engineer |
| Shipping a usable product | CLI, dashboard with tests, demo pages, documentation | Product analyst, forward-deployed engineer |

## Verified outcomes

All measured; the commands and conditions are in [QUALITY_REPORT.md](QUALITY_REPORT.md).

- **322 tests pass locally**, and line coverage of the package and the dashboard script is
  98%. CI passes on Python 3.11, 3.12 and 3.13.
- **Ground truth is recovered.** On synthetic churn data the true drivers are ranked first
  and the selected model reaches hold-out AUC 0.726 against 0.500 for the baseline.
- **Leakage is caught.** A column recorded after the outcome is flagged and excluded; forced
  back in, it would have produced an AUC of 1.000.
- **Noise is not oversold.** Pure-noise targets are reported as "no signal" in all 3 seeds
  tested, and random walks keep the naive forecast in all 3.
- **Hostile SQL is refused.** All 63 unsafe statements in the test suite are denied, and the
  same attacks fail at the database layer when the guard is bypassed.
- **Hostile data stays text.** With 12 hostile strings as column names, values and file
  name, the parsed report contains no link, image or HTML node.
- **The dashboard stays local.** A headless browser session through every section made 145
  requests, all to localhost.
- **Faster on large tables without changing the result.** On 100,000 rows, model evaluation
  took 10.4 s with the selection sample and 44.3 s without it in the same run, with the same
  hold-out AUC of 0.7396. 1,000,000 rows took 18.5 s.
- **Testing found real defects**: 20 are listed with their fixes, among them a shared
  database connection that failed all 240 queries in a threaded trial.

Not verified, and said so in the quality report: an independent review, browsers other than
headless Chrome, accessibility, Windows and macOS.

## Future improvements

1. An independent review of the statistics, the SQL guard and the dashboard.
2. A language-model translator behind the same guard, evaluated on a fixed question set
   against the rule-based one.
3. Probability calibration and class-imbalance handling, with the same baseline discipline.
4. A measured false-alarm rate for the "improves on the baseline" rule on small tables, and
   a rule that adapts its margin to the fold size.
5. Browser tests in CI, keyboard and screen-reader checks, narrow-screen layout.
6. Sign-in and per-user isolation, so the dashboard could run on a shared host.
7. Multi-table analysis with declared keys and grain.
8. A time limit on loading and profiling, and sampling for profiles of very large tables.

## How this was built

This project was specified, directed and verified by its owner and implemented with an AI
coding agent (Claude). The commit history shows this: commits carry the agent's co-author
trailer and a link to the session that produced them.

The owner's part was to decide what the tool must do and must never do (a baseline beside
every model, a leakage screen, no significance or money claims, read-only SQL, synthetic
data only), to set the acceptance rules (tests against trusted implementations, a coverage
gate, a quality report that contains only measured numbers and lists what is unverified),
to steer the work in milestones and to check the results. The agent wrote the code, the
tests and the documents under those rules, and reported defects it found on the way.

The verification is built so that it does not rest on trusting the agent: tests compare
with numpy, scipy and pandas rather than with the code's own output, CI repeats them on
GitHub's machines, and each claim in the quality report names the command behind it.

Before presenting the project, the owner should run `pytest`, `ads report` on the sample
data and the dashboard personally, read `sql_guard.py` and `modeling.py`, and be ready to
say which decisions were theirs. The answers below are written to be said truthfully on
that basis, not as claims of having typed the implementation.

## Resume bullet candidates

1. Specified, directed and verified an open-source Python analytics toolkit (DuckDB,
   scikit-learn, statsmodels, Streamlit, Plotly), implemented with an AI coding agent, that
   profiles a dataset, answers questions through guarded read-only SQL, benchmarks models
   against a no-skill baseline and writes a reproducible report; 322 automated tests, 98%
   line coverage, CI on Python 3.11, 3.12 and 3.13.
2. Defined statistical acceptance criteria and had them enforced on synthetic ground truth:
   a column recorded after the outcome is detected and excluded (hold-out AUC 0.726 instead
   of a misleading 1.000), pure-noise targets are reported as "no signal" in every tested
   seed, and a forecast is kept only if it beats naive methods in rolling backtests.
3. Directed the security and performance work on an AI-facing data tool: an allowlist SQL
   guard that refuses all 63 hostile statements in its test suite, report and chart output
   that keeps hostile column names inert, and a seeded selection sample that cut model
   evaluation on 100,000 rows from 44.3 s to 10.4 s with an unchanged hold-out AUC.

Wording to keep: "specified, directed and verified" and "implemented with an AI coding
agent". Do not add users, adoption, stars, revenue or time saved for anyone; none of that
was measured.

## 60-second elevator pitch

"When an analyst gets a spreadsheet and a deadline, the tempting path is to fit a model and
report the best number. AI assistants make that path even faster, and even riskier. Auto
Data Scientist is my project to make the careful path the default. You give it a table. It
tells you what is wrong with the data, answers questions with SQL that can only read, and
judges every model against a no-skill baseline. If a column leaks the answer, it drops it
and tells you what the score would have claimed. If there is no signal, it says so. Every
report comes with the command that reproduces it. I specified it, directed an AI coding
agent that implemented it, and made verification part of the specification: more than
three hundred tests against numpy, scipy and pandas, on synthetic data where the truth is
known. What I am proudest of is not a feature. It is the list of things the tool refuses to
say."

## 2-minute technical walkthrough

"I'll follow one file through the system. A CSV is validated and copied into an in-memory
DuckDB database. Rows with the wrong number of fields are rejected and counted, never
repaired silently. Then the connection is locked: no file access, no network, no
configuration changes.

Profiling infers a kind for each column and computes a quality score from five documented
components. Identifiers and free text are recognised and kept out of models.

Questions go through two gates. A rule-based translator handles simple aggregates and
refuses the rest; in the agent skill, the model writes the SQL instead. Either way, the
statement is parsed into a syntax tree and must be exactly one SELECT over known tables,
with no file or network functions. It then runs inside a transaction that is rolled back,
with a timeout and a row cap. So the allowlist and the locked database cover each other.

For prediction, twenty percent of the rows are held out first. A leakage screen drops
columns that predict the target almost perfectly on their own or are filled in only when
the outcome happened. Four candidates, including a no-skill baseline, are cross-validated as full
pipelines, so preprocessing is fitted inside each fold. A model counts as better only if it
wins every fold by a margin. On large tables the comparison runs on a seeded sample, but
the winner is refitted on all training rows and scored on the whole hold-out, and the
report says so.

Forecasts work the same way: Holt-Winters has to beat naive methods in rolling backtests,
or the naive forecast is reported.

Finally, the report. Every name from the data is escaped or put in a code span, because a
column called like a Markdown image should be printed, not fetched. And each report ships
with the query, a JSON copy and a script that regenerates it. The dashboard is a thin layer
over all of this, tested by driving the real page and comparing what it shows with what
the package computes."

## Interview questions

### 1. You built this with an AI coding agent. What did you actually do, and how do you know the code is right?

I wrote the specification and the rules: what the tool must do, what it must never claim,
which safeguards are mandatory, and how results have to be reported. The agent wrote the
implementation, the tests and the documents, in milestones that I directed. I do not ask
anyone to trust the agent, and I do not either. The tests compare against numpy, scipy and
pandas instead of the code's own output; the statistical tests use synthetic data where the
true answer is known; CI repeats everything on three Python versions; and the quality
report may only contain numbers that a command produced, with a section for what was not
verified. That report lists twenty defects found by testing, which tells me the tests bite.
What I would not claim is that I could have typed every module myself at that speed. What I
can do is explain each design decision, and show the evidence for each claim.

### 2. A model must beat the baseline in every fold and by a margin. Isn't that arbitrary?

The thresholds are a convention, but the structure is not. A baseline matters because a
score means nothing alone: 70% accuracy is poor if 70% of rows are one class. Winning every
fold asks for consistency, not one lucky split. The margin protects against differences
that are real but useless. The honest weakness is small data. During testing, a sample of
500 rows let pure noise pass the rule, because with validation folds of 100 rows a no-skill
AUC moves by about 0.058 from fold to fold, far more than the 0.02 margin. That is why the
selection sample has a floor of 5,000 rows, and why the quality report says the rule is
weak evidence on tables with a few hundred rows. The next step I would take is a margin
that scales with fold size, and a measured false-alarm rate.

### 3. How does the leakage screen work, and what can it miss?

It runs before any model is fitted and looks at each column alone. Three patterns are
flagged: a column that separates the outcome almost perfectly by itself, a column that is
only filled in when the outcome happened, such as a cancellation date, and a column named
after the target. Flagged columns are dropped and listed with the reason, and the user can
keep one explicitly, which the report then states. In the test data, a fee that is only
charged on cancellation takes the hold-out AUC from 0.726 to 1.000 when forced in. What it
misses: leakage spread across several columns, features computed with information from
after the prediction time, and target-related aggregates over the full table. Those need
knowledge of when each column is recorded, which is why the report also asks the business
owner to confirm it, and why a time-based split is offered.

### 4. You compare models on a sample for large tables. Doesn't that make the reported score optimistic or noisy?

The reported score does not come from the sample. Only the choice among candidates does.
Once a candidate is chosen, it is refitted on all training rows and scored once on the
whole hold-out, which was set aside before anything else. On the 100,000-row benchmark the
sampled and the full procedure picked the same model and produced the same hold-out AUC,
while the run time went from 44.3 to 10.4 seconds. The risk that remains is model
selection, not evaluation: if two candidates score very close together, another sample
could pick the other one. That is stated in the report's limitations, the sample is seeded
and its size is recorded, so the choice is reproducible. I would not use this for tuning
many hyperparameters, where selection noise accumulates.

### 5. A language model writes the SQL. Why an allowlist on the syntax tree instead of blocking dangerous keywords?

A keyword filter fails in both directions. It blocks a column called `update_date`, and it
misses whatever the author did not think of: a table function that reads a file, a pragma,
a second statement after a comment. The guard parses the statement and accepts only what it
recognises: one SELECT, known tables, functions that are not on the deny list, joins with a
condition. Anything unparseable is refused. Then I assume the guard is wrong somewhere. The
database connection itself cannot access files or the network and cannot be reconfigured,
and every query runs in a transaction that is rolled back with a time and row limit. The
tests send the same attacks straight to the connection, bypassing the guard, and they still
fail. I would not call it proven safe; SECURITY.md says that passing the tests does not
prove completeness.

### 6. What was the most interesting defect the testing found?

Three, for different reasons. First, Markdown injection: a column name written like an
image link ended up in the report as a real image, so anyone opening the report would have
made a request to a server named in the data. Now all data-derived text is escaped, and a
test parses the report and requires that no link, image or HTML node exists. Second, the
dashboard's page icon: a single line made every page load contact a font CDN, in a tool
that claims to work offline. It was found by recording browser requests, not by reading
code. Third, a shared database connection: with several sessions, every query in a trial
failed and the connection stayed broken. The lesson I take from all three is that the
integration points, where the engine meets a renderer, a browser or a second user, deserve
their own tests.

### 7. A stakeholder asks how much revenue the churn model will save. What do you say?

That the data cannot answer that yet, and what would. The model ranks customers by risk: in
the test data the top tenth holds 2.10 times the average churn rate. That is a targeting
result, not a savings result. Revenue depends on three things the table does not contain:
what a retained customer is worth, what the intervention costs, and whether the
intervention changes behaviour at all. The third one cannot come from a model trained on
observational data, because feature importance shows what the model relies on, not what
causes churn. So I would propose a pilot: contact a random half of the high-risk group,
hold out the other half, and measure the difference. The report already recommends this and
states that no monetary impact is estimated. I would rather give a range after a pilot than
a figure now that I would have to walk back.

### 8. On the sample data the forecast section reports a naive method. Isn't that a failure?

It is the tool doing its job. Three methods are scored on past periods they did not see.
The statistical model is used only if it beats the best naive method by a margin and in
most windows. On the built-in retail sample it does not, so the seasonal-naive forecast is
reported with its backtest error and a planning range. On a cleaner series in the test
suite, with a trend and a strong season, Holt-Winters wins clearly, with a backtest error of
27.8 against 96.3. For a planner, "last year's month is as good as the model" is useful:
it is cheaper to maintain and easier to explain. The failure would be to show the
sophisticated model anyway because it looks more impressive.

### 9. Your quality score has fixed weights. Why should anyone trust that number?

They should not trust it as a measurement of truth; it is a screening aid, and the
dashboard and the report both say so. Its value is that it is transparent and repeatable:
five components with written definitions, fixed weights, always displayed together, so
anyone can recompute the score and see which component pulled it down. A learned or
configurable weighting would look smarter and be harder to compare across files and teams.
What matters more than the score is the issue list next to it: rejected rows, values that
do not parse, constant columns, extreme values. In practice I would use the score to decide
where to look first across many files, and the issue list to decide what to fix.

### 10. How would you put this in front of fifty analysts at a client with sensitive data?

Not as it is, and I would say so first. The dashboard is a single-user tool: no sign-in,
shared caches between sessions, bound to localhost on purpose. For a team I would keep the
engine and change the shell. Authentication and per-user isolation in front; uploads in
storage the client controls, with retention rules; resource limits per user; an audit log
of queries. I would keep the two-layer query model, because the reason for it gets
stronger with more users. Before any of that I would run a pilot with three analysts and
their real questions, since the rule-based translator is narrow and I expect the first
request to be joins across tables. And I would ask for an independent security review,
which this project has not had yet. The honest pitch is a tested engine with a clear list
of what is missing for multi-user use.

## LinkedIn showcase post

**What to show**: the Predict screenshot (`demo/screenshots/predict.png`) or the refused
query (`demo/screenshots/ask_refused.png`). A picture of the tool saying "no" is more
memorable than a chart.

**Angle options**

- The refusal: "My data tool's best feature is the list of things it will not say."
- The method: "I gave an AI agent a specification instead of a prompt. Here is what the
  tests caught."
- The lesson: "The model scored 1.000. That was the bug."

**Structure**: one-line hook; the problem in two sentences; three concrete things the tool
does (baseline beside every model, leakage screen, read-only SQL); one honest sentence on
how it was built; one thing you learned; link to the repository.

**Draft**

> The model scored 1.000. That was the bug.
>
> A column in the data was only filled in after customers had already left. Any model looks
> brilliant with it and is useless in practice.
>
> Auto Data Scientist is my project to catch this kind of thing by default. It profiles a
> dataset, answers questions with SQL that can only read, and shows every model next to a
> no-skill baseline. If the data does not support a model, it says so. Each report comes with the
> command that reproduces it.
>
> How it was made: I wrote the specification and the acceptance rules, directed an AI
> coding agent through the implementation, and required every claim to be backed by a test
> or listed as unverified.
>
> What I learned: the most valuable part of the project is the list of things the tool
> refuses to say.
>
> Repository: (link)

**Keep it honest**: say "implemented with an AI coding agent"; quote only figures from the
quality report; do not mention users, adoption or business results, because there are none
to report; do not call it secure or production-ready, call it tested and say what is open.
Tag three or four relevant topics at most (for example data analytics, data science,
AI agents).
