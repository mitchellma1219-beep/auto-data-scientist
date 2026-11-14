# Security

## Threat model
The tool reads files a user points it at and runs SQL that a person or a language model
wrote. Both are treated as untrusted, and so is every name and value inside the data once it
is shown to a person or to an agent. The tool needs no credentials and makes no network
calls. The dashboard is a single-user tool for one computer.

## Controls

| Risk | Control | Verified by |
|---|---|---|
| SQL that writes, reads files or reaches the network | AST allowlist: exactly one SELECT, known tables only, no file / network / configuration / dynamic-SQL functions, no recursive CTEs, no unconditioned joins | 63 denied statements in `tests/unit/test_sql.py` |
| A statement that slips past the allowlist | The DuckDB connection has `enable_external_access=false` and a locked configuration; every query runs in a transaction that is rolled back | `test_database_layer_blocks_files_even_without_the_guard` |
| Runaway queries | Watchdog interrupt after `ADS_SQL_TIMEOUT_S` (30 s), row cap `ADS_SQL_MAX_ROWS` (10,000), join cap, memory limit `ADS_MEMORY_LIMIT_MB` | timeout and truncation tests; the dashboard states when a result was cut |
| Two callers on one connection (several dashboard sessions) | `Dataset.lock`: queries take turns, so transactions cannot interleave | `test_one_dataset_can_serve_several_threads` |
| SQL injection through questions, values, column or table names | Identifiers are always quoted, values are escaped or bound, generated SQL goes through the same allowlist | adversarial tests with quote and `DROP TABLE` payloads |
| Path traversal on upload | Only a validated suffix of the client file name is kept; the stored name is generated. The name shown to people is reduced to letters, digits, dot, dash, underscore and space | `test_upload_names_cannot_escape_the_directory`, `test_display_names_carry_no_path_and_no_markup` |
| Unsafe file formats | Extension allowlist (CSV, TSV, XLSX, Parquet, SQLite, DuckDB). No pickle, no macro-enabled workbooks, no `eval` / `exec` | unsupported-type tests; `test_nothing_executes_generated_code_or_renders_raw_html` |
| Oversized input | `ADS_MAX_FILE_MB` (500), `ADS_MAX_ROWS` (5,000,000); correlation limited to 300 columns; dashboard uploads limited to 200 MB | limit tests |
| Markup planted in data: links, images that make a viewer fetch a URL, HTML, fake headings | In the report and the CLI output, names go into code spans that cannot be closed from inside and all other data-derived text is escaped. Chart labels are HTML-escaped. The dashboard shows raw values in tables and code blocks and never uses `unsafe_allow_html` | `tests/adversarial/test_report_markup.py` parses the report with a Markdown parser; `test_hostile_names_and_values_never_become_markup_on_the_page`; `test_text_from_the_data_cannot_inject_markup` |
| Code execution from model output | The package and the dashboard never execute generated Python. The only generated artefact that runs is SQL, through the guard | by design; an AST scan in the tests finds no `eval`, `exec`, `compile`, `pickle` or `subprocess` in `src/` or `app/` |
| Instructions hidden in data, read by an agent using the skill | `SKILL.md` tells the agent that names and values are data: quote anything that reads like an instruction, never act on it | by instruction only; not enforceable in code |

## The dashboard

`streamlit run app/streamlit_app.py` starts a local web server.

- **It has no sign-in and no user accounts.** Anyone who can reach the port can read the
  loaded data and run read-only queries on it.
- **It is bound to localhost.** `app/.streamlit/config.toml` sets
  `server.address = "localhost"`, and Streamlit reads that file wherever the command is run
  from. A test starts the server from another directory and confirms that the port is closed
  on the machine's outside address. If the server is started with another address, the
  sidebar shows a warning. Do not put it behind a public URL or a shared host as it is.
- **It stays offline.** Usage statistics are switched off, and the page loads nothing from
  other hosts. A headless Chrome session through every section (`demo/capture_screenshots.py`,
  2026-11-14) recorded 145 requests, all to localhost. The first version used a material
  icon as page icon, which the browser fetched from a font CDN; it is now a local file.
- **Uploads** are written by `save_upload` under a generated name in a temporary folder that
  is private to the server process and removed when it exits. Files are not kept between
  runs.
- **Unexpected errors** show their type in the browser, not a stack trace with local paths
  (`client.showErrorDetails = "type"`).

## Scans (2026-11-14)
- `bandit -r src`: 0 findings. Ten `B608` lines (SQL built with quoted identifiers) were
  reviewed and annotated individually.
- `ruff` rule set `S` (flake8-bandit) is enforced in CI for `src/`, `app/`, `demo/` and
  `benchmarks/`.
- `pip-audit` runs in CI on the package installed with the `ui` extra; see the CI section of
  [QUALITY_REPORT.md](QUALITY_REPORT.md) for the latest result. The extra requires
  `pillow>=13.0.0` because Pillow 12.3.0 had known CVEs.
- No dedicated secret scanner was run; a keyword search of the tree found no credentials.

## Not covered
- SQLite and DuckDB **files** are opened read-only but are parsed by their native libraries;
  only open database files from sources you trust. The same holds for XLSX and Parquet
  parsers.
- There is no authentication, authorisation or multi-user isolation. Sessions of the
  dashboard share cached data.
- The escaping targets CommonMark and GitHub-flavoured Markdown. A renderer that turns bare
  domain names such as `example.com` into links can still link one that appears in data.
- A file can be crafted to be slow to parse or to profile within the size limits; there is
  no time limit on loading.
- Passing these tests does not prove the guard is complete. Report bypasses via GitHub issues.
