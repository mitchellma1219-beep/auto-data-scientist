# Security

## Threat model
The tool reads files a user points it at and runs SQL that a person or a language model
wrote. Both are treated as untrusted. It needs no credentials and makes no network calls.

## Controls

| Risk | Control | Verified by |
|---|---|---|
| SQL that writes, reads files or reaches the network | AST allowlist: exactly one SELECT, known tables only, no file / network / configuration / dynamic-SQL functions, no recursive CTEs, no unconditioned joins | 63 denied statements in `tests/unit/test_sql.py` |
| A statement that slips past the allowlist | The DuckDB connection has `enable_external_access=false` and a locked configuration; every query runs in a transaction that is rolled back | `test_database_layer_blocks_files_even_without_the_guard` |
| Runaway queries | Watchdog interrupt after `ADS_SQL_TIMEOUT_S` (30 s), row cap `ADS_SQL_MAX_ROWS` (10,000), join cap, memory limit `ADS_MEMORY_LIMIT_MB` | timeout and truncation tests |
| SQL injection through questions, values, column or table names | Identifiers are always quoted, values are escaped or bound, generated SQL goes through the same allowlist | adversarial tests with quote and `DROP TABLE` payloads |
| Path traversal on upload | Only a validated suffix of the client file name is kept; the stored name is generated | `test_upload_names_cannot_escape_the_directory` |
| Unsafe file formats | Extension allowlist (CSV, TSV, XLSX, Parquet, SQLite, DuckDB). No pickle, no macro-enabled workbooks, no `eval` / `exec` | unsupported-type tests |
| Oversized input | `ADS_MAX_FILE_MB` (500), `ADS_MAX_ROWS` (5,000,000); correlation limited to 300 columns | limit tests |
| Code execution from model output | The package never executes generated Python. The only generated artefact that runs is SQL, through the guard | by design |

## Scans (2026-11-13)
- `bandit -r src`: 0 findings. Nine `B608` lines (SQL built with quoted identifiers) were
  reviewed and annotated individually.
- `ruff` rule set `S` (flake8-bandit) is enforced in CI.
- `pip-audit` on the 26 runtime dependencies: no known vulnerabilities. With the `ui` extra
  (53 packages) the only findings were in Pillow 12.3.0; the extra now requires
  `pillow>=13.0.0`, which audits clean.

## Not covered
- SQLite and DuckDB **files** are opened read-only but are parsed by their native libraries;
  only open database files from sources you trust.
- There is no authentication or multi-user isolation. Do not expose the CLI or a future
  dashboard to the internet as-is.
- Passing these tests does not prove the guard is complete. Report bypasses via GitHub issues.
