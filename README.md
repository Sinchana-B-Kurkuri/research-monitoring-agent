# Research Monitoring Agent

Research monitoring with arXiv ingestion, normalized 384-dimensional embeddings, HIGH/BORDERLINE/LOW routing, human review, personalized digests, and feedback-driven preference learning.

The existing architecture is retained: a Python state machine, FastAPI/Jinja2/HTMX, PostgreSQL/pgvector, Rocchio updates, and guarded logistic regression.

**Start with [the dashboard workflow guide](docs/DASHBOARD_WORKFLOW.md).** It explains the private pilot directory, saved links, browser monitoring buttons, history, preference imports, and available-data evaluation. [Backend repairs](docs/BACKEND_REPAIRS.md) contains diagnostics and terminal verification commands. [The earlier codebase analysis](docs/CODEBASE_ANALYSIS.md) records the pre-repair findings.

## Quick start

```powershell
Set-Location 'C:\Users\sinch\Desktop\AI Project\research-monitoring-agent'
if (-not (Test-Path .\venv\Scripts\python.exe)) { python -m venv venv }
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e . --no-deps
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d db
.\venv\Scripts\python.exe -c "from research_monitor.database.session import init_db; init_db()"
.\venv\Scripts\python.exe scripts/run_phase3_loader.py
.\venv\Scripts\python.exe scripts/create_user.py --user-id researcher --name 'Researcher' --seed-text 'Autonomous AI agents, language model reasoning, reinforcement learning, and tool use.'
.\venv\Scripts\python.exe scripts/run_ui.py --host 127.0.0.1 --port 8000
```

Set DATABASE_URL in .env before initialization. The default Compose URL is postgresql+psycopg2://postgres:postgres@localhost:5432/research_monitor. The loader requires the existing data/corpus.parquet and data/embeddings.npy; the guide explains how to generate missing artifacts. It inserts missing papers without deleting existing records.

Save the personal review/digest/audit links printed by create_user.py. Missing tokens return 401, foreign resources return 403, and anonymous registration using an existing user ID returns 409. Access logs redact bearer tokens.

The web launcher also prints a private operator workspace link. Open it once to recover all local profiles and their personal links in your browser. Each profile's Dashboard now starts monitoring with a button, retains background progress, and links to paginated session/feedback history. Returning to Pilots & initialization preserves accessible profiles. Optional preference inheritance copies effective feedback only from a source profile you explicitly choose.

## Monitoring and summaries

```powershell
.\venv\Scripts\python.exe scripts/run_daily_monitoring.py --user-id researcher --once
# Historical replay:
.\venv\Scripts\python.exe scripts/run_cycle.py --user-id researcher --mode replay --as-of-date 2024-12-31 --limit 50
```

Live daily runs build digests automatically. After replay, open the review link and use its digest update button. Later acceptances enter an existing digest within its remaining capacity. Failed summaries remain retryable, and skipped reviews remain actionable. Defaults are 15 review tasks and 20 digest items per cycle.

Set GEMINI_API_KEY for actual summaries. Without a key, the digest shows original abstracts with summaries pending; mock summaries require explicit selection. Defaults are gemini-3.5-flash-lite, temperature 0.0, and prompt v1.0. The cache uses relational user/paper/model/prompt identity.

CALIBRATION_PATH defaults to data/calibrated_thresholds.json. Valid development calibration levels apply unless CALIBRATED_P_HIGH/CALIBRATED_P_LOW override them. Otherwise provisional P98/P90 levels apply. Adaptation uses effective labels from the SQL view and runs in the worker.

## Verification and data limits

The repaired suite passed **265 tests with no skips** against disposable PostgreSQL. Set TEST_DATABASE_URL explicitly to run PostgreSQL tests; tests never use the application's .env database URL or its live Gemini credential. Generated test outputs go into temporary directories. See the guide for commands.

The local sample currently contains **171 human labels out of 500 papers**. The Evaluation screen and default Phase 13 CLI produce available-data reports with explicit labeled-subset coverage. Add `--strict` to require the complete frozen sample for final research claims. Missing labels never become artificial negatives or disable application use. Phase 13 exits 2 when strict E1 is blocked, 1 on an evaluation error, and 0 when the requested analysis ran. Always inspect numbers_frozen to determine final evidence completion.

Explicit heuristic simulations use --simulated-oracle and record a run manifest. E5 is a theoretical token/cost calculation; E6 is ideal oracle review. Neither is measured human pilot evidence. Offline classifier ablations deliberately differ from the live promotion guard. Use a separate --results-dir for new analysis.

Historical paper drafts and result tables have not been regenerated and must not be treated as updated evidence.

## Scheduling and deployment

The daily runner supports --user-id all --once or --daemon --interval-hours 24 and can run under Task Scheduler/cron. Compose starts PostgreSQL only; this repository does not provision a hosted scheduler. Web, worker, and database remain separate processes. Hosted memory, latency, live arXiv availability, and Gemini output quality require deployment-specific verification.
