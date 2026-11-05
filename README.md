# Research Monitoring Agent

The research framework tracks arXiv papers against a pilot's interests. It routes high-relevance papers to a personalized digest, sends borderline papers to human review, and records low-relevance decisions in an audit trail. Gemini summarizes eligible papers. Review and digest feedback inform future relevance decisions.

**Start here:** [Run the dashboard locally](#local-setup) · [Use the dashboard](#dashboard-workflow) · [Run the tests](#tests) · [Read the reports](#reports)

## Architecture

| Area | Role |
| --- | --- |
| Ingestion and embeddings | The arXiv client fetches papers, and Sentence Transformers creates 384-dimensional embeddings with `all-MiniLM-L6-v2` by default. |
| Relevance and preference learning | A Python state machine scores each paper against the active interest profile. Percentile thresholds derived from the reference sample separate HIGH, BORDERLINE, and LOW. Rocchio updates the profile from feedback, and logistic regression is promoted only after validation. |
| Review and delivery | FastAPI serves Jinja2 templates and local HTMX assets. The dashboard starts monitoring and evaluation jobs and shows review, digest, audit, and history. |
| Persistence and summaries | SQLAlchemy, PostgreSQL, and pgvector store papers, embeddings, decisions, feedback, and model versions. The Google Gen AI SDK generates summaries for eligible papers and the database caches them. |

## Local setup

Use Python 3.11 or newer, Docker with the Compose plugin, and a terminal in the repository root. The examples use PowerShell. On macOS or Linux, use `./venv/bin/python` and your shell's equivalents for environment variables and file copies.

```powershell
if (-not (Test-Path .\venv\Scripts\python.exe)) { python -m venv venv }
.\venv\Scripts\python.exe -m pip install -e .
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit your local `.env` using [.env.example](.env.example) as a reference. The defaults match the local Compose database. Set `DATABASE_URL` only if your local connection differs.

| Variable | Local use |
| --- | --- |
| `DATABASE_URL` | Default is `postgresql+psycopg2://postgres:postgres@localhost:5432/research_monitor`, using the example Compose credentials. |
| `GEMINI_API_KEY` | Optional for local use. Add your key for generated summaries. With an empty key, the digest displays original abstracts and pending summaries. Restart the app after changing the key. |
| `TEST_DATABASE_URL` | Set this in the test terminal before running pytest. Use a separate disposable PostgreSQL database. |

The embedding model needs a local cache or network access for its initial download. Live ingestion needs arXiv access. Use corpus replay for a demonstration based on saved papers.

### Start PostgreSQL and load reference data

```powershell
docker compose up -d db
docker compose exec -T db pg_isready -U postgres -d research_monitor
```

Once PostgreSQL reports that it is accepting connections, reuse the existing pair `data/corpus.parquet` and `data/embeddings.npy`. If you do not have that pair, restore it from your research artifacts or generate a fresh pair with `scripts/run_phase2.py --max-results 10000` using the virtual environment. Generation writes those paths, so back up any existing corpus or embeddings first. The default corpus target leaves a separate evaluation frame alongside the 5,000 reference papers.

```powershell
.\venv\Scripts\python.exe scripts/run_phase3_loader.py
docker compose exec -T db psql -U postgres -d research_monitor -c "SELECT count(*) FROM papers WHERE is_reference AND embedding IS NOT NULL;"
```

The loader initializes the schema and inserts missing paper IDs while preserving existing papers and pilot records. The reference count should be at least 5,000 with default settings. To initialize the schema separately, run `python -c "from research_monitor.database.session import init_db; init_db()"` in the virtual environment.

### Start FastAPI and initialize a pilot

```powershell
.\venv\Scripts\python.exe scripts/run_ui.py --host 127.0.0.1 --port 8000
```

From another terminal, `Invoke-RestMethod http://127.0.0.1:8000/health` checks the app and database connection. Keep the server terminal open while using the dashboard.

### Dashboard workflow

1. Open the private workspace link printed in the terminal to see saved pilots. You can also open [Initialize a pilot](http://127.0.0.1:8000/onboard) directly.
2. Create a pilot with a short description of its research interests. The optional preference-memory selector copies effective feedback from a profile you can already access.
3. Open **Dashboard & settings**, choose arXiv categories, the lookback window, and a paper limit, then click **Start monitoring session**. Refreshing the page preserves job history.
4. Use **Review** to handle borderline papers and **Digest** to read delivered papers and provide feedback. **Audit** shows discarded papers, and **History** shows earlier cycles and feedback. **Back to pilots** returns to the workspace and its saved links.
5. Use **Evaluation** to run the available-data report. The report records label coverage and distinguishes partial results from the complete frozen evaluation.

For a terminal-based demonstration, replace `researcher` with an existing pilot ID:

```powershell
.\venv\Scripts\python.exe scripts/run_daily_monitoring.py --user-id researcher --once
.\venv\Scripts\python.exe scripts/run_cycle.py --user-id researcher --mode replay --as-of-date 2024-12-31 --limit 50
```

After a historical replay, open the review page and use its digest update button. The daily runner also supports `--daemon --interval-hours 24` for local scheduling.

## Tests

Use a disposable PostgreSQL/pgvector container so integration tests stay separate from pilot data. If port 55433 is occupied, choose another port in both the container command and the URL.

```powershell
docker run -d --rm --name research-monitor-local-tests -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=research_monitor_test -p 127.0.0.1:55433:5432 pgvector/pgvector:pg16
docker exec research-monitor-local-tests pg_isready -U postgres -d research_monitor_test
```

Wait for PostgreSQL to accept connections, then run:

```powershell
$env:TEST_DATABASE_URL = 'postgresql+psycopg2://postgres:postgres@127.0.0.1:55433/research_monitor_test'
.\venv\Scripts\python.exe -m pytest -q
Remove-Item Env:TEST_DATABASE_URL
docker stop research-monitor-local-tests
```

PostgreSQL integration tests skip without `TEST_DATABASE_URL`. The test fixtures isolate generated outputs and use no live Gemini credential.

## Research evaluation

Use the saved corpus, embeddings, evaluation sample, and human labels for research runs. If the evaluation sample is missing, run `scripts/prepare_evaluation_sample.py` in the virtual environment before labeling. Check label coverage with `scripts/label_papers.py --status`. Use `scripts/run_phase4_evaluation.py` to run baselines and calibrate thresholds on development labels. The default local calibration path is `data/calibrated_thresholds.json`.

```powershell
# Evaluate the available human labels and export pilot activity.
.\venv\Scripts\python.exe scripts/run_phase13_evaluation.py --results-dir data/local_evaluation/results --export-dir data/local_evaluation/export

# Run offline experiments, allowing a simulated oracle when labels are missing.
.\venv\Scripts\python.exe scripts/run_phase12_experiments.py --all --simulated-oracle --results-dir data/local_simulations
```

Add `--strict` to Phase 13 for the complete frozen-sample requirements. Check the report's label coverage and `numbers_frozen` before citing results. Simulations are separate from human pilot evidence. Use separate output directories for new runs to preserve the committed tables and figures.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/research_monitor/` | FastAPI, templates and assets, database, ingestion, relevance, review, summary, and evaluation modules |
| `scripts/` | Local startup, corpus preparation, monitoring, and evaluation commands |
| `tests/` | Unit and integration tests |
| `data/` | Local corpus, embeddings, labels, and runtime outputs. Git tracks only the directory placeholder. |
| `experiments/results/` | Committed research tables and figures |
| `docs/` | Abstract, initial report, final ACM paper, and editable source archive |

## Reports

1. [Abstract](docs/01-abstract.docx)
2. [Initial report](docs/02-initial-report.pdf)
3. [Final ACM research report](docs/03-final-report.pdf)

The [final LaTeX archive](docs/03-final-report-latex.zip) contains the editable paper, bibliography, figures, and build instructions.
