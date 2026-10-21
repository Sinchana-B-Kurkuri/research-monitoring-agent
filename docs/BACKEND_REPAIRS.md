# Backend diagnostics, repairs, and verification

21 October 2026. Work is on `codex/backend-repairs`. Nothing was pushed to GitHub. The existing module layout, state machine, database entities, and web/worker split were retained.

## Diagnostic summary and solution plan

The original isolated suite passed 217 tests and skipped eight PostgreSQL tests. Passing tests concealed module-boundary failures; some tests explicitly asserted unsafe behavior. The repair plan was to preserve the supplied work, add regressions, repair storage/startup, secure authentication and review transitions, reconcile digest delivery/retry, constrain monitoring inputs, and correct evaluation validity before final verification.

| Error and root cause | Implemented repair |
|---|---|
| Existing user ID returned its bearer token without authentication | Anonymous duplicate registration returns 409; authenticated retries require the existing token. CLI `--require-new` prevents concurrent signup from recovering another account's credentials. |
| Synchronous onboarding subprocess blocked the async request | Run it in the thread pool, retain timeout handling, and avoid exposing stderr/credentials to clients. |
| Partial profile recovery evaluated a NumPy vector as a Boolean | Use an explicit None check; repair missing access tokens through the trusted CLI path. |
| Fresh database initialization altered nonexistent tables and rolled back pgvector creation | Commit extension creation independently and guard legacy table alterations. Migration failures propagate. |
| ORM metadata created the effective-label view as a table | Remove its query-only mapping from table-creation metadata; initialize the SQL view idempotently. Refuse to delete unexpected nonempty legacy tables. |
| Loader called `next()` on a context manager | Use `with get_session()` and a testable `load_corpus()` function. |
| Loader deleted all papers and split category strings into characters | Insert only missing IDs; validate/normalize vectors and split comma-separated categories correctly. |
| Direct package/Uvicorn imports failed outside pytest | Add editable package configuration and template package data. Declare TestClient's HTTP dependency and missing typing imports. |
| Tests could contact the application's database and overwrite E5/E6 results | Use explicit `TEST_DATABASE_URL`; default to an unavailable test-only address. Route generated test results and access logs to temporary directories. |
| Existing digest returned before processing later review acceptances | Reconcile newly eligible items into the same digest, preserving delivered slots and the cap. |
| Summary failure was overwritten with DELIVERED and never retried | Keep summary pending while the abstract is visible; resume retries missing summaries. Daily runs revisit earlier pending cycles. |
| Reacting to an abstract changed its status and hid its missing summary from the retry scan | Also select delivered/reacted decisions without summaries; retry generation while preserving the user's reaction and existing digest item. |
| Acceptance background task summarized without ensuring delivery/cap | Reconcile the complete cycle digest in the background worker. |
| Repeated/terminal review actions could append duplicate events or revive expired tasks | Lock tasks; make repeated identical final actions idempotent; reject conflicting/expired transitions. |
| Skipped tasks had no action buttons | Keep skip pending and actionable; expire stale tasks when the queue is opened. |
| Foreign cycle IDs could be resumed | Verify cycle existence and ownership before mutations. |
| Classifier-routed papers were still selected by cosine under caps | Rank new queue/digest candidates using the active method's score with deterministic tie-breaking. |
| Missing API key silently persisted fictional mock summaries | Mock mode is explicit. Normal missing-key behavior keeps the original abstract and pending summary. |
| Concurrent resumes could generate/bill twice | Serialize resume using the cycle row lock, lock summary decisions, and avoid intermediate digest commits. Verified with two real PostgreSQL sessions. |
| Live discovery replayed historical/reference papers on empty days | Pass the fetched candidate set, including globally shared papers other users still need; empty means empty. |
| Multi-user execution fetched arXiv repeatedly | Fetch once per round, then deduplicate/embed/route for each user's outstanding candidates. |
| Live embedding fallback imported a nonexistent module; replay silently dropped embeddings | Use the existing embedder and align artifact vectors by paper ID; reject missing/invalid vectors. |
| Failures disappeared or still produced success; unknown users selected another account | Persist failed cycles after rollback, propagate stage failures, and reject unknown IDs. |
| Tracker counted dislike=-1 and replay days as live streaks | Count stored label 0 and only completed successful live days. |
| Review cap/calibration settings were disconnected | Honor configured queue caps and valid saved development percentile calibration, with explicit environment precedence. |
| Database percentile failure left the session aborted before fallback | Roll back the failed statement through a savepoint; verified against PostgreSQL with an intentionally invalid query. |
| Rocchio promotion baseline already contained held-out feedback | Recompute the baseline from each training fold and the original seed when evaluating an adapted centroid. |
| E1 treated unknown human labels as negatives | Require label coverage for every requested test/development paper and aligned embeddings. Final evaluation reports missing inputs explicitly. |
| Offline oracle silently used keywords for missing human ratings | Require explicit heuristic simulation; record coverage, sample hash, and simulation assumptions in a manifest. |
| Cosine and probability F1 used the same arbitrary 0.5 cutoff | Compare simulated scorers at a matched top-2-percent delivery volume. |
| Final simulation flag was unused; custom result paths were ignored in places | Execute requested E2-E6, parameterize destinations, and generate figures from that destination. |
| Readiness counted raw repeated feedback, unfinished/error cycles, or one participant's streak for all | Share a valid-live/effective-label eligibility contract; require every genuine participant's protocol and identity-matched survey. |
| Export ignored user filters, double-counted overlapping exclusions, and lost the first model activation | Apply the filter, use disjoint partition counts, exclude early feedback, and retain the first promoted-model timestamp. |
| Efficiency mixed replay/simulated/developer decisions and unrelated tasks/tokens | Filter all metrics to the same eligible population; distinguish queue/digest overflow and calculate actual task outcomes per cycle. Exclude mock token records. |
| Empty surveys/exports caused missing-key or empty-CSV errors | Return a consistent survey schema and readable empty CSV headers; validate survey identifiers. |
| Bearer tokens could leak through default access logs and referrers | Redact query/path tokens and set no-referrer/no-store response headers. |

The earlier analysis remains available as a historical record. Performance/scaling improvements such as UI pagination were not substituted for these correctness repairs. No hosted scheduler, paid model run, real participant data, or new research result was fabricated.

## Implementation commits

| Commit | Contents |
|---|---|
| `0e4cb5d` | Checkpoint of the user's existing tracked deployment/onboarding changes, before repairs |
| `3f26fe9` | Schema/view initialization, safe corpus loader, editable packaging, test isolation |
| `b85d0f7` | Authentication, review transitions, late digest delivery, summary retries, explicit mocks |
| `4945ec5` | Live candidate scope, embedding fallbacks, durable failures, calibration, tracker/runner corrections |
| `02fec0c` | Ground-truth coverage, simulation provenance, shared pilot eligibility, export and efficiency corrections |
| `ff45589` | Held-out baseline, SQL savepoints, bearer-link privacy, empty export/figure safeguards |

Exact code changes can be reviewed locally:

```powershell
git log --oneline --stat 9c39080..HEAD
git diff 0e4cb5d..HEAD -- src scripts tests pyproject.toml requirements.txt
git show b85d0f7
```

The documentation commit `dd52d97` adds this guide and the historical analysis. A subsequent focused retry commit covers reactions made before a summary is available and updates final verification to 248 passing tests. The existing untracked `report/` folder is intentionally not included in repair commits.

## 1. Install and configure

Use PowerShell:

```powershell
Set-Location 'C:\Users\sinch\Desktop\AI Project\research-monitoring-agent'
if (-not (Test-Path .\venv\Scripts\python.exe)) { python -m venv venv }
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e . --no-deps
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

For the existing local Compose database, put this in `.env`:

```text
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/research_monitor
```

Set `GEMINI_API_KEY` there only if actual generated summaries are wanted. Without it, the backend still delivers the author abstract and marks the summary pending. The embedding model needs a cached copy or network access for its initial download.

## 2. Initialize and load papers

```powershell
docker compose up -d db
.\venv\Scripts\python.exe -c "from research_monitor.database.session import init_db; init_db()"
.\venv\Scripts\python.exe scripts/run_phase3_loader.py
```

The current local corpus/embedding artifacts can be reused. If absent, generate them before running the loader:

```powershell
.\venv\Scripts\python.exe scripts/run_phase2.py --max-results 10000
.\venv\Scripts\python.exe scripts/run_phase3_loader.py
```

The default reference size is 5,000. A 10,000-paper corpus leaves a separate evaluation frame. The loader preserves existing IDs, and running it again reports zero newly inserted papers when the corpus is already loaded.

## 3. Create a user and start the web backend

```powershell
.\venv\Scripts\python.exe scripts/create_user.py --user-id researcher --name 'Researcher' --seed-text 'Autonomous AI agents, language model reasoning, reinforcement learning, and tool use.'
.\venv\Scripts\python.exe scripts/run_ui.py --host 127.0.0.1 --port 8000
```

Save the printed token links. An existing user's token is recoverable only through a trusted local operator's CLI or a request that already supplies the correct token. An anonymous form submission with an existing user ID receives 409.

In another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected fields: `status=ok`, `database=connected`, `version=0.9.0`. Open the personal review/digest/audit URLs. An unauthenticated digest request should return HTTP 401.

## 4. Verify routing, review, and delivery

For deterministic use of the saved corpus:

```powershell
.\venv\Scripts\python.exe scripts/run_cycle.py --user-id researcher --mode replay --as-of-date 2024-12-31 --limit 50
```

Open the review link and use its digest update button. HIGH papers enter the digest; BORDERLINE papers await review; LOW papers appear in the audit page. Accept a borderline paper, then open/update the digest again: it is added once if capacity remains. Reject/skip and like/dislike decisions preserve their distinct feedback sources.

For actual live retrieval:

```powershell
.\venv\Scripts\python.exe scripts/run_daily_monitoring.py --user-id researcher --once
```

Expected outcome is `[CYCLE_SUCCESS]`, `[CYCLE_EMPTY]`, or a recorded `[CYCLE_FAILURE]`. Live retrieval requires arXiv availability. A valid empty fetch does not route old corpus papers. The daily command also evaluates vector/classifier updates and retries older pending summaries.

## 5. Run every test against disposable PostgreSQL

Do not point `TEST_DATABASE_URL` at the pilot database: integration tests intentionally insert/update/delete their own fixtures.

```powershell
docker run -d --name research-monitor-tests -e POSTGRES_PASSWORD=backend_test -e POSTGRES_DB=research_monitor_test -p 127.0.0.1:55433:5432 pgvector/pgvector:pg16
docker exec research-monitor-tests pg_isready -U postgres -d research_monitor_test
$env:TEST_DATABASE_URL='postgresql+psycopg2://postgres:backend_test@127.0.0.1:55433/research_monitor_test'
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m pip check
```

Repeat `pg_isready` until it reports accepting connections. If the named container already exists, use `docker start research-monitor-tests`. If port 55433 is occupied, select another port and update the URL. Without `TEST_DATABASE_URL`, PostgreSQL-specific tests skip; other tests remain isolated from `.env`.

Optional model-offline mode, once the model is cached:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

The repair verification used a separately created container named `research-monitor-backend-tests`, never the existing application container. Its connection port was assigned dynamically. Tests now keep experiment outputs in temporary directories.

## 6. Evaluate without replacing historical evidence

```powershell
.\venv\Scripts\python.exe scripts/run_phase13_evaluation.py --results-dir data/verified_evaluation/results --export-dir data/verified_evaluation/export --survey-dir data/surveys --figures
$LASTEXITCODE
```

With the current 171/500 labels, expect `BLOCKED_INCOMPLETE_DATA`, a message identifying 329 missing labels, `numbers_frozen=false`, and exit code 2. This is the correct data-quality result. Exit 1 means an evaluation error. Exit 0 means the requested analysis ran, not necessarily that the pilot is complete. Inspect the generated JSON's readiness and completion fields.

Collect genuine missing ratings with the existing labeling CLI:

```powershell
.\venv\Scripts\python.exe scripts/label_papers.py --status
.\venv\Scripts\python.exe scripts/label_papers.py --user-id developer
```

For explicitly heuristic exploration, with separate result files:

```powershell
.\venv\Scripts\python.exe scripts/run_phase12_experiments.py --all --simulated-oracle --results-dir data/simulation_results
```

These are simulations. The manifest records the sample hash, human/heuristic label counts, and model/metric policy. E5 uses theoretical token/cost assumptions and E6 assumes ideal oracle review. Neither establishes measured production savings or real human review performance. Final human-label metrics require complete requested label coverage.

## Verification evidence

- Full suite: **248 passed, zero skipped, one third-party TestClient deprecation warning, 31.36 seconds**.
- Dependency check: no broken requirements.
- Syntax check: every Python source/script/test file compiled.
- Fresh PostgreSQL initialization succeeded twice with 13 tables and one effective-label view.
- Real loader inserted 10,000 papers into disposable PostgreSQL; user creation computed thresholds from 5,000 reference vectors.
- Replay smoke test: 50 decisions, HIGH=2, BORDERLINE=3, LOW=45 for the specific smoke interest paragraph. Counts naturally differ for another profile or subsequent run.
- HTTP smoke test: health/review/digest/audit 200, unauthenticated digest 401, anonymous duplicate onboarding 409, review acceptance 200.
- Existing digest grew from two items to three after a later acceptance; explicit mock generation was used only for this simulated smoke user.
- Actual Uvicorn access logs did not contain the smoke user's token.
- Two simultaneous PostgreSQL resumes produced one item, one summary, and one mock-generation call.
- A deliberately failed PostgreSQL percentile statement fell back successfully without leaving the session aborted.
- Phase 13 completed its export/report pipeline while holding E1 for missing human labels; no final numbers were frozen.

Gemini generation, live external arXiv availability, deployment memory/latency, and genuine pilot recruitment/observations were not established by these tests. Historical paper tables/figures remain unchanged and need separate regeneration after valid evidence is collected. The repository still has no provisioned hosted scheduler.
