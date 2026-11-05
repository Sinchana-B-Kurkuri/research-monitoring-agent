# Repository audit and map

This cleanup preserves the application at `1299132`: the existing website, routes, templates, assets, database code, workers, dependencies, and public script entry points are unchanged. The earlier backend changes remain in the Git history. See [backend repair notes](BACKEND_REPAIRS.md) for the detailed repair history and [codebase analysis](CODEBASE_ANALYSIS.md) for the pre-repair baseline.

## Repository map

| Area | Location and responsibility |
| --- | --- |
| Website | `src/research_monitor/ui/`: FastAPI routes, pilot access, Jinja templates, HTMX, CSS, and browser job launch. |
| Monitoring | `scripts/run_daily_monitoring.py`, `scripts/run_web_job.py`, and `src/research_monitor/ingestion/`: arXiv retrieval, preprocessing, embeddings, and job progress. |
| Ranking and adaptation | `src/research_monitor/relevance/`: scoring, three-way routing, reference thresholds, feedback-driven interest vectors, and classifier promotion. |
| Review and delivery | `src/research_monitor/review/` and `summary/`: queue actions, digest assembly, feedback, and gated Gemini calls. |
| Storage | `src/research_monitor/database/` and `docker-compose.yml`: schema, pgvector, effective feedback view, and local PostgreSQL. |
| Evaluation | `src/research_monitor/evaluation/`, `scripts/run_phase*_evaluation.py`, and `experiments/results/`: evaluation logic and historical outputs. Offline simulations are separate from live human evidence. |
| Tests and working notes | `tests/` uses isolated database fixtures and mocks. `scratch/` retains the original roadmap and a local label-audit utility. |
| Documents | [Project reports](reports/README.md), [dashboard workflow](DASHBOARD_WORKFLOW.md), and [Render deployment](RENDER_DEPLOYMENT.md). |

## Findings and actions

- The checkout had a local repair branch ahead of GitHub's `main`, with no divergence. Consolidate by fast-forwarding `main`, preserving the individual backend and UI commits. There was no repair branch on GitHub to remove.
- The three report deliverables were scattered at the root. They now live under `docs/reports/`, with the corrected full report, editable LaTeX archive, and historical material clearly separated. The report binaries were not edited. Historical documentation links that pointed to one computer now use repository-relative links.
- The local `AI/` directory is a separate nested workspace containing document build artifacts. It and the duplicate root `fig1.png` remain on disk and are excluded from this checkout through `.git/info/exclude`. Generated outputs and private local data are kept out of the repository. `.env.example` remains trackable while `.env` stays ignored.
- The workflow guide now describes the current warm theme and navigation. Two test files had excess end-of-file whitespace; their parsed Python code is identical after cleanup.
- A hosted service starting Uvicorn directly will not automatically create the new `background_jobs` table. The [Render guide](RENDER_DEPLOYMENT.md) shows a startup command that checks the table without logging a private workspace link. Existing hosted PostgreSQL data stays separate from the Git checkout. Preserve that database and verify its schema before any broader migration.
- The web service can be healthy even when model downloads, worker memory, or missing reference embeddings prevent onboarding or monitoring. The Render guide separates web startup checks from those hosted workflow checks. There is no hosted scheduler or live infrastructure configuration checked into this repository, and most Python dependencies remain unpinned. These are deployment limits, not reasons to change the current UI behavior during cleanup.
- The final report retains its original research claims. Its Gemini 1.5 Flash and temperature `0.2` wording differs from current code defaults (`gemini-3.5-flash-lite`, `0.0`). Hosted overrides and historical run settings were not verified. The historical cost assumptions and reported results should not be treated as current deployment measurements. The evaluation UI can report available labels without claiming full frozen evidence.

## Verification

- Compiled 91 tracked Python files in memory and checked 49 literal absolute project import targets. None were unresolved in the checkout. The installed environment passed `pip check`.
- Ran the full suite against a separate PostgreSQL/pgvector test container: **265 passed**, with one third-party Starlette TestClient deprecation warning. No production database or Gemini call was required.
- Started Uvicorn with the documented job-table bootstrap against the disposable database. Health, workspace, onboarding, stylesheet, logo, and HTMX returned HTTP 200. The unauthenticated dashboard returned 401. A trusted proxy HTTPS request passed the same-origin check and still required authentication.
- Checked Git text files and the incoming repair commits for common credential signatures without printing matches or reading `.env`. No candidates matched the checks. This was a limited signature scan, not a security certification.
- Verified both document archives, the abstract package, and the PDF headers. The final PDF matches the verified formatting deliverable byte for byte.

Render account settings and a live deploy were not inspected. Verify the service's branch, deployment commit, health, and one existing hosted pilot after publishing `main`.
