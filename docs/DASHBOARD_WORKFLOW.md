# Browser workflow and available-data evaluation

## Open the workspace

Start the existing web server with `venv\Scripts\python.exe scripts/run_ui.py`. Startup prints a private operator workspace URL. Open and bookmark that URL once: an HttpOnly, SameSite cookie retains operator access in this browser. The operator key is generated locally in ignored `data/local_runtime/workspace.key`, persists across restarts, and is never committed. Do not share that link.

The operator workspace lists every profile on this installation, including clearly marked simulations. It provides initialization dates, monitoring/digest/memory counts, and personal links. This is an account directory, not evidence that every account is a genuine human participant.

Ordinary personal links retain profile-level isolation. Opening one remembers that profile in this browser; anonymous visitors cannot list other profiles or recover their credentials. The browser's saved-profile cookie stores at most twelve bearer tokens, is HttpOnly, and lasts up to one year. Deleting cookies does not delete database records; the private operator workspace can recover links again.

Every personal page has **Pilots & initialization**, **Workspace**, **Dashboard**, **Review Queue**, **Digest**, **Audit**, **History**, and **Evaluation** navigation. Returning to initialization shows the profiles this browser can access alongside the new-profile form.

## Initialize and monitor

1. Open **Pilots & initialization** and complete the form.
2. Optionally choose a source profile under **Preference memory**. Only profiles you can access are offered. This copies effective labels as explicitly inherited memory; it does not alter the source profile or manufacture new human participation.
3. Open **Dashboard & settings**. Choose arXiv categories, a recent-day window (1–30 days), and a paper limit (1–50). Category suggestions include AI, robotics/vision, and materials/applied physics, using the [arXiv taxonomy](https://arxiv.org/category_taxonomy). The existing historical reference corpus remains AI-focused, so relevance thresholds outside that domain still require domain-specific validation.
4. Press **Start monitoring session**. No processing command needs to be copied into a terminal. The initialization success page also has a start button using the configured default categories.
5. Watch durable progress: fetching, embedding/routing, summaries, and preference updates. Navigation and refresh do not lose the saved job record.
6. Read the digest and review borderline papers. Accept/reject and like/dislike remain human actions. Skips create no negative label.

A job uses the existing daily monitoring functions in a separate Python subprocess. PostgreSQL supplies a processing-slot lock, serializing heavyweight browser jobs across server processes to avoid concurrent embedding-model memory spikes. Repeated starts of the same job kind for the same profile reuse its active job. Processing is bounded to fifteen minutes; interrupted records older than twenty minutes become retryable failures. This is a small local worker mechanism, not a hosted scheduler.

Monitoring is on demand. These changes do not provision an automatic daily OS task. Existing CLI/daemon scheduling remains available.

## History and learning

**History** retains monitoring cycles, individual routing decisions, errors/empty rounds, processing jobs, and feedback events. Pagination exposes older cycles, feedback, jobs, and papers. Each cycle links to its own digest rather than only the latest digest.

Feedback events are append-only. The effective-label view resolves one current preference per user/paper, with explicit user reactions taking precedence over review choices. Imported events carry `inherited:<source-profile>:<source-event>` provenance and no synthetic routing decision. Pilot readiness/export excludes these imported events from fresh human-evidence counts.

Saved feedback is evaluated **before** the next cycle scores papers. The original safeguards remain: first Rocchio update needs ten labels including three positives; initial classifier training needs thirty labels including eight positives and eight negatives, then must pass validation. Inherited memory can contribute to learning, but is identified separately in the dashboard.

Future Gemini summaries include nearest liked/disliked paper titles as preference context. The prompt treats those examples only as framing preferences, never as facts about the target abstract. Cache versions include the effective memory event ID when memory exists. Historical summaries are retained unless deliberately regenerated. Gemini calls have a sixty-second request timeout and bounded retries; logs record exception type/status without dumping provider messages or credentials.

HTMX 1.9.12 is served locally under `/static`, with its license and npm SHA-512 verification performed when vendored. Review interactions no longer depend on an external CDN loading.

## Evaluation without blocking application use

Press **Evaluate available data** in a profile's Evaluation page. Operational activity and memory counts are always visible. The shared benchmark uses the project's original research topic and currently human-labeled evaluation papers; it is separate from that profile's personal topic.

With the current 171/500 labels, the available-data report evaluated 50 labeled test papers and produced five baseline/gate comparisons. It reports `COMPLETED_PARTIAL`, records all label counts/scope in the report and CSV, and explicitly describes selection-bias limits. Missing labels are never assigned artificial negatives or replaced with AI judgments presented as human ground truth. A subset without enough classes receives an informative insufficient-data section while operational reports still run.

The CLI also defaults to available-data evaluation:

```powershell
.\venv\Scripts\python.exe scripts/run_phase13_evaluation.py --results-dir data/current_evaluation/results --export-dir data/current_evaluation/export
```

Add `--strict` to request the complete frozen sample for final research claims. `numbers_frozen` remains false until the full evidence/readiness requirements are satisfied. This does not disable initialization, monitoring, feedback, summaries, history, or partial reports. Backend API callers retain strict behavior unless they explicitly set `allow_partial_labels=True`.

Reports from browser jobs are isolated under ignored `data/web_evaluations/<job-id>/`. Existing historical report figures are not overwritten. A genuine evaluation error marks the job failed rather than reporting a successful benchmark.

## Verification

The implementation was verified with disposable PostgreSQL and authenticated HTTP requests, including concurrent start deduplication, global processing-slot serialization, foreign-history/job rejection, cookie-based profile recovery, cross-site write rejection, explicit feedback transfer/precedence, pre-cycle preference updates, and partial-vs-strict evaluation behavior. The full suite passed 265 tests with no skips and one third-party TestClient deprecation warning.

A real browser-triggered monitoring job for the existing robotics profile completed with four generated Gemini summaries. A real available-data evaluation completed with five benchmark rows. An earlier parallel stress attempt exceeded Windows paging capacity; its failed record is retained, and heavyweight web jobs are now serialized. Browser visual automation remained denied by a saved permission or managed policy, so browser layout and click behavior were not visually verified. No `.env` contents were inspected, no user feedback was invented, and no changes were pushed.
