# Phase 9 — Cold-Start Deployment & Pilot Launch Tracking

This operational document defines the protocol, daily run checklist, and verification rubric for executing the **7 consecutive daily monitoring cycles** required for pilot validation of the `research-monitoring-agent`.

---

## 1. Pilot Objectives & Guardrails

The goal of the cold-start pilot is to evaluate autonomous arXiv research monitoring with genuine human feedback under production conditions.

### Core Rules & Constraints
1. **Never fabricate pilot users, completed cycles, or results.** All entries in this checklist and tracker must correspond to real rows in the PostgreSQL database.
2. **Strict tenant isolation:** User access tokens protect private review queues, digests, and feedback. Never expose one user's data to another.
3. **Conservative operational bounds:**
   - Ingestion cap: 50 papers per daily cycle.
   - Review queue cap: maximum 15 borderline tasks per cycle.
   - Digest delivery cap: maximum 20 delivered papers per cycle.
   - Gated summarization: LLM summaries generated strictly for delivered papers.
4. **Zero-result runs:** Days with 0 new arXiv submissions (e.g., weekends or category lulls) are recorded as `[CYCLE_EMPTY]` and counted toward operational health without penalty.

---

## 2. Seven-Day Monitoring Checklist

Track each of the seven consecutive calendar days using the table below. To verify the current state against the live database at any time, run:

```powershell
python scripts/track_pilot.py --user-id <user_id>
```

| Day | Target Date | Cycle ID | Papers Fetched | New Ingested | High (Auto) | Borderline (Review) | Low (Discard) | Review Done? | Feedback 👍/👎 | Cycle Status | Verified By |
|:---:|:-----------:|:--------:|:--------------:|:------------:|:-----------:|:-------------------:|:-------------:|:------------:|:--------------:|:------------:|:-----------:|
| **1** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |
| **2** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |
| **3** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |
| **4** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |
| **5** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |
| **6** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |
| **7** | _Pending_ | _Pending_ | — | — | — | — | — | [ ] | 0 / 0 | `[PENDING]` | — |

*Note: Table rows start empty and must be updated with actual database outputs after running daily cycles.*

---

## 3. Daily Execution Runbook

Follow these four steps for each daily monitoring cycle:

### Step 1: Execute Daily Monitoring Cycle
Run the headless daily runner in PowerShell:

```powershell
python scripts/run_daily_monitoring.py --user-id <user_id> --once
```

**Expected outcomes:**
- `[CYCLE_SUCCESS]`: New papers retrieved from arXiv, deduplicated against database, embedded, scored against user's interest vector, and routed.
- `[CYCLE_EMPTY]`: 0 papers fetched from arXiv (e.g. arXiv weekend lull). Cycle completes cleanly.
- `[CYCLE_FAILURE]`: An error occurred (e.g., network timeout, database unreachable). Transaction is rolled back.

### Step 2: Human Review for Borderline Papers
Open the researcher's personalized review queue URL:
```text
http://localhost:8000/review?token=<user_access_token>
```
- Inspect candidate papers (title, abstract, similarity score, routing reason).
- Submit decisions: **Accept**, **Reject**, or **Skip**.
- Click **"Publish / Resume Digest"** to finalize accepted papers into the daily digest.

### Step 3: Digest Inspection & Feedback
Open the daily research digest URL:
```text
http://localhost:8000/digest?token=<user_access_token>
```
- Read delivered high-confidence papers and their interest-framed Gemini summaries.
- Provide interactive feedback: click **Like (👍)** or **Dislike (👎)**.
- Feedback is recorded into the `feedback_events` table and the `training_labels` view.

### Step 4: Audit & Verify Pilot Progress
Run the pilot progress tracker to inspect DB telemetry and update the streak:

```powershell
python scripts/track_pilot.py --user-id <user_id>
```

---

## 4. Automated Scheduling Options

If automated daily execution is preferred over running `--once` manually, two zero-dependency options are available:

### Option A: Python In-Process Daemon
Runs continuously in a lightweight loop without Celery, Redis, or Kubernetes:

```powershell
python scripts/run_daily_monitoring.py --user-id all --daemon --interval-hours 24
```

### Option B: Windows Task Scheduler
Create a scheduled task that executes once every 24 hours:

```powershell
$Action = New-ScheduledTaskAction -Execute "python.exe" -Argument "scripts/run_daily_monitoring.py --user-id all --once" -WorkingDirectory "$PWD"
$Trigger = New-ScheduledTaskTrigger -Daily -At 08:00
Register-ScheduledTask -TaskName "ResearchMonitorDaily" -Action $Action -Trigger $Trigger
```

### Option C: Linux / Crontab (on Cloud VM)
```bash
0 8 * * * cd /app && /venv/bin/python scripts/run_daily_monitoring.py --user-id all --once >> /var/log/daily_monitor.log 2>&1
```

---

## 5. Acceptance Criteria for Pilot Completion

The Phase 9 pilot is formally successful when:
1. **7 consecutive daily monitoring cycles** have executed without unhandled crashes.
2. At least **5 review decisions** (Accept or Reject) have been recorded by the pilot user.
3. At least **5 user feedback signals** (Like or Dislike) have been logged on delivered digest items.
4. The SQL view `training_labels` contains valid rows ready for future preference adaptation (Phase 10).
5. No cross-tenant data leakage or secret exposure occurred during the pilot period.
