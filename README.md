# Relevance-Gated Agentic Workflow for Autonomous Research Monitoring

An adaptive, autonomous AI research monitoring platform that discovers, embeds, ranks, gates, summarizes, and learns from user feedback on live arXiv research papers.

---

## Architecture & System Overview

```
[arXiv Live Submissions]
          │
          ▼
[Ingestion & Deduplication] ──── (Version strip, DB dedup, 384-d MiniLM embeddings)
          │
          ▼
[Relevance State Machine]   ──── (Cosine similarity vs. calibrated P_high / P_low)
   ├── HIGH (≥ t_high)      ────► [Auto-Delivered] ──► [Gated Gemini Flash Summary] ──► [Digest]
   ├── BORDERLINE           ────► [Human Review Queue (Max 15)] ──► [Accept / Reject / Skip]
   └── LOW (< t_low)        ────► [Audit Log (Discarded)]
                                                              │
                                                        [User Feedback] (👍 / 👎)
                                                              │
                                                              ▼
                                                   [training_labels SQL View]
```

### Core Architecture Highlights
- **Relevance Gating:** Two calibrated empirical percentile thresholds ($t_{high}, t_{low}$) route papers to prevent notification fatigue.
- **Gated LLM Summarization:** Google Gemini 2.5 Flash generates interest-framed summaries *only* for delivered papers, with MD5-cached persistence.
- **Lightweight Production Web Service:** FastAPI web portal operates with minimal memory and sub-100ms response times. PyTorch and `sentence_transformers` are **never** imported into the web process.
- **Hosted PostgreSQL + pgvector:** Resilient connection pooling (`pool_pre_ping=True`, `pool_recycle=300`) supports cloud databases (Neon, Supabase, AWS RDS, GCP Cloud SQL) and local Docker Compose.
- **Data Isolation:** Access-token authentication on all URLs ensures complete privacy between pilot users.

---

## Quickstart & Local Setup

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### 2. Database Setup
Ensure PostgreSQL with pgvector is running (either via local Docker Compose or cloud hosted instance):
```powershell
docker compose up -d
```
Set `DATABASE_URL` in `.env`:
```text
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/research_monitor
```

Initialize database tables and pgvector extension:
```powershell
python -c "from research_monitor.database.session import init_db; init_db()"
```

---

## Production Deployment (Phase 9)

### Production Start Command
Start the production ASGI web server using Uvicorn:
```powershell
uvicorn research_monitor.ui.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Health Check Endpoint
Cloud orchestrators and load balancers (Render, Fly.io, Railway, AWS, GCP) can monitor health via:
```text
GET /health
```
- **200 OK:** Database reachable (`{"status": "ok", "database": "connected", "version": "0.9.0"}`).
- **503 Service Unavailable:** Database connectivity failure (`{"status": "degraded", "database": "disconnected"}`).

### Required Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | Local postgres URL | Hosted PostgreSQL connection string (supports `?sslmode=require`) |
| `GEMINI_API_KEY` | `""` | Google Gemini API key for gated summaries (fallback heuristic active if omitted) |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model name for summarization |
| `PORT` | `8000` | Port for the web service |
| `HOST` | `0.0.0.0` | Host interface for web service |
| `DAILY_INGESTION_MAX_PAPERS` | `50` | Conservative ceiling on papers ingested per cycle |
| `REVIEW_QUEUE_MAX_TASKS` | `15` | Maximum borderline review tasks queued per cycle |
| `DIGEST_MAX_ITEMS` | `20` | Maximum delivered papers per digest |

---

## Cold-Start Pilot Onboarding

New researchers can be onboarded in seconds through either the Web UI or the CLI:

### Option A: Web Onboarding
1. Navigate to `http://localhost:8000/onboard` (or root `http://localhost:8000/`).
2. Enter User ID, Display Name, and a descriptive paragraph of research interests.
3. The system generates 384-dimensional seed embeddings, computes threshold boundaries, and displays your personalized access links.

### Option B: CLI Onboarding
```powershell
python scripts/create_user.py --user-id pilot_researcher --name "Dr. Alice Chen" --seed-text "Autonomous AI agents, reinforcement learning from human feedback, LLM tool use, reasoning, multi-agent systems."
```
The command displays:
- User ID and Access Token
- Personalized Review Queue URL: `http://localhost:8000/review?token=<TOKEN>`
- Personalized Digest URL: `http://localhost:8000/digest?token=<TOKEN>`
- Personalized Audit URL: `http://localhost:8000/audit?token=<TOKEN>`

---

## Daily Monitoring Execution

Execute daily arXiv monitoring cycles without Celery, Redis, or Kubernetes:

### 1. One-Shot Execution (Daily Cron / Task Scheduler)
Run a single daily monitoring cycle for all registered pilot users:
```powershell
python scripts/run_daily_monitoring.py --user-id all --once
```
Or for a specific pilot user:
```powershell
python scripts/run_daily_monitoring.py --user-id pilot_researcher --once
```

**Cycle Outcome Tags:**
- `[CYCLE_SUCCESS]`: New papers fetched, deduplicated, embedded, routed, and digests compiled.
- `[CYCLE_EMPTY]`: 0 papers fetched from arXiv (e.g. arXiv weekend lull). Handled cleanly as a valid cycle.
- `[CYCLE_FAILURE]`: Exception logged, error recorded, and transaction rolled back safely.

### 2. In-Process Background Daemon
To run continuously on a 24-hour schedule:
```powershell
python scripts/run_daily_monitoring.py --user-id all --daemon --interval-hours 24
```

---

## Pilot Tracking & 7-Day Checklist

Track real progress toward the **seven consecutive daily monitoring cycles** objective:

```powershell
python scripts/track_pilot.py --user-id pilot_researcher
```

This command queries the database directly to report:
- Active preference vector version and calibrated thresholds.
- Consecutive daily cycle streak (`[#------] 1 / 7 consecutive days`).
- Historical cycle outcomes (`[CYCLE_SUCCESS]`, `[CYCLE_EMPTY]`).
- Pending and completed review decisions (Accept / Reject / Skip).
- Digest delivery and user reaction telemetry (Likes 👍 / Dislikes 👎).

*See [`docs/pilot_tracking.md`](file:///c:/Users/sinch/Desktop/AI%20Project/research-monitoring-agent/docs/pilot_tracking.md) for the complete 7-day verification checklist.*

---

## Verification & Testing

Run compilation and the complete test suite:
```powershell
# Bytecode compilation check
python -m compileall src scripts tests -q

# Full test suite execution
pytest -v
```

---

## Troubleshooting & Rollback

### 1. Database Connection Drops
- **Symptom:** Web requests return 503 or logs show closed connection.
- **Resolution:** Hosted pools (Neon, Supabase) drop idle sockets. The engine is pre-configured with `pool_pre_ping=True` and `pool_recycle=300`. Verify `DATABASE_URL` includes `sslmode=require` if required by your provider.

### 2. Zero Results on arXiv Run
- **Symptom:** `[CYCLE_EMPTY]` is output by `run_daily_monitoring.py`.
- **Resolution:** arXiv does not publish new papers on Saturdays and Sundays (UTC). This is normal behavior and does not reset the 7-day monitoring streak.

### 3. Missing Gemini API Key
- **Symptom:** LLM summarization falls back to abstract excerpt.
- **Resolution:** Set `GEMINI_API_KEY` in `.env`. The system operates seamlessly in graceful degradation mode when no key is present.

### 4. Rollback Procedure
If a deployment needs to be rolled back:
```powershell
# Revert to last stable commit
git checkout HEAD~1

# Restart web service
uvicorn research_monitor.ui.app:app --host 0.0.0.0 --port 8000
```
Database migrations are additive; tables and views maintain backward compatibility.
