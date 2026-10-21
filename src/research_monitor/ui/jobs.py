"""Launch existing Python processing in a separate process, with durable status."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import subprocess
import sys
from uuid import uuid4
from contextlib import contextmanager
from sqlalchemy import text

from research_monitor.database.models import BackgroundJob, User
from research_monitor.database.session import SessionLocal, engine

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ("queued", "running")


@contextmanager
def processing_slot():
    """One heavyweight web worker at a time on this database installation."""
    if engine.dialect.name != "postgresql":
        yield
        return
    with engine.connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(75193042)"))
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(75193042)"))


def expire_interrupted_jobs(session, user_id):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=20)
    for job in session.query(BackgroundJob).filter(BackgroundJob.user_id == user_id,
            BackgroundJob.status.in_(ACTIVE), BackgroundJob.created_at < cutoff):
        job.status, job.phase = "failed", "Worker interrupted or exceeded its time limit"
        job.error = "This session did not finish. Its saved history is retained; start a new session."
        job.finished_at = datetime.now(timezone.utc)
    session.flush()


def enqueue_job(session, user_id, kind, options):
    session.query(User).filter_by(id=user_id).with_for_update().one()
    expire_interrupted_jobs(session, user_id)
    existing = session.query(BackgroundJob).filter(BackgroundJob.user_id == user_id,
        BackgroundJob.kind == kind, BackgroundJob.status.in_(ACTIVE)).first()
    if existing:
        session.commit()
        return existing, False
    job = BackgroundJob(id=uuid4().hex, user_id=user_id, kind=kind, options=options)
    session.add(job)
    session.commit()
    return job, True


def fail_job(job_id, message):
    with SessionLocal() as session:
        job = session.get(BackgroundJob, job_id)
        if job and job.status in ACTIVE:
            job.status, job.phase, job.error = "failed", "Session failed", message
            job.finished_at = datetime.now(timezone.utc)
            session.commit()


def run_background_job(job_id):
    process = None
    try:
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/run_web_job.py"), "--job-id", job_id],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with SessionLocal() as session:
            job = session.get(BackgroundJob, job_id)
            if job:
                job.worker_pid = process.pid
                session.commit()
        code = process.wait(timeout=900)
        if code:
            fail_job(job_id, "The processing worker stopped unexpectedly. Check database and network availability, then retry.")
    except subprocess.TimeoutExpired:
        if process:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
            else:
                process.kill()
        fail_job(job_id, "The session exceeded 15 minutes. Saved results remain available; retry later.")
    except Exception:
        fail_job(job_id, "Unable to start the processing worker. Restart the application and retry.")
