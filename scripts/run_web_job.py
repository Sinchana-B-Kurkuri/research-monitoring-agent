"""Process one durable browser job using the existing monitoring/evaluation logic."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research_monitor.database.models import BackgroundJob, Digest, Summary, User
from research_monitor.database.session import SessionLocal


def progress(job_id, phase):
    with SessionLocal() as session:
        job = session.get(BackgroundJob, job_id)
        job.phase = phase
        session.commit()


def execute_job(job_id):
    with SessionLocal() as session:
        job = session.query(BackgroundJob).filter_by(id=job_id).with_for_update().one()
        if job.status != "queued":
            return
        job.status, job.started_at = "running", datetime.now(timezone.utc)
        user_id, kind, options = job.user_id, job.kind, job.options
        session.commit()
    try:
        if kind == "monitor":
            from research_monitor.ingestion.live import fetch_live_papers
            from scripts.run_daily_monitoring import execute_daily_cycle_for_user
            from research_monitor.relevance.pipeline import record_cycle_failure
            now = datetime.now(timezone.utc)
            progress(job_id, "Fetching recent arXiv submissions")
            try:
                papers = fetch_live_papers(categories=options["categories"], max_results=options["max_results"],
                    start_date=(now-timedelta(days=options["days"])).strftime("%Y-%m-%d"), end_date=now.strftime("%Y-%m-%d"))
            except Exception as error:
                with SessionLocal() as session:
                    record_cycle_failure(session, user_id, "live", now.strftime("%Y-%m-%d"), error)
                raise RuntimeError("Paper retrieval failed. Check arXiv/network availability and retry.") from None
            with SessionLocal() as session:
                user = session.get(User, user_id)
                status, cycle_id, metrics = execute_daily_cycle_for_user(user, session,
                    options["categories"], options["max_results"], options["days"],
                    papers_override=papers, on_progress=lambda phase: progress(job_id, phase))
                if cycle_id is None:
                    raise RuntimeError("Monitoring failed during processing. Its failure record is retained in session history.")
                digest = session.query(Digest).filter_by(user_id=user_id, cycle_id=cycle_id).first()
                metrics["pending_summaries"] = sum(item.routing_decision.summary is None for item in digest.items) if digest else 0
                result = {"status": status, "cycle_id": cycle_id, "metrics": metrics}
        elif kind == "evaluation":
            from research_monitor.evaluation.final_evaluation import run_final_evaluation_suite
            import pandas as pd
            progress(job_id, "Evaluating available human labels and saved pilot activity")
            output = ROOT / "data" / "web_evaluations" / job_id
            with SessionLocal() as session:
                report = run_final_evaluation_suite(db_session=session, results_dir=output / "results",
                    pilot_export_dir=output / "export", allow_partial_labels=True)
            e1 = report.get("e1_gate_quality") or {}
            if e1.get("status") == "ERROR":
                raise RuntimeError("Benchmark processing failed. Close other heavy processes if memory is low, then retry; the operational report was retained.")
            table = []
            if e1.get("output_csv"):
                table = json.loads(pd.read_csv(e1["output_csv"]).to_json(orient="records"))
            result = {"status": report["completion_status"], "e1": e1,
                "label_coverage": report.get("label_coverage", {}), "numbers_frozen": report["numbers_frozen"],
                "benchmark_rows": table}
        else:
            raise RuntimeError("Unknown processing job type.")
        with SessionLocal() as session:
            job = session.get(BackgroundJob, job_id)
            job.status, job.phase, job.result = "completed", "Session completed", result
            job.cycle_id = result.get("cycle_id")
            job.finished_at = datetime.now(timezone.utc)
            session.commit()
    except Exception as error:
        from research_monitor.ui.jobs import fail_job
        message = str(error) if type(error) is RuntimeError else f"Processing failed ({type(error).__name__}). Check the service and try again."
        fail_job(job_id, message)
        raise SystemExit(1) from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    job_id = parser.parse_args().job_id
    from research_monitor.ui.jobs import processing_slot
    # Serialize heavyweight web workers across server processes. The existing
    # database supplies the lock; queued workers do not load embedding models.
    progress(job_id, "Waiting for the processing slot")
    with processing_slot():
        execute_job(job_id)
