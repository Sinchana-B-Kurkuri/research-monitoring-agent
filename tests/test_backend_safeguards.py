"""Privacy, held-out evaluation, and PostgreSQL concurrency regressions."""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

from research_monitor.database.models import Cycle, Digest, InterestVectorVersion, Paper, RoutingDecision, Summary, User
from research_monitor.database.session import init_db
from research_monitor.evaluation.final_evaluation import run_final_evaluation_suite
from research_monitor.relevance.classifier import _evaluate_repeated_stratified_oof
from research_monitor.review.digest import resume_after_review
from research_monitor.summary.summarizer import MockSummarizer
from research_monitor.ui.app import AccessTokenLogFilter


def test_access_logs_redact_query_and_path_tokens():
    for path in ("/review?token=private-value&action=accept", "/digest/private-value"):
        record = logging.LogRecord("uvicorn.access", 20, "", 1, '%s - "%s %s HTTP/%s" %s',
            ("client", "GET", path, "1.1", 200), None)
        AccessTokenLogFilter().filter(record)
        assert "private-value" not in record.getMessage()
        assert "[REDACTED]" in record.getMessage()


def test_fold_baseline_does_not_use_full_feedback_centroid():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 384)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.array([1] * 15 + [0] * 15)
    seed = x[0]
    options = dict(X=x, y=y, baseline_seed=seed, n_splits=3, n_repeats=1, c_candidates=(.3,))
    first = _evaluate_repeated_stratified_oof(active_centroid=seed, **options)
    second = _evaluate_repeated_stratified_oof(active_centroid=-seed, **options)
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[2], second[2])


def test_empty_pilot_exports_and_figure_generation_are_readable(tmp_path, monkeypatch):
    from scripts import generate_figures
    engine = create_engine("sqlite://")
    init_db(engine)
    results = tmp_path / "empty-results"
    with Session(engine) as db:
        report = run_final_evaluation_suite(db_session=db, results_dir=results,
            pilot_export_dir=tmp_path / "export", survey_dir=tmp_path / "surveys", run_offline_e1=False)
    monkeypatch.setattr(generate_figures, "RESULTS_DIR", results)
    assert not report["numbers_frozen"]
    assert pd.read_csv(results / "final_e7_pilot.csv").empty
    assert pd.read_csv(results / "final_pilot_before_after.csv").empty
    assert generate_figures.regenerate_all_figures() == []


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="Requires explicitly configured disposable PostgreSQL")
def test_postgres_concurrent_resume_creates_one_summary_and_item():
    engine = create_engine(os.environ["TEST_DATABASE_URL"])
    init_db(engine)
    uid = "repair_concurrency_" + uuid4().hex
    pid = uid + "_paper"
    with Session(engine) as db:
        user = User(id=uid, name="Concurrency test", is_simulated=True)
        db.add(user)
        db.flush()
        vector = InterestVectorVersion(user_id=uid, version=0, vector=[1.] + [0.]*383,
            p_high_level=98, p_low_level=90, t_high=.8, t_low=.3)
        cycle = Cycle(user_id=uid, mode="replay", as_of_date="2026-10-21", finished_at=datetime.now(timezone.utc))
        db.add_all([vector, cycle, Paper(id=pid, title="Test", abstract="Test abstract",
            submitted_date="2026-10-21", categories=["cs.AI"], embedding=[1.] + [0.]*383,
            embedding_model="test", is_reference=False)])
        db.flush()
        decision = RoutingDecision(user_id=uid, paper_id=pid, cycle_id=cycle.id,
            vector_version_id=vector.id, cosine_score=.9, thresholds_used={},
            route="HIGH", status="ROUTED_HIGH", reason="Concurrency regression")
        db.add(decision)
        db.commit()
        cycle_id = cycle.id
    summarizer = MockSummarizer()

    def resume(_):
        with Session(engine) as db:
            digest, _ = resume_after_review(uid, cycle_id, db, summarizer=summarizer)
            return len(digest.items)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(resume, range(2))) == [1, 1]
        with Session(engine) as db:
            assert db.query(Summary).filter_by(user_id=uid).count() == 1
        assert summarizer.call_count == 1
    finally:
        with Session(engine) as db:
            for digest in db.query(Digest).filter_by(user_id=uid):
                db.delete(digest)
            db.flush()
            db.delete(db.get(User, uid))
            db.flush()
            db.delete(db.get(Paper, pid))
            db.commit()
        engine.dispose()


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="Requires explicitly configured disposable PostgreSQL")
def test_postgres_threshold_fallback_recovers_aborted_statement(monkeypatch):
    from research_monitor.config import config
    from research_monitor.relevance.thresholds import compute_vector_thresholds
    engine = create_engine(os.environ["TEST_DATABASE_URL"])
    init_db(engine)
    pid = "repair_threshold_" + uuid4().hex
    monkeypatch.setattr(config, "REFERENCE_SAMPLE_SIZE", 1)
    with Session(engine) as db:
        db.add(Paper(id=pid, title="Reference", abstract="Reference abstract",
            submitted_date="2026-10-21", categories=["cs.AI"], embedding=[1.] + [0.]*383,
            embedding_model="test", is_reference=True))
        db.commit()
        execute = db.execute

        def fail_percentile(statement, *args, **kwargs):
            if "percentile_cont" in str(statement):
                return execute(text("SELECT deliberately_missing_column FROM papers"))
            return execute(statement, *args, **kwargs)

        try:
            monkeypatch.setattr(db, "execute", fail_percentile)
            thresholds = compute_vector_thresholds(np.array([1.] + [0.]*383), session=db)
            assert np.isfinite(thresholds.t_high)
            assert db.execute(text("SELECT 1")).scalar() == 1
        finally:
            db.rollback()
            db.delete(db.get(Paper, pid))
            db.commit()
    engine.dispose()


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="Requires explicitly configured disposable PostgreSQL")
def test_onboarding_releases_web_transaction_before_independent_worker(monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from research_monitor.ui.app import app, get_db

    engine = create_engine(os.environ["TEST_DATABASE_URL"])
    init_db(engine)
    uid = "repair_onboard_" + uuid4().hex
    with Session(engine) as web_session:
        def independent_worker(command, **kwargs):
            assert "--skip-db-init" in command
            assert not web_session.in_transaction()
            # Before the fix, the request's open SELECT blocks this otherwise
            # harmless schema statement until the onboarding timeout expires.
            with engine.begin() as connection:
                connection.execute(text("SET LOCAL lock_timeout = '1500ms'"))
                connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS access_token VARCHAR"))
            with Session(engine) as worker:
                worker.add(User(id=uid, name="Onboarding regression", access_token=uid, is_simulated=True))
                worker.commit()
            return SimpleNamespace(returncode=0)

        app.dependency_overrides[get_db] = lambda: web_session
        monkeypatch.setattr("subprocess.run", independent_worker)
        try:
            with TestClient(app) as client:
                response = client.post("/onboard", json={
                    "user_id": uid, "name": "Onboarding regression", "seed_text": "Robotic navigation",
                })
            assert response.status_code == 200
            assert response.json()["user_id"] == uid
        finally:
            app.dependency_overrides.clear()
            web_session.rollback()
            user = web_session.get(User, uid)
            if user:
                web_session.delete(user)
                web_session.commit()
    engine.dispose()
