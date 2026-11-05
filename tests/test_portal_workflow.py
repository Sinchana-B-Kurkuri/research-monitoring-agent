"""Browser workflow, access boundaries, preference continuity, and partial reports."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from research_monitor.database.models import BackgroundJob, FeedbackEvent, TrainingLabel
from research_monitor.relevance.user_manager import create_user
from research_monitor.summary.service import summarize_decision
from research_monitor.summary.summarizer import MockSummarizer
from research_monitor.ui.app import app, get_db
from research_monitor.ui.workspace import workspace_key
from tests.test_review_ui import db_session, sample_user, other_user, sample_cycle, _create_paper_and_decision


def client_for(session):
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def test_directory_requires_operator_or_remembered_profile(db_session, sample_user, other_user):
    try:
        with client_for(db_session) as client:
            page = client.get("/workspace")
            assert sample_user.access_token not in page.text and other_user.access_token not in page.text
            response = client.get("/workspace", params={"workspace_key": workspace_key()})
            assert response.status_code == 200
            assert "httponly" in response.history[0].headers["set-cookie"].lower()
            assert sample_user.name in response.text and other_user.name in response.text
            assert client.get("/", follow_redirects=False).headers["location"] == "/workspace"
    finally:
        app.dependency_overrides.clear()


def test_personal_link_remembers_only_that_profile_and_supports_back(db_session, sample_user, other_user):
    try:
        with client_for(db_session) as client:
            response = client.get("/digest", params={"token": sample_user.access_token})
            assert "httponly" in response.headers["set-cookie"].lower()
            page = client.get("/onboard")
            assert sample_user.name in page.text
            assert other_user.access_token not in page.text
            assert "Preference memory for this new pilot" in page.text
            assert client.get("/dashboard").status_code == 200
            assert client.get("/history").status_code == 200
            assert client.get("/static/htmx.min.js").status_code == 200
            assert 'src="/static/htmx.min.js"' in page.text
            client.cookies.clear()
            assert client.get("/dashboard").status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_hidden_profile_is_unlisted_but_direct_access_still_works(db_session, sample_user, other_user, tmp_path, monkeypatch):
    hidden = tmp_path / "hidden_pilots.txt"
    hidden.write_text(f"{sample_user.id}\n", encoding="utf-8")
    monkeypatch.setattr("research_monitor.ui.portal.HIDDEN_PROFILES_PATH", hidden)
    try:
        with client_for(db_session) as client:
            directory = client.get("/workspace", params={"workspace_key": workspace_key()})
            assert directory.status_code == 200
            assert sample_user.name not in directory.text
            assert other_user.name in directory.text
            assert sample_user.name not in client.get("/onboard").text
            assert client.get("/history", params={"token": sample_user.access_token}).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_monitor_button_is_idempotent_and_job_results_are_private(db_session, sample_user, other_user):
    launches = []
    try:
        with client_for(db_session) as client, patch("research_monitor.ui.portal.run_background_job", side_effect=launches.append):
            request = {"kind":"monitor", "categories":"cs.RO,cs.CV", "days":3, "max_results":20}
            first = client.post("/jobs", params={"token":sample_user.access_token}, json=request, headers={"Accept":"application/json"})
            second = client.post("/jobs", params={"token":sample_user.access_token}, json=request, headers={"Accept":"application/json"})
            assert first.status_code == 202 and first.json()["id"] == second.json()["id"]
            assert len(launches) == 1
            job = db_session.get(BackgroundJob, first.json()["id"])
            assert job.options["categories"] == ["cs.RO","cs.CV"]
            assert client.get(f"/jobs/{job.id}", params={"token":other_user.access_token}).status_code == 404
            assert client.get("/evaluation", params={"token":sample_user.access_token}).status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_cross_site_job_submission_is_rejected(db_session, sample_user):
    try:
        with client_for(db_session) as client:
            response = client.post("/jobs", params={"token":sample_user.access_token}, json={}, headers={"Origin":"https://another-site.example"})
        assert response.status_code == 403 and db_session.query(BackgroundJob).count() == 0
    finally:
        app.dependency_overrides.clear()


def test_foreign_cycle_and_unauthorized_preference_source_are_rejected(db_session, sample_user, other_user, sample_cycle):
    try:
        with client_for(db_session) as client, patch("subprocess.run") as worker:
            assert client.get(f"/history/{sample_cycle.id}", params={"token":other_user.access_token}).status_code == 404
            response = client.post("/onboard", json={"user_id":"new_profile", "name":"New profile", "seed_text":"Robotics", "inherit_from_user":sample_user.id})
            assert response.status_code == 403
            worker.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_failed_benchmark_marks_job_failed_instead_of_claiming_completion(db_session, sample_user, monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from scripts import run_web_job
    from research_monitor.ui import jobs
    from research_monitor.evaluation import final_evaluation
    factory = sessionmaker(bind=db_session.get_bind())
    monkeypatch.setattr(run_web_job,"SessionLocal",factory)
    monkeypatch.setattr(jobs,"SessionLocal",factory)
    monkeypatch.setattr(final_evaluation,"run_final_evaluation_suite",lambda **kwargs: {"e1_gate_quality":{"status":"ERROR","error":"Private details"}})
    db_session.add(BackgroundJob(id="failed_eval_test",user_id=sample_user.id,kind="evaluation",status="queued",options={}))
    db_session.commit()
    with pytest.raises(SystemExit):
        run_web_job.execute_job("failed_eval_test")
    db_session.expire_all()
    job = db_session.get(BackgroundJob,"failed_eval_test")
    assert job.status == "failed" and "Private details" not in job.error
