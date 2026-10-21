"""Runtime checks for digest recovery controls and bounded rendering queries."""
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import event

from research_monitor.config import config
from research_monitor.review.digest import resume_after_review
from research_monitor.summary.summarizer import MockSummarizer
from research_monitor.ui.app import app, get_db
from tests.test_review_ui import (
    db_session, sample_user, sample_cycle, _create_paper_and_decision,
)


def test_digest_renders_recovery_controls_without_per_paper_queries(db_session, sample_user, sample_cycle, monkeypatch):
    for index in range(10):
        _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id,
                                   f"runtime_{index}", "HIGH", .9, "ROUTED_HIGH")
    db_session.commit()
    resume_after_review(sample_user.id, sample_cycle.id, db_session,
                        summarizer=MockSummarizer(should_fail=True))
    token, cycle_id = sample_user.access_token, sample_cycle.id
    db_session.expire_all()
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    statements = []
    engine = db_session.get_bind()
    def count_query(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(engine, "before_cursor_execute", count_query)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as client:
            response = client.get("/digest", params={"token": token})
        assert response.status_code == 200
        assert "10 summaries pending" in response.text
        assert "Update Digest / Retry Summaries" in response.text
        assert f'/cycle/{cycle_id}/resume?' in response.text
        assert "this server has no Gemini API key configured" in response.text
        assert len(statements) <= 7  # auth, digest/items, latest cycle, reactions; independent of item count
    finally:
        event.remove(engine, "before_cursor_execute", count_query)
        app.dependency_overrides.clear()


def test_resume_reports_pending_generation_and_links_to_the_requested_cycle(db_session, sample_user, sample_cycle):
    _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id,
                               "pending_runtime", "HIGH", .9, "ROUTED_HIGH")
    db_session.commit()
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as client, patch("research_monitor.summary.service.get_summarizer", return_value=MockSummarizer(should_fail=True)):
            response = client.post(f"/cycle/{sample_cycle.id}/resume", params={"token": sample_user.access_token})
        assert response.status_code == 200
        assert "1 summaries pending" in response.text
        assert f"&amp;cycle_id={sample_cycle.id}" in response.text
        assert "original abstracts are available" in response.text
    finally:
        app.dependency_overrides.clear()
