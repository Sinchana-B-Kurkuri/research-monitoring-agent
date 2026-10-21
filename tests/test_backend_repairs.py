"""Regressions for failures at module boundaries, beyond individual stages."""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session

from research_monitor.database.models import Base, Paper
from research_monitor.database.session import init_db
from scripts.run_phase3_loader import load_corpus


def test_schema_initialization_is_repeatable_and_training_labels_is_a_view():
    engine = create_engine("sqlite://")
    init_db(engine)
    init_db(engine)
    assert "training_labels" not in Base.metadata.tables
    assert "training_labels" not in inspect(engine).get_table_names()
    assert "training_labels" in inspect(engine).get_view_names()


def test_corpus_loader_is_idempotent_and_preserves_existing_papers():
    engine = create_engine("sqlite://")
    init_db(engine)
    frame = pd.DataFrame([dict(paper_id="2401.00001", title="Paper", abstract="Abstract",
        submitted_date="2024-01-01", categories="cs.AI,cs.CL", is_reference=True)])
    with Session(engine) as db:
        assert load_corpus(db, frame, np.ones((1, 384))) == 1
        assert load_corpus(db, frame, np.ones((1, 384))) == 0
        paper = db.get(Paper, "2401.00001")
        assert paper.categories == ["cs.AI", "cs.CL"]
        assert paper.is_reference
        assert np.isclose(np.linalg.norm(paper.embedding), 1.0, atol=1e-6)


from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from research_monitor.database.models import Cycle, FeedbackEvent, InterestVectorVersion, ReviewTask, RoutingDecision, User
from research_monitor.review.digest import resume_after_review
from research_monitor.review.queue import build_review_queue, record_review_action
from research_monitor.summary.summarizer import GeminiSummarizer, MockSummarizer, get_summarizer
from research_monitor.ui.app import app, get_db
from tests.test_review_ui import db_session, sample_user, sample_cycle, _create_paper_and_decision


def test_anonymous_onboarding_cannot_recover_existing_credentials(db_session, sample_user):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app) as client, patch("subprocess.run") as process:
            response = client.post("/onboard", json=dict(user_id=sample_user.id, name="Unknown", seed_text="Unverified"))
        assert response.status_code == 409
        assert sample_user.access_token not in response.text
        process.assert_not_called()
    finally:
        app.dependency_overrides.clear()


def test_later_acceptance_updates_existing_digest_once(db_session, sample_user, sample_cycle):
    dec = _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id, "late", "BORDERLINE", .4, "ROUTED_BORDERLINE")
    db_session.commit()
    task = build_review_queue(sample_user.id, sample_cycle.id, db_session)[0]
    llm = MockSummarizer()
    digest, _ = resume_after_review(sample_user.id, sample_cycle.id, db_session, summarizer=llm)
    assert not digest.items
    record_review_action(task.id, "accept", sample_user.id, db_session)
    record_review_action(task.id, "accept", sample_user.id, db_session)
    digest, existed = resume_after_review(sample_user.id, sample_cycle.id, db_session, summarizer=llm)
    resume_after_review(sample_user.id, sample_cycle.id, db_session, summarizer=llm)
    assert existed and len(digest.items) == 1
    assert dec.status == "DELIVERED"
    assert digest.items[0].via == "review"
    assert llm.call_count == 1
    assert db_session.query(FeedbackEvent).count() == 1
    with pytest.raises(ValueError, match="completed"):
        record_review_action(task.id, "reject", sample_user.id, db_session)


def test_digest_retries_failed_summary_without_duplicate_delivery(db_session, sample_user, sample_cycle):
    dec = _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id, "retry", "HIGH", .9, "ROUTED_HIGH")
    db_session.commit()
    llm = MockSummarizer(fail_times=1)
    digest, _ = resume_after_review(sample_user.id, sample_cycle.id, db_session, summarizer=llm)
    assert len(digest.items) == 1 and dec.summary is None
    assert dec.status == "SUMMARIZE_PENDING"
    digest, _ = resume_after_review(sample_user.id, sample_cycle.id, db_session, summarizer=llm)
    assert len(digest.items) == 1 and dec.summary is not None
    assert dec.status == "DELIVERED" and llm.call_count == 2


def test_foreign_cycle_cannot_be_resumed(db_session, sample_user, sample_cycle):
    with pytest.raises(PermissionError):
        resume_after_review("another_user", sample_cycle.id, db_session)


def test_expired_review_cannot_be_accepted(db_session, sample_user, sample_cycle):
    dec = _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id, "expired", "BORDERLINE", .4, "ROUTED_BORDERLINE")
    db_session.commit()
    task = build_review_queue(sample_user.id, sample_cycle.id, db_session)[0]
    task.queued_at = datetime.now(timezone.utc) - timedelta(days=4)
    db_session.commit()
    with pytest.raises(ValueError, match="expired"):
        record_review_action(task.id, "accept", sample_user.id, db_session)
    assert dec.status == "REVIEW_EXPIRED"
    assert db_session.query(FeedbackEvent).count() == 0


def test_missing_api_key_never_implicitly_selects_mock(monkeypatch):
    from research_monitor import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    assert isinstance(get_summarizer(), GeminiSummarizer)


from research_monitor.ingestion.live import run_live_cycle
from research_monitor.relevance.state import PipelineState, PaperCandidate
from research_monitor.relevance.pipeline import embed_stage


def test_empty_live_fetch_never_routes_historical_reference_papers(db_session, sample_user):
    db_session.add(InterestVectorVersion(user_id=sample_user.id, version=0, vector=[1.] + [0.]*383,
        method="seed", p_high_level=98, p_low_level=90, t_high=.8, t_low=.3, is_active=True))
    db_session.add(Paper(id="historic", title="Old", abstract="Abstract", submitted_date="2024-01-01",
        categories=["cs.AI"], embedding=[1.] + [0.]*383, embedding_model="test", is_reference=True))
    db_session.commit()
    state, ingestion = run_live_cycle(sample_user.id, db_session, papers_override=[])
    assert ingestion.fetched == 0 and state.counts["total_routed"] == 0
    assert db_session.query(RoutingDecision).count() == 0


def test_replay_embedding_artifact_fallback_is_aligned_by_id(db_session, sample_user, tmp_path):
    corpus = tmp_path / "corpus.parquet"
    vectors = tmp_path / "embeddings.npy"
    pd.DataFrame({"paper_id": ["unrelated", "wanted"]}).to_parquet(corpus)
    np.save(vectors, np.array([[0.,1.] + [0.]*382, [1.] + [0.]*383]))
    state = PipelineState(user_id=sample_user.id, mode="replay")
    state.preprocessed = [PaperCandidate("wanted", "Title", "Abstract", "2024-01-01")]
    embed_stage(state, db_session, corpus_path=corpus, embeddings_path=vectors)
    assert len(state.embedded) == 1 and state.embedded[0].embedding[0] == 1
    assert db_session.get(Paper, "wanted") is not None


def test_live_fetch_failure_is_durably_recorded(db_session, sample_user):
    with patch("research_monitor.ingestion.live.fetch_live_papers", side_effect=RuntimeError("offline")):
        with pytest.raises(RuntimeError, match="offline"):
            run_live_cycle(sample_user.id, db_session)
    cycle = db_session.query(Cycle).one()
    assert cycle.counts["status"] == "failure" and cycle.finished_at is not None
