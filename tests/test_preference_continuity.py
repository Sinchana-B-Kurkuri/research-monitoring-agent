"""Explicit preference continuity and feedback timing regressions."""
from unittest.mock import patch
import numpy as np
from research_monitor.database.models import FeedbackEvent, TrainingLabel
from research_monitor.relevance.user_manager import create_user
from research_monitor.summary.service import summarize_decision
from research_monitor.summary.summarizer import MockSummarizer
from tests.test_review_ui import db_session, sample_user, sample_cycle, _create_paper_and_decision


def test_preference_import_preserves_provenance_and_later_feedback_overrides(db_session, sample_user, sample_cycle):
    decision = _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id, "memory_paper", "HIGH", .9, "DELIVERED")
    db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=decision.paper_id, decision_id=decision.id, source="user", label=1))
    db_session.commit()
    with patch("research_monitor.relevance.user_manager._generate_seed_vector", return_value=np.array([1.]+[0.]*383)):
        target = create_user("memory_target", "Memory Target", "Robotics", db_session,
            inherit_from_user=sample_user.id, reference_embeddings=np.eye(2,384))
    label = db_session.query(TrainingLabel).filter_by(user_id=target.id).one()
    assert label.label == 1 and label.origin.startswith("inherited:"+sample_user.id+":")
    inherited = db_session.query(FeedbackEvent).filter_by(user_id=target.id).one()
    assert inherited.decision_id is None
    db_session.add(FeedbackEvent(user_id=target.id, paper_id=decision.paper_id, source="reviewer", label=0))
    db_session.commit()
    db_session.expire_all()
    assert db_session.query(TrainingLabel).filter_by(user_id=target.id).one().label == 0
    assert db_session.query(TrainingLabel).filter_by(user_id=sample_user.id).one().label == 1


def test_summary_framing_uses_saved_feedback_as_preferences(db_session, sample_user, sample_cycle):
    liked = _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id, "liked_memory", "HIGH", .9, "DELIVERED")
    target = _create_paper_and_decision(db_session, sample_user.id, sample_cycle.id, "target_memory", "HIGH", .8, "ROUTED_HIGH")
    liked.paper.embedding = target.paper.embedding = [1.]+[0.]*383
    db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=liked.paper_id, source="user", label=1))
    db_session.commit()
    llm = MockSummarizer()
    summarize_decision(target.id, sample_user.id, db_session, summarizer=llm)
    assert liked.paper.title in llm.calls[0]["user_interests"]
    assert "never as facts about this paper" in llm.calls[0]["user_interests"]


def test_saved_preferences_are_updated_before_the_next_cycle_is_scored(db_session, sample_user):
    from types import SimpleNamespace
    from scripts.run_daily_monitoring import execute_daily_cycle_for_user
    order = []
    state = SimpleNamespace(cycle_id=999, counts={"discovered":0,"high":0,"borderline":0,"low":0})
    ingestion = SimpleNamespace(fetched=0, inserted=0, skipped_duplicates=0, embedded=0)
    def ingest(**kwargs):
        order.append("ingest")
        assert order[:2] == ["vector","classifier"]
        return state, ingestion
    def vector(**kwargs):
        order.append("vector")
        return SimpleNamespace(status="skipped",reason="Test")
    def classifier(**kwargs):
        order.append("classifier")
        return SimpleNamespace(status="skipped",reason="Test")
    with patch("research_monitor.relevance.preference.update_user_interest_vector",side_effect=vector), patch("research_monitor.relevance.classifier.train_and_evaluate_classifier",side_effect=classifier), patch("scripts.run_daily_monitoring.run_live_cycle",side_effect=ingest), patch("scripts.run_daily_monitoring.resume_after_review",return_value=(SimpleNamespace(items=[]),False)):
        status,_,_=execute_daily_cycle_for_user(sample_user,db_session,["cs.AI"],10,1)
    assert status == "[CYCLE_EMPTY]"
