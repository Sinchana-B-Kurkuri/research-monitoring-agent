"""
test_review_ui.py
~~~~~~~~~~~~~~~~~
Tests for Phase 6 Human Review, Digest, and Feedback UI:
  1. Review queue cap (15 tasks max) and overflow logging (DEFERRED_OVERFLOW).
  2. Review actions (Accept, Reject, Skip) and reviewer feedback event emission.
  3. Task expiry after 3 days (REVIEW_EXPIRED) and skip non-emission.
  4. Resume workflow and idempotency of resume_after_review.
  5. Digest cap (20 items max), ordering by score, via tags (auto/review), and overflow logging.
  6. Rejected and expired papers exclusion from digest.
  7. User feedback (Like/Dislike) and append-only event recording.
  8. Effective training labels view precedence rule (user reaction overrides reviewer).
  9. URL access-token authentication and cross-user access isolation (401 / 403).
  10. Audit page display of discarded LOW papers.
"""

from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_monitor.database.models import (
    Base,
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    Paper,
    ReviewTask,
    RoutingDecision,
    TrainingLabel,
    User,
    create_training_labels_view,
)
from research_monitor.relevance.state import DecisionStatus
from research_monitor.review.digest import (
    record_user_feedback,
    resume_after_review,
)
from research_monitor.review.labels import get_effective_training_labels
from research_monitor.review.queue import (
    build_review_queue,
    expire_stale_tasks,
    record_review_action,
)
from research_monitor.ui.app import app, get_db


@pytest.fixture
def db_session():
    """Isolated SQLite in-memory database with pgvector shim and training_labels view."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    create_training_labels_view(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_user(db_session):
    user = User(
        id="alice",
        name="Alice Researcher",
        access_token="alice_secret_token_123",
        is_simulated=False,
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def other_user(db_session):
    user = User(
        id="bob",
        name="Bob Researcher",
        access_token="bob_secret_token_456",
        is_simulated=False,
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def sample_cycle(db_session, sample_user):
    cycle = Cycle(
        user_id=sample_user.id,
        as_of_date="2024-06-30",
        mode="replay",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(cycle)
    db_session.commit()
    return cycle


def _create_paper_and_decision(
    db_session,
    user_id: str,
    cycle_id: int,
    paper_id: str,
    route: str,
    score: float,
    status: str,
) -> RoutingDecision:
    paper = db_session.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        paper = Paper(
            id=paper_id,
            title=f"Title for {paper_id}",
            abstract=f"Abstract text describing paper {paper_id}.",
            submitted_date="2024-06-01",
            categories=["cs.AI"],
            embedding=[0.0] * 384,
            embedding_model="test",
            is_reference=False,
        )
        db_session.add(paper)
        db_session.flush()

    dec = RoutingDecision(
        paper_id=paper_id,
        user_id=user_id,
        cycle_id=cycle_id,
        cosine_score=score,
        vector_version_id=1,
        method="threshold",
        thresholds_used={"t_high": 0.50, "t_low": 0.20},
        route=route,
        status=status,
        reason=f"Routing score {score:.4f} assigned route {route}",
    )
    db_session.add(dec)
    db_session.flush()
    return dec


class TestReviewQueueAndActions:
    def test_queue_cap_and_overflow(self, db_session, sample_user, sample_cycle):
        """Verifies requirement: Max 15 review tasks per user per cycle, overflow logged as DEFERRED_OVERFLOW."""
        # Create 20 borderline decisions
        for i in range(20):
            _create_paper_and_decision(
                db_session,
                user_id=sample_user.id,
                cycle_id=sample_cycle.id,
                paper_id=f"p_border_{i:02d}",
                route="BORDERLINE",
                score=0.30 + (i * 0.01),  # scores 0.30 to 0.49
                status=DecisionStatus.ROUTED_BORDERLINE.value,
            )
        db_session.commit()

        tasks = build_review_queue(
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            session=db_session,
            max_tasks=15,
        )

        assert len(tasks) == 15
        # Verify tasks are ordered by score DESC
        scores = [t.routing_decision.cosine_score for t in tasks]
        assert scores == sorted(scores, reverse=True)
        assert min(scores) >= 0.35

        # Check that the 5 overflow decisions became DEFERRED_OVERFLOW
        overflow_decs = (
            db_session.query(RoutingDecision)
            .filter(
                RoutingDecision.user_id == sample_user.id,
                RoutingDecision.cycle_id == sample_cycle.id,
                RoutingDecision.status == DecisionStatus.DEFERRED_OVERFLOW.value,
            )
            .all()
        )
        assert len(overflow_decs) == 5
        for od in overflow_decs:
            assert "Review queue cap (15) exceeded" in od.reason

    def test_review_action_accept(self, db_session, sample_user, sample_cycle):
        """Accept transitions to SUMMARIZE_PENDING and creates 1 reviewer feedback event (label=1)."""
        dec = _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_accept",
            route="BORDERLINE",
            score=0.42,
            status=DecisionStatus.ROUTED_BORDERLINE.value,
        )
        tasks = build_review_queue(sample_user.id, sample_cycle.id, db_session)
        task = tasks[0]

        record_review_action(task.id, "accept", sample_user.id, db_session)

        assert task.decision == "accept"
        assert dec.status == DecisionStatus.SUMMARIZE_PENDING.value

        events = db_session.query(FeedbackEvent).filter_by(paper_id="p_accept").all()
        assert len(events) == 1
        assert events[0].source == "reviewer"
        assert events[0].label == 1
        assert events[0].user_id == sample_user.id

    def test_review_action_reject(self, db_session, sample_user, sample_cycle):
        """Reject transitions to REVIEW_REJECTED and creates 1 reviewer feedback event (label=0)."""
        dec = _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_reject",
            route="BORDERLINE",
            score=0.31,
            status=DecisionStatus.ROUTED_BORDERLINE.value,
        )
        tasks = build_review_queue(sample_user.id, sample_cycle.id, db_session)
        task = tasks[0]

        record_review_action(task.id, "reject", sample_user.id, db_session)

        assert task.decision == "reject"
        assert dec.status == DecisionStatus.REVIEW_REJECTED.value

        events = db_session.query(FeedbackEvent).filter_by(paper_id="p_reject").all()
        assert len(events) == 1
        assert events[0].source == "reviewer"
        assert events[0].label == 0

    def test_review_action_skip(self, db_session, sample_user, sample_cycle):
        """Skip leaves task pending and creates NO feedback events."""
        dec = _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_skip",
            route="BORDERLINE",
            score=0.35,
            status=DecisionStatus.ROUTED_BORDERLINE.value,
        )
        tasks = build_review_queue(sample_user.id, sample_cycle.id, db_session)
        task = tasks[0]

        record_review_action(task.id, "skip", sample_user.id, db_session)

        assert task.decision == "skip"
        assert dec.status == DecisionStatus.PENDING_REVIEW.value

        events = db_session.query(FeedbackEvent).filter_by(paper_id="p_skip").all()
        assert len(events) == 0

    def test_task_expiry_after_3_days(self, db_session, sample_user, sample_cycle):
        """Tasks undecided/skipped after 3 days transition to REVIEW_EXPIRED with no feedback event."""
        dec = _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_expire",
            route="BORDERLINE",
            score=0.33,
            status=DecisionStatus.ROUTED_BORDERLINE.value,
        )
        tasks = build_review_queue(sample_user.id, sample_cycle.id, db_session)
        task = tasks[0]

        # Simulate task queued 4 days ago
        now = datetime.now(timezone.utc)
        task.queued_at = now - timedelta(days=4)
        db_session.commit()

        expired = expire_stale_tasks(sample_user.id, db_session, now=now)
        assert len(expired) == 1
        assert task.decision == "expired"
        assert task.expired_at is not None
        assert dec.status == DecisionStatus.REVIEW_EXPIRED.value

        events = db_session.query(FeedbackEvent).filter_by(paper_id="p_expire").all()
        assert len(events) == 0


class TestDigestAndResume:
    def test_digest_cap_and_overflow(self, db_session, sample_user, sample_cycle):
        """Verifies requirement: Max 20 items per digest, ordered by score, overflow logged as DEFERRED_OVERFLOW."""
        # 15 HIGH papers (scores 0.80 to 0.94)
        for i in range(15):
            _create_paper_and_decision(
                db_session,
                user_id=sample_user.id,
                cycle_id=sample_cycle.id,
                paper_id=f"p_high_{i:02d}",
                route="HIGH",
                score=0.80 + (i * 0.01),
                status=DecisionStatus.ROUTED_HIGH.value,
            )

        # 10 Review-accepted borderline papers (scores 0.40 to 0.49)
        for i in range(10):
            dec = _create_paper_and_decision(
                db_session,
                user_id=sample_user.id,
                cycle_id=sample_cycle.id,
                paper_id=f"p_rev_{i:02d}",
                route="BORDERLINE",
                score=0.40 + (i * 0.01),
                status=DecisionStatus.SUMMARIZE_PENDING.value,
            )
            task = ReviewTask(
                decision_id=dec.id,
                kind="borderline",
                decision="accept",
            )
            db_session.add(task)
        db_session.commit()

        digest, already_existed = resume_after_review(
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            session=db_session,
            max_digest_items=20,
        )

        assert already_existed is False
        assert len(digest.items) == 20

        # Items ordered by score DESC
        item_scores = [it.routing_decision.cosine_score for it in digest.items]
        assert item_scores == sorted(item_scores, reverse=True)

        # 15 HIGH auto-cleared (via=auto) + 5 review-accepted (via=review)
        vias = [it.via for it in digest.items]
        assert vias.count("auto") == 15
        assert vias.count("review") == 5

        # 5 review-accepted papers beyond cap 20 became DEFERRED_OVERFLOW
        overflow_decs = (
            db_session.query(RoutingDecision)
            .filter(
                RoutingDecision.user_id == sample_user.id,
                RoutingDecision.cycle_id == sample_cycle.id,
                RoutingDecision.status == DecisionStatus.DEFERRED_OVERFLOW.value,
            )
            .all()
        )
        assert len(overflow_decs) == 5
        for od in overflow_decs:
            assert "Digest cap (20) exceeded" in od.reason

    def test_resume_after_review_idempotency(self, db_session, sample_user, sample_cycle):
        """Verifies requirement: Re-running resume_after_review creates no duplicate digests or items."""
        _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_high_single",
            route="HIGH",
            score=0.91,
            status=DecisionStatus.ROUTED_HIGH.value,
        )
        db_session.commit()

        digest1, existed1 = resume_after_review(sample_user.id, sample_cycle.id, db_session)
        assert existed1 is False
        assert len(digest1.items) == 1

        digest2, existed2 = resume_after_review(sample_user.id, sample_cycle.id, db_session)
        assert existed2 is True
        assert digest2.id == digest1.id
        assert len(digest2.items) == 1

        all_digests = db_session.query(Digest).filter_by(user_id=sample_user.id).all()
        assert len(all_digests) == 1
        all_items = db_session.query(DigestItem).filter_by(digest_id=digest1.id).all()
        assert len(all_items) == 1

    def test_rejected_and_expired_excluded_from_digest(self, db_session, sample_user, sample_cycle):
        """Rejected and expired papers must never appear in the digest."""
        # 1 rejected paper
        dec_rej = _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_rej",
            route="BORDERLINE",
            score=0.45,
            status=DecisionStatus.REVIEW_REJECTED.value,
        )
        task_rej = ReviewTask(decision_id=dec_rej.id, decision="reject")
        db_session.add(task_rej)

        # 1 expired paper
        dec_exp = _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_exp",
            route="BORDERLINE",
            score=0.44,
            status=DecisionStatus.REVIEW_EXPIRED.value,
        )
        task_exp = ReviewTask(decision_id=dec_exp.id, decision="expired")
        db_session.add(task_exp)

        # 1 auto high paper
        _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_auto_good",
            route="HIGH",
            score=0.88,
            status=DecisionStatus.ROUTED_HIGH.value,
        )
        db_session.commit()

        digest, _ = resume_after_review(sample_user.id, sample_cycle.id, db_session)
        delivered_pids = [it.routing_decision.paper_id for it in digest.items]
        assert "p_auto_good" in delivered_pids
        assert "p_rej" not in delivered_pids
        assert "p_exp" not in delivered_pids

    def test_user_feedback_like_dislike(self, db_session, sample_user, sample_cycle):
        """User feedback appends exactly one event with source=user and updates status to REACTED."""
        _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_delivered",
            route="HIGH",
            score=0.92,
            status=DecisionStatus.ROUTED_HIGH.value,
        )
        db_session.commit()

        digest, _ = resume_after_review(sample_user.id, sample_cycle.id, db_session)
        item = digest.items[0]

        # 1. User Likes paper
        ev1 = record_user_feedback(item.id, "like", sample_user.id, db_session)
        assert ev1.source == "user"
        assert ev1.label == 1
        assert item.routing_decision.status == DecisionStatus.REACTED.value

        # 2. User changes mind to Dislike (appends new event, safe repeated requests)
        ev2 = record_user_feedback(item.id, "dislike", sample_user.id, db_session)
        assert ev2.source == "user"
        assert ev2.label == 0
        assert ev2.id != ev1.id

        events = (
            db_session.query(FeedbackEvent)
            .filter_by(paper_id="p_delivered", user_id=sample_user.id)
            .all()
        )
        assert len(events) == 2


class TestEffectiveTrainingLabels:
    def test_effective_labels_precedence_rule(self, db_session, sample_user):
        """
        Verifies requirement 7 / Section 7.3:
        1. Reviewer accept gives label 1.
        2. User dislike on same paper gives effective label 0 (user overrides reviewer).
        3. Subsequent user like gives effective label 1 (latest user reaction).
        4. Skip and expired produce NO label in training_labels.
        """
        paper_id = "paper_precedence_01"

        # Step 1: Reviewer accept (label = 1)
        fe_rev = FeedbackEvent(
            user_id=sample_user.id,
            paper_id=paper_id,
            source="reviewer",
            label=1,
            created_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        db_session.add(fe_rev)
        db_session.commit()

        labels = get_effective_training_labels(db_session, user_id=sample_user.id)
        assert len(labels) == 1
        assert labels[0].label == 1
        assert labels[0].origin == "reviewer"

        # Step 2: User dislike (label = 0) overrides reviewer accept
        fe_user_dislike = FeedbackEvent(
            user_id=sample_user.id,
            paper_id=paper_id,
            source="user",
            label=0,
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        db_session.add(fe_user_dislike)
        db_session.commit()

        labels = get_effective_training_labels(db_session, user_id=sample_user.id)
        assert len(labels) == 1
        assert labels[0].label == 0
        assert labels[0].origin == "user"
        assert labels[0].event_id == fe_user_dislike.id

        # Step 3: User subsequent like (label = 1) overrides earlier dislike
        fe_user_like = FeedbackEvent(
            user_id=sample_user.id,
            paper_id=paper_id,
            source="user",
            label=1,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        db_session.add(fe_user_like)
        db_session.commit()

        labels = get_effective_training_labels(db_session, user_id=sample_user.id)
        assert len(labels) == 1
        assert labels[0].label == 1
        assert labels[0].origin == "user"
        assert labels[0].event_id == fe_user_like.id

    def test_skipped_and_expired_produce_no_training_labels(self, db_session, sample_user, sample_cycle):
        """Skip and expired review tasks never write feedback events and never produce training labels."""
        dec_skip = _create_paper_and_decision(
            db_session, sample_user.id, sample_cycle.id, "p_skip_label", "BORDERLINE", 0.35, DecisionStatus.ROUTED_BORDERLINE.value
        )
        dec_exp = _create_paper_and_decision(
            db_session, sample_user.id, sample_cycle.id, "p_exp_label", "BORDERLINE", 0.34, DecisionStatus.ROUTED_BORDERLINE.value
        )
        tasks = build_review_queue(sample_user.id, sample_cycle.id, db_session)

        # Skip one
        record_review_action(tasks[0].id, "skip", sample_user.id, db_session)
        # Expire other
        tasks[1].queued_at = datetime.now(timezone.utc) - timedelta(days=4)
        db_session.commit()
        expire_stale_tasks(sample_user.id, db_session)

        labels = get_effective_training_labels(db_session, user_id=sample_user.id)
        assert len(labels) == 0


class TestFastAPIEndpointsAndAccessControl:
    @pytest.fixture
    def client(self, db_session):
        def override_get_db():
            try:
                yield db_session
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()

    def test_unauthenticated_requests_return_401(self, client):
        """Access without token or with invalid token returns 401."""
        # Missing token
        assert client.get("/review").status_code == 401
        assert client.get("/digest").status_code == 401
        assert client.get("/audit").status_code == 401

        # Invalid token
        assert client.get("/review?token=invalid_token").status_code == 401
        assert client.get("/digest/bad_token").status_code == 401
        assert client.get("/audit?token=fake").status_code == 401

    def test_user_data_isolation(self, client, db_session, sample_user, other_user, sample_cycle):
        """Alice cannot view or act on Bob's review tasks or digest items."""
        # Create a review task for Bob
        dec_bob = _create_paper_and_decision(
            db_session,
            user_id=other_user.id,
            cycle_id=sample_cycle.id,
            paper_id="bob_secret_paper",
            route="BORDERLINE",
            score=0.40,
            status=DecisionStatus.ROUTED_BORDERLINE.value,
        )
        task_bob = ReviewTask(decision_id=dec_bob.id, kind="borderline")
        db_session.add(task_bob)
        db_session.commit()

        # Alice attempts to review Bob's task using Alice's token -> 403 Forbidden
        resp = client.post(
            f"/review/{task_bob.id}/action?action=accept&token={sample_user.access_token}"
        )
        assert resp.status_code == 403
        assert "does not belong to user" in resp.text

    def test_review_htmx_action(self, client, db_session, sample_user, sample_cycle):
        """Review action via HTMX endpoint returns updated card component."""
        _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_htmx_review",
            route="BORDERLINE",
            score=0.45,
            status=DecisionStatus.ROUTED_BORDERLINE.value,
        )
        tasks = build_review_queue(sample_user.id, sample_cycle.id, db_session)
        task = tasks[0]

        resp = client.post(
            f"/review/{task.id}/action?action=accept&token={sample_user.access_token}"
        )
        assert resp.status_code == 200
        assert "Accepted" in resp.text
        assert "SUMMARIZE_PENDING" in resp.text

    def test_digest_and_feedback_htmx(self, client, db_session, sample_user, sample_cycle):
        """Digest item like/dislike via HTMX endpoint returns updated feedback buttons."""
        _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_htmx_digest",
            route="HIGH",
            score=0.89,
            status=DecisionStatus.ROUTED_HIGH.value,
        )
        db_session.commit()

        digest, _ = resume_after_review(sample_user.id, sample_cycle.id, db_session)
        item = digest.items[0]

        # Like
        resp = client.post(
            f"/digest/{item.id}/feedback?reaction=like&token={sample_user.access_token}"
        )
        assert resp.status_code == 200
        assert "active-like" in resp.text

        # Dislike
        resp2 = client.post(
            f"/digest/{item.id}/feedback?reaction=dislike&token={sample_user.access_token}"
        )
        assert resp2.status_code == 200
        assert "active-dislike" in resp2.text

    def test_audit_page_displays_low_papers(self, client, db_session, sample_user, sample_cycle):
        """Audit page displays discarded LOW papers with score and routing reason."""
        _create_paper_and_decision(
            db_session,
            user_id=sample_user.id,
            cycle_id=sample_cycle.id,
            paper_id="p_audit_low",
            route="LOW",
            score=0.08,
            status=DecisionStatus.DISCARDED.value,
        )
        db_session.commit()

        resp = client.get(f"/audit?token={sample_user.access_token}")
        assert resp.status_code == 200
        assert "p_audit_low" in resp.text
        assert "0.0800" in resp.text
        assert "LOW (DISCARDED)" in resp.text
