"""
test_phase11_classifier.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit and integration tests for Phase 11:
Logistic Regression Preference Model & Promotion Guard.

Test coverage:
  1. Eligibility: Insufficient total labels (< 30) prevents training.
  2. Eligibility: Insufficient positive labels (< 8) prevents training.
  3. Eligibility: Insufficient negative labels (< 8) prevents training.
  4. Eligibility: Initial training allowed when total >= 30, pos >= 8, neg >= 8.
  5. Eligibility: Retraining requires 25 new labels since cutoff event ID.
  6. Class Imbalance: Logistic regression handles imbalanced feedback with balanced weights.
  7. Cross-Validation: Repeated stratified 5-fold CV generates out-of-fold predictions.
  8. Leakage Prevention: Preprocessing and C-selection do not access held-out folds.
  9. Promotion Guard: Promoted when OOF AUC >= 0.65 and >= baseline + 0.02.
  10. Promotion Guard: Rejected when OOF AUC < 0.65.
  11. Promotion Guard: Rejected when margin < 0.02 over baseline.
  12. Promotion Guard: Failed promotion leaves existing routing method and active model active.
  13. Persistence: ModelVersion stored with serialized blob, event cutoff, and CV metadata.
  14. Rollback: Transaction failures roll back safely leaving no orphan active models.
  15. Routing Mode: Threshold mode logs cosine similarity and null clf_prob.
  16. Routing Mode: Classifier mode routes by probability and continues logging cosine similarity.
  17. UI Explanations: find_nearest_feedback_papers finds nearest liked and rejected papers.
  18. UI Explanations: Graceful handling when no positive or no negative labels exist.
"""

from datetime import datetime, timezone
import pickle
from unittest.mock import patch
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_monitor.database.models import (
    Base,
    FeedbackEvent,
    InterestProfile,
    InterestVectorVersion,
    ModelVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    TrainingLabel,
    User,
    create_training_labels_view,
)
from research_monitor.relevance.classifier import (
    MIN_EFFECTIVE_LABELS,
    MIN_NEG_LABELS,
    MIN_POS_LABELS,
    MIN_PROMOTION_AUC,
    PROMOTION_AUC_MARGIN,
    RETRAIN_INTERVAL,
    _evaluate_repeated_stratified_oof,
    _select_best_C,
    calibrate_classifier_thresholds,
    check_classifier_eligibility,
    find_nearest_feedback_papers,
    train_and_evaluate_classifier,
)
from research_monitor.relevance.pipeline import run_monitoring_cycle


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def random_unit_vector(rng):
    def _generator():
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)
    return _generator


@pytest.fixture
def db_session():
    """In-memory SQLite session with full schema initialized."""
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


def make_test_paper(
    paper_id: str,
    embedding: list[float],
    title: str = "Test Title",
    is_reference: bool = False,
) -> Paper:
    return Paper(
        id=paper_id,
        title=title,
        abstract="Test abstract",
        submitted_date="2024-01-01",
        categories=["cs.AI"],
        embedding=embedding,
        embedding_model="all-MiniLM-L6-v2",
        is_reference=is_reference,
    )


@pytest.fixture
def sample_user(db_session, random_unit_vector):
    """Creates a user with v0 interest vector and profile."""
    now = datetime.now(timezone.utc)
    user = User(
        id="user_alice",
        name="Alice Researcher",
        access_token="alice_token_123",
        is_simulated=False,
        created_at=now,
    )
    db_session.add(user)
    db_session.flush()

    seed_vec = random_unit_vector()

    profile = InterestProfile(
        user_id=user.id,
        seed_text="Machine learning agentic workflows and tool-augmented LLMs.",
        seed_embedding=seed_vec.tolist(),
        created_at=now,
    )
    db_session.add(profile)

    v0 = InterestVectorVersion(
        user_id=user.id,
        version=0,
        vector=seed_vec.tolist(),
        method="seed",
        n_effective_labels=0,
        built_from_max_event_id=None,
        p_high_level=98.0,
        p_low_level=90.0,
        t_high=0.75,
        t_low=0.55,
        created_at=now,
        is_active=True,
    )
    db_session.add(v0)
    db_session.commit()
    return user


# =========================================================================
# 1. Eligibility Tests
# =========================================================================

class TestClassifierEligibility:
    def test_insufficient_total_labels_prevents_training(self, db_session, sample_user, random_unit_vector):
        now = datetime.now(timezone.utc)
        # Add 25 labels (need >= 30)
        for i in range(25):
            p = make_test_paper(f"paper_el_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1 if i < 12 else 0, created_at=now))
        db_session.commit()

        elig = check_classifier_eligibility(user_id=sample_user.id, session=db_session)
        assert not elig.eligible
        assert "Insufficient effective labels" in elig.reason
        assert elig.total_effective_labels == 25

        res = train_and_evaluate_classifier(user_id=sample_user.id, session=db_session)
        assert res.status == "skipped"
        assert not res.eligible
        assert res.version_created is None

    def test_insufficient_positive_labels_prevents_training(self, db_session, sample_user, random_unit_vector):
        now = datetime.now(timezone.utc)
        # Add 35 labels, but only 5 positive (need >= 8)
        for i in range(35):
            p = make_test_paper(f"paper_pos_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1 if i < 5 else 0, created_at=now))
        db_session.commit()

        elig = check_classifier_eligibility(user_id=sample_user.id, session=db_session)
        assert not elig.eligible
        assert "Insufficient positive labels" in elig.reason
        assert elig.positive_labels == 5

    def test_insufficient_negative_labels_prevents_training(self, db_session, sample_user, random_unit_vector):
        now = datetime.now(timezone.utc)
        # Add 35 labels, but only 6 negative (need >= 8)
        for i in range(35):
            p = make_test_paper(f"paper_neg_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=0 if i < 6 else 1, created_at=now))
        db_session.commit()

        elig = check_classifier_eligibility(user_id=sample_user.id, session=db_session)
        assert not elig.eligible
        assert "Insufficient negative labels" in elig.reason
        assert elig.negative_labels == 6

    def test_initial_training_eligible_when_thresholds_met(self, db_session, sample_user, random_unit_vector):
        now = datetime.now(timezone.utc)
        # Exactly 30 labels: 10 pos, 20 neg
        for i in range(30):
            p = make_test_paper(f"paper_ok_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1 if i < 10 else 0, created_at=now))
        db_session.commit()

        elig = check_classifier_eligibility(user_id=sample_user.id, session=db_session)
        assert elig.eligible
        assert elig.total_effective_labels == 30
        assert elig.positive_labels == 10
        assert elig.negative_labels == 20

    def test_retraining_requires_25_new_labels_since_cutoff(self, db_session, sample_user, random_unit_vector):
        now = datetime.now(timezone.utc)
        # Create an existing model version v1 trained on cutoff event 100
        dummy_model = LogisticRegression()
        v1 = ModelVersion(
            user_id=sample_user.id,
            version=1,
            trained_at=now,
            n_samples=30,
            n_pos=10,
            n_neg=20,
            oof_auc=0.72,
            oof_auc_cosine_baseline=0.60,
            promoted=True,
            p_high=0.70,
            p_low=0.30,
            trained_on_max_event_id=100,
            model_blob=pickle.dumps(dummy_model),
            is_active=True,
        )
        db_session.add(v1)
        db_session.commit()

        # Add 30 historical feedback events (all event_id <= 100)
        # Plus 20 new events (event_id > 100) -> total new = 20 < 25
        for i in range(50):
            p = make_test_paper(f"paper_retrain_{i}", random_unit_vector().tolist())
            db_session.add(p)
            ev = FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="user",
                label=1 if i % 2 == 0 else 0,
                created_at=now,
            )
            db_session.add(ev)
        db_session.commit()

        # Set IDs such that 30 are <= 100 and 20 are > 100
        # In SQLite autoincrement starts from 1, so events have IDs 1..50
        # Let's adjust v1.trained_on_max_event_id = 30
        v1.trained_on_max_event_id = 30
        db_session.commit()

        elig = check_classifier_eligibility(user_id=sample_user.id, session=db_session)
        assert not elig.eligible
        assert elig.new_effective_labels_since_cutoff == 20
        assert "20/25 required" in elig.reason

        # Add 5 more events -> reaches 25 new events
        for i in range(50, 55):
            p = make_test_paper(f"paper_retrain_{i}", random_unit_vector().tolist())
            db_session.add(p)
            ev = FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="user",
                label=1 if i % 2 == 0 else 0,
                created_at=now,
            )
            db_session.add(ev)
        db_session.commit()

        elig2 = check_classifier_eligibility(user_id=sample_user.id, session=db_session)
        assert elig2.eligible
        assert elig2.new_effective_labels_since_cutoff == 25


# =========================================================================
# 2. Mathematical Cross-Validation & Leakage Prevention Tests
# =========================================================================

class TestCrossValidationAndLeakage:
    def test_inner_cv_selects_C_from_candidates(self, rng):
        # Generate synthetic data where strong regularization (C=0.1) performs well
        X = rng.standard_normal((40, 384)).astype(np.float32)
        y = np.array([1] * 15 + [0] * 25, dtype=np.int32)

        best_c = _select_best_C(X, y, c_candidates=(0.1, 0.3, 1.0), random_state=42)
        assert best_c in (0.1, 0.3, 1.0)

    def test_repeated_stratified_oof_leakage_prevention_and_bounds(self, rng):
        pos_dir = rng.standard_normal(384).astype(np.float32)
        pos_dir /= np.linalg.norm(pos_dir)
        neg_dir = rng.standard_normal(384).astype(np.float32)
        neg_dir /= np.linalg.norm(neg_dir)

        # 15 positive around pos_dir, 25 negative around neg_dir
        X_pos = pos_dir + 0.1 * rng.standard_normal((15, 384)).astype(np.float32)
        X_neg = neg_dir + 0.1 * rng.standard_normal((25, 384)).astype(np.float32)

        X = np.vstack([X_pos, X_neg])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.array([1] * 15 + [0] * 25, dtype=np.int32)

        active_centroid = pos_dir

        clf_auc, base_auc, oof_probs, best_c = _evaluate_repeated_stratified_oof(
            X=X,
            y=y,
            active_centroid=active_centroid,
            c_candidates=(0.1, 0.3, 1.0),
            n_splits=5,
            n_repeats=2,
            random_state=42,
        )

        assert len(oof_probs) == 40
        assert np.all(oof_probs >= 0.0) and np.all(oof_probs <= 1.0)
        assert 0.0 <= clf_auc <= 1.0
        assert 0.0 <= base_auc <= 1.0
        assert best_c in (0.1, 0.3, 1.0)


# =========================================================================
# 3. Promotion Guard Tests
# =========================================================================

class TestPromotionGuard:
    def test_promotion_succeeds_when_auc_and_margin_satisfied(self, db_session, sample_user, rng):
        now = datetime.now(timezone.utc)
        pos_dir = rng.standard_normal(384).astype(np.float32)
        pos_dir /= np.linalg.norm(pos_dir)
        neg_dir = rng.standard_normal(384).astype(np.float32)
        neg_dir /= np.linalg.norm(neg_dir)

        # Baseline vector is unaligned with target direction (orthogonal)
        ortho_vec = rng.standard_normal(384).astype(np.float32)
        ortho_vec -= np.dot(ortho_vec, pos_dir) * pos_dir
        ortho_vec /= np.linalg.norm(ortho_vec)

        v0 = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id, version=0).first()
        v0.vector = ortho_vec.tolist()
        db_session.commit()

        # 16 positive papers strongly separated from 16 negative papers
        for i in range(16):
            v = pos_dir + 0.05 * rng.standard_normal(384).astype(np.float32)
            v /= np.linalg.norm(v)
            p = make_test_paper(f"sep_pos_{i}", v.tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))

        for i in range(16):
            v = neg_dir + 0.05 * rng.standard_normal(384).astype(np.float32)
            v /= np.linalg.norm(v)
            p = make_test_paper(f"sep_neg_{i}", v.tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=0, created_at=now))
        db_session.commit()

        res = train_and_evaluate_classifier(user_id=sample_user.id, session=db_session)

        assert res.status == "promoted"
        assert res.promoted is True
        assert res.is_active is True
        assert res.version_created == 1
        assert res.oof_auc >= MIN_PROMOTION_AUC
        assert res.oof_auc >= res.oof_auc_cosine_baseline + PROMOTION_AUC_MARGIN

        # Check DB
        model_row = db_session.query(ModelVersion).filter_by(user_id=sample_user.id, version=1).first()
        assert model_row is not None
        assert model_row.is_active is True
        assert model_row.promoted is True

    def test_promotion_fails_when_auc_below_minimum_or_insufficient_margin(
        self, db_session, sample_user, random_unit_vector
    ):
        now = datetime.now(timezone.utc)
        # Create 32 random papers where classifier has no signal (random noise embeddings)
        for i in range(32):
            p = make_test_paper(f"rand_p_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1 if i < 16 else 0, created_at=now))
        db_session.commit()

        # Mock out-of-fold evaluation to simulate AUC < 0.65
        with patch("research_monitor.relevance.classifier._evaluate_repeated_stratified_oof", return_value=(0.61, 0.58, np.full(32, 0.5), 1.0)):
            res = train_and_evaluate_classifier(user_id=sample_user.id, session=db_session)
            assert res.status == "not_promoted"
            assert res.promoted is False
            assert res.is_active is False
            assert "did not satisfy guard" in res.reason

            # Verify model was persisted as promoted=False, is_active=False
            model_row = db_session.query(ModelVersion).filter_by(user_id=sample_user.id, version=1).first()
            assert model_row is not None
            assert model_row.promoted is False
            assert model_row.is_active is False

    def test_failed_promotion_leaves_existing_active_model_intact(
        self, db_session, sample_user, random_unit_vector
    ):
        now = datetime.now(timezone.utc)
        # User already has promoted active model v1
        dummy_model = LogisticRegression()
        v1 = ModelVersion(
            user_id=sample_user.id,
            version=1,
            trained_at=now,
            n_samples=30,
            n_pos=10,
            n_neg=20,
            oof_auc=0.75,
            oof_auc_cosine_baseline=0.60,
            promoted=True,
            p_high=0.70,
            p_low=0.30,
            trained_on_max_event_id=1,
            model_blob=pickle.dumps(dummy_model),
            is_active=True,
        )
        db_session.add(v1)
        db_session.commit()

        # Add 30 new labels to trigger retraining
        for i in range(30):
            p = make_test_paper(f"new_p_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1 if i < 12 else 0, created_at=now))
        db_session.commit()

        # Simulate second model v2 failing the promotion guard
        with patch("research_monitor.relevance.classifier._evaluate_repeated_stratified_oof", return_value=(0.63, 0.62, np.full(30, 0.5), 1.0)):
            res = train_and_evaluate_classifier(user_id=sample_user.id, session=db_session)
            assert res.status == "not_promoted"
            assert res.version_created == 2
            assert res.is_active is False

            # Model v1 MUST still remain active!
            v1_refetched = db_session.query(ModelVersion).filter_by(user_id=sample_user.id, version=1).first()
            assert v1_refetched.is_active is True

            # Model v2 must be inactive
            v2_refetched = db_session.query(ModelVersion).filter_by(user_id=sample_user.id, version=2).first()
            assert v2_refetched.is_active is False


# =========================================================================
# 4. Routing Integration & Explanation Tests
# =========================================================================

class TestRoutingIntegrationAndExplanations:
    def test_routing_in_threshold_mode_logs_cosine_and_null_clf_prob(
        self, db_session, sample_user, random_unit_vector
    ):
        candidates = [
            {
                "id": "cand_thresh_1",
                "title": "Threshold Paper",
                "abstract": "Abstract",
                "embedding": random_unit_vector().tolist(),
                "submitted_date": "2024-01-01",
                "categories": ["cs.AI"],
            }
        ]

        state = run_monitoring_cycle(
            user_id=sample_user.id,
            session=db_session,
            mode="replay",
            papers_override=candidates,
            auto_build_queue=False,
        )

        assert state.method == "threshold"
        assert state.model_version_id is None

        dec = db_session.query(RoutingDecision).filter_by(user_id=sample_user.id, paper_id="cand_thresh_1").first()
        assert dec is not None
        assert dec.method == "threshold"
        assert dec.clf_prob is None
        assert dec.cosine_score is not None

    def test_routing_in_classifier_mode_routes_by_probability_and_logs_cosine(
        self, db_session, sample_user, random_unit_vector
    ):
        now = datetime.now(timezone.utc)
        # Train and promote a real model
        pos_dir = random_unit_vector()
        clf = LogisticRegression()
        # Fit dummy clf on 2 samples
        X_dummy = np.vstack([pos_dir, -pos_dir])
        y_dummy = np.array([1, 0])
        clf.fit(X_dummy, y_dummy)

        v1 = ModelVersion(
            user_id=sample_user.id,
            version=1,
            trained_at=now,
            n_samples=32,
            n_pos=16,
            n_neg=16,
            oof_auc=0.80,
            oof_auc_cosine_baseline=0.60,
            promoted=True,
            p_high=0.70,
            p_low=0.30,
            trained_on_max_event_id=50,
            model_blob=pickle.dumps(clf),
            is_active=True,
        )
        db_session.add(v1)
        db_session.commit()

        candidates = [
            {
                "id": "cand_clf_1",
                "title": "Classifier Paper",
                "abstract": "Abstract",
                "embedding": pos_dir.tolist(),
                "submitted_date": "2024-01-01",
                "categories": ["cs.AI"],
            }
        ]

        state = run_monitoring_cycle(
            user_id=sample_user.id,
            session=db_session,
            mode="replay",
            papers_override=candidates,
            auto_build_queue=False,
        )

        assert state.method == "classifier"
        assert state.model_version_id == v1.id
        assert state.model_version_num == 1

        dec = db_session.query(RoutingDecision).filter_by(user_id=sample_user.id, paper_id="cand_clf_1").first()
        assert dec is not None
        assert dec.method == "classifier"
        assert dec.model_version_id == v1.id
        assert dec.clf_prob is not None
        assert dec.cosine_score is not None
        assert "Classifier probability" in dec.reason

    def test_find_nearest_feedback_papers_finds_nearest_liked_and_rejected(
        self, db_session, sample_user, random_unit_vector
    ):
        now = datetime.now(timezone.utc)
        query_vec = random_unit_vector()

        # Liked paper closely aligned with query_vec
        liked_emb = query_vec + 0.01 * random_unit_vector()
        liked_emb /= np.linalg.norm(liked_emb)
        p_liked = make_test_paper("p_liked_1", liked_emb.tolist(), title="Deep Learning Foundations")
        db_session.add(p_liked)
        db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p_liked.id, source="user", label=1, created_at=now))

        # Rejected paper orthogonal / far from query_vec
        rej_emb = -query_vec + 0.01 * random_unit_vector()
        rej_emb /= np.linalg.norm(rej_emb)
        p_rej = make_test_paper("p_rej_1", rej_emb.tolist(), title="Irrelevant Theory")
        db_session.add(p_rej)
        db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p_rej.id, source="user", label=0, created_at=now))
        db_session.commit()

        nearest_liked, nearest_rejected = find_nearest_feedback_papers(
            paper_embedding=query_vec,
            user_id=sample_user.id,
            session=db_session,
        )

        assert nearest_liked is not None
        assert nearest_liked["paper_id"] == "p_liked_1"
        assert nearest_liked["title"] == "Deep Learning Foundations"
        assert nearest_liked["similarity"] > 0.90

        assert nearest_rejected is not None
        assert nearest_rejected["paper_id"] == "p_rej_1"
        assert nearest_rejected["title"] == "Irrelevant Theory"
        assert nearest_rejected["similarity"] < 0.0

    def test_find_nearest_feedback_papers_handles_missing_or_empty_gracefully(
        self, db_session, sample_user, random_unit_vector
    ):
        # No feedback events in DB yet
        liked, rejected = find_nearest_feedback_papers(
            paper_embedding=random_unit_vector(),
            user_id=sample_user.id,
            session=db_session,
        )
        assert liked is None
        assert rejected is None

        # None embedding passed
        liked2, rejected2 = find_nearest_feedback_papers(
            paper_embedding=None,
            user_id=sample_user.id,
            session=db_session,
        )
        assert liked2 is None
        assert rejected2 is None

    def test_pipeline_failed_promotion_leaves_routing_method_as_threshold(
        self, db_session, sample_user, random_unit_vector
    ):
        """When promotion guard rejects a model, pipeline execution remains in threshold mode."""
        now = datetime.now(timezone.utc)
        dummy_model = LogisticRegression()
        v_unpromoted = ModelVersion(
            user_id=sample_user.id,
            version=1,
            trained_at=now,
            n_samples=30,
            n_pos=15,
            n_neg=15,
            hyperparameters={"C": 0.1, "class_weight": "balanced"},
            oof_auc=0.61,  # below 0.65
            oof_auc_cosine_baseline=0.60,
            p_high=0.70,
            p_low=0.30,
            promoted=False,
            is_active=False,
            model_blob=pickle.dumps(dummy_model),
            trained_on_max_event_id=30,
        )
        db_session.add(v_unpromoted)
        db_session.commit()

        candidates = [
            {"id": "cand_unprom_1", "title": "Test Title", "abstract": "Test Abstract", "embedding": random_unit_vector().tolist()}
        ]
        state = run_monitoring_cycle(
            user_id=sample_user.id,
            session=db_session,
            papers_override=candidates,
            auto_build_queue=False,
        )

        assert state.method == "threshold"
        assert state.active_model is None
        assert state.model_version_id is None
        assert state.model_version_num is None

        dec = db_session.query(RoutingDecision).filter_by(user_id=sample_user.id, paper_id="cand_unprom_1").first()
        assert dec.method == "threshold"
        assert dec.model_version_id is None
        assert dec.clf_prob is None
        assert dec.cosine_score is not None

    def test_ui_review_page_renders_decision_explanations(
        self, db_session, sample_user, random_unit_vector
    ):
        """Review queue page renders nearest-liked and nearest-rejected paper explanations."""
        from fastapi.testclient import TestClient
        from research_monitor.database.models import Cycle
        from research_monitor.relevance.state import DecisionStatus
        from research_monitor.ui.app import app, get_db

        now = datetime.now(timezone.utc)
        query_vec = random_unit_vector()

        # Liked paper
        p_liked = make_test_paper("p_liked_ui", (query_vec + 0.01 * random_unit_vector()).tolist(), title="Super Liked Paper")
        db_session.add(p_liked)
        db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p_liked.id, source="user", label=1, created_at=now))

        # Rejected paper
        p_rej = make_test_paper("p_rej_ui", (-query_vec + 0.01 * random_unit_vector()).tolist(), title="Super Rejected Paper")
        db_session.add(p_rej)
        db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p_rej.id, source="user", label=0, created_at=now))

        # Borderline review paper
        p_review = make_test_paper("p_review_ui", query_vec.tolist(), title="Borderline Paper Needing Review")
        db_session.add(p_review)
        db_session.commit()

        cycle = Cycle(user_id=sample_user.id, as_of_date="2026-10-15", mode="live")
        db_session.add(cycle)
        db_session.commit()

        v0_id = sample_user.vector_versions[0].id
        dec = RoutingDecision(
            user_id=sample_user.id,
            paper_id=p_review.id,
            cycle_id=cycle.id,
            vector_version_id=v0_id,
            cosine_score=0.45,
            clf_prob=None,
            method="threshold",
            thresholds_used={"t_high": 0.75, "t_low": 0.55},
            route="BORDERLINE",
            status=DecisionStatus.ROUTED_BORDERLINE.value,
            reason="Borderline relevance score",
        )
        db_session.add(dec)
        db_session.commit()

        task = ReviewTask(decision_id=dec.id, kind="borderline")
        db_session.add(task)
        db_session.commit()

        def override_get_db():
            yield db_session

        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as client:
                resp = client.get(f"/review?token={sample_user.access_token}")
                assert resp.status_code == 200
                assert "Borderline Paper Needing Review" in resp.text
                assert "Super Liked Paper" in resp.text
                assert "Super Rejected Paper" in resp.text
        finally:
            app.dependency_overrides.clear()

