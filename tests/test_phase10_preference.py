"""
test_phase10_preference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit and integration tests for Phase 10:
Preference Memory — Interest Vector (Rocchio Centroid).

Test coverage:
  1. Positive feedback moves the centroid toward positive paper embeddings.
  2. Negative feedback moves the centroid away from negative paper embeddings.
  3. The negative term is omitted when there are no negative examples.
  4. The original seed vector remains unchanged.
  5. The output is a finite, normalized 384-dimensional vector.
  6. Insufficient labels prevent the first update (< 10 total).
  7. Fewer than 3 positive labels prevent the first update (< 3 positive).
  8. Exactly 10 effective labels with at least 3 positives allow the first update.
  9. A subsequent update does not happen before 25 new effective labels and does happen when reached.
  10. Repeated events for one paper count as one effective label.
  11. The latest user feedback takes precedence over reviewer feedback.
  12. Users' feedback is never mixed.
  13. The correct reference sample is used to recalculate both thresholds.
  14. Old versions are preserved and only the intended version is active.
  15. built_from_max_event_id is stored correctly.
  16. Re-running without new qualifying feedback creates no duplicate version.
  17. The new version is used in the next cycle, while historical routing decisions retain their IDs.
  18. Invalid embeddings, a degenerate centroid, missing reference papers and transaction failures are handled safely.
"""

from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_monitor.database.models import (
    Base,
    FeedbackEvent,
    InterestProfile,
    InterestVectorVersion,
    Paper,
    RoutingDecision,
    TrainingLabel,
    User,
    create_training_labels_view,
)
from research_monitor.relevance.pipeline import run_monitoring_cycle
from research_monitor.relevance.preference import (
    FIRST_UPDATE_MIN_LABELS,
    FIRST_UPDATE_MIN_POS,
    ROCCHIO_ALPHA,
    ROCCHIO_BETA,
    ROCCHIO_GAMMA,
    SUBSEQUENT_UPDATE_INTERVAL,
    VectorUpdateEligibility,
    VectorUpdateResult,
    check_vector_update_eligibility,
    compute_rocchio_centroid,
    get_user_seed_embedding,
    update_user_interest_vector,
)


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


@pytest.fixture
def reference_embeddings(rng):
    """Synthetic reference sample matrix of 200 unit vectors."""
    raw = rng.standard_normal((200, 384)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_test_paper(
    paper_id: str,
    embedding: list[float],
    title: str = "Test Title",
    is_reference: bool = False,
) -> Paper:
    """Helper to construct a valid Paper entity for tests."""
    return Paper(
        id=paper_id,
        title=title,
        abstract="Test Abstract",
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
# 1. Pure Mathematical Rocchio Tests
# =========================================================================

class TestRocchioMathematics:
    def test_positive_feedback_moves_centroid_toward_positive_embeddings(self, random_unit_vector):
        c0 = random_unit_vector()

        # Create distinct positive papers in a specific direction
        target_dir = random_unit_vector()
        pos_embeddings = [target_dir, target_dir]

        updated = compute_rocchio_centroid(
            c0=c0,
            positive_embeddings=pos_embeddings,
            negative_embeddings=[],
        )

        initial_sim = float(np.dot(c0, target_dir))
        updated_sim = float(np.dot(updated, target_dir))

        assert updated_sim > initial_sim, "Centroid must move toward positive examples"

    def test_negative_feedback_moves_centroid_away_from_negative_embeddings(self, random_unit_vector):
        c0 = random_unit_vector()
        neg_dir = random_unit_vector()
        neg_embeddings = [neg_dir, neg_dir]

        updated = compute_rocchio_centroid(
            c0=c0,
            positive_embeddings=[],
            negative_embeddings=neg_embeddings,
        )

        initial_sim = float(np.dot(c0, neg_dir))
        updated_sim = float(np.dot(updated, neg_dir))

        assert updated_sim < initial_sim, "Centroid must move away from negative examples"

    def test_negative_term_omitted_when_no_negative_examples(self, random_unit_vector):
        c0 = random_unit_vector()
        p1 = random_unit_vector()

        # Compute with empty list
        c_empty = compute_rocchio_centroid(c0=c0, positive_embeddings=[p1], negative_embeddings=[])
        # Compute with default ()
        c_default = compute_rocchio_centroid(c0=c0, positive_embeddings=[p1])

        np.testing.assert_allclose(c_empty, c_default, rtol=1e-6)

        # Expected manual calculation: alpha * c0 + beta * p1
        expected_raw = ROCCHIO_ALPHA * c0 + ROCCHIO_BETA * p1
        expected = expected_raw / np.linalg.norm(expected_raw)
        np.testing.assert_allclose(c_empty, expected, rtol=1e-5)

    def test_original_seed_vector_remains_unchanged(self, random_unit_vector):
        c0 = random_unit_vector()
        c0_copy = c0.copy()
        pos = [random_unit_vector() for _ in range(3)]
        neg = [random_unit_vector() for _ in range(2)]

        _ = compute_rocchio_centroid(c0=c0, positive_embeddings=pos, negative_embeddings=neg)

        np.testing.assert_array_equal(c0, c0_copy, "Seed vector must not be mutated in-place")

    def test_output_is_finite_normalized_384_dimensional_vector(self, random_unit_vector):
        c0 = random_unit_vector()
        pos = [random_unit_vector() for _ in range(5)]
        neg = [random_unit_vector() for _ in range(3)]

        centroid = compute_rocchio_centroid(c0=c0, positive_embeddings=pos, negative_embeddings=neg)

        assert centroid.shape == (384,)
        assert centroid.dtype == np.float32
        assert np.all(np.isfinite(centroid))
        norm = float(np.linalg.norm(centroid))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_degenerate_near_zero_centroid_raises_value_error(self):
        # Construct exact cancellation: alpha * c0 - gamma * neg = 0
        c0 = np.zeros(384, dtype=np.float32)
        c0[0] = 1.0

        neg = np.zeros(384, dtype=np.float32)
        neg[0] = 1.0  # normalized

        # Force alpha * c0 - gamma * neg = 0 by choosing alpha=0.25, gamma=0.25
        with pytest.raises(ValueError, match="Degenerate near-zero centroid"):
            compute_rocchio_centroid(
                c0=c0,
                positive_embeddings=[],
                negative_embeddings=[neg],
                alpha=0.25,
                beta=0.0,
                gamma=0.25,
            )

    def test_invalid_dimensions_and_non_finite_inputs_raise_cleanly(self):
        with pytest.raises(ValueError, match="dimension 384"):
            compute_rocchio_centroid(c0=np.ones(10), positive_embeddings=[])

        nan_vec = np.zeros(384, dtype=np.float32)
        nan_vec[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            compute_rocchio_centroid(c0=nan_vec, positive_embeddings=[])


# =========================================================================
# 2. Update Trigger Policy Tests
# =========================================================================

class TestUpdateTriggers:
    def test_insufficient_labels_prevent_first_update(self, db_session, sample_user, random_unit_vector):
        # Add 9 effective labels (need 10)
        now = datetime.now(timezone.utc)
        for i in range(9):
            p = make_test_paper(f"paper_test_{i}", random_unit_vector().tolist(), f"Title {i}")
            db_session.add(p)
            ev = FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="reviewer",
                label=1 if i < 5 else 0,
                created_at=now,
            )
            db_session.add(ev)
        db_session.commit()

        eligibility = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert not eligibility.eligible
        assert "Insufficient effective labels" in eligibility.reason
        assert eligibility.total_effective_labels == 9

        # update_user_interest_vector should skip safely
        res = update_user_interest_vector(user_id=sample_user.id, session=db_session)
        assert res.status == "skipped"
        assert not res.eligible
        assert res.version_created is None

    def test_fewer_than_3_positive_labels_prevent_first_update(self, db_session, sample_user, random_unit_vector):
        # Add 12 effective labels, but only 2 positive (need >= 3)
        now = datetime.now(timezone.utc)
        for i in range(12):
            p = make_test_paper(f"paper_test_pos_{i}", random_unit_vector().tolist(), f"Title {i}")
            db_session.add(p)
            ev = FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="reviewer",
                label=1 if i < 2 else 0,  # Exactly 2 positive
                created_at=now,
            )
            db_session.add(ev)
        db_session.commit()

        eligibility = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert not eligibility.eligible
        assert "Insufficient positive labels" in eligibility.reason
        assert eligibility.positive_labels == 2

    def test_exactly_10_effective_labels_with_at_least_3_positives_allows_first_update(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        # Add exactly 10 effective labels: 3 positive, 7 negative
        now = datetime.now(timezone.utc)
        for i in range(10):
            p = make_test_paper(f"paper_exact_10_{i}", random_unit_vector().tolist(), f"Title {i}")
            db_session.add(p)
            ev = FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="reviewer",
                label=1 if i < 3 else 0,
                created_at=now,
            )
            db_session.add(ev)
        db_session.commit()

        eligibility = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert eligibility.eligible
        assert eligibility.total_effective_labels == 10
        assert eligibility.positive_labels == 3
        assert eligibility.negative_labels == 7

        res = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )

        assert res.status == "updated"
        assert res.eligible
        assert res.version_created == 1
        assert res.n_effective_labels_used == 10
        assert res.n_positive_labels_used == 3
        assert res.n_negative_labels_used == 7
        assert res.is_active_for_next_cycle

    def test_subsequent_update_requires_25_new_labels(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        now = datetime.now(timezone.utc)
        # 1. Trigger first update with 10 papers
        for i in range(10):
            p = make_test_paper(f"paper_v1_{i}", random_unit_vector().tolist(), f"Title {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        res1 = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res1.version_created == 1
        cutoff_v1 = res1.built_from_max_event_id

        # 2. Add 24 new labels (need 25)
        for i in range(24):
            p = make_test_paper(f"paper_v2_wait_{i}", random_unit_vector().tolist(), f"Title {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=0, created_at=now))
        db_session.commit()

        # Check eligibility
        eligibility = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert not eligibility.eligible
        assert eligibility.new_effective_labels_since_cutoff == 24
        assert "24/25 required" in eligibility.reason

        res_skip = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res_skip.status == "skipped"
        assert res_skip.version_created is None

        # 3. Add the 25th label
        p25 = make_test_paper("paper_v2_25th", random_unit_vector().tolist(), "Title 25th")
        db_session.add(p25)
        db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p25.id, source="user", label=1, created_at=now))
        db_session.commit()

        eligibility2 = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert eligibility2.eligible
        assert eligibility2.new_effective_labels_since_cutoff == 25

        res2 = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res2.status == "updated"
        assert res2.version_created == 2
        assert res2.n_effective_labels_used == 35  # All 35 effective labels used
        assert res2.built_from_max_event_id > cutoff_v1


# =========================================================================
# 3. Feedback Precedence & Multi-Tenant Isolation Tests
# =========================================================================

class TestFeedbackPrecedenceAndIsolation:
    def test_repeated_events_for_one_paper_count_as_one_effective_label(
        self, db_session, sample_user, random_unit_vector
    ):
        p = make_test_paper("paper_repeated", random_unit_vector().tolist(), "Multi-reaction Paper")
        db_session.add(p)
        now = datetime.now(timezone.utc)

        # 5 consecutive user reactions for the same paper
        for i in range(5):
            db_session.add(
                FeedbackEvent(
                    user_id=sample_user.id,
                    paper_id=p.id,
                    source="user",
                    label=1 if i % 2 == 0 else 0,
                    created_at=now,
                )
            )
        db_session.commit()

        eligibility = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert eligibility.total_effective_labels == 1, "Must count effective labels, not raw events"

    def test_latest_user_feedback_takes_precedence_over_reviewer_feedback(
        self, db_session, sample_user, random_unit_vector
    ):
        p = make_test_paper("paper_precedence", random_unit_vector().tolist(), "Precedence Paper")
        db_session.add(p)
        now = datetime.now(timezone.utc)

        # 1. Reviewer rejects paper (label=0)
        db_session.add(
            FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="reviewer",
                label=0,
                created_at=now,
            )
        )
        db_session.commit()

        labels1 = db_session.query(TrainingLabel).filter_by(user_id=sample_user.id).all()
        assert len(labels1) == 1
        assert labels1[0].label == 0
        assert labels1[0].origin == "reviewer"

        # 2. Later user likes paper (label=1)
        db_session.add(
            FeedbackEvent(
                user_id=sample_user.id,
                paper_id=p.id,
                source="user",
                label=1,
                created_at=now,
            )
        )
        db_session.commit()

        labels2 = db_session.query(TrainingLabel).filter_by(user_id=sample_user.id).all()
        assert len(labels2) == 1
        assert labels2[0].label == 1
        assert labels2[0].origin == "user"

    def test_users_feedback_is_never_mixed(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        # Create second user
        now = datetime.now(timezone.utc)
        user_bob = User(id="user_bob", name="Bob", is_simulated=False, created_at=now)
        db_session.add(user_bob)
        db_session.add(
            InterestProfile(
                user_id=user_bob.id,
                seed_text="Bob's interests",
                seed_embedding=random_unit_vector().tolist(),
                created_at=now,
            )
        )
        db_session.add(
            InterestVectorVersion(
                user_id=user_bob.id,
                version=0,
                vector=random_unit_vector().tolist(),
                method="seed",
                p_high_level=98.0,
                p_low_level=90.0,
                t_high=0.75,
                t_low=0.55,
                created_at=now,
                is_active=True,
            )
        )

        # Add 10 positive labels for Alice
        for i in range(10):
            p = make_test_paper(f"paper_alice_{i}", random_unit_vector().tolist(), f"Paper A {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))

        # Add 5 labels for Bob
        for i in range(5):
            p = make_test_paper(f"paper_bob_{i}", random_unit_vector().tolist(), f"Paper B {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=user_bob.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        # Alice should be eligible for v1
        elig_alice = check_vector_update_eligibility(user_id=sample_user.id, session=db_session)
        assert elig_alice.eligible
        assert elig_alice.total_effective_labels == 10

        # Bob should NOT be eligible (only 5 labels)
        elig_bob = check_vector_update_eligibility(user_id=user_bob.id, session=db_session)
        assert not elig_bob.eligible
        assert elig_bob.total_effective_labels == 5


# =========================================================================
# 4. State Persistence, Activation & Next-Cycle Idempotency Tests
# =========================================================================

class TestPersistenceAndCycleActivation:
    def test_old_versions_preserved_and_only_new_version_is_active(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        now = datetime.now(timezone.utc)
        for i in range(10):
            p = make_test_paper(f"paper_persist_{i}", random_unit_vector().tolist(), f"P {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        res = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res.version_created == 1

        all_versions = (
            db_session.query(InterestVectorVersion)
            .filter_by(user_id=sample_user.id)
            .order_by(InterestVectorVersion.version.asc())
            .all()
        )
        assert len(all_versions) == 2
        # v0 must be preserved but inactive
        assert all_versions[0].version == 0
        assert not all_versions[0].is_active
        assert all_versions[0].method == "seed"

        # v1 must be active
        assert all_versions[1].version == 1
        assert all_versions[1].is_active
        assert all_versions[1].method == "rocchio"
        assert all_versions[1].n_effective_labels == 10

    def test_built_from_max_event_id_stored_correctly(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        now = datetime.now(timezone.utc)
        events = []
        for i in range(10):
            p = make_test_paper(f"paper_event_id_{i}", random_unit_vector().tolist(), f"P {i}")
            db_session.add(p)
            ev = FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now)
            db_session.add(ev)
            events.append(ev)
        db_session.commit()

        expected_max_event_id = max(e.id for e in events)

        res = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res.status == "updated"
        assert res.built_from_max_event_id == expected_max_event_id

        v1 = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id, version=1).first()
        assert v1.built_from_max_event_id == expected_max_event_id

    def test_rerunning_without_new_qualifying_feedback_creates_no_duplicate_version(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        now = datetime.now(timezone.utc)
        for i in range(10):
            p = make_test_paper(f"paper_dup_{i}", random_unit_vector().tolist(), f"P {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        # Run 1: updates to v1
        res1 = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res1.status == "updated"
        assert res1.version_created == 1

        # Run 2 immediately: must be skipped
        res2 = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert res2.status == "skipped"
        assert res2.version_created is None

        # Verify exactly 2 versions in DB (v0 and v1)
        count = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id).count()
        assert count == 2

    def test_correct_reference_sample_used_to_recalculate_both_thresholds(
        self, db_session, sample_user, random_unit_vector
    ):
        now = datetime.now(timezone.utc)
        # Create 100 reference papers and 50 non-reference papers
        rng = np.random.default_rng(123)
        ref_embeddings = []
        for i in range(100):
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            ref_embeddings.append(v)
            p = make_test_paper(f"ref_paper_{i}", v.tolist(), is_reference=True)
            db_session.add(p)

        for i in range(50):
            v = rng.standard_normal(384).astype(np.float32)
            v = v / np.linalg.norm(v)
            p = make_test_paper(f"non_ref_paper_{i}", v.tolist(), is_reference=False)
            db_session.add(p)

        # Add 10 positive labels
        for i in range(10):
            p = make_test_paper(f"paper_lbl_{i}", random_unit_vector().tolist(), f"L {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        # Update vector passing the explicit 100 reference embeddings
        ref_matrix = np.array(ref_embeddings, dtype=np.float32)
        res = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=ref_matrix,
        )
        assert res.status == "updated"
        assert res.t_high is not None
        assert res.t_low is not None
        assert res.t_high >= res.t_low

        # Manually compute percentiles on the reference matrix
        v1 = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id, version=1).first()
        centroid_arr = np.array(v1.vector, dtype=np.float32)
        scores = ref_matrix @ centroid_arr
        expected_t_high = float(np.percentile(scores, 98.0))
        expected_t_low = float(np.percentile(scores, 90.0))

        assert res.t_high == pytest.approx(expected_t_high, abs=1e-5)
        assert res.t_low == pytest.approx(expected_t_low, abs=1e-5)

    def test_new_version_used_in_next_cycle_and_historical_decisions_retained(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        now = datetime.now(timezone.utc)
        v0 = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id, version=0).first()

        # Prepare candidate papers for Cycle 1
        cycle1_candidates = [
            {
                "id": f"c1_paper_{i}",
                "title": f"C1 Paper {i}",
                "abstract": f"Abstract {i}",
                "embedding": random_unit_vector().tolist(),
                "submitted_date": "2024-01-01",
                "categories": ["cs.AI"],
            }
            for i in range(5)
        ]

        # Execute Cycle 1 (uses v0)
        state_c1 = run_monitoring_cycle(
            user_id=sample_user.id,
            session=db_session,
            mode="replay",
            papers_override=cycle1_candidates,
            auto_build_queue=False,
        )
        assert state_c1.vector_version_num == 0
        assert state_c1.vector_version_id == v0.id

        # Verify decisions in Cycle 1 point to v0.id
        decisions_c1 = db_session.query(RoutingDecision).filter_by(user_id=sample_user.id).all()
        assert len(decisions_c1) == 5
        for d in decisions_c1:
            assert d.vector_version_id == v0.id

        # Now simulate user reviewing and providing 10 positive labels
        for i in range(10):
            p = make_test_paper(f"paper_feedback_{i}", random_unit_vector().tolist(), f"F {i}")
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        # Trigger vector update -> produces v1
        update_res = update_user_interest_vector(
            user_id=sample_user.id,
            session=db_session,
            reference_embeddings=reference_embeddings,
        )
        assert update_res.status == "updated"
        assert update_res.version_created == 1
        v1_id = update_res.vector_version_id

        # Historical decisions from Cycle 1 MUST still retain v0.id
        decisions_after = (
            db_session.query(RoutingDecision)
            .filter(RoutingDecision.paper_id.in_([c["id"] for c in cycle1_candidates]))
            .all()
        )
        for d in decisions_after:
            assert d.vector_version_id == v0.id

        # Execute Cycle 2 (must now autonomously pick up v1!)
        cycle2_candidates = [
            {
                "id": f"c2_paper_{i}",
                "title": f"C2 Paper {i}",
                "abstract": f"Abstract {i}",
                "embedding": random_unit_vector().tolist(),
                "submitted_date": "2024-01-02",
                "categories": ["cs.AI"],
            }
            for i in range(5)
        ]

        state_c2 = run_monitoring_cycle(
            user_id=sample_user.id,
            session=db_session,
            mode="replay",
            papers_override=cycle2_candidates,
            auto_build_queue=False,
        )
        assert state_c2.vector_version_num == 1
        assert state_c2.vector_version_id == v1_id

        # Decisions for Cycle 2 must record v1_id
        decisions_c2 = (
            db_session.query(RoutingDecision)
            .filter(RoutingDecision.paper_id.in_([c["id"] for c in cycle2_candidates]))
            .all()
        )
        assert len(decisions_c2) == 5
        for d in decisions_c2:
            assert d.vector_version_id == v1_id

    def test_safe_handling_of_invalid_embeddings_missing_reference_and_failures(
        self, db_session, sample_user, random_unit_vector
    ):
        now = datetime.now(timezone.utc)
        # 10 feedback events referencing non-existent paper records (missing embeddings)
        for i in range(10):
            db_session.add(
                FeedbackEvent(
                    user_id=sample_user.id,
                    paper_id=f"paper_missing_record_{i}",
                    source="user",
                    label=1,
                    created_at=now,
                )
            )
        db_session.commit()

        # Update should safely skip due to no valid positive embeddings found
        res = update_user_interest_vector(user_id=sample_user.id, session=db_session)
        assert res.status == "skipped"
        assert "No valid positive paper embeddings" in res.reason

        # Non-existent user returns error status
        res_nonexistent = update_user_interest_vector(user_id="nobody", session=db_session)
        assert res_nonexistent.status == "error"
        assert "does not exist" in res_nonexistent.reason

        # Transaction failure during update rolls back and returns error status
        with patch.object(db_session, "commit", side_effect=RuntimeError("Simulated DB connection drop")):
            res_fail = update_user_interest_vector(
                user_id=sample_user.id,
                session=db_session,
            )
            # Since skipped occurs before commit when embeddings are missing, let's test with valid embeddings
            # but simulated commit failure
            pass

    def test_transaction_rollback_on_commit_failure(
        self, db_session, sample_user, random_unit_vector, reference_embeddings
    ):
        now = datetime.now(timezone.utc)
        for i in range(10):
            p = make_test_paper(f"paper_tx_{i}", random_unit_vector().tolist())
            db_session.add(p)
            db_session.add(FeedbackEvent(user_id=sample_user.id, paper_id=p.id, source="user", label=1, created_at=now))
        db_session.commit()

        with patch.object(db_session, "commit", side_effect=RuntimeError("Simulated DB connection drop")):
            res = update_user_interest_vector(
                user_id=sample_user.id,
                session=db_session,
                reference_embeddings=reference_embeddings,
            )
            assert res.status == "error"
            assert "Simulated DB connection drop" in res.reason

        # Ensure no partially committed v1 exists
        v1 = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id, version=1).first()
        assert v1 is None
        # v0 remains active
        v0 = db_session.query(InterestVectorVersion).filter_by(user_id=sample_user.id, version=0).first()
        assert v0.is_active
