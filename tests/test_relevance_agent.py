"""
test_relevance_agent.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for Phase 5:
  - Exact router boundary tests (equality and inequalities)
  - State machine node transitions (discover -> preprocess -> embed -> score -> route)
  - Reference percentile threshold verification
  - User creation and vector version 0 initialization
  - Auditable decision persistence (including LOW papers)
  - Re-running cycles prevents duplicate decisions
"""

from datetime import datetime, timezone
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from research_monitor.database.models import (
    Base,
    Cycle,
    InterestProfile,
    InterestVectorVersion,
    Paper,
    RoutingDecision,
    User,
)
from research_monitor.relevance.pipeline import (
    discover_stage,
    embed_stage,
    preprocess_stage,
    route_stage,
    run_monitoring_cycle,
    score_stage,
)
from research_monitor.relevance.router import route_score
from research_monitor.relevance.state import (
    DecisionStatus,
    PaperCandidate,
    PipelineState,
    Route,
)
from research_monitor.relevance.thresholds import (
    compute_vector_thresholds,
)
from research_monitor.relevance.user_manager import (
    create_user,
    get_active_vector_version,
)


@pytest.fixture
def in_memory_session():
    """Isolated in-memory SQLite session for unit testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# =========================================================================
# 1. Router Boundary Tests (Roadmap Section 7.2 & 9.1)
# =========================================================================

class TestRouterBoundaries:
    def test_score_strictly_above_t_high(self):
        res = route_score(score=0.55, t_high=0.50, t_low=0.30, p_high=98.0, p_low=90.0)
        assert res.route == Route.HIGH
        assert res.status == DecisionStatus.ROUTED_HIGH
        assert "Cleared gate for automatic delivery" in res.reason

    def test_score_exactly_equal_t_high_routes_high(self):
        """Boundary condition: score == t_high must route to HIGH."""
        res = route_score(score=0.5000, t_high=0.5000, t_low=0.3000, p_high=98.0, p_low=90.0)
        assert res.route == Route.HIGH
        assert res.status == DecisionStatus.ROUTED_HIGH
        assert ">=" in res.reason

    def test_score_strictly_below_t_low(self):
        res = route_score(score=0.25, t_high=0.50, t_low=0.30, p_high=98.0, p_low=90.0)
        assert res.route == Route.LOW
        assert res.status == DecisionStatus.DISCARDED
        assert "Discarded paper below relevance floor" in res.reason

    def test_score_exactly_equal_t_low_routes_low(self):
        """Boundary condition: score == t_low must route to LOW."""
        res = route_score(score=0.3000, t_high=0.5000, t_low=0.3000, p_high=98.0, p_low=90.0)
        assert res.route == Route.LOW
        assert res.status == DecisionStatus.DISCARDED
        assert "<=" in res.reason

    def test_score_strictly_between_thresholds_routes_borderline(self):
        res = route_score(score=0.40, t_high=0.50, t_low=0.30, p_high=98.0, p_low=90.0)
        assert res.route == Route.BORDERLINE
        assert res.status == DecisionStatus.ROUTED_BORDERLINE
        assert "borderline band" in res.reason
        assert "Queued for human review" in res.reason


# =========================================================================
# 2. Reference Percentile Threshold Computation (Section 7.2)
# =========================================================================

class TestThresholdComputation:
    def test_reference_percentiles_calculation(self):
        rng = np.random.default_rng(42)
        # 5,000 reference embeddings, L2 normalized
        raw_embs = rng.standard_normal((5000, 384)).astype(np.float32)
        ref_embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

        # Seed vector
        raw_seed = rng.standard_normal(384).astype(np.float32)
        seed_vec = raw_seed / np.linalg.norm(raw_seed)

        thresholds = compute_vector_thresholds(
            vector=seed_vec,
            reference_embeddings=ref_embs,
            p_high=98.0,
            p_low=90.0,
        )

        assert thresholds.p_high == 98.0
        assert thresholds.p_low == 90.0
        assert thresholds.t_high > thresholds.t_low

        # Verification per Roadmap Section 9.1:
        # About (100 - P_high)% of reference papers score >= t_high
        scores = np.dot(ref_embs, seed_vec)
        high_share = float(np.mean(scores >= thresholds.t_high))
        assert high_share == pytest.approx(0.02, abs=0.005)

        low_share = float(np.mean(scores <= thresholds.t_low))
        assert low_share == pytest.approx(0.90, abs=0.005)


# =========================================================================
# 3. User Creation and Vector Version 0 Initialization
# =========================================================================

class TestUserManager:
    def test_create_user_and_vector_v0(self, in_memory_session):
        rng = np.random.default_rng(42)
        ref_embs = rng.standard_normal((100, 384)).astype(np.float32)
        ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)

        user = create_user(
            user_id="alice_01",
            name="Alice Researcher",
            seed_text="LLM reasoning and multi-agent tool execution.",
            session=in_memory_session,
            is_simulated=True,
            reference_embeddings=ref_embs,
        )

        assert user.id == "alice_01"
        assert user.name == "Alice Researcher"
        assert user.is_simulated is True

        # Check InterestProfile
        profile = in_memory_session.query(InterestProfile).filter_by(user_id="alice_01").first()
        assert profile is not None
        assert profile.seed_text == "LLM reasoning and multi-agent tool execution."
        assert len(profile.seed_embedding) == 384

        # Check InterestVectorVersion
        v0 = get_active_vector_version("alice_01", in_memory_session)
        assert v0 is not None
        assert v0.version == 0
        assert v0.method == "seed"
        assert v0.is_active is True
        assert v0.t_high > v0.t_low
        assert v0.p_high_level == 98.0
        assert v0.p_low_level == 90.0


# =========================================================================
# 4. Pipeline State Machine Node Transitions
# =========================================================================

class TestPipelineNodes:
    def test_explicit_pipeline_flow(self, in_memory_session):
        rng = np.random.default_rng(123)
        v = rng.standard_normal(384).astype(np.float32)
        v = v / np.linalg.norm(v)

        state = PipelineState(
            user_id="bob",
            cycle_id=1,
            mode="replay",
            as_of_date="2024-06-30",
            active_vector=v,
            vector_version_id=1,
            vector_version_num=0,
            t_high=0.50,
            t_low=0.20,
            p_high_level=98.0,
            p_low_level=90.0,
        )

        # Candidate papers fixture
        # paper1: high (score ~0.9), paper2: borderline (score ~0.35), paper3: low (score ~ -0.1)
        emb_high = v.copy()
        # Create vector with exact 0.35 cosine similarity to v
        noise = rng.standard_normal(384).astype(np.float32)
        noise = noise - np.dot(noise, v) * v
        noise = noise / np.linalg.norm(noise)
        target_sim = 0.35
        emb_borderline = (target_sim * v + np.sqrt(1.0 - target_sim**2) * noise).astype(np.float32)
        emb_low = -v.copy()

        override_papers = [
            {"id": "p_high", "title": "  Agent Paper  ", "abstract": "Reasoning tools.", "embedding": emb_high},
            {"id": "p_border", "title": "Border Paper", "abstract": "General ML.", "embedding": emb_borderline},
            {"id": "p_low", "title": "Low Paper", "abstract": "Unrelated topic.", "embedding": emb_low},
        ]

        # 1. Discover
        discover_stage(state, in_memory_session, papers_override=override_papers)
        assert len(state.discovered) == 3
        assert state.counts["discovered"] == 3

        # 2. Preprocess
        preprocess_stage(state)
        assert len(state.preprocessed) == 3
        assert state.preprocessed[0].title == "Agent Paper"  # whitespace stripped

        # 3. Embed
        embed_stage(state, in_memory_session)
        assert len(state.embedded) == 3
        for c in state.embedded:
            assert c.embedding is not None

        # 4. Score
        score_stage(state)
        assert len(state.scored) == 3
        assert state.scored[0].score == pytest.approx(1.0, abs=1e-4)
        assert state.scored[1].score == pytest.approx(0.35, abs=0.05)
        assert state.scored[2].score == pytest.approx(-1.0, abs=1e-4)

        # 5. Route
        route_stage(state)
        assert len(state.routed) == 3
        assert state.routed[0].route == Route.HIGH
        assert state.routed[0].status == DecisionStatus.ROUTED_HIGH

        assert state.routed[1].route == Route.BORDERLINE
        assert state.routed[1].status == DecisionStatus.ROUTED_BORDERLINE

        assert state.routed[2].route == Route.LOW
        assert state.routed[2].status == DecisionStatus.DISCARDED

        assert state.counts["high"] == 1
        assert state.counts["borderline"] == 1
        assert state.counts["low"] == 1


# =========================================================================
# 5. Full Monitoring Cycle & Duplicate Prevention
# =========================================================================

class TestMonitoringCycleExecution:
    def test_run_monitoring_cycle_persists_decisions_including_low(self, in_memory_session):
        rng = np.random.default_rng(77)
        ref_embs = rng.standard_normal((50, 384)).astype(np.float32)
        ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)

        user = create_user(
            user_id="cycle_user",
            name="Cycle Tester",
            seed_text="Autonomous agents and planning systems.",
            session=in_memory_session,
            reference_embeddings=ref_embs,
        )

        v = in_memory_session.query(InterestVectorVersion).filter_by(user_id=user.id).first()
        t_high = v.t_high
        t_low = v.t_low

        # Create paper records in DB
        v_active = np.array(v.vector, dtype=np.float32)
        p1 = Paper(
            id="paper_h",
            title="High paper",
            abstract="Active research",
            submitted_date="2024-01-01",
            categories=["cs.AI"],
            embedding=v_active.tolist(),
            embedding_model="test",
            is_reference=False,
        )
        p2 = Paper(
            id="paper_l",
            title="Low paper",
            abstract="Distant research",
            submitted_date="2024-01-01",
            categories=["cs.AI"],
            embedding=(-v_active).tolist(),
            embedding_model="test",
            is_reference=False,
        )
        in_memory_session.add_all([p1, p2])
        in_memory_session.commit()

        # Run cycle
        state = run_monitoring_cycle(
            user_id=user.id,
            session=in_memory_session,
            mode="replay",
            as_of_date="2024-01-01",
        )

        assert state.counts["total_routed"] == 2
        assert state.counts["high"] == 1
        assert state.counts["low"] == 1

        # Verify decisions in DB
        decisions = in_memory_session.query(RoutingDecision).filter_by(user_id=user.id).all()
        assert len(decisions) == 2

        d_map = {d.paper_id: d for d in decisions}
        assert d_map["paper_h"].route == "HIGH"
        assert d_map["paper_h"].status == "ROUTED_HIGH"
        assert d_map["paper_l"].route == "LOW"
        assert d_map["paper_l"].status == "DISCARDED"

        # Check cycle record in DB
        cycle = in_memory_session.query(Cycle).filter_by(id=state.cycle_id).first()
        assert cycle is not None
        assert cycle.finished_at is not None
        assert cycle.counts["high"] == 1
        assert cycle.counts["low"] == 1

    def test_rerun_cycle_creates_no_duplicate_decisions(self, in_memory_session):
        """Verifies requirement: Re-running a cycle creates no duplicate decisions."""
        rng = np.random.default_rng(88)
        ref_embs = rng.standard_normal((50, 384)).astype(np.float32)
        ref_embs = ref_embs / np.linalg.norm(ref_embs, axis=1, keepdims=True)

        user = create_user(
            user_id="idempotent_user",
            name="Idempotent Tester",
            seed_text="AI agents and reasoning.",
            session=in_memory_session,
            reference_embeddings=ref_embs,
        )
        v = in_memory_session.query(InterestVectorVersion).filter_by(user_id=user.id).first()
        v_vec = np.array(v.vector, dtype=np.float32)

        paper = Paper(
            id="idemp_p1",
            title="Single Paper",
            abstract="Testing idempotency",
            submitted_date="2024-01-01",
            categories=["cs.AI"],
            embedding=v_vec.tolist(),
            embedding_model="test",
            is_reference=False,
        )
        in_memory_session.add(paper)
        in_memory_session.commit()

        # First cycle execution: routes 1 paper
        state1 = run_monitoring_cycle(
            user_id=user.id,
            session=in_memory_session,
            mode="replay",
        )
        assert state1.counts["total_routed"] == 1
        decisions1 = in_memory_session.query(RoutingDecision).filter_by(user_id=user.id).all()
        assert len(decisions1) == 1

        # Second cycle execution: paper is already routed, discovers 0 new candidates
        state2 = run_monitoring_cycle(
            user_id=user.id,
            session=in_memory_session,
            mode="replay",
        )
        assert state2.counts["discovered"] == 0
        assert state2.counts["total_routed"] == 0

        # Total decisions in DB remains exactly 1
        decisions2 = in_memory_session.query(RoutingDecision).filter_by(user_id=user.id).all()
        assert len(decisions2) == 1

        # Direct DB duplicate insertion attempt must fail via UniqueConstraint
        duplicate_decision = RoutingDecision(
            paper_id="idemp_p1",
            user_id=user.id,
            cycle_id=state2.cycle_id,
            cosine_score=0.9,
            vector_version_id=v.id,
            thresholds_used={},
            route="HIGH",
            status="ROUTED_HIGH",
            reason="Duplicate attempt",
            created_at=datetime.now(timezone.utc),
        )
        in_memory_session.add(duplicate_decision)
        with pytest.raises(IntegrityError):
            in_memory_session.commit()
        in_memory_session.rollback()
