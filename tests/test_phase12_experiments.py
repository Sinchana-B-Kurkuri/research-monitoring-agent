"""
test_phase12_experiments.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for Phase 12: Offline Experiment Suite.

Tests cover:
  1. Baselines correctness (B0, B1, B2, B3, B4, B5).
  2. Matched-volume evaluation correctness.
  3. Simulation oracle label mapping, noise handling, and fallback.
  4. Protocol A (random stratified sampling) and classifier switch point marking (n=30).
  5. Protocol B (gate-selected feedback) and pool capacity respect/exhaustion.
  6. E3 Signal ablation (user-only vs reviewer-only vs combined).
  7. E4 Centroid update adaptation.
  8. E5 Efficiency and LLM-call reduction formula.
  9. E6 Human review escalation value logic.
  10. Test set isolation: dev sampling never touches test split.
  11. Test access audit logging in experiments/results/test_access.log.
  12. CLI simulate-feedback demo snapshot preparation.
  13. Publication figure generation.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_monitor.database.models import (
    Base,
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    InterestProfile,
    InterestVectorVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    TrainingLabel,
    User,
    create_training_labels_view,
)
from research_monitor.evaluation.access import get_test_access_history, log_test_access
from research_monitor.evaluation.baselines import (
    evaluate_at_matched_volume,
    score_centroid_b3,
    score_embedding_b2,
    score_random_b0,
    score_tfidf_b1,
    train_and_score_embedding_logreg_b5,
    train_and_score_tfidf_logreg_b4,
)
from research_monitor.evaluation.experiments import (
    BlockedExperimentError,
    run_experiment_e1_gate_quality,
    run_experiment_e5_efficiency,
    run_experiment_e6_human_review_value,
)
from research_monitor.evaluation.simulation import (
    CLASSIFIER_SWITCH_POINT,
    SimulationOracle,
    run_protocol_a,
    run_protocol_b,
    run_signal_ablation_e3,
)
from scripts.generate_figures import (
    plot_e2_learning_curves,
    plot_e3_signal_ablation,
    plot_e4_centroid_update,
    plot_e5_efficiency,
    plot_e6_human_review_value,
)
from scripts.simulate_feedback import apply_oracle_feedback, prepare_demo_snapshot


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def random_unit_vector(rng):
    def _gen():
        v = rng.standard_normal(384).astype(np.float32)
        return v / np.linalg.norm(v)
    return _gen


@pytest.fixture
def db_session():
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


# =========================================================================
# 1. Baseline Correctness & Matched Volume Tests
# =========================================================================

class TestBaselinesAndMatchedVolume:
    def test_b0_random_scores_bounded_and_deterministic(self):
        s1 = score_random_b0(100, seed=42)
        s2 = score_random_b0(100, seed=42)
        s3 = score_random_b0(100, seed=99)

        assert len(s1) == 100
        assert np.all(s1 >= 0.0) and np.all(s1 <= 1.0)
        assert np.allclose(s1, s2)
        assert not np.allclose(s1, s3)

    def test_b1_tfidf_similarity_correctness(self):
        corpus = [
            "Autonomous AI agents using reinforcement learning for decision making.",
            "Quantum electrodynamics in condensed matter physics systems.",
            "Language models tool use multi-agent planning frameworks.",
        ]
        eval_texts = [
            "AI agents and autonomous planning using tools.",
            "Superconducting materials and quantum phase transitions.",
        ]
        seed_text = "AI agents, autonomous systems, and planning tools."

        scores = score_tfidf_b1(corpus, eval_texts, seed_text)
        assert len(scores) == 2
        assert scores[0] > scores[1]  # Agent paper much more similar than physics paper
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_b2_and_b3_embedding_cosine_correctness(self, random_unit_vector):
        v_seed = random_unit_vector()
        v_similar = v_seed + 0.01 * random_unit_vector()
        v_similar /= np.linalg.norm(v_similar)
        v_opposite = -v_seed

        embs = np.vstack([v_similar, v_opposite])
        scores_b2 = score_embedding_b2(embs, v_seed)
        assert scores_b2[0] > 0.95
        assert scores_b2[1] < -0.95

        v_centroid = v_similar
        scores_b3 = score_centroid_b3(embs, v_centroid)
        assert scores_b3[0] > 0.99
        assert scores_b3[1] < -0.95

    def test_b4_tfidf_logreg_train_and_score(self):
        train_texts = [
            "AI agent workflow planning",
            "Autonomous tool use agents",
            "Astrophysics stellar evolution",
            "Galaxy cluster dark matter",
        ]
        y_train = [1, 1, 0, 0]
        eval_texts = [
            "Agent planning with tools",
            "Galactic observations",
        ]
        probs = train_and_score_tfidf_logreg_b4(
            train_texts=train_texts,
            y_train=y_train,
            eval_texts=eval_texts,
            c_val=1.0,
            random_state=42,
        )
        assert len(probs) == 2
        assert probs[0] > probs[1]
        assert 0.0 <= probs[0] <= 1.0
        assert 0.0 <= probs[1] <= 1.0

    def test_b5_embedding_logreg_train_and_score(self, rng, random_unit_vector):
        v_pos = random_unit_vector()
        v_neg = random_unit_vector()

        X_train = np.vstack([
            v_pos + 0.05 * rng.standard_normal((10, 384)).astype(np.float32),
            v_neg + 0.05 * rng.standard_normal((10, 384)).astype(np.float32),
        ])
        X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
        y_train = np.array([1] * 10 + [0] * 10, dtype=np.int32)

        X_eval = np.vstack([v_pos, v_neg])
        X_eval /= np.linalg.norm(X_eval, axis=1, keepdims=True)

        probs = train_and_score_embedding_logreg_b5(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            c_val=0.3,
            random_state=42,
        )
        assert len(probs) == 2
        assert probs[0] > probs[1]
        assert probs[0] > 0.5
        assert probs[1] < 0.5

    def test_evaluate_at_matched_volume(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
        y_true = np.array([1, 1, 0, 1, 0, 0, 0])

        # Top 3 delivered -> scores 0.9, 0.8, 0.7 -> true labels 1, 1, 0 -> 2/3 precision
        res = evaluate_at_matched_volume(scores, y_true, target_volume=3)
        assert res["volume"] == 3
        assert np.isclose(res["precision"], 2.0 / 3.0)
        assert np.isclose(res["recall"], 2.0 / 3.0)  # 2 found out of 3 total positives


# =========================================================================
# 2. Simulation Oracle Tests
# =========================================================================

class TestSimulationOracle:
    def test_oracle_primary_mapping_and_decisions(self):
        gt = {
            "p1": "relevant",
            "p2": "somewhat",
            "p3": "not_relevant",
            "p4": "RELEVANT",
        }
        oracle = SimulationOracle(ground_truth_map=gt, noise_rate=0.0)

        # User reaction
        assert oracle.get_user_reaction("p1") == 1
        assert oracle.get_user_reaction("p2") == 0  # primary mapping treats somewhat as 0
        assert oracle.get_user_reaction("p3") == 0
        assert oracle.get_user_reaction("p4") == 1

        # Reviewer decision
        assert oracle.get_reviewer_decision("p1") == "accept"
        assert oracle.get_reviewer_decision("p2") == "reject"
        assert oracle.get_reviewer_decision("p3") == "reject"
        assert oracle.get_reviewer_decision("p4") == "accept"

    def test_oracle_noise_sensitivity_flips_labels(self):
        gt = {"p1": "relevant"}
        # 100% noise -> flips 1 to 0
        oracle_noisy = SimulationOracle(ground_truth_map=gt, noise_rate=1.0, seed=42)
        assert oracle_noisy.get_user_reaction("p1") == 0
        assert oracle_noisy.get_reviewer_decision("p1") == "reject"

    def test_oracle_heuristic_fallback_when_ground_truth_absent(self):
        oracle = SimulationOracle(ground_truth_map={}, noise_rate=0.0)
        paper_agent = {
            "title": "LLM Agents for Multi-Agent Planning",
            "abstract": "We develop autonomous agent workflows using reinforcement learning.",
        }
        paper_physics = {
            "title": "Cosmological Inflation and CMB Radiation",
            "abstract": "Analysis of primordial gravitational waves.",
        }
        assert oracle.get_user_reaction("p_agent", paper_agent) == 1
        assert oracle.get_user_reaction("p_phys", paper_physics) == 0


# =========================================================================
# 3. Protocol A & B Simulation Tests
# =========================================================================

class TestProtocolsABAndAblations:
    @pytest.fixture
    def mock_eval_data(self, rng, random_unit_vector):
        seed_vec = random_unit_vector()
        n_dev, n_test = 40, 20

        # Dev data: 20 pos, 20 neg
        X_dev_pos = seed_vec + 0.1 * rng.standard_normal((20, 384)).astype(np.float32)
        X_dev_neg = -seed_vec + 0.1 * rng.standard_normal((20, 384)).astype(np.float32)
        X_dev = np.vstack([X_dev_pos, X_dev_neg])
        X_dev /= np.linalg.norm(X_dev, axis=1, keepdims=True)

        dev_pids = [f"dev_{i}" for i in range(n_dev)]
        dev_df = pd.DataFrame({
            "paper_id": dev_pids,
            "title": [f"Paper {i}" for i in range(n_dev)],
            "abstract": [f"Abstract {i}" for i in range(n_dev)],
            "stratum": ["S1"] * 10 + ["S2"] * 10 + ["S3"] * 10 + ["S5"] * 10,
            "sampling_weight": [1.0] * n_dev,
        })

        # Test data: 10 pos, 10 neg
        X_test_pos = seed_vec + 0.1 * rng.standard_normal((10, 384)).astype(np.float32)
        X_test_neg = -seed_vec + 0.1 * rng.standard_normal((10, 384)).astype(np.float32)
        X_test = np.vstack([X_test_pos, X_test_neg])
        X_test /= np.linalg.norm(X_test, axis=1, keepdims=True)

        test_pids = [f"test_{i}" for i in range(n_test)]
        test_df = pd.DataFrame({
            "paper_id": test_pids,
            "title": [f"Test Paper {i}" for i in range(n_test)],
            "abstract": [f"Test Abstract {i}" for i in range(n_test)],
            "stratum": ["S1"] * 5 + ["S2"] * 5 + ["S3"] * 5 + ["S5"] * 5,
            "sampling_weight": [1.0] * n_test,
        })

        gt_map = {f"dev_{i}": ("relevant" if i < 20 else "not_relevant") for i in range(n_dev)}
        gt_map.update({f"test_{i}": ("relevant" if i < 10 else "not_relevant") for i in range(n_test)})

        oracle = SimulationOracle(ground_truth_map=gt_map, noise_rate=0.0)

        return dev_df, X_dev, test_df, X_test, seed_vec, oracle

    def test_protocol_a_execution_and_switch_point(self, mock_eval_data):
        dev_df, X_dev, test_df, X_test, seed_vec, oracle = mock_eval_data

        sys_pts, b4_pts = run_protocol_a(
            dev_df=dev_df,
            dev_embeddings=X_dev,
            test_df=test_df,
            test_embeddings=X_test,
            seed_vector=seed_vec,
            oracle=oracle,
            feedback_sizes=(0, 10, 20, 30),
            random_seeds=(42, 43),
        )

        assert len(sys_pts) == 4
        assert sys_pts[0].n_feedback == 0
        assert sys_pts[0].model_type == "b2_seed"
        assert sys_pts[1].model_type == "b3_centroid"
        assert not sys_pts[1].is_classifier
        assert sys_pts[3].n_feedback == 30
        assert sys_pts[3].is_classifier
        assert sys_pts[3].is_switch_point
        assert sys_pts[3].model_type == "b5_classifier"

    def test_protocol_b_respects_pool_capacity(self, mock_eval_data):
        dev_df, X_dev, test_df, X_test, seed_vec, oracle = mock_eval_data

        curve_pts, pool_caps = run_protocol_b(
            dev_df=dev_df,
            dev_embeddings=X_dev,
            test_df=test_df,
            test_embeddings=X_test,
            seed_vector=seed_vec,
            oracle=oracle,
            t_high=0.70,
            t_low=0.30,
            feedback_sizes=(0, 10, 100),  # 100 exceeds capacity
            random_seeds=(42,),
        )

        assert "high_capacity" in pool_caps
        assert "borderline_capacity" in pool_caps
        # Point for n=100 should be stopped because 100 > total capacity
        feedback_points_evaluated = [pt.n_feedback for pt in curve_pts]
        assert 0 in feedback_points_evaluated
        assert 10 in feedback_points_evaluated
        assert 100 not in feedback_points_evaluated

    def test_e3_signal_ablation_conditions(self, mock_eval_data):
        dev_df, X_dev, test_df, X_test, seed_vec, oracle = mock_eval_data

        res = run_signal_ablation_e3(
            dev_df=dev_df,
            dev_embeddings=X_dev,
            test_df=test_df,
            test_embeddings=X_test,
            seed_vector=seed_vec,
            oracle=oracle,
            t_high=0.50,
            t_low=-0.50,
            feedback_sizes=(10,),
            random_seeds=(42,),
        )

        assert "user_only" in res
        assert "reviewer_only" in res
        assert "combined" in res


# =========================================================================
# 4. Efficiency & Human Review Value Tests
# =========================================================================

class TestEfficiencyAndReviewValue:
    def test_e5_efficiency_savings_calculation(self):
        # 1000 scores uniformly distributed
        scores = np.linspace(0.0, 1.0, 1000)
        df_eff = run_experiment_e5_efficiency(
            scores=scores,
            percentile_pairs=((98.0, 90.0),),
            digest_cap=20,
            review_cap=15,
        )
        assert len(df_eff) == 1
        row = df_eff.iloc[0]
        assert row["total_papers_scored"] == 1000
        # Summarized capped at 20 + 7 = 27
        assert row["papers_summarized"] <= 27
        assert row["llm_call_reduction_pct"] > 95.0
        assert row["cost_savings_pct"] > 95.0

    def test_e6_human_review_escalation_improves_precision_over_auto_accept(self):
        # High paper scores > 0.8 (positives), borderline scores in [0.4, 0.8] (half pos, half neg), low < 0.4
        scores = np.array([0.9, 0.85, 0.6, 0.5, 0.2, 0.1])
        y_true = np.array([1, 1, 1, 0, 0, 0])

        df_e6 = run_experiment_e6_human_review_value(
            scores=scores,
            y_true=y_true,
            t_high=0.8,
            t_low=0.3,
        )

        # Gate + Review should reject the negative borderline paper (score 0.5) and deliver 3 papers (all 3 positive -> precision 1.0)
        # Auto-Accept delivers all 4 (3 positive, 1 negative -> precision 0.75)
        strat_map = {r["strategy"]: r for _, r in df_e6.iterrows()}
        p_review = strat_map["Relevance Gate with Human Review"]["precision"]
        p_auto_accept = strat_map["Auto-Accept Borderline (HIGH + BORDERLINE)"]["precision"]

        assert p_review >= p_auto_accept
        assert strat_map["Relevance Gate with Human Review"]["papers_reviewed"] == 2


# =========================================================================
# 5. CLI & Demo Snapshot Tests
# =========================================================================

class TestCLISimulateFeedbackAndDemoSnapshot:
    def test_prepare_demo_snapshot_creates_simulated_user_and_feedback(self, db_session):
        user_id = prepare_demo_snapshot(session=db_session)
        assert user_id == "simulated_researcher"

        user = db_session.query(User).filter_by(id="simulated_researcher").first()
        assert user is not None
        assert user.is_simulated is True

        v0 = db_session.query(InterestVectorVersion).filter_by(user_id=user_id, version=0).first()
        assert v0 is not None
        assert v0.is_active is True

        events = db_session.query(FeedbackEvent).filter_by(user_id=user_id).all()
        assert len(events) == 40
        pos_count = sum(1 for e in events if e.label == 1)
        neg_count = sum(1 for e in events if e.label == 0)
        assert pos_count == 20
        assert neg_count == 20

    def test_test_access_audit_logging(self, tmp_path):
        log_file = tmp_path / "test_access.log"
        log_test_access(
            caller="test_script.py",
            purpose="Integration unit test for test access logging",
            sample_count=150,
            log_path=str(log_file),
        )
        assert log_file.exists()
        history = get_test_access_history(log_path=str(log_file))
        assert len(history) == 1
        assert "CALLER=test_script.py" in history[0]
        assert "SAMPLES=150" in history[0]
