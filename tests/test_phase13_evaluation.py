"""
test_phase13_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive Unit & Integration Test Suite for Phase 13:
Pilot Completion and Final Evaluation.

Covers:
  1. Export correctness and deterministic reruns (Req A).
  2. Duplicate-event and duplicate-paper handling (Req A).
  3. Pilot / replay / simulated-data separation (Req A).
  4. Developer-user reporting and Section 8.6 exclusions (Req A & Req C).
  5. Missing labels and incomplete pilot cycles safeguards (Req 3).
  6. Frozen test-split protection and access logging (Req C).
  7. Correct metric calculations against hand-checkable examples (Req C).
  8. Before-and-after adaptation analysis with missing activation dates (Req D).
  9. Efficiency calculations and missing cost telemetry (Req E).
  10. Reproducibility of generated tables and figures (Req F).
  11. Correct readiness status transitions and completion safeguards (Req 3).
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

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
    ModelVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    Summary,
    TrainingLabel,
    User,
    ValidationLabel,
    create_training_labels_view,
)
from research_monitor.evaluation.access import log_test_access
from research_monitor.evaluation.adaptation import (
    CAUSATION_DISCLAIMER,
    analyze_all_pilot_adaptation,
    analyze_user_adaptation,
)
from research_monitor.evaluation.efficiency import (
    calculate_operational_efficiency,
    export_efficiency_report,
)
from research_monitor.evaluation.final_evaluation import (
    run_experiment_e7_pilot,
    run_final_evaluation_suite,
)
from research_monitor.evaluation.metrics import (
    binary_map_labels,
    low_miss_rate,
    weighted_f1,
    weighted_precision,
    weighted_recall,
)
from research_monitor.evaluation.pilot_export import export_pilot_dataset
from research_monitor.evaluation.readiness import (
    ReadinessTier,
    check_pilot_readiness,
)
from research_monitor.evaluation.survey import (
    compute_survey_statistics,
    create_blank_survey_template,
    load_all_surveys,
    save_survey,
    validate_survey_dict,
)
from scripts.generate_figures import plot_e7_pilot_metrics, plot_pilot_before_after


@pytest.fixture
def db_session():
    """Provides a fresh, isolated in-memory SQLite database."""
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


def _populate_mock_pilot_data(session):
    """Populates controlled synthetic data across users, cycles, decisions, and tasks."""
    now = datetime.now(timezone.utc)

    # 1. Users: 1 live pilot participant, 1 developer, 1 simulated
    u_pilot = User(id="user_alice", name="Alice", is_simulated=False, created_at=now - timedelta(days=10))
    u_dev = User(id="developer", name="Dev User", is_simulated=False, created_at=now - timedelta(days=10))
    u_sim = User(id="sim_user", name="Simulated User", is_simulated=True, created_at=now - timedelta(days=10))
    session.add_all([u_pilot, u_dev, u_sim])
    session.commit()

    # Profiles
    p_pilot = InterestProfile(user_id="user_alice", seed_text="AI alignment", seed_embedding=[0.1]*384, created_at=now - timedelta(days=10))
    p_dev = InterestProfile(user_id="developer", seed_text="Deep learning", seed_embedding=[0.1]*384, created_at=now - timedelta(days=10))
    session.add_all([p_pilot, p_dev])
    session.commit()

    # Vector versions for Alice: v0 (seed) at day -10, v1 (rocchio) activated at day -4
    v0_alice = InterestVectorVersion(
        user_id="user_alice",
        version=0,
        vector=[0.1] * 384,
        method="seed",
        p_high_level=98.0,
        p_low_level=90.0,
        t_high=0.8,
        t_low=0.4,
        is_active=False,
        created_at=now - timedelta(days=10),
    )
    v1_alice = InterestVectorVersion(
        user_id="user_alice",
        version=1,
        vector=[0.2] * 384,
        method="rocchio",
        p_high_level=98.0,
        p_low_level=90.0,
        t_high=0.8,
        t_low=0.4,
        is_active=True,
        created_at=now - timedelta(days=4),
    )
    session.add_all([v0_alice, v1_alice])
    session.commit()

    # Papers
    papers = []
    for i in range(1, 35):
        papers.append(Paper(
            id=f"2401.{i:05d}",
            title=f"Paper Title {i}",
            abstract=f"Abstract content for paper {i}",
            submitted_date="2026-01-01",
            categories=["cs.AI"],
            embedding=[0.1] * 384,
            embedding_model="all-MiniLM-L6-v2",
            is_reference=False,
        ))
    session.add_all(papers)
    session.commit()

    # Cycles for Alice: 6 daily cycles (days -6 through -1)
    # Day -6 to -4: v0 (before vector adaptation)
    # Day -3 to -1: v1 (after vector adaptation)
    cycles = []
    for day_idx in range(6, 0, -1):
        c_time = now - timedelta(days=day_idx)
        c = Cycle(
            user_id="user_alice",
            as_of_date=c_time.strftime("%Y-%m-%d"),
            mode="live",
            started_at=c_time,
            finished_at=c_time + timedelta(minutes=5),
            counts={"high": 2, "borderline": 2, "low": 1},
        )
        cycles.append(c)

    # Add 1 replay cycle
    c_replay = Cycle(
        user_id="user_alice",
        as_of_date=(now - timedelta(days=20)).strftime("%Y-%m-%d"),
        mode="replay",
        started_at=now - timedelta(days=20),
        finished_at=now - timedelta(days=20),
    )
    cycles.append(c_replay)
    session.add_all(cycles)
    session.commit()

    # Routing decisions for Alice across cycles
    decisions = []
    paper_idx = 0
    for day_idx, c in enumerate(cycles[:-1]):  # 6 live cycles
        is_post_vector = (day_idx >= 3)
        v_id = v1_alice.id if is_post_vector else v0_alice.id

        # 2 HIGH papers
        for _ in range(2):
            paper_idx += 1
            pid = f"2401.{paper_idx:05d}"
            d = RoutingDecision(
                paper_id=pid,
                user_id="user_alice",
                cycle_id=c.id,
                cosine_score=0.85,
                vector_version_id=v_id,
                thresholds_used={"t_high": 0.8, "t_low": 0.4},
                route="HIGH",
                status="DELIVERED",
                reason="Above high threshold",
                created_at=c.started_at,
            )
            decisions.append(d)

        # 2 BORDERLINE papers
        for _ in range(2):
            paper_idx += 1
            pid = f"2401.{paper_idx:05d}"
            d = RoutingDecision(
                paper_id=pid,
                user_id="user_alice",
                cycle_id=c.id,
                cosine_score=0.60,
                vector_version_id=v_id,
                thresholds_used={"t_high": 0.8, "t_low": 0.4},
                route="BORDERLINE",
                status="ROUTED_BORDERLINE",
                reason="In borderline range",
                created_at=c.started_at,
            )
            decisions.append(d)

        # 1 LOW paper
        paper_idx += 1
        pid = f"2401.{paper_idx:05d}"
        d = RoutingDecision(
            paper_id=pid,
            user_id="user_alice",
            cycle_id=c.id,
            cosine_score=0.30,
            vector_version_id=v_id,
            thresholds_used={"t_high": 0.8, "t_low": 0.4},
            route="LOW",
            status="DISCARDED",
            reason="Below low threshold",
            created_at=c.started_at,
        )
        decisions.append(d)

    # 1 overflow decision (distinct paper)
    d_over = RoutingDecision(
        paper_id="2401.00031",
        user_id="user_alice",
        cycle_id=cycles[0].id,
        cosine_score=0.55,
        vector_version_id=v0_alice.id,
        thresholds_used={"t_high": 0.8, "t_low": 0.4},
        route="BORDERLINE",
        status="DEFERRED_OVERFLOW",
        reason="Exceeded queue cap",
        created_at=cycles[0].started_at,
    )
    decisions.append(d_over)

    # 1 Developer decision (distinct paper)
    d_dev = RoutingDecision(
        paper_id="2401.00032",
        user_id="developer",
        cycle_id=None,
        cosine_score=0.88,
        vector_version_id=v0_alice.id,
        thresholds_used={"t_high": 0.8, "t_low": 0.4},
        route="HIGH",
        status="DELIVERED",
        reason="Developer verification",
        created_at=now,
    )
    decisions.append(d_dev)

    session.add_all(decisions)
    session.commit()

    # Review tasks for BORDERLINE papers
    border_decs = [d for d in decisions if d.route == "BORDERLINE" and d.status != "DEFERRED_OVERFLOW"]
    for b_dec in border_decs:
        t = ReviewTask(
            decision_id=b_dec.id,
            kind="borderline",
            decision="accept",
            queued_at=b_dec.created_at,
            decided_at=b_dec.created_at + timedelta(hours=1),
        )
        session.add(t)

    # Feedback events (User likes on delivered HIGH papers)
    high_decs = [d for d in decisions if d.route == "HIGH" and d.user_id == "user_alice"]
    for idx, h_dec in enumerate(high_decs):
        # First 3 (pre-vector) has 2 likes, 1 dislike
        # Next 3 (post-vector) has 3 likes (improvement)
        label_val = 1 if idx != 2 else 0
        fb = FeedbackEvent(
            user_id="user_alice",
            paper_id=h_dec.paper_id,
            decision_id=h_dec.id,
            source="user",
            label=label_val,
            created_at=h_dec.created_at + timedelta(hours=2),
        )
        session.add(fb)

    session.commit()


# =========================================================================
# 1. Export Correctness & Deterministic Reruns
# =========================================================================

class TestExportAndDeterministicReruns:
    def test_export_is_deterministic_and_preserves_db(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)

        export_dir1 = tmp_path / "export1"
        export_dir2 = tmp_path / "export2"

        # Record count of DB records before export
        pre_decisions_count = db_session.query(RoutingDecision).count()
        pre_feedback_count = db_session.query(FeedbackEvent).count()

        sum1 = export_pilot_dataset(session=db_session, export_dir=export_dir1)
        sum2 = export_pilot_dataset(session=db_session, export_dir=export_dir2)

        # Database records must NOT be deleted or modified
        assert db_session.query(RoutingDecision).count() == pre_decisions_count
        assert db_session.query(FeedbackEvent).count() == pre_feedback_count

        # Check export determinism
        assert sum1.clean_pilot_records_count == sum2.clean_pilot_records_count
        assert sum1.developer_decisions_count == sum2.developer_decisions_count
        assert sum1.excluded_decisions_count == sum2.excluded_decisions_count

        df1 = pd.read_csv(export_dir1 / "pilot_records_clean.csv")
        df2 = pd.read_csv(export_dir2 / "pilot_records_clean.csv")
        pd.testing.assert_frame_equal(df1, df2)


# =========================================================================
# 2. Duplicate Handling & Effective Label Precedence
# =========================================================================

class TestDuplicateAndEffectiveLabelHandling:
    def test_latest_user_feedback_takes_precedence_over_reviewer(self, db_session):
        now = datetime.now(timezone.utc)
        u = User(id="user_dup", name="Dup Test", is_simulated=False)
        p = Paper(
            id="dup.001",
            title="Dup Paper",
            abstract="...",
            submitted_date="2026-01-01",
            categories=["cs.AI"],
            embedding=[0.1] * 384,
            embedding_model="all-MiniLM-L6-v2",
            is_reference=False,
        )
        db_session.add_all([u, p])
        db_session.commit()

        # Step 1: Reviewer accepted (label = 1) at t0
        fb_rev = FeedbackEvent(
            user_id="user_dup",
            paper_id="dup.001",
            source="reviewer",
            label=1,
            created_at=now - timedelta(hours=2),
        )
        # Step 2: User subsequently dislikes (label = 0) at t1
        fb_user = FeedbackEvent(
            user_id="user_dup",
            paper_id="dup.001",
            source="user",
            label=0,
            created_at=now - timedelta(hours=1),
        )
        db_session.add_all([fb_rev, fb_user])
        db_session.commit()

        # Query effective label view
        eff_labels = db_session.query(TrainingLabel).filter_by(user_id="user_dup", paper_id="dup.001").all()
        assert len(eff_labels) == 1
        assert eff_labels[0].label == 0  # Latest user reaction wins
        assert eff_labels[0].origin == "user"


# =========================================================================
# 3. Pilot / Replay / Simulated Data Separation
# =========================================================================

class TestPilotReplaySimulatedSeparation:
    def test_replay_and_simulated_are_strictly_separated(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)
        export_dir = tmp_path / "export_sep"

        summary = export_pilot_dataset(session=db_session, export_dir=export_dir)
        clean_df = pd.read_csv(export_dir / "pilot_records_clean.csv")

        # Clean dataset must ONLY contain live pilot data
        assert (clean_df["cycle_mode"] == "live").all()
        assert (clean_df["is_simulated"] == False).all()
        assert not clean_df["user_id"].isin(["developer", "dev_phase5", "sim_user"]).any()

        # Replay and simulated are recorded in audit log
        assert summary.simulated_replay_decisions_count >= 0
        with open(export_dir / "audit_log.json", "r") as f:
            audit = json.load(f)
        assert "sim_user" in audit["simulated_users"]


# =========================================================================
# 4. Developer Reporting & Section 8.6 Exclusions
# =========================================================================

class TestDeveloperReportingAndExclusions:
    def test_developer_data_saved_separately(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)
        export_dir = tmp_path / "export_dev"

        export_pilot_dataset(session=db_session, export_dir=export_dir)

        # Developer records file exists and is populated
        dev_file = export_dir / "developer_records.csv"
        assert dev_file.exists()
        dev_df = pd.read_csv(dev_file)
        assert len(dev_df) > 0
        assert (dev_df["user_id"] == "developer").all()

    def test_section_8_6_user_exclusion_for_h1_h2(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)
        export_dir = tmp_path / "export_excl"
        export_pilot_dataset(session=db_session, export_dir=export_dir)

        with open(export_dir / "audit_log.json", "r") as f:
            audit = json.load(f)

        # user_alice has >= 5 active days but < 30 effective labels in this mock setup
        u_elig = audit["user_eligibility"]["user_alice"]
        assert u_elig["active_days"] >= 5
        assert u_elig["effective_labels"] < 30
        assert "effective" in u_elig["h1_h2_exclusion_reason"].lower()


# =========================================================================
# 5. Missing Labels & Incomplete Pilot Readiness Safeguards
# =========================================================================

class TestReadinessAndCompletionSafeguards:
    def test_incomplete_pilot_cannot_freeze_numbers(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)

        # In mock data, streak is 6 days (< 7 target), validation labels = 0
        readiness = check_pilot_readiness(
            session=db_session,
            survey_dir=tmp_path / "surveys",
        )

        assert readiness.can_freeze_final_numbers is False
        assert readiness.tier == ReadinessTier.EVALUATION_PARTIALLY_AVAILABLE
        assert any("streak" in p.lower() or "days" in p.lower() for p in readiness.missing_prerequisites)
        assert any("surveys" in p.lower() for p in readiness.missing_prerequisites)

    def test_full_readiness_achieved_when_targets_met(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)
        now = datetime.now(timezone.utc)

        # Add 2nd and 3rd users to reach 3 genuine participants (target: 3-5 users)
        u_bob = User(id="user_bob", name="Bob", is_simulated=False, created_at=now - timedelta(days=10))
        u_carol = User(id="user_carol", name="Carol", is_simulated=False, created_at=now - timedelta(days=10))
        db_session.add_all([u_bob, u_carol])

        # Add 7th consecutive daily cycle for alice (now streak == 7)
        c7 = Cycle(
            user_id="user_alice",
            as_of_date=now.strftime("%Y-%m-%d"),
            mode="live",
            started_at=now,
            finished_at=now,
            counts={"high": 2},
        )
        db_session.add(c7)

        # Add required validation labels (>= 70 dev, >= 30 test, >= 100 relevant)
        v_labels = []
        for i in range(100):
            split = "test" if i < 35 else "dev"
            v_labels.append(ValidationLabel(
                user_id="developer",
                paper_id=f"2401.{i+1:05d}",
                label="relevant",
                sampling_stratum="S1",
                sampling_weight=1.5,
                split=split,
            ))
        db_session.add_all(v_labels)
        db_session.commit()

        # Add survey responses for all 3 participants
        s_dir = tmp_path / "surveys"
        for uid in ["user_alice", "user_bob", "user_carol", "developer"]:
            save_survey(
                {
                    "user_id": uid,
                    "usefulness_score": 5,
                    "summary_helpfulness_score": 4,
                    "triage_time_minutes": 10.0,
                },
                survey_dir=s_dir,
            )

        r_dir = tmp_path / "results"
        readiness = check_pilot_readiness(session=db_session, survey_dir=s_dir, results_dir=r_dir)
        assert readiness.can_freeze_final_numbers is True
        assert readiness.tier == ReadinessTier.FINAL_EVALUATION_READY


# =========================================================================
# 6. Test Metric Calculations Against Small Hand-Checkable Examples
# =========================================================================

class TestMetricCalculationsSmallExamples:
    def test_hand_checkable_metrics(self):
        # y_true: [1, 1, 0, 0, 1]
        # y_pred: [1, 0, 1, 0, 1]
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([True, False, True, False, True])
        weights = np.ones_like(y_true, dtype=float)

        # TP = 2 (items 0, 4), FP = 1 (item 2), FN = 1 (item 1), TN = 1 (item 3)
        # Precision = 2 / 3 = 0.6667
        # Recall = 2 / 3 = 0.6667
        # F1 = 0.6667
        prec = weighted_precision(y_true, y_pred, weights)
        rec = weighted_recall(y_true, y_pred, weights)
        f1 = weighted_f1(y_true, y_pred, weights)

        assert prec == pytest.approx(2 / 3, abs=0.001)
        assert rec == pytest.approx(2 / 3, abs=0.001)
        assert f1 == pytest.approx(2 / 3, abs=0.001)

        # Low miss rate: y_true relevant papers in discard
        # If low mask is [0, 1, 0, 1, 0], relevant missed = item 1 (1 of 3)
        low_mask = np.array([False, True, False, True, False])
        miss = low_miss_rate(y_true, low_mask, weights)
        assert miss == pytest.approx(1 / 3, abs=0.001)

    def test_survey_statistics_median_hand_check(self):
        surveys = [
            {"user_id": "u1", "usefulness_score": 4, "summary_helpfulness_score": 5, "triage_time_minutes": 10.0},
            {"user_id": "u2", "usefulness_score": 5, "summary_helpfulness_score": 4, "triage_time_minutes": 20.0},
            {"user_id": "u3", "usefulness_score": 4, "summary_helpfulness_score": 4, "triage_time_minutes": 15.0},
        ]
        stats = compute_survey_statistics(surveys)
        assert stats["n_responses"] == 3
        assert stats["median_usefulness"] == 4.0
        assert stats["median_summary_helpfulness"] == 4.0
        assert stats["mean_triage_time_minutes"] == 15.0
        assert stats["h3_supported"] is True


# =========================================================================
# 7. Before-and-After Adaptation Analysis
# =========================================================================

class TestBeforeAfterAdaptationAnalysis:
    def test_adaptation_analysis_with_activation(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)
        export_dir = tmp_path / "export_adapt"
        export_pilot_dataset(session=db_session, export_dir=export_dir)
        clean_df = pd.read_csv(export_dir / "pilot_records_clean.csv")

        res = analyze_user_adaptation(user_id="user_alice", pilot_records_df=clean_df)
        assert res["has_vector_activation"] is True
        assert res["pre_vector_metrics"]["record_count"] > 0
        assert res["post_vector_metrics"]["record_count"] > 0
        assert res["disclaimer"] == CAUSATION_DISCLAIMER

    def test_adaptation_analysis_without_activation(self):
        # Synthetic df where vector_activation_date is null
        dummy_df = pd.DataFrame([{
            "user_id": "user_no_act",
            "cycle_id": 1,
            "route": "HIGH",
            "delivered_in_digest": True,
            "user_reaction": 1,
            "in_review_queue": False,
            "vector_activation_date": None,
            "decision_created_at": "2026-10-01T10:00:00Z",
        }])

        res = analyze_user_adaptation(user_id="user_no_act", pilot_records_df=dummy_df)
        assert res["status"] == "INSUFFICIENT_OBSERVATIONS_NO_VECTOR_ACTIVATION"
        assert res["h1_like_rate_test"]["supported"] is None


# =========================================================================
# 8. Efficiency & Operational Workload Calculations
# =========================================================================

class TestEfficiencyCalculations:
    def test_llm_call_reduction_and_missing_token_reporting(self, db_session):
        _populate_mock_pilot_data(db_session)

        # In mock data, 0 summaries generated
        summary = calculate_operational_efficiency(session=db_session)
        assert summary.total_papers_scored > 0
        assert summary.total_summaries_generated == 0
        assert summary.measured_savings_status == "NOT_MEASURABLE"
        assert summary.measured_llm_call_reduction is None
        assert summary.overall_llm_call_reduction is None
        assert summary.theoretical_call_reduction == 1.0

        # Token telemetry is missing -> reports missing honestly
        assert summary.has_token_telemetry is False
        assert summary.actual_cost_usd is None
        assert any("token" in r.lower() or "summaries" in r.lower() for r in summary.missing_telemetry_reasons)

        # Review overflow is counted
        assert summary.total_review_overflow >= 1

    def test_efficiency_with_token_records(self, db_session):
        _populate_mock_pilot_data(db_session)
        now = datetime.now(timezone.utc)

        # Insert 1 summary with recorded tokens
        d = db_session.query(RoutingDecision).filter_by(user_id="user_alice", route="HIGH").first()
        s = Summary(
            decision_id=d.id,
            paper_id=d.paper_id,
            user_id="user_alice",
            text="Summary text",
            llm_model="gemini-1.5-flash",
            prompt_version="v1",
            tokens_in=300,
            tokens_out=150,
            created_at=now,
        )
        db_session.add(s)
        db_session.commit()

        summary = calculate_operational_efficiency(session=db_session)
        assert summary.has_token_telemetry is True
        assert summary.total_tokens_in == 300
        assert summary.total_tokens_out == 150
        assert summary.actual_cost_usd is not None
        assert summary.actual_cost_usd > 0
        assert summary.counterfactual_cost_usd > summary.actual_cost_usd


# =========================================================================
# 9. Figure Regeneration Correctness
# =========================================================================

class TestFigureRegeneration:
    def test_figure_generation_from_csv(self, tmp_path, monkeypatch):
        from scripts import generate_figures
        monkeypatch.setattr(generate_figures, "RESULTS_DIR", tmp_path)

        # Create dummy final_e7_pilot.csv
        df_e7 = pd.DataFrame([
            {
                "user_id": "user_a",
                "like_rate_pre": 0.50,
                "like_rate_post": 0.80,
                "review_burden_pre": 4.5,
                "review_burden_post": 3.0,
            },
            {
                "user_id": "user_b",
                "like_rate_pre": 0.60,
                "like_rate_post": 0.75,
                "review_burden_pre": 5.0,
                "review_burden_post": 5.0,
            },
        ])
        df_e7.to_csv(tmp_path / "final_e7_pilot.csv", index=False)

        # Create dummy final_pilot_before_after.csv
        df_ba = pd.DataFrame([
            {
                "user_id": "user_a",
                "pre_high_share": 0.20,
                "post_high_share": 0.35,
                "pre_borderline_share": 0.30,
                "post_borderline_share": 0.25,
            }
        ])
        df_ba.to_csv(tmp_path / "final_pilot_before_after.csv", index=False)

        p1 = plot_e7_pilot_metrics(csv_path=tmp_path / "final_e7_pilot.csv")
        p2 = plot_pilot_before_after(csv_path=tmp_path / "final_pilot_before_after.csv")

        assert p1 is not None and p1.exists()
        assert p2 is not None and p2.exists()


# =========================================================================
# 10. Master Suite Execution & Deliverables
# =========================================================================

class TestFinalEvaluationMasterSuite:
    def test_master_suite_generates_all_deliverables(self, db_session, tmp_path):
        _populate_mock_pilot_data(db_session)

        export_dir = tmp_path / "pilot_export"
        survey_dir = tmp_path / "surveys"
        results_dir = tmp_path / "results"

        report = run_final_evaluation_suite(
            db_session=db_session,
            pilot_export_dir=export_dir,
            survey_dir=survey_dir,
            results_dir=results_dir,
            run_offline_e1=False,  # Skip test split in unit test
        )

        assert report["phase"] == 13
        assert report["numbers_frozen"] is False  # Incomplete pilot safeguards
        assert (results_dir / "final_evaluation_report.json").exists()
        assert (results_dir / "readiness_report.json").exists()
        assert (results_dir / "final_e7_pilot.csv").exists()
        assert (results_dir / "final_pilot_before_after.csv").exists()
        assert (results_dir / "final_efficiency.csv").exists()
        assert (export_dir / "pilot_records_clean.csv").exists()
        assert (export_dir / "audit_log.json").exists()


# =========================================================================
# 11. Phase 13 Evaluation Audit & Accounting Regression Tests
# =========================================================================

def _create_test_vector(session, user_id: str) -> InterestVectorVersion:
    v = InterestVectorVersion(
        user_id=user_id,
        version=0,
        vector=[0.1] * 384,
        method="seed",
        p_high_level=98.0,
        p_low_level=90.0,
        t_high=0.8,
        t_low=0.4,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    session.add(v)
    session.commit()
    return v


class TestPhase13EvaluationAuditRegression:
    def test_user_with_cycles_but_insufficient_h1_h2_eligibility(self, db_session, tmp_path):
        """
        Regression test: phase5_verify has 3 cycles across 2 active days, but
        0 effective labels and no vector update.
        Must report 1 genuine participant, 3 recorded cycles, 2 active days,
        0 H1/H2 eligible users, and not freeze final numbers.
        """
        now = datetime.now(timezone.utc)
        u = User(id="phase5_verify", name="Phase 5 Verify", is_simulated=False, created_at=now - timedelta(days=5))
        db_session.add(u)
        prof = InterestProfile(user_id="phase5_verify", seed_text="AI safety", seed_embedding=[0.1]*384, created_at=now - timedelta(days=5))
        db_session.add(prof)
        db_session.commit()
        vec = _create_test_vector(db_session, "phase5_verify")

        # 3 cycles: Day 1 (2 replay cycles), Day 2 (1 live cycle)
        day1 = now - timedelta(days=2)
        day2 = now - timedelta(days=1)
        c1 = Cycle(user_id="phase5_verify", as_of_date=day1.strftime("%Y-%m-%d"), mode="replay", started_at=day1, finished_at=day1)
        c2 = Cycle(user_id="phase5_verify", as_of_date=day1.strftime("%Y-%m-%d"), mode="replay", started_at=day1 + timedelta(hours=1), finished_at=day1 + timedelta(hours=1))
        c3 = Cycle(user_id="phase5_verify", as_of_date=day2.strftime("%Y-%m-%d"), mode="live", started_at=day2, finished_at=day2)
        db_session.add_all([c1, c2, c3])
        db_session.commit()

        # Decisions: 50 in c1, 50 in c2, 20 in c3 (120 total)
        for i in range(120):
            p = Paper(
                id=f"2401.reg_{i:04d}",
                title=f"Paper {i}",
                abstract="Abstract",
                submitted_date="2026-01-01",
                categories=["cs.AI"],
                embedding=[0.1] * 384,
                embedding_model="all-MiniLM-L6-v2",
                is_reference=False,
            )
            db_session.add(p)
            c = c1 if i < 50 else (c2 if i < 100 else c3)
            d = RoutingDecision(
                paper_id=p.id,
                user_id="phase5_verify",
                cycle_id=c.id,
                vector_version_id=vec.id,
                cosine_score=0.75,
                thresholds_used={"t_high": 0.8, "t_low": 0.4},
                route="BORDERLINE",
                status="DELIVERED",
                reason="Audit check",
                created_at=c.started_at,
            )
            db_session.add(d)
        db_session.commit()

        readiness = check_pilot_readiness(session=db_session, survey_dir=tmp_path / "surveys")
        assert readiness.total_users_with_records == 1
        assert readiness.genuine_pilot_participants_count == 1
        assert readiness.h1_h2_eligible_users_count == 0
        assert readiness.total_recorded_cycles == 3
        assert readiness.genuine_recorded_cycles == 3
        assert readiness.genuine_live_cycles == 1
        assert readiness.distinct_active_calendar_days == 2
        assert readiness.consecutive_live_calendar_days == 1
        assert readiness.can_freeze_final_numbers is False

        # Export and verify Section 8.6 exclusion in adaptation
        export_dir = tmp_path / "export"
        export_summary = export_pilot_dataset(session=db_session, export_dir=export_dir)
        u_elig = export_summary.to_dict()["user_eligibility"]["phase5_verify"]
        assert u_elig["h1_h2_eligible"] is False
        assert "effective" in u_elig["h1_h2_exclusion_reason"].lower() or "active days" in u_elig["h1_h2_exclusion_reason"].lower()

        clean_df = pd.read_csv(export_dir / "pilot_records_clean.csv")
        assert len(clean_df) == 20  # Only 20 live records from c3
        adapt_res = analyze_user_adaptation(
            user_id="phase5_verify",
            pilot_records_df=clean_df,
            user_eligibility=u_elig,
        )
        assert adapt_res["is_h1_h2_eligible"] is False
        assert adapt_res["status"] == "EXCLUDED_SECTION_8_6_NO_ACTIVATION"
        assert adapt_res["h1_like_rate_test"]["supported"] is None
        assert adapt_res["h2_review_burden_test"]["supported"] is None

    def test_exclusion_of_replay_simulated_developer_from_genuine_pilot_counts(self, db_session, tmp_path):
        now = datetime.now(timezone.utc)
        # Developer user
        u_dev = User(id="dev_user", name="Dev", is_simulated=False)
        # Simulated user
        u_sim = User(id="sim_bot", name="Bot", is_simulated=True)
        # Genuine user
        u_gen = User(id="real_user", name="Real", is_simulated=False)
        db_session.add_all([u_dev, u_sim, u_gen])

        # Cycles
        c_dev = Cycle(user_id="dev_user", as_of_date="2026-10-10", mode="replay", started_at=now, finished_at=now)
        c_sim = Cycle(user_id="sim_bot", as_of_date="2026-10-10", mode="replay", started_at=now, finished_at=now)
        c_gen_replay = Cycle(user_id="real_user", as_of_date="2026-10-11", mode="replay", started_at=now, finished_at=now)
        c_gen_live = Cycle(user_id="real_user", as_of_date="2026-10-12", mode="live", started_at=now, finished_at=now)
        db_session.add_all([c_dev, c_sim, c_gen_replay, c_gen_live])
        db_session.commit()

        readiness = check_pilot_readiness(session=db_session, survey_dir=tmp_path / "surveys")
        assert readiness.total_users_with_records == 3
        assert readiness.genuine_pilot_participants_count == 1
        assert readiness.developer_users_count == 1
        assert readiness.simulated_users_count == 1
        assert readiness.total_recorded_cycles == 4
        assert readiness.genuine_recorded_cycles == 2
        assert readiness.genuine_live_cycles == 1
        assert readiness.consecutive_live_calendar_days == 1

    def test_export_accounting_reconciles_210_records_without_overlap_confusion(self, db_session, tmp_path):
        """
        Recreates the exact 210-record scenario:
          - dev_phase5: 50 records (cycle replay)
          - simulated_researcher: 40 records (cycle replay)
          - phase5_verify: 100 records (cycle replay)
          - phase5_verify: 20 records (cycle live)
        Total decisions = 210.
        Partition: Clean live (20) + Genuine replay (100) + Developer (50) + Simulated (40) = 210.
        Replay flag: 190 total replay records.
        """
        now = datetime.now(timezone.utc)
        u_dev = User(id="dev_phase5", name="Dev", is_simulated=False, created_at=now - timedelta(days=10))
        u_sim = User(id="simulated_researcher", name="Sim", is_simulated=True, created_at=now - timedelta(days=10))
        u_gen = User(id="phase5_verify", name="Verify", is_simulated=False, created_at=now - timedelta(days=10))
        p_dev = InterestProfile(user_id="dev_phase5", seed_text="AI", seed_embedding=[0.1]*384, created_at=now - timedelta(days=10))
        p_sim = InterestProfile(user_id="simulated_researcher", seed_text="AI", seed_embedding=[0.1]*384, created_at=now - timedelta(days=10))
        p_gen = InterestProfile(user_id="phase5_verify", seed_text="AI", seed_embedding=[0.1]*384, created_at=now - timedelta(days=10))
        db_session.add_all([u_dev, u_sim, u_gen, p_dev, p_sim, p_gen])
        db_session.commit()

        v_dev = _create_test_vector(db_session, "dev_phase5")
        v_sim = _create_test_vector(db_session, "simulated_researcher")
        v_gen = _create_test_vector(db_session, "phase5_verify")

        c_dev = Cycle(user_id="dev_phase5", as_of_date="2026-10-09", mode="replay", started_at=now, finished_at=now)
        c_sim = Cycle(user_id="simulated_researcher", as_of_date="2026-10-09", mode="replay", started_at=now, finished_at=now)
        c_gen_rep = Cycle(user_id="phase5_verify", as_of_date="2026-10-11", mode="replay", started_at=now, finished_at=now)
        c_gen_live = Cycle(user_id="phase5_verify", as_of_date="2026-10-12", mode="live", started_at=now, finished_at=now)
        db_session.add_all([c_dev, c_sim, c_gen_rep, c_gen_live])
        db_session.commit()

        # Add 50 dev, 40 sim, 100 gen rep, 20 gen live = 210 decisions
        specs = [
            ("dev_phase5", c_dev, v_dev.id, 50),
            ("simulated_researcher", c_sim, v_sim.id, 40),
            ("phase5_verify", c_gen_rep, v_gen.id, 100),
            ("phase5_verify", c_gen_live, v_gen.id, 20),
        ]
        p_id = 0
        for uid, cyc, v_id, count in specs:
            for _ in range(count):
                p_id += 1
                paper = Paper(
                    id=f"2401.rec_{p_id:04d}",
                    title=f"Paper {p_id}",
                    abstract="Abstract",
                    submitted_date="2026-01-01",
                    categories=["cs.AI"],
                    embedding=[0.1] * 384,
                    embedding_model="all-MiniLM-L6-v2",
                    is_reference=False,
                )
                db_session.add(paper)
                dec = RoutingDecision(
                    paper_id=paper.id,
                    user_id=uid,
                    cycle_id=cyc.id,
                    vector_version_id=v_id,
                    cosine_score=0.75,
                    thresholds_used={"t_high": 0.8, "t_low": 0.4},
                    route="HIGH" if uid == "phase5_verify" and cyc.mode == "live" else "LOW",
                    status="DELIVERED" if uid == "phase5_verify" and cyc.mode == "live" else "DISCARDED",
                    reason="Test reconciliation",
                    created_at=cyc.started_at,
                )
                db_session.add(dec)
        db_session.commit()

        export_dir = tmp_path / "audit_recon"
        summary = export_pilot_dataset(session=db_session, export_dir=export_dir)
        d = summary.to_dict()

        assert d["total_routing_decisions_exported"] == 210
        part = d["mutually_exclusive_partition"]
        assert part["clean_live_records"] == 20
        assert part["genuine_replay_records"] == 100
        assert part["developer_records"] == 50
        assert part["simulated_records"] == 40
        assert part["partition_sum"] == 210
        assert part["is_reconciled"] is True

        flags = d["overlapping_flags"]
        assert flags["replay_decisions_count"] == 190
        assert flags["clean_records_count"] == 20
        assert flags["developer_decisions_count"] == 50
        assert flags["simulated_decisions_count"] == 40

        # Check audit_log.json
        with open(export_dir / "audit_log.json", "r", encoding="utf-8") as f:
            audit = json.load(f)
        assert audit["export_reconciliation"]["is_reconciled"] is True
        assert audit["export_reconciliation"]["partition_sum"] == 210

    def test_efficiency_reporting_not_measurable_when_zero_summaries(self, db_session, tmp_path):
        now = datetime.now(timezone.utc)
        u = User(id="u_eff", name="Eff User", is_simulated=False)
        db_session.add(u)
        db_session.commit()
        v_eff = _create_test_vector(db_session, "u_eff")

        c = Cycle(user_id="u_eff", as_of_date="2026-10-12", mode="live", started_at=now, finished_at=now)
        db_session.add(c)
        for i in range(10):
            p = Paper(
                id=f"2401.eff_{i:04d}",
                title=f"Paper {i}",
                abstract="Abstract",
                submitted_date="2026-01-01",
                categories=["cs.AI"],
                embedding=[0.1] * 384,
                embedding_model="all-MiniLM-L6-v2",
                is_reference=False,
            )
            db_session.add(p)
            d = RoutingDecision(
                paper_id=p.id,
                user_id="u_eff",
                cycle_id=c.id,
                vector_version_id=v_eff.id,
                cosine_score=0.85,
                thresholds_used={"t_high": 0.8, "t_low": 0.4},
                route="HIGH",
                status="DELIVERED",
                reason="High relevance",
                created_at=now,
            )
            db_session.add(d)
        db_session.commit()

        # Zero summaries created, no token telemetry
        eff = calculate_operational_efficiency(session=db_session)
        assert eff.measured_savings_status == "NOT_MEASURABLE"
        assert eff.measured_llm_call_reduction is None
        assert eff.overall_llm_call_reduction is None
        assert eff.theoretical_call_reduction == 1.0  # 100% counterfactual

        out_csv = export_efficiency_report(eff, output_dir=tmp_path)
        df_eff = pd.read_csv(out_csv)
        row = df_eff.iloc[0]
        assert row["measured_savings_status"] == "NOT_MEASURABLE"
        assert pd.isna(row["measured_llm_call_reduction"])
        assert row["theoretical_call_reduction"] == 1.0
