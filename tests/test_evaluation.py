"""
test_evaluation.py
~~~~~~~~~~~~~~~~~~
Unit and integration tests for Phase 4:
  - Label validation and normalization
  - Reference percentiles and strata allocation
  - Stratified sampling and weight reproduction (sum(n_h * w_h) == N_frame)
  - Dev/test 70/30 stratified splitting and reproducibility
  - Baselines B0, B1, B2 score calculation
  - Weighted metrics: precision, recall, F1, PR-AUC, LOW miss rate, BORDERLINE composition
  - Stratified bootstrap confidence intervals
  - Threshold calibration on dev data only
  - Test-set access audit logging
  - Database ValidationLabel round-trip
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from research_monitor.database.models import Base, ValidationLabel
from research_monitor.evaluation.access import (
    get_test_access_history,
    log_test_access,
)
from research_monitor.evaluation.baselines import (
    score_embedding_b2,
    score_random_b0,
    score_tfidf_b1,
)
from research_monitor.evaluation.calibration import (
    CalibrationResult,
    calibrate_thresholds,
    get_provisional_calibration,
    load_calibration_result,
    save_calibration_result,
    sensitivity_analysis,
)
from research_monitor.evaluation.labeling import (
    export_labels_to_csv,
    get_label_statistics,
    get_labeled_paper_ids,
    load_latest_labels,
    normalize_label,
    normalize_paper_id,
    record_validation_label,
)
from research_monitor.evaluation.metrics import (
    binary_map_labels,
    borderline_composition,
    low_miss_rate,
    stratified_bootstrap_ci,
    weighted_f1,
    weighted_pr_auc,
    weighted_precision,
    weighted_recall,
)
from research_monitor.evaluation.sampling import (
    ReferencePercentiles,
    assign_strata,
    compute_reference_percentiles,
    draw_stratified_evaluation_sample,
    split_strata_dev_test,
    verify_sampling_weights,
)


# =========================================================================
# 1. Label Validation & Normalization
# =========================================================================

class TestLabeling:
    def test_normalize_valid_labels(self):
        assert normalize_label("relevant") == "relevant"
        assert normalize_label("RELEVANT") == "relevant"
        assert normalize_label("1") == "relevant"

        assert normalize_label("somewhat") == "somewhat"
        assert normalize_label("somewhat relevant") == "somewhat"
        assert normalize_label("2") == "somewhat"

        assert normalize_label("not_relevant") == "not_relevant"
        assert normalize_label("not relevant") == "not_relevant"
        assert normalize_label("not") == "not_relevant"
        assert normalize_label("NOT") == "not_relevant"
        assert normalize_label("3") == "not_relevant"

    def test_normalize_invalid_label_raises(self):
        with pytest.raises(ValueError):
            normalize_label("maybe")
        with pytest.raises(ValueError):
            normalize_label("4")
        with pytest.raises(ValueError):
            normalize_label("")

    def test_record_validation_label_dict(self):
        rec = record_validation_label(
            paper_id="2401.00001",
            label="relevant",
            stratum="S1",
            sampling_weight=1.19,
            split="dev",
            notes="Strong agent paper",
        )
        assert rec["paper_id"] == "2401.00001"
        assert rec["label"] == "relevant"
        assert rec["stratum"] == "S1"
        assert rec["sampling_weight"] == 1.19
        assert rec["split"] == "dev"
        assert rec["notes"] == "Strong agent paper"

    def test_record_validation_label_invalid_stratum_raises(self):
        with pytest.raises(ValueError, match="Invalid stratum"):
            record_validation_label("2401.00001", "relevant", "S6", 1.0, "dev")

    def test_record_validation_label_invalid_split_raises(self):
        with pytest.raises(ValueError, match="Invalid split"):
            record_validation_label("2401.00001", "relevant", "S1", 1.0, "val")

    def test_export_and_load_csv_backups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_records = [
                {"paper_id": "p1", "label": "relevant", "stratum": "S1", "sampling_weight": 1.2, "split": "dev"},
                {"paper_id": "p2", "label": "not_relevant", "stratum": "S5", "sampling_weight": 25.0, "split": "test"},
            ]
            t_path, latest_path = export_labels_to_csv(sample_records, output_dir=tmpdir)
            assert t_path.exists()
            assert latest_path.exists()

            loaded = pd.read_csv(latest_path)
            assert len(loaded) == 2
            assert loaded.iloc[0]["paper_id"] == "p1"

    def test_get_label_statistics_shortfall(self):
        df_empty = pd.DataFrame()
        stats_empty = get_label_statistics(df_empty)
        assert stats_empty["overall_shortfall"] == 100
        assert stats_empty["test_shortfall"] == 30
        assert stats_empty["target_overall_met"] is False

        sample_df = pd.DataFrame([
            {"label": "relevant", "split": "dev"},
            {"label": "relevant", "split": "test"},
            {"label": "somewhat", "split": "dev"},
            {"label": "not_relevant", "split": "test"},
        ])
        stats = get_label_statistics(sample_df)
        assert stats["total_count"] == 4
        assert stats["relevant_count"] == 2
        assert stats["test_relevant_count"] == 1
        assert stats["overall_shortfall"] == 98
        assert stats["test_shortfall"] == 29

    def test_normalize_paper_id_preserves_and_recovers_trailing_zero(self):
        # ArXiv ID from 2015+ (YYMM.NNNNN) where trailing zero was dropped by float parsing
        assert normalize_paper_id("2411.1151") == "2411.11510"
        assert normalize_paper_id("2411.11510") == "2411.11510"
        assert normalize_paper_id("2412.10320") == "2412.10320"
        assert normalize_paper_id("2412.1032") == "2412.10320"
        # Non-arXiv or normal string
        assert normalize_paper_id("test-paper-1") == "test-paper-1"
        # Matches valid_pids if provided
        assert normalize_paper_id("2411.1151", valid_pids={"2411.11510"}) == "2411.11510"

    def test_queue_skips_labeled_paper_and_presents_unlabeled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Existing label for p1
            existing = [
                {"user_id": "developer", "paper_id": "2411.11510", "label": "relevant", "stratum": "S1", "sampling_weight": 1.0, "split": "dev"}
            ]
            export_labels_to_csv(existing, output_dir=tmpdir)

            # Sample has 3 papers: p1 (labeled), p2 (unlabeled), p3 (unlabeled)
            sample_df = pd.DataFrame([
                {"paper_id": "2411.11510", "title": "Paper 1", "stratum": "S1", "split": "dev"},
                {"paper_id": "2412.02316", "title": "Paper 2", "stratum": "S1", "split": "dev"},
                {"paper_id": "2412.10320", "title": "Paper 3", "stratum": "S2", "split": "test"},
            ])

            labeled_pids = get_labeled_paper_ids(session=None, user_id="developer", output_dir=tmpdir)
            assert "2411.11510" in labeled_pids

            sample_df["_norm_pid"] = sample_df["paper_id"].apply(normalize_paper_id)
            unlabeled_df = sample_df[~sample_df["_norm_pid"].isin(labeled_pids)].copy()

            # p1 is skipped, p2 and p3 are presented
            assert len(unlabeled_df) == 2
            assert "2411.11510" not in unlabeled_df["paper_id"].tolist()
            assert unlabeled_df["paper_id"].tolist() == ["2412.02316", "2412.10320"]

    def test_restarting_session_does_not_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Label p1
            rec1 = record_validation_label(
                paper_id="2411.11510",
                label="relevant",
                stratum="S1",
                sampling_weight=1.0,
                split="dev",
                user_id="developer",
            )
            export_labels_to_csv([rec1], output_dir=tmpdir)

            # Restart session: load labels
            df_restarted = load_latest_labels(output_dir=tmpdir)
            assert len(df_restarted) == 1
            assert df_restarted.iloc[0]["paper_id"] == "2411.11510"

            # Session 2: Label p2
            rec2 = record_validation_label(
                paper_id="2412.02316",
                label="not_relevant",
                stratum="S1",
                sampling_weight=1.0,
                split="dev",
                user_id="developer",
            )
            current_records = df_restarted.to_dict(orient="records")
            current_records.append(rec2)
            export_labels_to_csv(current_records, output_dir=tmpdir)

            # Restart session again: load labels
            df_restarted2 = load_latest_labels(output_dir=tmpdir)
            assert len(df_restarted2) == 2

            # Simulate saving duplicates or re-saving: verify no duplicate rows
            export_labels_to_csv(current_records + [rec1], output_dir=tmpdir)
            df_restarted3 = load_latest_labels(output_dir=tmpdir)
            assert len(df_restarted3) == 2
            stats = get_label_statistics(df_restarted3)
            assert stats["total_count"] == 2

    def test_multi_user_queue_isolation_in_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec_dev = record_validation_label(
                paper_id="2411.11510",
                label="relevant",
                stratum="S1",
                sampling_weight=1.0,
                split="dev",
                user_id="developer",
            )
            export_labels_to_csv([rec_dev], output_dir=tmpdir)

            # For developer, paper is labeled
            dev_pids = get_labeled_paper_ids(session=None, user_id="developer", output_dir=tmpdir)
            assert "2411.11510" in dev_pids

            # For reviewer_2, paper is NOT labeled
            rev2_pids = get_labeled_paper_ids(session=None, user_id="reviewer_2", output_dir=tmpdir)
            assert "2411.11510" not in rev2_pids

            # Reviewer 2 labels the same paper with their own opinion
            rec_rev2 = record_validation_label(
                paper_id="2411.11510",
                label="not_relevant",
                stratum="S1",
                sampling_weight=1.0,
                split="dev",
                user_id="reviewer_2",
            )
            export_labels_to_csv([rec_dev, rec_rev2], output_dir=tmpdir)

            latest_df = load_latest_labels(output_dir=tmpdir)
            assert len(latest_df) == 2
            assert set(latest_df["user_id"]) == {"developer", "reviewer_2"}



# =========================================================================
# 2. Reference Percentiles & Stratified Sampling
# =========================================================================

class TestSampling:
    def test_compute_reference_percentiles(self):
        ref_scores = np.linspace(0.1, 0.9, 1000)
        p = compute_reference_percentiles(ref_scores)
        assert p.p50 == pytest.approx(0.5, abs=0.01)
        assert p.p75 == pytest.approx(0.7, abs=0.01)
        assert p.p90 == pytest.approx(0.82, abs=0.01)
        assert p.p98 == pytest.approx(0.884, abs=0.01)

    def test_assign_strata(self):
        percentiles = ReferencePercentiles(p50=0.3, p75=0.5, p90=0.7, p98=0.9)
        test_scores = np.array([0.95, 0.75, 0.55, 0.35, 0.15])
        strata = assign_strata(test_scores, percentiles)
        assert list(strata) == ["S1", "S2", "S3", "S4", "S5"]

    def test_draw_stratified_evaluation_sample_and_weights(self):
        rng = np.random.default_rng(42)
        n_papers = 5000
        eval_df = pd.DataFrame({
            "paper_id": [f"paper_{i}" for i in range(n_papers)],
            "title": [f"Title {i}" for i in range(n_papers)],
        })
        scores = rng.uniform(0.0, 1.0, size=n_papers)
        ref_scores = rng.uniform(0.0, 1.0, size=5000)

        targets = {"S1": 100, "S2": 120, "S3": 100, "S4": 80, "S5": 100}
        sampled_df, strata_meta = draw_stratified_evaluation_sample(
            eval_df, scores, ref_scores, target_allocation=targets, seed=42
        )

        assert len(sampled_df) == 500
        counts = sampled_df["stratum"].value_counts().to_dict()
        for s, target_n in targets.items():
            assert counts[s] == target_n

        # Check weight reproduction
        is_valid, reproduced_pop = verify_sampling_weights(sampled_df, expected_population=5000)
        assert is_valid
        assert reproduced_pop == pytest.approx(5000.0, abs=1e-3)

    def test_sampling_reproducibility(self):
        rng = np.random.default_rng(123)
        n = 1000
        df = pd.DataFrame({"paper_id": [f"p_{i}" for i in range(n)]})
        scores = rng.uniform(0.0, 1.0, size=n)
        ref_scores = rng.uniform(0.0, 1.0, size=1000)

        sample1, _ = draw_stratified_evaluation_sample(df, scores, ref_scores, seed=42)
        sample2, _ = draw_stratified_evaluation_sample(df, scores, ref_scores, seed=42)
        sample3, _ = draw_stratified_evaluation_sample(df, scores, ref_scores, seed=99)

        assert sample1["paper_id"].tolist() == sample2["paper_id"].tolist()
        assert sample1["paper_id"].tolist() != sample3["paper_id"].tolist()

    def test_split_strata_dev_test(self):
        strata_col = (
            ["S1"] * 100 + ["S2"] * 120 + ["S3"] * 100 + ["S4"] * 80 + ["S5"] * 100
        )
        sample_df = pd.DataFrame({
            "paper_id": [f"p_{i}" for i in range(500)],
            "stratum": strata_col,
            "sampling_weight": [1.0] * 500,
        })

        split_df = split_strata_dev_test(sample_df, dev_ratio=0.70, seed=42)
        assert "split" in split_df.columns
        assert len(split_df) == 500

        dev_count = (split_df["split"] == "dev").sum()
        test_count = (split_df["split"] == "test").sum()

        assert dev_count == 350
        assert test_count == 150

        # Check each stratum ratio
        s1 = split_df[split_df["stratum"] == "S1"]
        assert (s1["split"] == "dev").sum() == 70
        assert (s1["split"] == "test").sum() == 30

        s2 = split_df[split_df["stratum"] == "S2"]
        assert (s2["split"] == "dev").sum() == 84
        assert (s2["split"] == "test").sum() == 36


# =========================================================================
# 3. Baselines B0, B1, B2
# =========================================================================

class TestBaselines:
    def test_b0_random_baseline(self):
        scores1 = score_random_b0(100, seed=42)
        scores2 = score_random_b0(100, seed=42)
        scores3 = score_random_b0(100, seed=99)

        assert len(scores1) == 100
        assert np.all((scores1 >= 0.0) & (scores1 <= 1.0))
        assert np.array_equal(scores1, scores2)
        assert not np.array_equal(scores1, scores3)

    def test_b1_tfidf_baseline(self):
        corpus = [
            "Reinforcement learning and agent reasoning with large language models",
            "Deep neural networks for image segmentation and computer vision",
            "Multi-agent communication and autonomous planning systems",
            "Quantum physics and semiconductor device manufacturing",
        ]
        eval_papers = [
            "Autonomous multi-agent reinforcement learning reasoning",
            "Quantum error correction in superconducting qubits",
        ]
        seed_text = "AI agents, autonomous systems, reinforcement learning, and planning"

        scores = score_tfidf_b1(corpus, eval_papers, seed_text)
        assert len(scores) == 2
        # The agent paper should have significantly higher score than quantum physics
        assert scores[0] > scores[1]
        assert scores[0] > 0.1

    def test_b2_embedding_cosine_baseline(self):
        # 3 papers, 384 dimensions
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(3, 384))
        # L2 normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        seed_vec = embeddings[0].copy()  # Paper 0 is identical to seed vector

        scores = score_embedding_b2(embeddings, seed_vec)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] < 1.0


# =========================================================================
# 4. Weighted Metrics & Stratified Bootstrap CI
# =========================================================================

class TestMetrics:
    def test_binary_map_labels(self):
        labels = ["relevant", "somewhat", "not_relevant", "RELEVANT"]
        primary = binary_map_labels(labels, lenient=False)
        assert list(primary) == [1, 0, 0, 1]

        lenient = binary_map_labels(labels, lenient=True)
        assert list(lenient) == [1, 1, 0, 1]

    def test_weighted_precision_recall_f1(self):
        # 4 papers, weights [1, 2, 1, 4]
        # y_true = [1, 1, 0, 0]
        # selected = [True, True, True, False] -> selected: 0, 1, 2
        y_true = np.array([1, 1, 0, 0])
        weights = np.array([1.0, 2.0, 1.0, 4.0])
        selected = np.array([True, True, True, False])

        # selected weights: 1 + 2 + 1 = 4
        # relevant selected weights: 1*1 + 2*1 + 0*1 = 3
        # precision: 3 / 4 = 0.75
        prec = weighted_precision(y_true, selected, weights)
        assert prec == pytest.approx(0.75)

        # total relevant weights: 1*1 + 2*1 = 3
        # recall: 3 / 3 = 1.0
        rec = weighted_recall(y_true, selected, weights)
        assert rec == pytest.approx(1.0)

        # f1: 2 * 0.75 * 1.0 / (0.75 + 1.0) = 1.5 / 1.75 = 0.8571
        f1 = weighted_f1(y_true, selected, weights)
        assert f1 == pytest.approx(1.5 / 1.75)

    def test_low_miss_rate(self):
        y_true = np.array([1, 1, 0, 0])
        weights = np.array([2.0, 2.0, 5.0, 5.0])
        # Suppose paper 0 is routed LOW
        low_mask = np.array([True, False, False, True])

        # Relevant in LOW: paper 0 (weight 2.0)
        # Total relevant weight: paper 0 (2.0) + paper 1 (2.0) = 4.0
        # Miss rate: 2.0 / 4.0 = 0.50
        miss = low_miss_rate(y_true, low_mask, weights)
        assert miss == pytest.approx(0.50)

    def test_borderline_composition(self):
        raw_labels = np.array(["relevant", "somewhat", "not_relevant", "not_relevant"])
        weights = np.array([2.0, 3.0, 5.0, 10.0])
        borderline_mask = np.array([True, True, True, False])

        comp = borderline_composition(raw_labels, borderline_mask, weights)
        # Total weight in borderline: 2 + 3 + 5 = 10. Total weight overall: 20
        assert comp["share"] == pytest.approx(0.50)
        assert comp["pct_relevant"] == pytest.approx(2.0 / 10.0)
        assert comp["pct_somewhat"] == pytest.approx(3.0 / 10.0)
        assert comp["pct_not_relevant"] == pytest.approx(5.0 / 10.0)

    def test_stratified_bootstrap_ci(self):
        df = pd.DataFrame({
            "stratum": ["S1"] * 50 + ["S2"] * 50,
            "y": [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40,
        })
        metric_fn = lambda d: float(d["y"].mean())

        lower, upper = stratified_bootstrap_ci(df, metric_fn, n_resamples=200, seed=42)
        assert 0.0 <= lower <= upper <= 1.0
        assert lower < 0.5 < upper


# =========================================================================
# 5. Threshold Calibration (Section 7.2 Objective)
# =========================================================================

class TestCalibration:
    def test_calibrate_thresholds_objective(self):
        # Construct synthetic dev split
        # High scoring papers are mostly relevant, low scoring papers are not_relevant
        rng = np.random.default_rng(42)
        n = 350
        scores = np.concatenate([
            rng.uniform(0.5, 0.9, size=70),   # S1
            rng.uniform(0.35, 0.5, size=84),  # S2
            rng.uniform(0.25, 0.35, size=70), # S3
            rng.uniform(0.18, 0.25, size=56), # S4
            rng.uniform(0.05, 0.18, size=70), # S5
        ])
        labels = (
            ["relevant"] * 60 + ["somewhat"] * 10 +  # S1: mostly relevant
            ["relevant"] * 25 + ["somewhat"] * 25 + ["not_relevant"] * 34 + # S2
            ["somewhat"] * 15 + ["not_relevant"] * 55 + # S3
            ["not_relevant"] * 56 + # S4
            ["not_relevant"] * 70   # S5
        )
        weights = (
            [1.2] * 70 +
            [3.1] * 84 +
            [7.2] * 70 +
            [15.4] * 56 +
            [25.6] * 70
        )

        dev_df = pd.DataFrame({
            "score": scores,
            "label": labels,
            "sampling_weight": weights,
            "stratum": (
                ["S1"] * 70 + ["S2"] * 84 + ["S3"] * 70 + ["S4"] * 56 + ["S5"] * 70
            ),
        })

        ref_scores = rng.uniform(0.05, 0.9, size=5000)

        result = calibrate_thresholds(
            dev_df=dev_df,
            ref_scores=ref_scores,
            target_precision=0.80,
            target_miss_rate=0.10,
        )

        assert isinstance(result, CalibrationResult)
        assert result.p_low < result.p_high
        assert result.t_low < result.t_high
        assert 0.0 <= result.high_precision <= 1.0
        assert 0.0 <= result.low_miss_rate <= 1.0

    def test_save_and_load_calibration_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibrated_thresholds.json"
            res = CalibrationResult(
                p_high=98.0,
                p_low=90.0,
                t_high=0.4714,
                t_low=0.3639,
                high_precision=0.85,
                high_recall=0.40,
                high_f1=0.54,
                low_miss_rate=0.04,
                borderline_share=0.08,
                borderline_composition={"share": 0.08, "pct_relevant": 0.2, "pct_somewhat": 0.3, "pct_not_relevant": 0.5},
                target_precision_met=True,
                target_miss_rate_met=True,
                dev_sample_size=350,
                notes="Test calibration",
            )
            saved_path = save_calibration_result(res, filepath=path)
            assert saved_path.exists()

            loaded = load_calibration_result(filepath=path)
            assert loaded is not None
            assert loaded.p_high == 98.0
            assert loaded.p_low == 90.0
            assert loaded.high_precision == 0.85
            assert loaded.status == "calibrated"

    def test_provisional_calibration_and_aiming_for_five_percent(self):
        ref_scores = np.linspace(0.1, 0.9, 100)
        prov = get_provisional_calibration(ref_scores)
        assert prov.status == "provisional"
        assert prov.p_high == 98.0
        assert prov.p_low == 90.0
        assert prov.data_split == "none"

        # Calibration on synthetic dev df
        df = pd.DataFrame({
            "score": [0.85, 0.75, 0.65, 0.35, 0.20],
            "label": ["relevant", "relevant", "relevant", "not", "not_relevant"],
            "sampling_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        calib = calibrate_thresholds(
            dev_df=df,
            ref_scores=ref_scores,
            target_precision=0.60,
            target_miss_rate=0.10,
        )
        assert calib.status == "calibrated"
        assert calib.data_split == "dev"
        assert calib.low_miss_rate <= 0.10


# =========================================================================
# 6. Test Set Freeze & Access Control Logging
# =========================================================================

class TestAccessLogging:
    def test_log_and_read_test_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test_access.log"

            log_test_access(
                caller="test_module",
                purpose="Baseline E1 reporting",
                sample_count=150,
                log_path=log_file,
            )

            history = get_test_access_history(log_path=log_file)
            assert len(history) == 1
            assert "CALLER=test_module" in history[0]
            assert "PURPOSE=Baseline E1 reporting" in history[0]
            assert "SAMPLES=150" in history[0]

            # Second access
            log_test_access(
                caller="second_module",
                purpose="Final audit verification",
                sample_count=150,
                log_path=log_file,
            )
            history2 = get_test_access_history(log_path=log_file)
            assert len(history2) == 2

    def test_log_test_access_fails_closed(self, monkeypatch):
        """Verify that audit logging failure raises RuntimeError and halts access (fail-closed)."""
        def broken_open(*args, **kwargs):
            raise PermissionError("Simulated disk write failure")

        monkeypatch.setattr("builtins.open", broken_open)
        with pytest.raises(RuntimeError, match="fail-closed"):
            log_test_access(
                caller="unauthorized_caller",
                purpose="Attempted unlogged access",
                sample_count=150,
            )


# =========================================================================
# 7. Database ValidationLabel Model Schema & Round-trip
# =========================================================================

class TestValidationLabelModel:
    def test_validation_label_in_memory_db(self):
        """Test ValidationLabel with an isolated SQLite database."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        lbl = ValidationLabel(
            user_id="developer",
            paper_id="2401.99999",
            label="relevant",
            stratum="S1",
            sampling_weight=1.19,
            split="dev",
            notes="Agent tool paper",
        )
        session.add(lbl)
        session.commit()

        retrieved = session.query(ValidationLabel).filter_by(paper_id="2401.99999").first()
        assert retrieved is not None
        assert retrieved.user_id == "developer"
        assert retrieved.label == "relevant"
        assert retrieved.stratum == "S1"
        assert retrieved.sampling_stratum == "S1"
        assert retrieved.sampling_weight == pytest.approx(1.19)
        assert retrieved.split == "dev"
        assert retrieved.notes == "Agent tool paper"
        assert isinstance(retrieved.created_at, datetime)
        assert isinstance(retrieved.labelled_at, datetime)

        session.close()

    def test_validation_label_unique_constraint(self):
        """Verify that duplicate (user_id, paper_id) violates database unique constraint."""
        from sqlalchemy.exc import IntegrityError

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        lbl1 = ValidationLabel(
            user_id="developer",
            paper_id="2401.00001",
            label="relevant",
            sampling_stratum="S1",
            sampling_weight=1.19,
            split="dev",
        )
        session.add(lbl1)
        session.commit()

        # Duplicate insertion for same user + paper
        lbl2 = ValidationLabel(
            user_id="developer",
            paper_id="2401.00001",
            label="not",
            sampling_stratum="S2",
            sampling_weight=3.20,
            split="dev",
        )
        session.add(lbl2)
        with pytest.raises(IntegrityError):
            session.commit()

        session.rollback()
        session.close()

    def test_record_validation_label_session_persistence(self):
        """Verify record_validation_label saves and updates in database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # 1. Insert initial label
        rec1 = record_validation_label(
            paper_id="2401.12345",
            label="relevant",
            stratum="S3",
            sampling_weight=2.0,
            split="dev",
            notes="Initial label",
            session=session,
            user_id="researcher_1",
        )
        assert rec1["label"] == "relevant"

        retrieved = session.query(ValidationLabel).filter_by(paper_id="2401.12345", user_id="researcher_1").first()
        assert retrieved is not None
        assert retrieved.label == "relevant"
        assert retrieved.sampling_stratum == "S3"
        assert retrieved.stratum == "S3"
        assert retrieved.notes == "Initial label"

        # 2. Update existing label for same user + paper
        rec2 = record_validation_label(
            paper_id="2401.12345",
            label="somewhat",
            stratum="S3",
            sampling_weight=2.0,
            split="dev",
            notes="Updated label",
            session=session,
            user_id="researcher_1",
        )
        assert rec2["label"] == "somewhat"

        all_matches = session.query(ValidationLabel).filter_by(paper_id="2401.12345", user_id="researcher_1").all()
        assert len(all_matches) == 1
        assert all_matches[0].label == "somewhat"
        assert all_matches[0].notes == "Updated label"

        session.close()
