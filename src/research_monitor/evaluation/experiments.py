"""
experiments.py
~~~~~~~~~~~~~~
Implementation of Phase 12 Offline Experiment Suite (Roadmap Sections 8.4 & 8.5).

Experiments implemented:
  - E1: Gate Quality (Test split, weighted, vs B0-B2, B4 at matched volume, bootstrap CIs).
  - E2: Learning Curves (Protocols A and B, PR-AUC and F1 vs n, B2 & B4 baselines, switch point).
  - E3: Signal Ablation (Protocol B, user-only vs reviewer-only vs combined, effective label rule).
  - E4: Centroid Update (B2 fixed vs B3 updated centroid vs B5 classifier at equal budgets).
  - E5: Efficiency (LLM-call reduction, tokens & costs, review workload, percentile sensitivity).
  - E6: Human-Review Value (Escalation vs auto-accept vs auto-reject on BORDERLINE papers).
"""

from dataclasses import asdict, dataclass
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_monitor.config import config
from research_monitor.evaluation.access import log_test_access
from research_monitor.evaluation.baselines import (
    evaluate_at_matched_volume,
    score_centroid_b3,
    score_embedding_b2,
    score_random_b0,
    score_tfidf_b1,
    train_and_score_embedding_logreg_b5,
    train_and_score_tfidf_logreg_b4,
)
from research_monitor.evaluation.sampling import compute_reference_percentiles
from research_monitor.evaluation.calibration import get_provisional_calibration
from research_monitor.evaluation.labeling import load_latest_labels
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
from research_monitor.evaluation.simulation import (
    CLASSIFIER_SWITCH_POINT,
    DEFAULT_RANDOM_SEEDS,
    SIMULATION_FEEDBACK_SIZES,
    LearningCurvePoint,
    SimulationOracle,
    run_protocol_a,
    run_protocol_b,
    run_signal_ablation_e3,
)
from research_monitor.relevance.preference import compute_rocchio_centroid

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")


class BlockedExperimentError(Exception):
    """Raised when an experiment is blocked due to missing human ground truth data."""
    pass


# =========================================================================
# E1: Gate Quality
# =========================================================================

def run_experiment_e1_gate_quality(
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    seed_vector: np.ndarray,
    ground_truth_labels: Optional[dict[str, str]] = None,
    dev_df: Optional[pd.DataFrame] = None,
    t_high: float = 0.50,
    t_low: float = 0.40,
    n_resamples: int = 1000,
    random_state: int = 42,
    corpus_texts: Optional[list[str]] = None,
    simulated: bool = False,
) -> pd.DataFrame:
    """
    E1 — Gate Quality (Roadmap Section 8.5):
    Evaluates Three-Way Gate on test split (weighted) against baselines B0, B1, B2, B4
    at matched volume, reporting 95% stratified bootstrap confidence intervals.
    """
    log_test_access(
        caller="run_experiment_e1_gate_quality",
        purpose="Evaluating E1 gate quality and matched-volume baselines on test split",
        sample_count=len(test_df),
    )

    if ground_truth_labels is None:
        raise BlockedExperimentError(
            "E1 is BLOCKED: Ground-truth human labels are required on the test split. "
            "No human labels found in data/labels/latest_labels.csv."
        )

    # Resolve test ground truth labels
    pids = test_df["paper_id"].astype(str).values
    raw_labels = np.array([ground_truth_labels.get(pid, "not_relevant") for pid in pids])
    y_test = binary_map_labels(raw_labels)
    weights = test_df["sampling_weight"].values if "sampling_weight" in test_df.columns else None

    test_texts = (test_df["title"].fillna("") + " " + test_df["abstract"].fillna("")).tolist()

    # 1. Evaluate Gate (B2 embedding cosine with thresholds)
    gate_scores = score_embedding_b2(test_embeddings, seed_vector)
    high_mask = gate_scores >= t_high
    low_mask = gate_scores <= t_low
    border_mask = ~high_mask & ~low_mask

    target_volume = int(np.sum(high_mask))
    if target_volume == 0:
        target_volume = max(1, int(0.05 * len(test_df)))

    gate_prec = float(weighted_precision(y_test, high_mask, weights))
    gate_rec = float(weighted_recall(y_test, high_mask, weights))
    gate_f1 = float(weighted_f1(y_test, high_mask, weights))
    gate_pr_auc = float(weighted_pr_auc(y_test, gate_scores, weights))
    gate_roc_auc = float(roc_auc_score(y_test, gate_scores, sample_weight=weights))
    gate_miss = float(low_miss_rate(y_test, low_mask, weights))
    b_comp = borderline_composition(raw_labels, border_mask, weights)

    # Stratified bootstrap CI for Gate PR-AUC
    eval_df = pd.DataFrame({
        "y_true": y_test,
        "score": gate_scores,
        "weight": weights,
        "stratum": test_df["stratum"].values if "stratum" in test_df.columns else "all",
    })

    def _calc_pr_auc(sub_df: pd.DataFrame) -> float:
        return weighted_pr_auc(sub_df["y_true"].values, sub_df["score"].values, sub_df["weight"].values)

    ci_pr_auc = stratified_bootstrap_ci(
        df=eval_df,
        metric_fn=_calc_pr_auc,
        strata_col="stratum",
        n_resamples=n_resamples,
        seed=random_state,
    )

    # 2. Baseline B0: Random routing at matched volume
    b0_scores = score_random_b0(len(test_df), seed=random_state)
    b0_res = evaluate_at_matched_volume(b0_scores, y_test, target_volume=target_volume, weights=weights)
    b0_pr_auc = float(weighted_pr_auc(y_test, b0_scores, weights))
    b0_roc_auc = float(roc_auc_score(y_test, b0_scores, sample_weight=weights))

    # 3. Baseline B1: TF-IDF cosine similarity to seed interest text
    all_texts = (corpus_texts or []) + test_texts
    b1_scores = score_tfidf_b1(all_texts, test_texts, config.SEED_INTEREST)
    b1_res = evaluate_at_matched_volume(b1_scores, y_test, target_volume=target_volume, weights=weights)
    b1_pr_auc = float(weighted_pr_auc(y_test, b1_scores, weights))
    b1_roc_auc = float(roc_auc_score(y_test, b1_scores, sample_weight=weights))

    # 4. Baseline B2: Embedding cosine at matched volume
    b2_res = evaluate_at_matched_volume(gate_scores, y_test, target_volume=target_volume, weights=weights)

    # 5. Baseline B4: TF-IDF + Logistic Regression (if dev texts provided)
    if dev_df is not None and len(dev_df) > 0:
        dev_pids = dev_df["paper_id"].astype(str).values
        y_dev = np.array([1 if ground_truth_labels.get(pid) == "relevant" else 0 for pid in dev_pids])
        dev_texts = (dev_df["title"].fillna("") + " " + dev_df["abstract"].fillna("")).tolist()
        b4_scores = train_and_score_tfidf_logreg_b4(
            train_texts=dev_texts,
            y_train=y_dev,
            eval_texts=test_texts,
            corpus_texts=corpus_texts,
            random_state=random_state,
        )
        b4_res = evaluate_at_matched_volume(b4_scores, y_test, target_volume=target_volume, weights=weights)
        b4_pr_auc = float(weighted_pr_auc(y_test, b4_scores, weights))
        b4_roc_auc = float(roc_auc_score(y_test, b4_scores, sample_weight=weights))
    else:
        b4_res = {"precision": np.nan, "recall": np.nan, "f1": np.nan}
        b4_pr_auc, b4_roc_auc = np.nan, np.nan

    records = [
        {
            "model_or_baseline": "Three-Way Gate (HIGH)",
            "matched_volume": target_volume,
            "precision": gate_prec,
            "recall": gate_rec,
            "f1": gate_f1,
            "pr_auc": gate_pr_auc,
            "pr_auc_ci_lower": ci_pr_auc[0],
            "pr_auc_ci_upper": ci_pr_auc[1],
            "roc_auc": gate_roc_auc,
            "low_miss_rate": gate_miss,
            "borderline_share": b_comp["share"],
            "borderline_pct_relevant": b_comp["pct_relevant"],
            "borderline_pct_somewhat": b_comp["pct_somewhat"],
            "borderline_pct_not_relevant": b_comp["pct_not_relevant"],
            "is_simulated": simulated,
        },
        {
            "model_or_baseline": "B0: Random (Matched Volume)",
            "matched_volume": target_volume,
            "precision": b0_res["precision"],
            "recall": b0_res["recall"],
            "f1": b0_res["f1"],
            "pr_auc": b0_pr_auc,
            "pr_auc_ci_lower": np.nan,
            "pr_auc_ci_upper": np.nan,
            "roc_auc": b0_roc_auc,
            "low_miss_rate": np.nan,
            "borderline_share": np.nan,
            "borderline_pct_relevant": np.nan,
            "borderline_pct_somewhat": np.nan,
            "borderline_pct_not_relevant": np.nan,
            "is_simulated": simulated,
        },
        {
            "model_or_baseline": "B1: TF-IDF Cosine (Matched Volume)",
            "matched_volume": target_volume,
            "precision": b1_res["precision"],
            "recall": b1_res["recall"],
            "f1": b1_res["f1"],
            "pr_auc": b1_pr_auc,
            "pr_auc_ci_lower": np.nan,
            "pr_auc_ci_upper": np.nan,
            "roc_auc": b1_roc_auc,
            "low_miss_rate": np.nan,
            "borderline_share": np.nan,
            "borderline_pct_relevant": np.nan,
            "borderline_pct_somewhat": np.nan,
            "borderline_pct_not_relevant": np.nan,
            "is_simulated": simulated,
        },
        {
            "model_or_baseline": "B2: Seed Embedding Cosine (Matched Volume)",
            "matched_volume": target_volume,
            "precision": b2_res["precision"],
            "recall": b2_res["recall"],
            "f1": b2_res["f1"],
            "pr_auc": gate_pr_auc,
            "pr_auc_ci_lower": np.nan,
            "pr_auc_ci_upper": np.nan,
            "roc_auc": gate_roc_auc,
            "low_miss_rate": np.nan,
            "borderline_share": np.nan,
            "borderline_pct_relevant": np.nan,
            "borderline_pct_somewhat": np.nan,
            "borderline_pct_not_relevant": np.nan,
            "is_simulated": simulated,
        },
        {
            "model_or_baseline": "B4: TF-IDF + LogReg (Matched Volume)",
            "matched_volume": target_volume,
            "precision": b4_res["precision"],
            "recall": b4_res["recall"],
            "f1": b4_res["f1"],
            "pr_auc": b4_pr_auc,
            "pr_auc_ci_lower": np.nan,
            "pr_auc_ci_upper": np.nan,
            "roc_auc": b4_roc_auc,
            "low_miss_rate": np.nan,
            "borderline_share": np.nan,
            "borderline_pct_relevant": np.nan,
            "borderline_pct_somewhat": np.nan,
            "borderline_pct_not_relevant": np.nan,
            "is_simulated": simulated,
        },
    ]

    df_out = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "e1_gate_quality.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved E1 Gate Quality results to %s", out_csv)
    return df_out


# =========================================================================
# E2: Learning Curves (Protocols A and B)
# =========================================================================

def run_experiment_e2_learning_curves(
    dev_df: pd.DataFrame,
    dev_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    seed_vector: np.ndarray,
    oracle: SimulationOracle,
    t_high: float,
    t_low: float,
    feedback_sizes: Sequence[int] = SIMULATION_FEEDBACK_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    corpus_texts: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    E2 — Learning Curves under Protocols A and B (Roadmap Section 8.5).
    Evaluates PR-AUC and F1 against feedback sample size, comparing with B2 and B4,
    marking the classifier switch point at n=30.
    """
    log_test_access(
        caller="run_experiment_e2_learning_curves",
        purpose="Evaluating E2 learning curves under Protocols A & B on test split",
        sample_count=len(test_df),
    )

    # 1. Protocol A
    sys_a_pts, b4_a_pts = run_protocol_a(
        dev_df=dev_df,
        dev_embeddings=dev_embeddings,
        test_df=test_df,
        test_embeddings=test_embeddings,
        seed_vector=seed_vector,
        oracle=oracle,
        feedback_sizes=feedback_sizes,
        random_seeds=random_seeds,
        corpus_texts=corpus_texts,
    )

    # 2. Protocol B
    sys_b_pts, pool_caps = run_protocol_b(
        dev_df=dev_df,
        dev_embeddings=dev_embeddings,
        test_df=test_df,
        test_embeddings=test_embeddings,
        seed_vector=seed_vector,
        oracle=oracle,
        t_high=t_high,
        t_low=t_low,
        feedback_sizes=feedback_sizes,
        random_seeds=random_seeds,
        corpus_texts=corpus_texts,
    )

    rows = []
    for pt in sys_a_pts:
        d = asdict(pt)
        d["series"] = "System (Protocol A - Random)"
        d["is_simulated"] = True
        rows.append(d)

    for pt in b4_a_pts:
        d = asdict(pt)
        d["series"] = "B4: TF-IDF + LogReg (Protocol A)"
        d["is_simulated"] = True
        rows.append(d)

    for pt in sys_b_pts:
        d = asdict(pt)
        d["series"] = "System (Protocol B - Gate Selected)"
        d["is_simulated"] = True
        rows.append(d)

    df_out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "e2_learning_curves.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved E2 Learning Curves results to %s", out_csv)
    return df_out


# =========================================================================
# E3: Signal Ablation
# =========================================================================

def run_experiment_e3_signal_ablation(
    dev_df: pd.DataFrame,
    dev_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    seed_vector: np.ndarray,
    oracle: SimulationOracle,
    t_high: float,
    t_low: float,
    feedback_sizes: Sequence[int] = (10, 20, 30, 50),
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
) -> pd.DataFrame:
    """
    E3 — Signal Ablation (Roadmap Section 8.5):
    Compares user-only, reviewer-only, and combined feedback under Protocol B
    at matched sample sizes, evaluating the effective-label rule.
    """
    log_test_access(
        caller="run_experiment_e3_signal_ablation",
        purpose="Evaluating E3 signal ablation on test split",
        sample_count=len(test_df),
    )

    raw_results = run_signal_ablation_e3(
        dev_df=dev_df,
        dev_embeddings=dev_embeddings,
        test_df=test_df,
        test_embeddings=test_embeddings,
        seed_vector=seed_vector,
        oracle=oracle,
        t_high=t_high,
        t_low=t_low,
        feedback_sizes=feedback_sizes,
        random_seeds=random_seeds,
    )

    rows = []
    for cond, pts in raw_results.items():
        for pt in pts:
            pt["is_simulated"] = True
            rows.append(pt)

    df_out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "e3_signal_ablation.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved E3 Signal Ablation results to %s", out_csv)
    return df_out


# =========================================================================
# E4: Centroid Update
# =========================================================================

def run_experiment_e4_centroid_update(
    dev_df: pd.DataFrame,
    dev_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    seed_vector: np.ndarray,
    oracle: SimulationOracle,
    feedback_sizes: Sequence[int] = (10, 20, 30, 50),
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
) -> pd.DataFrame:
    """
    E4 — Centroid Update (Roadmap Section 8.5):
    Evaluates B3 (updated centroid) vs B2 (fixed seed vector) and B5 (classifier)
    at equal feedback budgets, testing whether centroid adaptation helps before
    the classifier is eligible.
    """
    log_test_access(
        caller="run_experiment_e4_centroid_update",
        purpose="Evaluating E4 centroid update vs fixed seed & classifier on test split",
        sample_count=len(test_df),
    )

    test_weights = test_df["sampling_weight"].values if "sampling_weight" in test_df.columns else None

    # Derive test ground truth labels
    y_test = np.array([
        oracle.evaluate_paper(str(pid), {"title": r.get("title", ""), "abstract": r.get("abstract", "")})
        for pid, (_, r) in zip(test_df["paper_id"].values, test_df.iterrows())
    ], dtype=np.int32)

    dev_labels = np.array([
        oracle.evaluate_paper(pid, {"title": r["title"], "abstract": r["abstract"]})
        for pid, (_, r) in zip(dev_df["paper_id"].values, dev_df.iterrows())
    ], dtype=np.int32)

    # Constant B2 baseline
    b2_scores = score_embedding_b2(test_embeddings, seed_vector)
    b2_pr_auc = float(weighted_pr_auc(y_test, b2_scores, test_weights))
    b2_roc_auc = float(roc_auc_score(y_test, b2_scores, sample_weight=test_weights))
    b2_f1 = float(weighted_f1(y_test, b2_scores >= 0.5, test_weights))

    rows = []
    # Add B2 baseline row
    rows.append({
        "feedback_size": 0,
        "model": "B2: Fixed Seed Vector",
        "mean_pr_auc": b2_pr_auc,
        "std_pr_auc": 0.0,
        "mean_roc_auc": b2_roc_auc,
        "std_roc_auc": 0.0,
        "mean_f1": b2_f1,
        "std_f1": 0.0,
        "is_simulated": True,
    })

    dev_n = len(dev_df)
    for n in feedback_sizes:
        if n > dev_n:
            continue

        b3_pr_aucs, b3_roc_aucs, b3_f1s = [], [], []
        b5_pr_aucs, b5_roc_aucs, b5_f1s = [], [], []

        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            sample_idx = rng.choice(dev_n, size=n, replace=False)
            X_s = dev_embeddings[sample_idx]
            y_s = dev_labels[sample_idx]

            # B3 Centroid
            centroid = compute_rocchio_centroid(
                c0=seed_vector,
                positive_embeddings=X_s[y_s == 1],
                negative_embeddings=X_s[y_s == 0],
            )
            b3_scores = score_centroid_b3(test_embeddings, centroid)
            b3_pr_aucs.append(float(weighted_pr_auc(y_test, b3_scores, test_weights)))
            b3_roc_aucs.append(float(roc_auc_score(y_test, b3_scores, sample_weight=test_weights)))
            b3_f1s.append(float(weighted_f1(y_test, b3_scores >= 0.5, test_weights)))

            # B5 Classifier
            b5_scores = train_and_score_embedding_logreg_b5(
                X_train=X_s, y_train=y_s, X_eval=test_embeddings, c_val=0.3, random_state=seed
            )
            b5_pr_aucs.append(float(weighted_pr_auc(y_test, b5_scores, test_weights)))
            b5_roc_aucs.append(float(roc_auc_score(y_test, b5_scores, sample_weight=test_weights)))
            b5_f1s.append(float(weighted_f1(y_test, b5_scores >= 0.5, test_weights)))

        rows.append({
            "feedback_size": n,
            "model": "B3: Updated Centroid (Rocchio)",
            "mean_pr_auc": float(np.mean(b3_pr_aucs)),
            "std_pr_auc": float(np.std(b3_pr_aucs)),
            "mean_roc_auc": float(np.mean(b3_roc_aucs)),
            "std_roc_auc": float(np.std(b3_roc_aucs)),
            "mean_f1": float(np.mean(b3_f1s)),
            "std_f1": float(np.std(b3_f1s)),
            "is_simulated": True,
        })

        rows.append({
            "feedback_size": n,
            "model": "B5: Classifier (Embedding LogReg)",
            "mean_pr_auc": float(np.mean(b5_pr_aucs)),
            "std_pr_auc": float(np.std(b5_pr_aucs)),
            "mean_roc_auc": float(np.mean(b5_roc_aucs)),
            "std_roc_auc": float(np.std(b5_roc_aucs)),
            "mean_f1": float(np.mean(b5_f1s)),
            "std_f1": float(np.std(b5_f1s)),
            "is_simulated": True,
        })

    df_out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "e4_centroid_update.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved E4 Centroid Update results to %s", out_csv)
    return df_out


# =========================================================================
# E5: Efficiency & Workload Reduction
# =========================================================================

def run_experiment_e5_efficiency(
    scores: np.ndarray,
    percentile_pairs: Sequence[tuple[float, float]] = ((98.0, 90.0), (95.0, 85.0), (99.0, 95.0)),
    digest_cap: int = 20,
    review_cap: int = 15,
    avg_tokens_per_abstract: int = 250,
    avg_tokens_per_summary: int = 150,
    cost_per_1m_input_usd: float = 0.075,
    cost_per_1m_output_usd: float = 0.30,
) -> pd.DataFrame:
    """
    E5 — Efficiency (Roadmap Section 8.5):
    Measures LLM-call reduction against summarize-everything, token and cost savings,
    review workload, and sensitivity to percentile levels.
    """
    n_total = len(scores)
    rows = []

    # Summarize everything baseline
    full_llm_calls = n_total
    full_input_tokens = full_llm_calls * avg_tokens_per_abstract
    full_output_tokens = full_llm_calls * avg_tokens_per_summary
    full_cost_usd = (full_input_tokens / 1_000_000 * cost_per_1m_input_usd) + (
        full_output_tokens / 1_000_000 * cost_per_1m_output_usd
    )

    for p_high, p_low in percentile_pairs:
        t_high = float(np.percentile(scores, p_high))
        t_low = float(np.percentile(scores, p_low))

        high_count = int(np.sum(scores >= t_high))
        low_count = int(np.sum(scores <= t_low))
        border_count = n_total - high_count - low_count

        # Caps enforcement
        summarized_high = min(high_count, digest_cap)
        queued_border = min(border_count, review_cap)
        # Assuming ~50% borderline acceptance under balanced review
        accepted_border = queued_border // 2
        total_summarized = min(digest_cap, summarized_high + accepted_border)

        llm_call_reduction = 1.0 - (total_summarized / max(1, full_llm_calls))
        actual_input_tokens = total_summarized * avg_tokens_per_abstract
        actual_output_tokens = total_summarized * avg_tokens_per_summary
        actual_cost_usd = (actual_input_tokens / 1_000_000 * cost_per_1m_input_usd) + (
            actual_output_tokens / 1_000_000 * cost_per_1m_output_usd
        )
        cost_savings_pct = 1.0 - (actual_cost_usd / max(1e-6, full_cost_usd))

        rows.append({
            "p_high": p_high,
            "p_low": p_low,
            "t_high": t_high,
            "t_low": t_low,
            "total_papers_scored": n_total,
            "high_routed": high_count,
            "borderline_routed": border_count,
            "low_discarded": low_count,
            "review_queue_workload": queued_border,
            "review_queue_overflow": max(0, border_count - review_cap),
            "papers_summarized": total_summarized,
            "unmitigated_llm_calls": full_llm_calls,
            "llm_call_reduction_pct": float(llm_call_reduction * 100.0),
            "estimated_cost_usd": float(actual_cost_usd),
            "unmitigated_cost_usd": float(full_cost_usd),
            "cost_savings_pct": float(cost_savings_pct * 100.0),
        })

    df_out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "e5_efficiency.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved E5 Efficiency results to %s", out_csv)
    return df_out


# =========================================================================
# E6: Human-Review Value
# =========================================================================

def run_experiment_e6_human_review_value(
    scores: np.ndarray,
    y_true: np.ndarray,
    t_high: float,
    t_low: float,
    weights: Optional[np.ndarray] = None,
    oracle: Optional[SimulationOracle] = None,
    paper_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    E6 — Human-Review Value (Roadmap Section 8.5):
    Evaluates whether escalation of borderline papers improves decision quality:
      1. Gate + Review (deliver HIGH + oracle-accepted BORDERLINE)
      2. Auto-Accept Borderline (deliver HIGH + all BORDERLINE)
      3. Auto-Reject Borderline (deliver HIGH only)
    """
    if weights is None:
        weights = np.ones(len(scores), dtype=np.float32)

    high_mask = scores >= t_high
    low_mask = scores <= t_low
    border_mask = ~high_mask & ~low_mask

    # 1. Auto-Reject Borderline (HIGH only)
    pred_auto_reject = high_mask
    p_reject = weighted_precision(y_true, pred_auto_reject, weights)
    r_reject = weighted_recall(y_true, pred_auto_reject, weights)
    f1_reject = weighted_f1(y_true, pred_auto_reject, weights)

    # 2. Auto-Accept Borderline (HIGH + all BORDERLINE)
    pred_auto_accept = high_mask | border_mask
    p_accept = weighted_precision(y_true, pred_auto_accept, weights)
    r_accept = weighted_recall(y_true, pred_auto_accept, weights)
    f1_accept = weighted_f1(y_true, pred_auto_accept, weights)

    # 3. Human Review (HIGH + oracle accepted BORDERLINE)
    # Reviewer accepts when y_true == 1 (or oracle decision)
    if oracle is not None and paper_ids is not None:
        accepted_border_mask = np.zeros(len(scores), dtype=bool)
        for i in np.where(border_mask)[0]:
            dec = oracle.get_reviewer_decision(paper_ids[i])
            if dec == "accept":
                accepted_border_mask[i] = True
    else:
        # Standard oracle accepts relevant borderline papers
        accepted_border_mask = border_mask & (y_true == 1)

    pred_review = high_mask | accepted_border_mask
    p_review = weighted_precision(y_true, pred_review, weights)
    r_review = weighted_recall(y_true, pred_review, weights)
    f1_review = weighted_f1(y_true, pred_review, weights)

    n_reviewed = int(np.sum(border_mask))

    rows = [
        {
            "strategy": "Auto-Reject Borderline (HIGH Only)",
            "papers_delivered": int(np.sum(pred_auto_reject)),
            "papers_reviewed": 0,
            "precision": float(p_reject),
            "recall": float(r_reject),
            "f1": float(f1_reject),
            "description": "Baseline: Discards all borderline papers without review.",
        },
        {
            "strategy": "Auto-Accept Borderline (HIGH + BORDERLINE)",
            "papers_delivered": int(np.sum(pred_auto_accept)),
            "papers_reviewed": 0,
            "precision": float(p_accept),
            "recall": float(r_accept),
            "f1": float(f1_accept),
            "description": "Baseline: Delivers all borderline papers without human filter.",
        },
        {
            "strategy": "Relevance Gate with Human Review",
            "papers_delivered": int(np.sum(pred_review)),
            "papers_reviewed": n_reviewed,
            "precision": float(p_review),
            "recall": float(r_review),
            "f1": float(f1_review),
            "description": "Proposed system: Escalates borderline papers to human review.",
        },
    ]

    df_out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "e6_human_review_value.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved E6 Human Review Value results to %s", out_csv)
    return df_out
