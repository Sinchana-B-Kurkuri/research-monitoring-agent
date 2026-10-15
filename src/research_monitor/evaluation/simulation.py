"""
simulation.py
~~~~~~~~~~~~~
Simulation Protocols A & B and Oracle Feedback Generation (Roadmap Section 8.4).

Follows Section 8.4:
  - Oracle: Simulated user likes a delivered paper if primary label is 'relevant';
    simulated reviewer accepts a borderline paper under the same rule.
    Optional noise sensitivity flips decisions with probability noise_rate.
  - Protocol A: Stratified-randomly sample labelled development papers for feedback.
  - Protocol B: Simulate feedback selection through the cold-start gate:
    user signals from HIGH papers and reviewer signals from BORDERLINE papers.
    Respects pool capacity and stops curves when a pool is exhausted.
  - Feedback sample sizes: [0, 10, 20, 30, 50, 75, 100, 150] (capped by pool size).
  - 10 random seeds per sample size; reports means and confidence bands.
  - Below classifier eligibility threshold (< 30 labels), system curve uses
    cosine-to-centroid (B3); at >= 30, switches to embedding logistic regression (B5).
  - All oracle-generated outputs are clearly marked as [SIMULATED].
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research_monitor.evaluation.baselines import (
    score_centroid_b3,
    score_embedding_b2,
    train_and_score_embedding_logreg_b5,
    train_and_score_tfidf_logreg_b4,
)
from research_monitor.evaluation.metrics import (
    binary_map_labels,
    stratified_bootstrap_ci,
    weighted_f1,
    weighted_pr_auc,
)
from research_monitor.relevance.preference import compute_rocchio_centroid

logger = logging.getLogger(__name__)

SIMULATION_FEEDBACK_SIZES = (0, 10, 20, 30, 50, 75, 100, 150)
CLASSIFIER_SWITCH_POINT = 30
DEFAULT_RANDOM_SEEDS = tuple(range(42, 52))  # 10 deterministic seeds


class SimulationOracle:
    """
    Simulated human feedback oracle (Roadmap Section 8.4).

    Rules:
      - Primary mapping: 'relevant' -> 1 (Like / Accept),
        'somewhat' or 'not_relevant' -> 0 (Dislike / Reject).
      - Noise sensitivity (optional): Flips label with probability noise_rate.
      - Heuristic fallback: If ground truth label is missing, uses seed-keyword alignment.
    """

    def __init__(
        self,
        ground_truth_map: Optional[dict[str, str]] = None,
        noise_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.ground_truth_map = ground_truth_map or {}
        self.noise_rate = max(0.0, min(1.0, float(noise_rate)))
        self.rng = np.random.default_rng(seed)

    def evaluate_paper(
        self,
        paper_id: str,
        paper_meta: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Evaluate a single paper. Returns 1 (relevant/positive) or 0 (non-relevant/negative).
        """
        # 1. Check ground-truth map
        raw_label = self.ground_truth_map.get(str(paper_id))
        if raw_label is not None:
            canonical = str(raw_label).strip().lower()
            val = 1 if canonical == "relevant" else 0
        elif paper_meta is not None:
            # Fallback heuristic: check title + abstract for core seed interests
            text = (
                str(paper_meta.get("title", "")) + " " + str(paper_meta.get("abstract", ""))
            ).lower()
            keywords = [
                "agent",
                "autonomous",
                "reinforcement learning",
                "large language model",
                "llm",
                "reasoning",
                "multi-agent",
                "planning",
                "tool use",
            ]
            match_count = sum(1 for kw in keywords if kw in text)
            val = 1 if match_count >= 2 else 0
        else:
            val = 0

        # 2. Apply noise sensitivity if specified
        if self.noise_rate > 0.0 and self.rng.uniform(0.0, 1.0) < self.noise_rate:
            val = 1 - val

        return int(val)

    def get_user_reaction(
        self,
        paper_id: str,
        paper_meta: Optional[dict[str, Any]] = None,
    ) -> int:
        """Returns 1 for Like, 0 for Dislike."""
        return self.evaluate_paper(paper_id, paper_meta)

    def get_reviewer_decision(
        self,
        paper_id: str,
        paper_meta: Optional[dict[str, Any]] = None,
    ) -> str:
        """Returns 'accept' if positive, 'reject' if negative."""
        val = self.evaluate_paper(paper_id, paper_meta)
        return "accept" if val == 1 else "reject"


@dataclass
class LearningCurvePoint:
    """Evaluation metrics for a single feedback sample size across seeds."""
    n_feedback: int
    protocol: str  # 'A' or 'B'
    model_type: str  # 'b2_seed', 'b3_centroid', 'b5_classifier', or 'b4_tfidf'
    mean_pr_auc: float
    std_pr_auc: float
    ci_lower_pr_auc: float
    ci_upper_pr_auc: float
    mean_roc_auc: float
    std_roc_auc: float
    mean_f1: float
    std_f1: float
    is_classifier: bool
    is_switch_point: bool
    pool_exhausted: bool = False
    details_per_seed: list[dict[str, Any]] = field(default_factory=list)


def run_protocol_a(
    dev_df: pd.DataFrame,
    dev_embeddings: np.ndarray,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    seed_vector: np.ndarray,
    oracle: SimulationOracle,
    feedback_sizes: Sequence[int] = SIMULATION_FEEDBACK_SIZES,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    corpus_texts: Optional[list[str]] = None,
) -> tuple[list[LearningCurvePoint], list[LearningCurvePoint]]:
    """
    Protocol A: Random Stratified Feedback Selection (Section 8.4).

    For each n in feedback_sizes:
      Sample n development papers at random across strata.
      Derive oracle labels.
      If n < 30: System = B3 (Rocchio centroid update).
      If n >= 30: System = B5 (Embedding logistic regression).
      Literature baseline = B4 (TF-IDF + Logistic Regression).
      Evaluate on frozen test split (weighted metrics).

    Returns:
      (system_curve_points, b4_curve_points)
    """
    dev_n = len(dev_df)
    test_weights = test_df["sampling_weight"].values if "sampling_weight" in test_df.columns else None

    # Derive test ground truth labels
    y_test = np.array([
        oracle.evaluate_paper(str(pid), {"title": r.get("title", ""), "abstract": r.get("abstract", "")})
        for pid, (_, r) in zip(test_df["paper_id"].values, test_df.iterrows())
    ], dtype=np.int32)

    dev_texts = (dev_df["title"].fillna("") + " " + dev_df["abstract"].fillna("")).tolist()
    test_texts = (test_df["title"].fillna("") + " " + test_df["abstract"].fillna("")).tolist()

    # Pre-extract dev oracle labels
    dev_labels = np.array([
        oracle.evaluate_paper(pid, {"title": r["title"], "abstract": r["abstract"]})
        for pid, (_, r) in zip(dev_df["paper_id"].values, dev_df.iterrows())
    ], dtype=np.int32)

    # Constant B2 baseline on test split
    b2_scores = score_embedding_b2(test_embeddings, seed_vector)
    b2_pr_auc = float(weighted_pr_auc(y_test, b2_scores, test_weights))
    b2_roc_auc = float(roc_auc_score(y_test, b2_scores, sample_weight=test_weights))
    b2_f1 = float(weighted_f1(y_test, b2_scores >= 0.5, test_weights))

    system_points: list[LearningCurvePoint] = []
    b4_points: list[LearningCurvePoint] = []

    for n in feedback_sizes:
        if n > dev_n:
            logger.info("Protocol A: requested n=%d exceeds dev pool size (%d). Stopping.", n, dev_n)
            break

        if n == 0:
            # Baseline point: fixed seed vector
            pt = LearningCurvePoint(
                n_feedback=0,
                protocol="A",
                model_type="b2_seed",
                mean_pr_auc=b2_pr_auc,
                std_pr_auc=0.0,
                ci_lower_pr_auc=b2_pr_auc,
                ci_upper_pr_auc=b2_pr_auc,
                mean_roc_auc=b2_roc_auc,
                std_roc_auc=0.0,
                mean_f1=b2_f1,
                std_f1=0.0,
                is_classifier=False,
                is_switch_point=False,
            )
            system_points.append(pt)
            b4_points.append(pt)
            continue

        is_clf = (n >= CLASSIFIER_SWITCH_POINT)
        is_switch = (n == CLASSIFIER_SWITCH_POINT)

        sys_pr_aucs, sys_roc_aucs, sys_f1s = [], [], []
        b4_pr_aucs, b4_roc_aucs, b4_f1s = [], [], []

        for seed in random_seeds:
            rng = np.random.default_rng(seed)

            # Stratified sampling across strata if available, else random
            if "stratum" in dev_df.columns:
                sampled_indices = []
                strata_groups = dev_df.groupby("stratum").indices
                strata_keys = list(strata_groups.keys())
                k_per_stratum = max(1, n // len(strata_keys))
                for s_key in strata_keys:
                    pool_idx = strata_groups[s_key]
                    take = min(k_per_stratum, len(pool_idx))
                    sampled_indices.extend(rng.choice(pool_idx, size=take, replace=False))
                
                # Adjust to exact n
                if len(sampled_indices) > n:
                    sampled_indices = list(rng.choice(sampled_indices, size=n, replace=False))
                elif len(sampled_indices) < n:
                    remaining = list(set(range(dev_n)) - set(sampled_indices))
                    needed = n - len(sampled_indices)
                    sampled_indices.extend(rng.choice(remaining, size=min(needed, len(remaining)), replace=False))
                sample_idx = np.array(sampled_indices)
            else:
                sample_idx = rng.choice(dev_n, size=n, replace=False)

            X_sample = dev_embeddings[sample_idx]
            y_sample = dev_labels[sample_idx]
            sample_txts = [dev_texts[i] for i in sample_idx]

            # 1. System Evaluation
            if not is_clf:
                # B3 Centroid update
                pos_embs = X_sample[y_sample == 1]
                neg_embs = X_sample[y_sample == 0]
                updated_centroid = compute_rocchio_centroid(
                    c0=seed_vector,
                    positive_embeddings=pos_embs,
                    negative_embeddings=neg_embs,
                    alpha=1.0,
                    beta=0.75,
                    gamma=0.25,
                )
                scores = score_centroid_b3(test_embeddings, updated_centroid)
                m_type = "b3_centroid"
            else:
                # B5 Logistic Regression
                scores = train_and_score_embedding_logreg_b5(
                    X_train=X_sample,
                    y_train=y_sample,
                    X_eval=test_embeddings,
                    c_val=0.3,
                    random_state=seed,
                )
                m_type = "b5_classifier"

            sys_pr_aucs.append(float(weighted_pr_auc(y_test, scores, test_weights)))
            sys_roc_aucs.append(float(roc_auc_score(y_test, scores, sample_weight=test_weights)))
            sys_f1s.append(float(weighted_f1(y_test, scores >= 0.5, test_weights)))

            # 2. B4 TF-IDF LogReg Evaluation
            b4_scores = train_and_score_tfidf_logreg_b4(
                train_texts=sample_txts,
                y_train=y_sample,
                eval_texts=test_texts,
                corpus_texts=corpus_texts,
                c_val=1.0,
                random_state=seed,
            )
            b4_pr_aucs.append(float(weighted_pr_auc(y_test, b4_scores, test_weights)))
            b4_roc_aucs.append(float(roc_auc_score(y_test, b4_scores, sample_weight=test_weights)))
            b4_f1s.append(float(weighted_f1(y_test, b4_scores >= 0.5, test_weights)))

        # Aggregate System stats
        sys_mean_pr = float(np.mean(sys_pr_aucs))
        sys_std_pr = float(np.std(sys_pr_aucs))
        sys_ci_low = max(0.0, sys_mean_pr - 1.96 * sys_std_pr / np.sqrt(len(random_seeds)))
        sys_ci_high = min(1.0, sys_mean_pr + 1.96 * sys_std_pr / np.sqrt(len(random_seeds)))

        system_points.append(LearningCurvePoint(
            n_feedback=n,
            protocol="A",
            model_type=m_type,
            mean_pr_auc=sys_mean_pr,
            std_pr_auc=sys_std_pr,
            ci_lower_pr_auc=sys_ci_low,
            ci_upper_pr_auc=sys_ci_high,
            mean_roc_auc=float(np.mean(sys_roc_aucs)),
            std_roc_auc=float(np.std(sys_roc_aucs)),
            mean_f1=float(np.mean(sys_f1s)),
            std_f1=float(np.std(sys_f1s)),
            is_classifier=is_clf,
            is_switch_point=is_switch,
        ))

        # Aggregate B4 stats
        b4_mean_pr = float(np.mean(b4_pr_aucs))
        b4_std_pr = float(np.std(b4_pr_aucs))
        b4_ci_low = max(0.0, b4_mean_pr - 1.96 * b4_std_pr / np.sqrt(len(random_seeds)))
        b4_ci_high = min(1.0, b4_mean_pr + 1.96 * b4_std_pr / np.sqrt(len(random_seeds)))

        b4_points.append(LearningCurvePoint(
            n_feedback=n,
            protocol="A",
            model_type="b4_tfidf",
            mean_pr_auc=b4_mean_pr,
            std_pr_auc=b4_std_pr,
            ci_lower_pr_auc=b4_ci_low,
            ci_upper_pr_auc=b4_ci_high,
            mean_roc_auc=float(np.mean(b4_roc_aucs)),
            std_roc_auc=float(np.std(b4_roc_aucs)),
            mean_f1=float(np.mean(b4_f1s)),
            std_f1=float(np.std(b4_f1s)),
            is_classifier=False,
            is_switch_point=False,
        ))

    return system_points, b4_points


def run_protocol_b(
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
) -> tuple[list[LearningCurvePoint], dict[str, int]]:
    """
    Protocol B: Gate-Selected Feedback Simulation (Section 8.4).

    Partitions dev pool using cold-start gate:
      - HIGH pool (score >= t_high): delivers to user -> user reactions (Likes/Dislikes)
      - BORDERLINE pool (t_low < score < t_high): review queue -> reviewer decisions (Accept/Reject)
      - LOW pool (score <= t_low): discarded -> no feedback

    Respects pool capacity and stops curves when a pool is exhausted.
    Below n=30, uses B3 (centroid update); at n>=30, switches to B5 (embedding logistic regression).

    Returns:
      (learning_curve_points, pool_capacities)
    """
    test_weights = test_df["sampling_weight"].values if "sampling_weight" in test_df.columns else None

    # Derive test ground truth labels
    y_test = np.array([
        oracle.evaluate_paper(str(pid), {"title": r.get("title", ""), "abstract": r.get("abstract", "")})
        for pid, (_, r) in zip(test_df["paper_id"].values, test_df.iterrows())
    ], dtype=np.int32)

    # Compute cold-start scores on dev pool to partition into pools
    dev_scores = score_embedding_b2(dev_embeddings, seed_vector)
    high_pool = np.where(dev_scores >= t_high)[0]
    borderline_pool = np.where((dev_scores > t_low) & (dev_scores < t_high))[0]

    pool_capacities = {
        "high_capacity": len(high_pool),
        "borderline_capacity": len(borderline_pool),
        "total_gate_capacity": len(high_pool) + len(borderline_pool),
    }

    logger.info(
        "Protocol B pools: HIGH=%d, BORDERLINE=%d, total capacity=%d",
        len(high_pool),
        len(borderline_pool),
        pool_capacities["total_gate_capacity"],
    )

    # Pre-extract oracle feedback labels
    dev_labels = np.array([
        oracle.evaluate_paper(pid, {"title": r["title"], "abstract": r["abstract"]})
        for pid, (_, r) in zip(dev_df["paper_id"].values, dev_df.iterrows())
    ], dtype=np.int32)

    # Constant B2 baseline on test split
    b2_scores = score_embedding_b2(test_embeddings, seed_vector)
    b2_pr_auc = float(weighted_pr_auc(y_test, b2_scores, test_weights))
    b2_roc_auc = float(roc_auc_score(y_test, b2_scores, sample_weight=test_weights))
    b2_f1 = float(weighted_f1(y_test, b2_scores >= 0.5, test_weights))

    curve_points: list[LearningCurvePoint] = []

    for n in feedback_sizes:
        if n == 0:
            curve_points.append(LearningCurvePoint(
                n_feedback=0,
                protocol="B",
                model_type="b2_seed",
                mean_pr_auc=b2_pr_auc,
                std_pr_auc=0.0,
                ci_lower_pr_auc=b2_pr_auc,
                ci_upper_pr_auc=b2_pr_auc,
                mean_roc_auc=b2_roc_auc,
                std_roc_auc=0.0,
                mean_f1=b2_f1,
                std_f1=0.0,
                is_classifier=False,
                is_switch_point=False,
            ))
            continue

        if n > pool_capacities["total_gate_capacity"]:
            logger.info("Protocol B: requested n=%d exceeds total gate capacity (%d). Stopping curve.", n, pool_capacities["total_gate_capacity"])
            break

        is_clf = (n >= CLASSIFIER_SWITCH_POINT)
        is_switch = (n == CLASSIFIER_SWITCH_POINT)

        pr_aucs, roc_aucs, f1s = [], [], []

        for seed in random_seeds:
            rng = np.random.default_rng(seed)

            # Sample equally from HIGH (user) and BORDERLINE (reviewer) pools up to capacities
            n_high_target = n // 2
            n_border_target = n - n_high_target

            # Check individual pool limits
            take_high = min(n_high_target, len(high_pool))
            take_border = min(n_border_target, len(borderline_pool))

            # If one pool was smaller, compensate from the other pool if possible
            if take_high < n_high_target:
                take_border = min(len(borderline_pool), take_border + (n_high_target - take_high))
            elif take_border < n_border_target:
                take_high = min(len(high_pool), take_high + (n_border_target - take_border))

            sampled_high = rng.choice(high_pool, size=take_high, replace=False) if take_high > 0 else np.empty(0, dtype=int)
            sampled_border = rng.choice(borderline_pool, size=take_border, replace=False) if take_border > 0 else np.empty(0, dtype=int)

            sample_idx = np.concatenate([sampled_high, sampled_border])

            X_sample = dev_embeddings[sample_idx]
            y_sample = dev_labels[sample_idx]

            if not is_clf:
                # B3 Centroid update
                pos_embs = X_sample[y_sample == 1]
                neg_embs = X_sample[y_sample == 0]
                updated_centroid = compute_rocchio_centroid(
                    c0=seed_vector,
                    positive_embeddings=pos_embs,
                    negative_embeddings=neg_embs,
                    alpha=1.0,
                    beta=0.75,
                    gamma=0.25,
                )
                scores = score_centroid_b3(test_embeddings, updated_centroid)
                m_type = "b3_centroid"
            else:
                # B5 Logistic Regression
                scores = train_and_score_embedding_logreg_b5(
                    X_train=X_sample,
                    y_train=y_sample,
                    X_eval=test_embeddings,
                    c_val=0.3,
                    random_state=seed,
                )
                m_type = "b5_classifier"

            pr_aucs.append(float(weighted_pr_auc(y_test, scores, test_weights)))
            roc_aucs.append(float(roc_auc_score(y_test, scores, sample_weight=test_weights)))
            f1s.append(float(weighted_f1(y_test, scores >= 0.5, test_weights)))

        mean_pr = float(np.mean(pr_aucs))
        std_pr = float(np.std(pr_aucs))
        ci_low = max(0.0, mean_pr - 1.96 * std_pr / np.sqrt(len(random_seeds)))
        ci_high = min(1.0, mean_pr + 1.96 * std_pr / np.sqrt(len(random_seeds)))

        curve_points.append(LearningCurvePoint(
            n_feedback=n,
            protocol="B",
            model_type=m_type,
            mean_pr_auc=mean_pr,
            std_pr_auc=std_pr,
            ci_lower_pr_auc=ci_low,
            ci_upper_pr_auc=ci_high,
            mean_roc_auc=float(np.mean(roc_aucs)),
            std_roc_auc=float(np.std(roc_aucs)),
            mean_f1=float(np.mean(f1s)),
            std_f1=float(np.std(f1s)),
            is_classifier=is_clf,
            is_switch_point=is_switch,
        ))

    return curve_points, pool_capacities


def run_signal_ablation_e3(
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
) -> dict[str, list[dict[str, Any]]]:
    """
    E3 — Signal Ablation under Protocol B (Roadmap Section 8.5):
    Compares:
      1. User-only (drawn from HIGH pool)
      2. Reviewer-only (drawn from BORDERLINE pool)
      3. Combined (effective-label rule, 50/50 split)
    at matched sample sizes.

    Returns:
      dict with keys 'user_only', 'reviewer_only', 'combined'.
    """
    test_weights = test_df["sampling_weight"].values if "sampling_weight" in test_df.columns else None

    # Derive test ground truth labels
    y_test = np.array([
        oracle.evaluate_paper(str(pid), {"title": r.get("title", ""), "abstract": r.get("abstract", "")})
        for pid, (_, r) in zip(test_df["paper_id"].values, test_df.iterrows())
    ], dtype=np.int32)

    dev_scores = score_embedding_b2(dev_embeddings, seed_vector)
    high_pool = np.where(dev_scores >= t_high)[0]
    border_pool = np.where((dev_scores > t_low) & (dev_scores < t_high))[0]

    dev_labels = np.array([
        oracle.evaluate_paper(pid, {"title": r["title"], "abstract": r["abstract"]})
        for pid, (_, r) in zip(dev_df["paper_id"].values, dev_df.iterrows())
    ], dtype=np.int32)

    results: dict[str, list[dict[str, Any]]] = {
        "user_only": [],
        "reviewer_only": [],
        "combined": [],
    }

    for n in feedback_sizes:
        for condition, pool in [
            ("user_only", high_pool),
            ("reviewer_only", border_pool),
        ]:
            if n > len(pool):
                logger.info("E3: Condition '%s' at n=%d exceeds pool capacity (%d). Skipping.", condition, n, len(pool))
                continue

            pr_list, roc_list, f1_list = [], [], []
            for seed in random_seeds:
                rng = np.random.default_rng(seed)
                sample_idx = rng.choice(pool, size=n, replace=False)
                X_s, y_s = dev_embeddings[sample_idx], dev_labels[sample_idx]

                if n < CLASSIFIER_SWITCH_POINT:
                    centroid = compute_rocchio_centroid(
                        c0=seed_vector,
                        positive_embeddings=X_s[y_s == 1],
                        negative_embeddings=X_s[y_s == 0],
                    )
                    scores = score_centroid_b3(test_embeddings, centroid)
                else:
                    scores = train_and_score_embedding_logreg_b5(
                        X_train=X_s, y_train=y_s, X_eval=test_embeddings, c_val=0.3, random_state=seed
                    )

                pr_list.append(float(weighted_pr_auc(y_test, scores, test_weights)))
                roc_list.append(float(roc_auc_score(y_test, scores, sample_weight=test_weights)))
                f1_list.append(float(weighted_f1(y_test, scores >= 0.5, test_weights)))

            results[condition].append({
                "n_feedback": n,
                "condition": condition,
                "mean_pr_auc": float(np.mean(pr_list)),
                "std_pr_auc": float(np.std(pr_list)),
                "mean_roc_auc": float(np.mean(roc_list)),
                "mean_f1": float(np.mean(f1_list)),
            })

        # Combined condition
        if n <= (len(high_pool) + len(border_pool)):
            pr_list, roc_list, f1_list = [], [], []
            for seed in random_seeds:
                rng = np.random.default_rng(seed)
                n_h = min(n // 2, len(high_pool))
                n_b = min(n - n_h, len(border_pool))
                if n_b < (n - n_h) and (len(high_pool) - n_h) > 0:
                    n_h = min(len(high_pool), n_h + (n - n_h - n_b))

                s_h = rng.choice(high_pool, size=n_h, replace=False) if n_h > 0 else np.empty(0, dtype=int)
                s_b = rng.choice(border_pool, size=n_b, replace=False) if n_b > 0 else np.empty(0, dtype=int)
                sample_idx = np.concatenate([s_h, s_b])

                X_s, y_s = dev_embeddings[sample_idx], dev_labels[sample_idx]
                if n < CLASSIFIER_SWITCH_POINT:
                    centroid = compute_rocchio_centroid(
                        c0=seed_vector,
                        positive_embeddings=X_s[y_s == 1],
                        negative_embeddings=X_s[y_s == 0],
                    )
                    scores = score_centroid_b3(test_embeddings, centroid)
                else:
                    scores = train_and_score_embedding_logreg_b5(
                        X_train=X_s, y_train=y_s, X_eval=test_embeddings, c_val=0.3, random_state=seed
                    )

                pr_list.append(float(weighted_pr_auc(y_test, scores, test_weights)))
                roc_list.append(float(roc_auc_score(y_test, scores, sample_weight=test_weights)))
                f1_list.append(float(weighted_f1(y_test, scores >= 0.5, test_weights)))

            results["combined"].append({
                "n_feedback": n,
                "condition": "combined",
                "mean_pr_auc": float(np.mean(pr_list)),
                "std_pr_auc": float(np.std(pr_list)),
                "mean_roc_auc": float(np.mean(roc_list)),
                "mean_f1": float(np.mean(f1_list)),
            })

    return results
