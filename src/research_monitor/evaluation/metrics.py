"""
metrics.py
~~~~~~~~~~
Weighted evaluation metrics and stratified bootstrap confidence intervals (Phase 4).

Follows Roadmap Section 8.1 and 8.2:
  - Weighted precision, recall, F1
  - Weighted PR-AUC (Average Precision)
  - LOW miss rate (fraction of relevant papers routed to LOW)
  - BORDERLINE composition and share
  - Stratified bootstrap confidence intervals (1,000 resamples within strata)
"""

from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def binary_map_labels(
    labels: list[str] | pd.Series | np.ndarray,
    lenient: bool = False,
) -> np.ndarray:
    """
    Map categorical labels ('relevant', 'somewhat', 'not_relevant') to binary 0/1.

    Args:
        labels: Array of labels.
        lenient: If False (primary mapping), relevant=1, somewhat/not_relevant=0.
                 If True (lenient sensitivity), relevant+somewhat=1, not_relevant=0.

    Returns:
        1D numpy array of 0 and 1 ints.
    """
    s = pd.Series(labels).astype(str).str.lower().str.strip()
    if not lenient:
        return (s == "relevant").astype(int).values
    else:
        return (s.isin(["relevant", "somewhat"])).astype(int).values


def weighted_precision(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted precision of selected papers.

    Precision_w = sum_{i in selected} (w_i * y_i) / sum_{i in selected} w_i
    """
    sel_w = weights[selected_mask]
    if len(sel_w) == 0 or sel_w.sum() == 0:
        return 0.0
    return float(np.sum(sel_w * y_true[selected_mask]) / np.sum(sel_w))


def weighted_recall(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted recall of selected papers against all relevant papers in population.

    Recall_w = sum_{i in selected} (w_i * y_i) / sum_{all} (w_i * y_i)
    """
    total_rel_w = np.sum(weights * y_true)
    if total_rel_w == 0:
        return 0.0
    return float(np.sum(weights[selected_mask] * y_true[selected_mask]) / total_rel_w)


def weighted_f1(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted F1 score: 2 * Prec_w * Rec_w / (Prec_w + Rec_w).
    """
    prec = weighted_precision(y_true, selected_mask, weights)
    rec = weighted_recall(y_true, selected_mask, weights)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def weighted_pr_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted PR-AUC (Average Precision score) with sample weights.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return float(average_precision_score(y_true, scores, sample_weight=weights))
    except Exception:
        return 0.0


def low_miss_rate(
    y_true: np.ndarray,
    low_mask: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Fraction of relevant papers that are routed to LOW.

    MissRate_w = sum_{i in LOW} (w_i * y_i) / sum_{all} (w_i * y_i)
    """
    total_rel_w = np.sum(weights * y_true)
    if total_rel_w == 0:
        return 0.0
    return float(np.sum(weights[low_mask] * y_true[low_mask]) / total_rel_w)


def borderline_composition(
    labels: np.ndarray | pd.Series,
    borderline_mask: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """
    Report the breakdown and share of papers routed to BORDERLINE.

    Returns:
        Dict with keys:
          'share': weighted share of all papers routed to BORDERLINE
          'pct_relevant': weighted % of borderline papers that are relevant
          'pct_somewhat': weighted % of borderline papers that are somewhat relevant
          'pct_not_relevant': weighted % of borderline papers that are not relevant
    """
    total_w = np.sum(weights)
    if total_w == 0:
        return {"share": 0.0, "pct_relevant": 0.0, "pct_somewhat": 0.0, "pct_not_relevant": 0.0}

    b_weights = weights[borderline_mask]
    b_total_w = np.sum(b_weights)
    share = float(b_total_w / total_w)

    if b_total_w == 0:
        return {"share": 0.0, "pct_relevant": 0.0, "pct_somewhat": 0.0, "pct_not_relevant": 0.0}

    b_labels = pd.Series(labels)[borderline_mask].astype(str).str.lower().str.strip().values
    rel_w = float(np.sum(b_weights[b_labels == "relevant"]))
    some_w = float(np.sum(b_weights[b_labels == "somewhat"]))
    not_w = float(np.sum(b_weights[np.isin(b_labels, ["not_relevant", "not"])]))

    return {
        "share": share,
        "pct_relevant": float(rel_w / b_total_w),
        "pct_somewhat": float(some_w / b_total_w),
        "pct_not_relevant": float(not_w / b_total_w),
    }


def stratified_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    strata_col: str = "stratum",
    n_resamples: int = 1_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute stratified bootstrap confidence intervals (Section 8.2).

    Resamples with replacement independently within each stratum,
    evaluating metric_fn on each resampled DataFrame.

    Args:
        df: DataFrame containing samples, strata, weights, labels, and scores.
        metric_fn: Callable taking a DataFrame and returning a scalar float metric.
        strata_col: Column identifying strata (default: 'stratum').
        n_resamples: Number of bootstrap iterations (default: 1,000).
        ci: Confidence level (default: 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (ci_lower, ci_upper)
    """
    if len(df) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    bootstrap_estimates = np.empty(n_resamples, dtype=float)

    # Pre-index strata
    strata_groups = {s: g for s, g in df.groupby(strata_col)}

    for i in range(n_resamples):
        resampled_subsets = []
        for stratum, group in strata_groups.items():
            n = len(group)
            if n > 0:
                idx = rng.choice(n, size=n, replace=True)
                resampled_subsets.append(group.iloc[idx])

        resampled_df = pd.concat(resampled_subsets, ignore_index=True)
        try:
            val = metric_fn(resampled_df)
        except Exception:
            val = 0.0
        bootstrap_estimates[i] = val

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1.0 - alpha)))

    return lower, upper
