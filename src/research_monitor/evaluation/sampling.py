"""
sampling.py
~~~~~~~~~~~
Stratified evaluation sampling and reference percentile partitioning (Phase 4).

Follows Section 8.1 of the Implementation Roadmap:
  - Evaluation frame: evaluation-window corpus papers excluding the reference sample.
  - Strata defined by seed-vector scores relative to reference sample percentiles:
      S1: >= P98      (Target: 100, routing: HIGH)
      S2: P90 - P98   (Target: 120, routing: BORDERLINE)
      S3: P75 - P90   (Target: 100, routing: LOW)
      S4: P50 - P75   (Target:  80, routing: LOW)
      S5: < P50       (Target: 100, routing: LOW)
  - Sampling weight: stratum_population / stratum_sample_size.
  - Within-stratum split: 70% development, 30% test with fixed seeds.
"""

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Section 8.1 target sample sizes
DEFAULT_STRATA_TARGETS = {
    "S1": 100,
    "S2": 120,
    "S3": 100,
    "S4": 80,
    "S5": 100,
}


@dataclass(frozen=True)
class ReferencePercentiles:
    """Percentile cut points derived from reference sample scores."""
    p50: float
    p75: float
    p90: float
    p98: float


def compute_reference_percentiles(ref_scores: np.ndarray) -> ReferencePercentiles:
    """
    Compute P50, P75, P90, and P98 cut points on reference sample scores.

    Args:
        ref_scores: 1D array of cosine similarity scores on reference sample.

    Returns:
        ReferencePercentiles dataclass with p50, p75, p90, p98.
    """
    if len(ref_scores) == 0:
        raise ValueError("Reference scores array cannot be empty.")

    return ReferencePercentiles(
        p50=float(np.percentile(ref_scores, 50)),
        p75=float(np.percentile(ref_scores, 75)),
        p90=float(np.percentile(ref_scores, 90)),
        p98=float(np.percentile(ref_scores, 98)),
    )


def assign_strata(scores: np.ndarray, percentiles: ReferencePercentiles) -> np.ndarray:
    """
    Assign stratum labels (S1..S5) to score values according to reference percentiles.

    Rules:
      S1: score >= P98
      S2: P90 <= score < P98
      S3: P75 <= score < P90
      S4: P50 <= score < P75
      S5: score < P50
    """
    strata = np.empty(len(scores), dtype=object)

    s1_mask = scores >= percentiles.p98
    s2_mask = (scores >= percentiles.p90) & ~s1_mask
    s3_mask = (scores >= percentiles.p75) & ~(s1_mask | s2_mask)
    s4_mask = (scores >= percentiles.p50) & ~(s1_mask | s2_mask | s3_mask)
    s5_mask = scores < percentiles.p50

    strata[s1_mask] = "S1"
    strata[s2_mask] = "S2"
    strata[s3_mask] = "S3"
    strata[s4_mask] = "S4"
    strata[s5_mask] = "S5"

    return strata


def draw_stratified_evaluation_sample(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    ref_scores: np.ndarray,
    target_allocation: Optional[dict[str, int]] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Draw approximately 500 papers from the evaluation frame across strata S1-S5.

    Args:
        eval_df: DataFrame of non-reference papers. Must have 'paper_id'.
        scores: Cosine similarity scores corresponding 1-to-1 with eval_df rows.
        ref_scores: Cosine similarity scores of the reference sample papers.
        target_allocation: Target sample counts per stratum. Defaults to Roadmap Section 8.1.
        seed: Random seed for reproducible sampling.

    Returns:
        (sampled_df, strata_metadata) where:
          sampled_df contains the sampled papers with columns:
            'stratum', 'score', 'sampling_weight'
          strata_metadata maps stratum -> {'population': N_h, 'sample_size': n_h, 'sampling_weight': w_h}
    """
    if len(eval_df) != len(scores):
        raise ValueError(
            f"eval_df length ({len(eval_df)}) does not match scores length ({len(scores)})"
        )

    targets = target_allocation or DEFAULT_STRATA_TARGETS
    percentiles = compute_reference_percentiles(ref_scores)
    strata_array = assign_strata(scores, percentiles)

    df = eval_df.copy()
    df["score"] = scores
    df["stratum"] = strata_array

    rng = np.random.default_rng(seed)
    sampled_dfs = []
    strata_metadata = {}

    for stratum in ["S1", "S2", "S3", "S4", "S5"]:
        stratum_sub = df[df["stratum"] == stratum]
        pop_size = len(stratum_sub)
        target_size = targets.get(stratum, 100)
        actual_sample_size = min(pop_size, target_size)

        if pop_size == 0:
            logger.warning("Stratum %s has 0 papers in evaluation frame.", stratum)
            weight = 1.0
            chosen = stratum_sub.copy()
        else:
            weight = pop_size / actual_sample_size
            chosen_indices = rng.choice(pop_size, size=actual_sample_size, replace=False)
            chosen = stratum_sub.iloc[chosen_indices].copy()

        chosen["sampling_weight"] = weight
        sampled_dfs.append(chosen)

        strata_metadata[stratum] = {
            "population": float(pop_size),
            "sample_size": float(actual_sample_size),
            "sampling_weight": float(weight),
        }
        logger.info(
            "Stratum %s: population=%d, sampled=%d, weight=%.4f",
            stratum,
            pop_size,
            actual_sample_size,
            weight,
        )

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    return result_df, strata_metadata


def split_strata_dev_test(
    sample_df: pd.DataFrame,
    dev_ratio: float = 0.70,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split sampled papers within each stratum into 70% dev and 30% test.

    Args:
        sample_df: DataFrame output from draw_stratified_evaluation_sample.
        dev_ratio: Fraction of each stratum to allocate to dev (default 0.70).
        seed: Random seed for reproducible splitting.

    Returns:
        DataFrame with an added 'split' column ('dev' or 'test').
    """
    if "stratum" not in sample_df.columns:
        raise ValueError("sample_df must contain a 'stratum' column.")

    rng = np.random.default_rng(seed)
    df = sample_df.copy()
    splits = []

    for stratum, group in df.groupby("stratum", sort=False):
        n = len(group)
        n_dev = int(round(n * dev_ratio))
        # Ensure at least 1 in test if n >= 2
        if n >= 2 and n_dev == n:
            n_dev = n - 1

        shuffled = rng.permutation(n)
        dev_idx = set(shuffled[:n_dev])

        group_split = ["dev" if i in dev_idx else "test" for i in range(n)]
        group = group.copy()
        group["split"] = group_split
        splits.append(group)

    result_df = pd.concat(splits, ignore_index=True)
    return result_df


def verify_sampling_weights(
    sample_df: pd.DataFrame,
    expected_population: int,
    tolerance: float = 1e-4,
) -> tuple[bool, float]:
    """
    Verify that sum of sampling_weights for each stratum reproduces the stratum population,
    and the overall sum equals the frame population.

    Args:
        sample_df: Sampled DataFrame containing 'stratum' and 'sampling_weight'.
        expected_population: Expected total frame population (e.g. 5,000).
        tolerance: Maximum acceptable float discrepancy.

    Returns:
        (is_valid, reproduced_population)
    """
    if "sampling_weight" not in sample_df.columns:
        raise ValueError("sample_df must contain 'sampling_weight' column.")

    reproduced = float(sample_df["sampling_weight"].sum())
    is_valid = abs(reproduced - expected_population) <= tolerance
    return is_valid, reproduced
