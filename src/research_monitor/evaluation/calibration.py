"""
calibration.py
~~~~~~~~~~~~~~
Threshold calibration on development data (Phase 4, Section 7.2).

Selects percentile thresholds (P_high, P_low) of the reference sample distribution:
  - P_high: smallest percentile with weighted HIGH precision >= 0.80
  - P_low:  largest percentile (< P_high) with weighted LOW miss rate <= 0.10 (aiming for <= 0.05)
  - Uses DEVELOPMENT DATA ONLY. Never uses the frozen test split.
  - Reports BORDERLINE share and composition.
  - Supports persistence of calibrated levels.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research_monitor.evaluation.metrics import (
    binary_map_labels,
    borderline_composition,
    low_miss_rate,
    weighted_f1,
    weighted_precision,
    weighted_recall,
)

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("data/calibrated_thresholds.json")


@dataclass
class CalibrationResult:
    """Outcome of threshold calibration on development split."""
    p_high: float
    p_low: float
    t_high: float
    t_low: float
    high_precision: float
    high_recall: float
    high_f1: float
    low_miss_rate: float
    borderline_share: float
    borderline_composition: dict[str, float]
    target_precision_met: bool
    target_miss_rate_met: bool
    dev_sample_size: int
    notes: str
    status: str = "calibrated"  # 'provisional' or 'calibrated'
    data_split: str = "dev"  # 'dev' or 'none'


def get_provisional_calibration(
    ref_scores: np.ndarray,
    p_high: float = 98.0,
    p_low: float = 90.0,
) -> CalibrationResult:
    """
    Construct provisional thresholds (Section 7.2 provisional levels: P_high=98, P_low=90).
    Used before human ground-truth labeling and development-split calibration are completed.
    """
    t_high = float(np.percentile(ref_scores, p_high))
    t_low = float(np.percentile(ref_scores, p_low))
    return CalibrationResult(
        p_high=float(p_high),
        p_low=float(p_low),
        t_high=t_high,
        t_low=t_low,
        high_precision=0.0,
        high_recall=0.0,
        high_f1=0.0,
        low_miss_rate=0.0,
        borderline_share=0.0,
        borderline_composition={"share": 0.0, "pct_relevant": 0.0, "pct_somewhat": 0.0, "pct_not_relevant": 0.0},
        target_precision_met=False,
        target_miss_rate_met=False,
        dev_sample_size=0,
        notes="Provisional reference percentiles (P_high=98.0, P_low=90.0) prior to human labeling calibration.",
        status="provisional",
        data_split="none",
    )


def calibrate_thresholds(
    dev_df: pd.DataFrame,
    ref_scores: np.ndarray,
    score_col: str = "score",
    label_col: str = "label",
    weight_col: str = "sampling_weight",
    target_precision: float = 0.80,
    target_miss_rate: float = 0.10,
    p_high_grid: Optional[list[float]] = None,
    p_low_grid: Optional[list[float]] = None,
) -> CalibrationResult:
    """
    Calibrate (P_high, P_low) using development data only.

    Args:
        dev_df: DataFrame of development set papers with labels, scores, weights.
        ref_scores: 1D array of reference sample scores under active seed vector.
        score_col: Column with paper scores.
        label_col: Column with ground truth labels ('relevant', 'somewhat', 'not'/'not_relevant').
        weight_col: Column with sampling weights.
        target_precision: Target precision for HIGH auto-delivery (default: 0.80).
        target_miss_rate: Max acceptable fraction of relevant papers in LOW (default: 0.10, aiming for <= 0.05).
        p_high_grid: Percentiles to search for P_high (default: 80.0 to 99.8 in 0.2 steps).
        p_low_grid: Percentiles to search for P_low (default: 50.0 to 97.0 in 0.5 steps).

    Returns:
        CalibrationResult dataclass.
    """
    if len(dev_df) == 0:
        raise ValueError("dev_df cannot be empty for calibration.")

    y_true = binary_map_labels(dev_df[label_col].values, lenient=False)
    scores = dev_df[score_col].values
    weights = dev_df[weight_col].values
    labels_raw = dev_df[label_col].values

    if p_high_grid is None:
        p_high_grid = [round(x, 2) for x in np.arange(80.0, 99.81, 0.2)]
    if p_low_grid is None:
        p_low_grid = [round(x, 2) for x in np.arange(50.0, 97.01, 0.5)]

    # 1. Select P_high: smallest percentile with precision >= target_precision
    valid_highs = []
    all_high_evals = []

    for p in p_high_grid:
        t = float(np.percentile(ref_scores, p))
        high_mask = scores >= t
        prec = weighted_precision(y_true, high_mask, weights)
        rec = weighted_recall(y_true, high_mask, weights)
        all_high_evals.append((p, t, prec, rec))
        if prec >= target_precision:
            valid_highs.append((p, t, prec, rec))

    if valid_highs:
        # Smallest P_high that meets target precision
        best_p_high, best_t_high, best_prec, best_rec = min(
            valid_highs, key=lambda x: x[0]
        )
        target_prec_met = True
    else:
        # Best achievable precision
        best_p_high, best_t_high, best_prec, best_rec = max(
            all_high_evals, key=lambda x: x[2]
        )
        target_prec_met = False
        logger.warning(
            "Target precision %.2f unattainable on dev split. Best achievable: %.4f at P_high=%.1f",
            target_precision,
            best_prec,
            best_p_high,
        )

    # 2. Select P_low: largest percentile (< P_high) with miss rate <= target_miss_rate
    valid_lows = []
    all_low_evals = []

    for p in p_low_grid:
        if p >= best_p_high:
            continue
        t = float(np.percentile(ref_scores, p))
        low_mask = scores <= t
        miss = low_miss_rate(y_true, low_mask, weights)
        all_low_evals.append((p, t, miss))
        if miss <= target_miss_rate:
            valid_lows.append((p, t, miss))

    if valid_lows:
        # Largest P_low that keeps miss rate <= target
        # Prefer <= 0.05 if possible (aiming for <= 0.05 per Section 7.2)
        ambitious_lows = [x for x in valid_lows if x[2] <= 0.05]
        if ambitious_lows:
            best_p_low, best_t_low, best_miss = max(ambitious_lows, key=lambda x: x[0])
        else:
            best_p_low, best_t_low, best_miss = max(valid_lows, key=lambda x: x[0])
        target_miss_met = True
    elif all_low_evals:
        # Lowest achievable miss rate
        best_p_low, best_t_low, best_miss = min(all_low_evals, key=lambda x: x[2])
        target_miss_met = False
        logger.warning(
            "Target miss rate %.2f unattainable on dev split. Best achievable: %.4f at P_low=%.1f",
            target_miss_rate,
            best_miss,
            best_p_low,
        )
    else:
        # Fallback if no low candidates below p_high
        best_p_low = max(50.0, best_p_high - 8.0)
        best_t_low = float(np.percentile(ref_scores, best_p_low))
        low_mask = scores <= best_t_low
        best_miss = low_miss_rate(y_true, low_mask, weights)
        target_miss_met = best_miss <= target_miss_rate

    # Compute overall metrics under chosen thresholds
    high_mask = scores >= best_t_high
    low_mask = scores <= best_t_low
    borderline_mask = ~high_mask & ~low_mask

    f1 = weighted_f1(y_true, high_mask, weights)
    comp = borderline_composition(labels_raw, borderline_mask, weights)

    notes = (
        f"Calibrated on {len(dev_df)} dev labels. "
        f"P_high={best_p_high} (prec={best_prec:.3f}), P_low={best_p_low} (miss={best_miss:.3f})."
    )

    result = CalibrationResult(
        p_high=float(best_p_high),
        p_low=float(best_p_low),
        t_high=float(best_t_high),
        t_low=float(best_t_low),
        high_precision=float(best_prec),
        high_recall=float(best_rec),
        high_f1=float(f1),
        low_miss_rate=float(best_miss),
        borderline_share=float(comp["share"]),
        borderline_composition=comp,
        target_precision_met=target_prec_met,
        target_miss_rate_met=target_miss_met,
        dev_sample_size=len(dev_df),
        notes=notes,
        status="calibrated",
        data_split="dev",
    )

    logger.info("Calibration complete: %s", notes)
    return result


def sensitivity_analysis(
    dev_df: pd.DataFrame,
    ref_scores: np.ndarray,
    p_high_values: tuple[float, ...] = (95.0, 98.0, 99.0),
    p_low_values: tuple[float, ...] = (85.0, 90.0, 92.0),
    score_col: str = "score",
    label_col: str = "label",
    weight_col: str = "sampling_weight",
) -> list[dict[str, float]]:
    """
    Run sensitivity analysis across a grid of (P_high, P_low) values (Section 7.2).
    """
    y_true = binary_map_labels(dev_df[label_col].values, lenient=False)
    scores = dev_df[score_col].values
    weights = dev_df[weight_col].values

    rows = []
    for ph in p_high_values:
        th = float(np.percentile(ref_scores, ph))
        for pl in p_low_values:
            if pl >= ph:
                continue
            tl = float(np.percentile(ref_scores, pl))
            high_mask = scores >= th
            low_mask = scores <= tl
            b_mask = ~high_mask & ~low_mask

            prec = weighted_precision(y_true, high_mask, weights)
            rec = weighted_recall(y_true, high_mask, weights)
            f1 = weighted_f1(y_true, high_mask, weights)
            miss = low_miss_rate(y_true, low_mask, weights)
            b_share = float(np.sum(weights[b_mask]) / np.sum(weights))

            rows.append({
                "P_high": ph,
                "P_low": pl,
                "t_high": th,
                "t_low": tl,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "low_miss_rate": miss,
                "borderline_share": b_share,
            })
    return rows


def save_calibration_result(
    result: CalibrationResult,
    filepath: Path | str = CALIBRATION_FILE,
) -> Path:
    """Save CalibrationResult as JSON."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Saved calibration result to %s", out_path)
    return out_path


def load_calibration_result(
    filepath: Path | str = CALIBRATION_FILE,
) -> Optional[CalibrationResult]:
    """Load CalibrationResult from JSON if file exists."""
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibrationResult(**data)
