"""
run_phase4_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~
Evaluates Baselines (B0, B1, B2) and calibrates threshold percentiles (P_high, P_low).

Rules & Workflow:
  1. Uses development split ONLY for threshold calibration and tuning.
  2. The test split remains frozen and is only accessed if `--evaluate-test` is explicitly passed.
  3. Every test set access is audited in `experiments/results/test_access.log`.
  4. Never fabricates labels. Reports status honestly if labels are incomplete.
  5. Computes weighted metrics: Precision, Recall, F1, PR-AUC, LOW miss rate, BORDERLINE composition.
  6. Computes stratified bootstrap confidence intervals (1,000 resamples).
  7. Saves calibrated levels to `data/calibrated_thresholds.json`.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pandas as pd

from research_monitor.config import config
from research_monitor.evaluation.access import log_test_access
from research_monitor.evaluation.baselines import (
    score_embedding_b2,
    score_random_b0,
    score_tfidf_b1,
)
from research_monitor.evaluation.calibration import (
    calibrate_thresholds,
    save_calibration_result,
    sensitivity_analysis,
)
from research_monitor.evaluation.labeling import (
    get_label_statistics,
    load_latest_labels,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_scorer(
    y_true: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    t_high: float,
    t_low: float,
    raw_labels: np.ndarray,
) -> dict[str, float]:
    """Compute all Phase 4 weighted metrics for a given scoring method."""
    high_mask = scores >= t_high
    low_mask = scores <= t_low
    b_mask = ~high_mask & ~low_mask

    prec = weighted_precision(y_true, high_mask, weights)
    rec = weighted_recall(y_true, high_mask, weights)
    f1 = weighted_f1(y_true, high_mask, weights)
    pr_auc = weighted_pr_auc(y_true, scores, weights)
    miss = low_miss_rate(y_true, low_mask, weights)
    b_comp = borderline_composition(raw_labels, b_mask, weights)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        "low_miss_rate": miss,
        "borderline_share": b_comp["share"],
        "borderline_pct_relevant": b_comp["pct_relevant"],
        "borderline_pct_somewhat": b_comp["pct_somewhat"],
        "borderline_pct_not_relevant": b_comp["pct_not_relevant"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 Baselines (B0-B2) and Threshold Calibration."
    )
    parser.add_argument(
        "--sample-path",
        default="data/evaluation_sample.parquet",
        help="Path to evaluation sample parquet artifact.",
    )
    parser.add_argument(
        "--corpus-path",
        default="data/corpus.parquet",
        help="Path to full corpus artifact (for B1 TF-IDF fitting).",
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/embeddings.npy",
        help="Path to embeddings numpy artifact.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate on the frozen test split and log access to test_access.log.",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample_path)
    if not sample_path.exists():
        csv_fallback = sample_path.with_suffix(".csv")
        if csv_fallback.exists():
            sample_df = pd.read_csv(csv_fallback)
        else:
            logger.error(
                "Evaluation sample %s not found. Run scripts/prepare_evaluation_sample.py first.",
                sample_path,
            )
            sys.exit(1)
    else:
        sample_df = pd.read_parquet(sample_path)

    corpus_file = Path(args.corpus_path)
    emb_file = Path(args.embeddings_path)
    if not corpus_file.exists() or not emb_file.exists():
        logger.error("Missing %s or %s.", corpus_file, emb_file)
        sys.exit(1)

    logger.info("Loading corpus and embeddings for baseline scoring...")
    corpus_df = pd.read_parquet(corpus_file)
    embeddings = np.load(emb_file)

    ref_mask = corpus_df["is_reference"].values
    ref_embeddings = embeddings[ref_mask]

    # Encode seed vector for B2
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    seed_vec = model.encode(config.SEED_INTEREST, convert_to_numpy=True)
    seed_vec = seed_vec / np.linalg.norm(seed_vec)

    ref_scores = score_embedding_b2(ref_embeddings, seed_vec)

    # Align evaluation sample with embeddings
    corpus_pid_to_idx = {pid: i for i, pid in enumerate(corpus_df["paper_id"].values)}
    sample_indices = [corpus_pid_to_idx[pid] for pid in sample_df["paper_id"].values]
    sample_embeddings = embeddings[sample_indices]

    # Compute baseline scores on evaluation sample
    logger.info("Computing B0 (Random), B1 (TF-IDF), B2 (Embedding Cosine) baseline scores...")
    sample_texts = (sample_df["title"].fillna("") + " " + sample_df["abstract"].fillna("")).tolist()
    corpus_texts = (corpus_df["title"].fillna("") + " " + corpus_df["abstract"].fillna("")).tolist()

    b0_scores = score_random_b0(len(sample_df), seed=config.EVALUATION_SEED)
    b1_scores = score_tfidf_b1(corpus_texts, sample_texts, config.SEED_INTEREST)
    b2_scores = score_embedding_b2(sample_embeddings, seed_vec)

    sample_df["b0_score"] = b0_scores
    sample_df["b1_score"] = b1_scores
    sample_df["b2_score"] = b2_scores

    # Check for ground truth labels
    labels_df = load_latest_labels()

    if labels_df is None or len(labels_df) == 0:
        print("\n" + "=" * 65)
        print("  PHASE 4 EVALUATION PIPELINE - STATUS: AWAITING LABELS")
        print("=" * 65)
        print("  Evaluation sample prepared: 500 papers across S1-S5.")
        print("  Baselines B0, B1, and B2 scores computed successfully.")
        print("\n  GROUND TRUTH LABELS:")
        print("  No human ground-truth labels have been recorded yet.")
        print("  Never fabricating artificial model labels as ground truth.")
        print("\n  NEXT STEP TO PROCEED WITH CALIBRATION:")
        print("  Run the labeling tool:")
        print("    .\\venv\\Scripts\\python.exe scripts\\label_papers.py")
        print("\n  Target: at least 100 relevant labels overall (at least 30 in test split).")
        print("=" * 65 + "\n")
        return

    # Merge labels with sample
    stats = get_label_statistics(labels_df)
    labeled_pids = set(labels_df["paper_id"].astype(str).tolist())
    merged = sample_df[sample_df["paper_id"].astype(str).isin(labeled_pids)].copy()
    merged = merged.merge(
        labels_df[["paper_id", "label"]], on="paper_id", how="left"
    )

    dev_df = merged[merged["split"] == "dev"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    print("\n" + "=" * 65)
    print("  PHASE 4 GROUND TRUTH PROGRESS")
    print("=" * 65)
    print(f"  Total Labeled:      {len(merged)}")
    print(f"  Dev Split Labeled:  {len(dev_df)}")
    print(f"  Test Split Labeled: {len(test_df)}")
    print(f"  Relevant Count:     {stats['relevant_count']} (Target >= 100)")
    print(f"  Test Relevant:      {stats['test_relevant_count']} (Target >= 30)")
    print("-" * 65)

    if len(dev_df) < 10 or stats["relevant_count"] == 0:
        print("  Insufficient dev labels to perform threshold calibration.")
        print("  Please label more papers using: python scripts/label_papers.py")
        print("=" * 65 + "\n")
        return

    # 1. Calibrate on DEV split only
    logger.info("Calibrating thresholds on DEVELOPMENT split (P_high, P_low)...")
    calib_result = calibrate_thresholds(
        dev_df=dev_df,
        ref_scores=ref_scores,
        score_col="b2_score",
        label_col="label",
        weight_col="sampling_weight",
        target_precision=config.CALIBRATION_TARGET_PRECISION,
        target_miss_rate=config.CALIBRATION_TARGET_MISS_RATE,
    )
    save_calibration_result(calib_result)

    print("\n" + "=" * 65)
    print("  DEVELOPMENT SPLIT THRESHOLD CALIBRATION RESULTS")
    print("=" * 65)
    print(f"  Calibrated P_high:   {calib_result.p_high:.1f}th percentile (score t_high = {calib_result.t_high:.4f})")
    print(f"  Calibrated P_low:    {calib_result.p_low:.1f}th percentile (score t_low = {calib_result.t_low:.4f})")
    print(f"  HIGH Precision:      {calib_result.high_precision:.4f} (Target >= {config.CALIBRATION_TARGET_PRECISION:.2f}: {calib_result.target_precision_met})")
    print(f"  HIGH Recall:         {calib_result.high_recall:.4f}")
    print(f"  HIGH F1:             {calib_result.high_f1:.4f}")
    print(f"  LOW Miss Rate:       {calib_result.low_miss_rate:.4f} (Target <= {config.CALIBRATION_TARGET_MISS_RATE:.2f}: {calib_result.target_miss_rate_met})")
    print(f"  BORDERLINE Share:    {calib_result.borderline_share * 100:.1f}%")
    b_comp = calib_result.borderline_composition
    print(f"  BORDERLINE Mix:      {b_comp['pct_relevant']*100:.1f}% rel, {b_comp['pct_somewhat']*100:.1f}% some, {b_comp['pct_not_relevant']*100:.1f}% not")
    print("-" * 65)
    print("  Sensitivity Analysis (Section 7.2):")
    sens = sensitivity_analysis(dev_df, ref_scores, score_col="b2_score")
    for s in sens[:4]:
        print(f"    P_high={s['P_high']:.0f}, P_low={s['P_low']:.0f} -> Prec={s['precision']:.3f}, Miss={s['low_miss_rate']:.3f}, B-Share={s['borderline_share']*100:.1f}%")
    print("=" * 65)

    # 2. Test split evaluation (only if explicitly requested)
    if not args.evaluate_test:
        print("\n[NOTE] Test split remains FROZEN. Pass '--evaluate-test' for final test reporting.")
        return

    # Log test access
    log_path = log_test_access(
        caller="run_phase4_evaluation.py",
        purpose="Final baseline and calibrated threshold evaluation on frozen test split",
        sample_count=len(test_df),
    )
    logger.info("Logged test split access to %s.", log_path)

    y_test = binary_map_labels(test_df["label"].values)
    w_test = test_df["sampling_weight"].values
    raw_test_labels = test_df["label"].values

    test_b0 = evaluate_scorer(y_test, test_df["b0_score"].values, w_test, calib_result.t_high, calib_result.t_low, raw_test_labels)
    test_b1 = evaluate_scorer(y_test, test_df["b1_score"].values, w_test, calib_result.t_high, calib_result.t_low, raw_test_labels)
    test_b2 = evaluate_scorer(y_test, test_df["b2_score"].values, w_test, calib_result.t_high, calib_result.t_low, raw_test_labels)

    # Bootstrap 95% CI on B2 PR-AUC and F1
    def b2_f1_fn(df_resample: pd.DataFrame) -> float:
        y_r = binary_map_labels(df_resample["label"].values)
        w_r = df_resample["sampling_weight"].values
        mask_r = df_resample["b2_score"].values >= calib_result.t_high
        return weighted_f1(y_r, mask_r, w_r)

    def b2_prauc_fn(df_resample: pd.DataFrame) -> float:
        y_r = binary_map_labels(df_resample["label"].values)
        w_r = df_resample["sampling_weight"].values
        return weighted_pr_auc(y_r, df_resample["b2_score"].values, w_r)

    f1_ci = stratified_bootstrap_ci(test_df, b2_f1_fn, n_resamples=1000, seed=config.EVALUATION_SEED)
    prauc_ci = stratified_bootstrap_ci(test_df, b2_prauc_fn, n_resamples=1000, seed=config.EVALUATION_SEED)

    print("\n" + "=" * 65)
    print("  FROZEN TEST SPLIT - FINAL BASELINE RESULTS (E1)")
    print("=" * 65)
    print(f"  {'Metric':<20} {'B0 (Random)':<14} {'B1 (TF-IDF)':<14} {'B2 (Embedding)'}")
    print("-" * 65)
    print(f"  {'PR-AUC':<20} {test_b0['pr_auc']:<14.4f} {test_b1['pr_auc']:<14.4f} {test_b2['pr_auc']:.4f} (95% CI: [{prauc_ci[0]:.3f}, {prauc_ci[1]:.3f}])")
    print(f"  {'HIGH Precision':<20} {test_b0['precision']:<14.4f} {test_b1['precision']:<14.4f} {test_b2['precision']:.4f}")
    print(f"  {'HIGH Recall':<20} {test_b0['recall']:<14.4f} {test_b1['recall']:<14.4f} {test_b2['recall']:.4f}")
    print(f"  {'HIGH F1':<20} {test_b0['f1']:<14.4f} {test_b1['f1']:<14.4f} {test_b2['f1']:.4f} (95% CI: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}])")
    print(f"  {'LOW Miss Rate':<20} {test_b0['low_miss_rate']:<14.4f} {test_b1['low_miss_rate']:<14.4f} {test_b2['low_miss_rate']:.4f}")
    print(f"  {'BORDERLINE Share':<20} {test_b0['borderline_share']*100:<13.1f}% {test_b1['borderline_share']*100:<13.1f}% {test_b2['borderline_share']*100:.1f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
