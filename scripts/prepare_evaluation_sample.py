"""
prepare_evaluation_sample.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generates the stratified evaluation sample for Phase 4 human labeling.

Steps:
  1. Loads `data/corpus.parquet` and `data/embeddings.npy`.
  2. Embeds seed-interest text using the project sentence-transformer model.
  3. Computes reference sample scores and reference percentiles (P50, P75, P90, P98).
  4. Isolates the evaluation frame (papers with is_reference=False).
  5. Draws stratified sample targeting 500 papers across S1-S5 (Section 8.1).
  6. Calculates exact stratum populations, sample sizes, and sampling weights.
  7. Splits each stratum 70% development and 30% test (seed=42).
  8. Verifies sampling weights reproduce the frame population (5,000).
  9. Saves `data/evaluation_sample.parquet` and `data/evaluation_sample.csv`.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from research_monitor.config import config
from research_monitor.evaluation.sampling import (
    DEFAULT_STRATA_TARGETS,
    compute_reference_percentiles,
    draw_stratified_evaluation_sample,
    split_strata_dev_test,
    verify_sampling_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare stratified evaluation sample for Phase 4 labeling."
    )
    parser.add_argument(
        "--corpus-path",
        default="data/corpus.parquet",
        help="Path to corpus Parquet artifact.",
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/embeddings.npy",
        help="Path to corpus embeddings numpy artifact.",
    )
    parser.add_argument(
        "--output-path",
        default="data/evaluation_sample.parquet",
        help="Destination path for sampled evaluation papers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.EVALUATION_SEED,
        help="Random seed for reproducible sampling and splitting.",
    )
    args = parser.parse_args()

    corpus_file = Path(args.corpus_path)
    emb_file = Path(args.embeddings_path)

    if not corpus_file.exists() or not emb_file.exists():
        logger.error(
            "Missing input artifacts. Ensure %s and %s exist.",
            corpus_file,
            emb_file,
        )
        sys.exit(1)

    logger.info("Loading corpus from %s...", corpus_file)
    corpus_df = pd.read_parquet(corpus_file)

    logger.info("Loading embeddings from %s...", emb_file)
    embeddings = np.load(emb_file)

    if len(corpus_df) != len(embeddings):
        logger.error(
            "Corpus row count (%d) does not match embedding count (%d).",
            len(corpus_df),
            len(embeddings),
        )
        sys.exit(1)

    logger.info(
        "Corpus has %d papers. Checking is_reference column...", len(corpus_df)
    )
    if "is_reference" not in corpus_df.columns:
        logger.error("Corpus missing 'is_reference' column. Re-run Phase 2 reference sampling.")
        sys.exit(1)

    ref_mask = corpus_df["is_reference"].values
    ref_count = int(ref_mask.sum())
    eval_count = int((~ref_mask).sum())
    logger.info("Reference sample: %d papers. Evaluation frame: %d papers.", ref_count, eval_count)

    # 1. Embed seed interest text
    logger.info("Loading embedding model %s to encode seed interest text...", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    seed_vec = model.encode(config.SEED_INTEREST, convert_to_numpy=True)
    seed_norm = np.linalg.norm(seed_vec)
    if seed_norm > 0:
        seed_vec = seed_vec / seed_norm

    # 2. Score all papers (dot product since embeddings are L2 normalized)
    all_scores = embeddings @ seed_vec
    corpus_df["score"] = all_scores.astype(np.float32)

    ref_scores = all_scores[ref_mask]
    eval_df = corpus_df[~ref_mask].copy().reset_index(drop=True)
    eval_scores = all_scores[~ref_mask]

    # 3. Compute reference percentiles
    percentiles = compute_reference_percentiles(ref_scores)
    logger.info(
        "Reference percentiles: P50=%.4f, P75=%.4f, P90=%.4f, P98=%.4f",
        percentiles.p50,
        percentiles.p75,
        percentiles.p90,
        percentiles.p98,
    )

    # 4. Draw stratified sample
    sampled_df, strata_meta = draw_stratified_evaluation_sample(
        eval_df=eval_df,
        scores=eval_scores,
        ref_scores=ref_scores,
        target_allocation=DEFAULT_STRATA_TARGETS,
        seed=args.seed,
    )

    # 5. Split 70% dev / 30% test
    final_df = split_strata_dev_test(
        sample_df=sampled_df,
        dev_ratio=config.EVALUATION_DEV_RATIO,
        seed=args.seed,
    )

    # 6. Verify sampling weights
    is_valid_weights, rep_pop = verify_sampling_weights(
        final_df, expected_population=eval_count
    )
    if not is_valid_weights:
        logger.error(
            "Weight verification failed! Reproduced population=%.2f, expected=%d",
            rep_pop,
            eval_count,
        )
        sys.exit(1)

    logger.info(
        "Weights successfully verified: reproduced population = %.1f (expected %d).",
        rep_pop,
        eval_count,
    )

    # 7. Save artifacts
    out_parquet = Path(args.output_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_parquet, index=False)

    out_csv = out_parquet.with_suffix(".csv")
    final_df.to_csv(out_csv, index=False)

    logger.info("Saved %d evaluation papers to %s and %s", len(final_df), out_parquet, out_csv)

    # Print summary
    print("\n" + "=" * 65)
    print("  PHASE 4 EVALUATION SAMPLE SUMMARY")
    print("=" * 65)
    print(f"  Reference Percentiles: P50={percentiles.p50:.4f}, P75={percentiles.p75:.4f}, P90={percentiles.p90:.4f}, P98={percentiles.p98:.4f}")
    print(f"  Total Frame Population: {eval_count}")
    print(f"  Total Sampled:          {len(final_df)}")
    print("-" * 65)
    print(f"  {'Stratum':<8} {'Population':<12} {'Sample':<8} {'Weight':<10} {'Dev':<6} {'Test':<6}")
    print("-" * 65)
    for stratum in ["S1", "S2", "S3", "S4", "S5"]:
        sub = final_df[final_df["stratum"] == stratum]
        meta = strata_meta[stratum]
        dev_n = int((sub["split"] == "dev").sum())
        test_n = int((sub["split"] == "test").sum())
        print(
            f"  {stratum:<8} {int(meta['population']):<12} {len(sub):<8} "
            f"{meta['sampling_weight']:<10.4f} {dev_n:<6} {test_n:<6}"
        )
    print("=" * 65)
    print(f"  Total Dev papers:  {int((final_df['split'] == 'dev').sum())}")
    print(f"  Total Test papers: {int((final_df['split'] == 'test').sum())}")
    print(f"  Weight check: sum(n_h * w_h) = {rep_pop:.1f} (matches population: {is_valid_weights})")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
