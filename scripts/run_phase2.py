"""
scripts/run_phase2.py
~~~~~~~~~~~~~~~~~~~~~
End-to-end Phase 2 pipeline:

  1. Fetch papers from arXiv API (streaming, incremental)
  2. Preprocess: deduplicate, normalize, filter withdrawn/empty
  3. Save processed corpus to Parquet
  4. Generate L2-normalised embeddings
  5. Save embeddings as .npy
  6. Create deterministic reference sample
  7. Save reference IDs to text artifact
  8. Print data-quality report

Usage (from repo root, venv active):
    python scripts/run_phase2.py [--max-results N]

Default fetches 5,000 papers. Use --max-results to scale up.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the src/ package importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_monitor.config import config
from research_monitor.ingestion.fetcher import fetch_papers
from research_monitor.ingestion.preprocessor import preprocess, strip_version
from research_monitor.ingestion.embedder import embed_dataframe
from research_monitor.ingestion.reference_sample import (
    create_reference_sample,
    save_reference_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
REFERENCE_IDS_PATH = DATA_DIR / "reference_sample_ids.txt"


def run(max_results: int) -> None:
    categories = [c.strip() for c in config.CORPUS_CATEGORIES.split(",")]
    start_date = config.CORPUS_START_DATE
    end_date = config.CORPUS_END_DATE
    sample_size = config.REFERENCE_SAMPLE_SIZE
    seed = config.REFERENCE_SAMPLE_SEED

    report = {
        "categories": categories,
        "date_range": f"{start_date} -> {end_date}",
        "max_results_requested": max_results,
    }

    # ── 1. Fetch ────────────────────────────────────────────────────────────
    logger.info("=== Phase 2: Fetch ===")
    t_fetch_start = time.perf_counter()
    raw_stream = fetch_papers(
        categories=categories,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )

    # ── 2. Preprocess (streaming: keep flat memory footprint) ────────────────
    logger.info("=== Phase 2: Preprocess ===")
    clean_stream = preprocess(raw_stream)

    # Collect into a list so we can measure counts and save to Parquet.
    # For very large corpora this could be chunked, but for ≤50k it's fine.
    rows = list(clean_stream)
    t_fetch_end = time.perf_counter()

    report["raw_fetched"] = max_results          # upper bound; API may return less
    report["after_preprocessing"] = len(rows)
    report["fetch_preprocess_time_s"] = round(t_fetch_end - t_fetch_start, 1)

    if not rows:
        logger.error("No papers survived preprocessing. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Category counts
    cat_counts: dict[str, int] = {}
    for cat_list in df["categories"]:
        for cat in cat_list.split(","):
            cat = cat.strip()
            if cat in categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
    report["category_counts"] = cat_counts

    # Date-range summary
    report["submitted_date_min"] = df["submitted_date"].min()
    report["submitted_date_max"] = df["submitted_date"].max()

    # ── 3. Save corpus to Parquet ────────────────────────────────────────────
    logger.info("=== Phase 2: Save Parquet corpus ===")
    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(CORPUS_PATH, index=False, engine="pyarrow")
    report["corpus_parquet"] = str(CORPUS_PATH)
    report["final_corpus_size"] = len(df)
    logger.info("Corpus saved: %d papers -> %s", len(df), CORPUS_PATH)

    # ── 4. Embed ─────────────────────────────────────────────────────────────
    logger.info("=== Phase 2: Embed ===")
    _, embeddings, embed_stats = embed_dataframe(df, model_name=config.EMBEDDING_MODEL)
    report.update(embed_stats)

    # ── 5. Save embeddings ───────────────────────────────────────────────────
    np.save(EMBEDDINGS_PATH, embeddings)
    report["embeddings_path"] = str(EMBEDDINGS_PATH)
    logger.info("Embeddings saved -> %s  shape=%s", EMBEDDINGS_PATH, embeddings.shape)

    # Example norm verification
    sample_norm = float(np.linalg.norm(embeddings[0]))
    report["sample_embedding_norm"] = round(sample_norm, 6)

    # ── 6. Reference sample ───────────────────────────────────────────────────
    logger.info("=== Phase 2: Reference Sample ===")
    if len(df) < sample_size:
        logger.warning(
            "Corpus (%d) is smaller than requested reference sample (%d). "
            "Using all papers as reference; increase max_results for a proper run.",
            len(df), sample_size,
        )
        actual_size = len(df)
    else:
        actual_size = sample_size

    df, ref_df = create_reference_sample(df, sample_size=actual_size, seed=seed)

    # ── 7. Save reference IDs ─────────────────────────────────────────────────
    save_reference_ids(ref_df, REFERENCE_IDS_PATH)
    report["reference_sample_size"] = len(ref_df)
    report["reference_sample_seed"] = seed
    report["reference_ids_path"] = str(REFERENCE_IDS_PATH)

    # Re-save corpus Parquet now that it includes the is_reference column
    df.to_parquet(CORPUS_PATH, index=False, engine="pyarrow")

    # ── 8. Data-quality report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 - DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Categories:                {report['categories']}")
    print(f"  Date range:                {report['date_range']}")
    print(f"  Max results requested:     {report['max_results_requested']}")
    print(f"  After preprocessing:       {report['after_preprocessing']}")
    print(f"  Category counts:           {report['category_counts']}")
    print(f"  Submitted date min:        {report['submitted_date_min']}")
    print(f"  Submitted date max:        {report['submitted_date_max']}")
    print(f"  Final corpus size:         {report['final_corpus_size']}")
    print(f"  Fetch+preprocess time (s): {report['fetch_preprocess_time_s']}")
    print()
    print(f"  Embedding count:           {report['n_embedded']}")
    print(f"  Embedding dimension:       {report['embedding_dim']}")
    print(f"  Embedding time (s):        {report['embedding_time_s']}")
    print(f"  Throughput (papers/s):     {report['papers_per_second']}")
    print(f"  Sample embedding norm:     {report['sample_embedding_norm']}")
    print()
    print(f"  Reference sample size:     {report['reference_sample_size']}")
    print(f"  Reference sample seed:     {report['reference_sample_seed']}")
    print(f"  Reference IDs saved to:    {report['reference_ids_path']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 corpus pipeline.")
    parser.add_argument(
        "--max-results",
        type=int,
        default=5_000,
        help="Maximum papers to fetch from arXiv API (default: 5000).",
    )
    args = parser.parse_args()
    run(args.max_results)
