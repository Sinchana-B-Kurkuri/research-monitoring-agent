"""
reference_sample.py
~~~~~~~~~~~~~~~~~~~
Creates a deterministic reference sample from the processed corpus.

Rules:
  - Exactly REFERENCE_SAMPLE_SIZE papers (default 5,000) are selected.
  - Selection uses a fixed random seed for full reproducibility.
  - The same processed corpus + seed always produces the same paper IDs.
  - Reference papers are flagged with is_reference=True in the corpus.
  - Reference paper IDs are written to a separate text artifact.
  - Reference papers must NOT appear in the future human-labelling frame.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_reference_sample(
    df: pd.DataFrame,
    sample_size: int = 5_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select a deterministic reference sample from the corpus DataFrame.

    Args:
        df:          Full processed corpus DataFrame (must have 'paper_id').
        sample_size: Exact number of papers in the reference sample.
        seed:        Random seed for reproducibility.

    Returns:
        (corpus_df, reference_df) where:
          corpus_df    — full corpus with 'is_reference' boolean column added
          reference_df — subset DataFrame containing only reference papers

    Raises:
        ValueError: if the corpus has fewer than sample_size valid papers.
    """
    if len(df) < sample_size:
        raise ValueError(
            f"Corpus has only {len(df)} papers but reference sample requires {sample_size}. "
            "Fetch more papers or reduce REFERENCE_SAMPLE_SIZE."
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=sample_size, replace=False)
    indices_set = set(indices.tolist())

    df = df.copy()
    df["is_reference"] = [i in indices_set for i in range(len(df))]

    reference_df = df[df["is_reference"]].reset_index(drop=True)

    logger.info(
        "Reference sample: size=%d  seed=%d  is_reference column added to corpus.",
        len(reference_df), seed,
    )
    return df, reference_df


def save_reference_ids(reference_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Write the reference paper IDs to a plain-text file, one ID per line.

    This artifact is the authoritative record of which papers are in the
    reference set. It is intentionally separate from the Parquet corpus
    so it can be version-controlled independently (it's small).

    Args:
        reference_df: DataFrame of reference papers (must have 'paper_id').
        output_path:  Destination path for the ID list.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = reference_df["paper_id"].tolist()
    output_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    logger.info("Wrote %d reference IDs to %s", len(ids), output_path)


def load_reference_ids(path: str | Path) -> set[str]:
    """Load reference sample IDs from a text artifact into a set."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def get_labelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset of the corpus that is eligible for human labelling.

    Reference papers are excluded from this frame to prevent contamination
    of future threshold calibration.
    """
    return df[~df["is_reference"]].reset_index(drop=True)
