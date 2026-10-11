"""
thresholds.py
~~~~~~~~~~~~~
Reference percentile threshold computation for user interest vectors (Phase 5).

Follows Roadmap Section 7.2:
  - For the active vector, score the 5,000 reference papers.
  - t_high is the P_high percentile (default 98.0 provisional).
  - t_low is the P_low percentile (default 90.0 provisional).
  - Thresholds are recomputed automatically whenever the interest vector changes.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.config import config
from research_monitor.database.models import Paper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    p_high: float
    p_low: float
    t_high: float
    t_low: float

    def to_dict(self) -> dict[str, float]:
        return {
            "p_high": float(self.p_high),
            "p_low": float(self.p_low),
            "t_high": float(self.t_high),
            "t_low": float(self.t_low),
        }


def load_reference_embeddings(
    session: Optional[Session] = None,
    corpus_path: Path | str = "data/corpus.parquet",
    embeddings_path: Path | str = "data/embeddings.npy",
) -> np.ndarray:
    """
    Load the 5,000 reference paper embeddings.
    First tries the database Paper records where is_reference == True.
    If database is not populated or offline, falls back to the Parquet + NumPy artifacts.

    Returns:
        2D numpy array of shape (5000, 384), L2-normalized.
    """
    if session is not None:
        try:
            ref_papers = (
                session.query(Paper.embedding)
                .filter(Paper.is_reference.is_(True))
                .all()
            )
            if len(ref_papers) >= config.REFERENCE_SAMPLE_SIZE:
                logger.info(
                    "Loaded %d reference embeddings from PostgreSQL database.",
                    len(ref_papers),
                )
                embeddings = np.array([p[0] for p in ref_papers], dtype=np.float32)
                # Ensure L2 normalization
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return embeddings / norms
        except Exception as e:
            logger.warning(
                "Could not load reference embeddings from DB (%s); falling back to file artifacts.",
                e,
            )

    # Fallback to local artifacts
    c_path = Path(corpus_path)
    e_path = Path(embeddings_path)
    if not c_path.exists() or not e_path.exists():
        raise FileNotFoundError(
            f"Cannot load reference embeddings: missing {c_path} or {e_path}."
        )

    corpus_df = pd.read_parquet(c_path)
    all_embeddings = np.load(e_path)
    ref_mask = corpus_df["is_reference"].values
    ref_embs = all_embeddings[ref_mask]

    norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ref_embs / norms


def compute_vector_thresholds(
    vector: np.ndarray,
    reference_embeddings: Optional[np.ndarray] = None,
    p_high: Optional[float] = None,
    p_low: Optional[float] = None,
    session: Optional[Session] = None,
) -> Thresholds:
    """
    Score the 5,000 reference papers under an interest vector and compute (t_high, t_low).

    Args:
        vector: 1D normalized interest vector (length 384).
        reference_embeddings: Optional pre-loaded reference embeddings matrix.
        p_high: Percentile level for HIGH route (defaults to config.active_p_high).
        p_low: Percentile level for LOW route (defaults to config.active_p_low).
        session: Optional SQLAlchemy DB session for reference lookup.

    Returns:
        Thresholds dataclass.
    """
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Interest vector has zero norm.")
    v = v / norm

    ref_embs = (
        reference_embeddings
        if reference_embeddings is not None
        else load_reference_embeddings(session=session)
    )

    # Cosine similarity is dot product because vectors are L2-normalized
    ref_scores = np.dot(ref_embs, v)

    p_h = p_high if p_high is not None else config.active_p_high
    p_l = p_low if p_low is not None else config.active_p_low

    t_h = float(np.percentile(ref_scores, p_h))
    t_l = float(np.percentile(ref_scores, p_l))

    logger.info(
        "Computed vector thresholds: P_high=%.1f -> t_high=%.4f, P_low=%.1f -> t_low=%.4f (ref_count=%d)",
        p_h,
        t_h,
        p_l,
        t_l,
        len(ref_scores),
    )

    return Thresholds(
        p_high=float(p_h),
        p_low=float(p_l),
        t_high=t_h,
        t_low=t_l,
    )
