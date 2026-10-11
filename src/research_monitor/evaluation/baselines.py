"""
baselines.py
~~~~~~~~~~~~
Implementation of Phase 4 Baselines (Section 8.3 of Roadmap):
  - B0: Random routing baseline (uniform score at matched volume).
  - B1: Keyword/TF-IDF cosine similarity to seed-interest text.
  - B2: Sentence-transformers embedding cosine similarity to fixed seed vector.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def score_random_b0(
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """
    B0 Baseline: Generate reproducible uniform random relevance scores.

    Args:
        n_samples: Number of papers to score.
        seed: Random seed for reproducibility.

    Returns:
        1D numpy array of scores in [0.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=n_samples)


def score_tfidf_b1(
    train_corpus_texts: list[str] | pd.Series,
    eval_texts: list[str] | pd.Series,
    seed_interest_text: str,
    max_features: Optional[int] = 10_000,
) -> np.ndarray:
    """
    B1 Baseline: TF-IDF cosine similarity between papers and seed interest text.

    Args:
        train_corpus_texts: Texts (e.g. title + abstract) used to fit vocabulary/IDF.
        eval_texts: Texts of papers to be scored.
        seed_interest_text: Seed interest paragraph.
        max_features: Vocabulary size cap.

    Returns:
        1D numpy array of cosine similarity scores in [0.0, 1.0].
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
    )
    # Fit vectorizer on available corpus texts
    vectorizer.fit(train_corpus_texts)

    eval_matrix = vectorizer.transform(eval_texts)
    seed_vec = vectorizer.transform([seed_interest_text])

    # Compute cosine similarity
    similarities = cosine_similarity(eval_matrix, seed_vec).flatten()
    return similarities.astype(np.float32)


def score_embedding_b2(
    paper_embeddings: np.ndarray,
    seed_vector: np.ndarray,
) -> np.ndarray:
    """
    B2 Baseline: Embedding cosine similarity to the fixed seed vector.

    Since both paper embeddings and seed vector are L2-normalized,
    cosine similarity is exactly the inner product (dot product).

    Args:
        paper_embeddings: (N, 384) array of L2-normalized vectors.
        seed_vector: (384,) L2-normalized seed interest vector.

    Returns:
        1D numpy array of cosine similarity scores.
    """
    # Ensure 1D seed vector
    v = seed_vector.flatten()
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm

    # Inner product
    scores = paper_embeddings @ v
    return scores.astype(np.float32)
