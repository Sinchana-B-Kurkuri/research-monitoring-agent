"""
embedder.py
~~~~~~~~~~~
Batch-generates L2-normalised embeddings for paper text.

Reuses the same embedding function signature established in Phase 1
(prototype.py:get_embeddings) so Phase 1 tests remain valid.

New in Phase 2:
  - accepts a batch_size parameter for incremental processing
  - records and returns embedding-generation timing
  - works directly on a pandas DataFrame
"""

import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Module-level model cache: load once per process, reuse across calls.
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info("Loading SentenceTransformer model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2-D array of embeddings row-wise.

    Rows with zero norm are left as-is (no division by zero).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of text strings and return L2-normalised float32 embeddings.

    This is the canonical embedding function for the project.
    Phase 1's prototype.get_embeddings() delegates here in spirit;
    tests that call prototype.get_embeddings() still work unchanged.

    Args:
        texts:         List of strings to embed.
        model_name:    SentenceTransformer model identifier.
        batch_size:    Encoding batch size.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32, L2-normalised.
    """
    model = _get_model(model_name)
    raw = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return l2_normalize(raw)


def embed_dataframe(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    text_col: Optional[str] = None,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Add embeddings to a corpus DataFrame and return timing statistics.

    The input DataFrame must have 'title' and 'abstract' columns.
    Embeddings are computed for `title + " " + abstract`.

    Args:
        df:          Corpus DataFrame.
        model_name:  SentenceTransformer model name.
        batch_size:  Batch size for encoding.
        text_col:    Optional pre-computed text column name; if None, one is built.

    Returns:
        (df, embeddings, stats)  where:
          df          — original DataFrame (unchanged)
          embeddings  — np.ndarray (N, dim), L2-normalised
          stats       — dict with timing and dimension info
    """
    if text_col and text_col in df.columns:
        texts = df[text_col].tolist()
    else:
        texts = (df["title"] + " " + df["abstract"]).tolist()

    logger.info("Embedding %d texts with model '%s' (batch_size=%d)…",
                len(texts), model_name, batch_size)

    t0 = time.perf_counter()
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size,
                             show_progress=True)
    elapsed = time.perf_counter() - t0

    stats = {
        "n_embedded": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "embedding_time_s": round(elapsed, 2),
        "papers_per_second": round(len(embeddings) / elapsed, 1) if elapsed > 0 else 0,
    }
    logger.info(
        "Embedding done in %.1fs  (%.1f papers/s)  dim=%d",
        elapsed, stats["papers_per_second"], stats["embedding_dim"],
    )
    return df, embeddings, stats
