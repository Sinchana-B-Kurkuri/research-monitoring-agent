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


def score_centroid_b3(
    paper_embeddings: np.ndarray,
    centroid_vector: np.ndarray,
) -> np.ndarray:
    """
    B3 Baseline: Embedding cosine similarity to the updated interest centroid vector.

    Args:
        paper_embeddings: (N, 384) array of L2-normalized paper vectors.
        centroid_vector: (384,) updated Rocchio centroid vector.

    Returns:
        1D numpy array of cosine similarity scores.
    """
    v = centroid_vector.flatten()
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm

    scores = paper_embeddings @ v
    return scores.astype(np.float32)


def train_and_score_tfidf_logreg_b4(
    train_texts: list[str] | pd.Series,
    y_train: np.ndarray | list[int],
    eval_texts: list[str] | pd.Series,
    corpus_texts: Optional[list[str] | pd.Series] = None,
    max_features: int = 10_000,
    c_val: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    B4 Baseline: TF-IDF + Logistic Regression trained on feedback (arXiv-Sanity style).

    Args:
        train_texts: Texts of feedback training papers.
        y_train: Binary labels for training papers (1=relevant, 0=non-relevant).
        eval_texts: Texts of papers to score.
        corpus_texts: Optional background corpus to initialize vocabulary and IDF.
        max_features: TF-IDF feature cap.
        c_val: Regularization parameter.
        random_state: Random state for deterministic fitting.

    Returns:
        1D numpy array of predicted probabilities in [0.0, 1.0].
    """
    y_arr = np.asarray(y_train, dtype=np.int32)
    n_eval = len(eval_texts)
    if n_eval == 0:
        return np.empty(0, dtype=np.float32)

    # If training data has fewer than 2 classes, output prior class probability
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        prior = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        return np.full(n_eval, prior, dtype=np.float32)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
    )

    fit_texts = list(train_texts)
    if corpus_texts is not None and len(corpus_texts) > 0:
        fit_texts = list(corpus_texts) + fit_texts

    vectorizer.fit(fit_texts)
    X_train_tfidf = vectorizer.transform(train_texts)
    X_eval_tfidf = vectorizer.transform(eval_texts)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=c_val,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_train_tfidf, y_arr)

    # Return predicted probability of class 1
    if 1 in clf.classes_:
        pos_idx = int(np.where(clf.classes_ == 1)[0][0])
        probs = clf.predict_proba(X_eval_tfidf)[:, pos_idx]
    else:
        probs = np.zeros(n_eval, dtype=np.float32)

    return probs.astype(np.float32)


def train_and_score_embedding_logreg_b5(
    X_train: np.ndarray,
    y_train: np.ndarray | list[int],
    X_eval: np.ndarray,
    c_val: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    B5 Baseline: Embedding Logistic Regression trained on feedback subsets.

    Args:
        X_train: (N_train, 384) normalized embedding matrix.
        y_train: Binary labels (1=relevant, 0=non-relevant).
        X_eval: (N_eval, 384) normalized embedding matrix.
        c_val: Regularization parameter C.
        random_state: Random state for deterministic fitting.

    Returns:
        1D numpy array of predicted probabilities in [0.0, 1.0].
    """
    y_arr = np.asarray(y_train, dtype=np.int32)
    n_eval = len(X_eval)
    if n_eval == 0:
        return np.empty(0, dtype=np.float32)

    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        prior = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        return np.full(n_eval, prior, dtype=np.float32)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=c_val,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_train, y_arr)

    if 1 in clf.classes_:
        pos_idx = int(np.where(clf.classes_ == 1)[0][0])
        probs = clf.predict_proba(X_eval)[:, pos_idx]
    else:
        probs = np.zeros(n_eval, dtype=np.float32)

    return probs.astype(np.float32)


def evaluate_at_matched_volume(
    scores: np.ndarray,
    y_true: np.ndarray,
    target_volume: int,
    weights: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Evaluate ranking at matched delivered volume (Section 8.2 of Roadmap).

    The top target_volume papers with the highest scores are treated as positive
    predictions (delivered), and the rest as negative predictions.

    Args:
        scores: Continuous predicted scores or probabilities.
        y_true: Ground truth binary labels (1=relevant, 0=non-relevant).
        target_volume: Number of top-ranked papers to deliver.
        weights: Optional sampling weights per paper.

    Returns:
        dict with precision, recall, f1, and target_volume.
    """
    n = len(scores)
    if n == 0 or target_volume <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "volume": 0}

    k = min(target_volume, n)
    ranked_indices = np.argsort(-scores)  # descending order
    pred_mask = np.zeros(n, dtype=bool)
    pred_mask[ranked_indices[:k]] = True

    from research_monitor.evaluation.metrics import (
        weighted_f1,
        weighted_precision,
        weighted_recall,
    )

    if weights is None:
        weights = np.ones(n, dtype=np.float32)

    prec = weighted_precision(y_true, pred_mask, weights)
    rec = weighted_recall(y_true, pred_mask, weights)
    f1 = weighted_f1(y_true, pred_mask, weights)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "volume": int(k),
    }
