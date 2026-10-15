"""
classifier.py
~~~~~~~~~~~~~
Phase 11 — Logistic Regression Preference Model & Promotion Guard.

Follows Roadmap Section 6, 7.5, and 7.8:
  - Trains scikit-learn LogisticRegression on normalized 384-dimensional paper embeddings:
      * L2 regularization, class_weight='balanced'.
      * Hyperparameter C selected from {0.1, 0.3, 1.0} using inner cross-validation.
  - Eligibility requirements:
      * At least 30 effective labels (50 preferred), with at least 8 positive and 8 negative labels.
      * Retrain after every 25 new effective labels since previous model's trained_on_max_event_id.
  - Repeated stratified 5-fold cross-validation (out-of-fold predictions):
      * Strict leakage prevention: preprocessing and C-selection never use held-out fold data.
      * Evaluates classifier OOF ROC-AUC against cosine-to-current-centroid baseline on the exact same folds.
  - Promotion Guard:
      * Promotes only if oof_auc >= 0.65 AND oof_auc >= baseline_auc + 0.02.
      * If guard fails, model is stored as promoted=False and is_active=False; existing routing method remains active.
  - Threshold Calibration:
      * Starts with provisional p_high=0.70, p_low=0.30.
      * Calibrates with Roadmap Section 7.2 objective using dev labels or out-of-fold predictions.
  - Nearest-liked and nearest-rejected paper explanations for the review UI.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import pickle
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    InterestVectorVersion,
    ModelVersion,
    Paper,
    TrainingLabel,
    User,
    ValidationLabel,
)
from research_monitor.relevance.preference import get_user_seed_embedding
from research_monitor.relevance.user_manager import get_active_vector_version
from research_monitor.review.labels import get_effective_training_labels

logger = logging.getLogger(__name__)

# Phase 11 Roadmap Constants (Section 7.5)
MIN_EFFECTIVE_LABELS = 30
PREFERRED_EFFECTIVE_LABELS = 50
MIN_POS_LABELS = 8
MIN_NEG_LABELS = 8
RETRAIN_INTERVAL = 25
C_CANDIDATES = (0.1, 0.3, 1.0)
MIN_PROMOTION_AUC = 0.65
PROMOTION_AUC_MARGIN = 0.02

DEFAULT_PROVISIONAL_P_HIGH = 0.70
DEFAULT_PROVISIONAL_P_LOW = 0.30


@dataclass(frozen=True)
class ClassifierEligibility:
    """Eligibility telemetry report for preference classifier training."""
    user_id: str
    eligible: bool
    reason: str
    total_effective_labels: int
    positive_labels: int
    negative_labels: int
    new_effective_labels_since_cutoff: int
    current_model_version: Optional[int]
    cutoff_event_id: Optional[int]


@dataclass
class ClassifierTrainingResult:
    """Outcome report for a classifier training attempt."""
    user_id: str
    eligible: bool
    status: str  # 'promoted', 'not_promoted', 'skipped', 'error'
    reason: str
    version_created: Optional[int] = None
    model_version_id: Optional[int] = None
    promoted: bool = False
    oof_auc: float = 0.0
    oof_auc_cosine_baseline: float = 0.0
    best_C: float = 1.0
    n_samples: int = 0
    n_pos: int = 0
    n_neg: int = 0
    p_high: float = DEFAULT_PROVISIONAL_P_HIGH
    p_low: float = DEFAULT_PROVISIONAL_P_LOW
    threshold_calibration_source: str = "provisional"
    trained_on_max_event_id: Optional[int] = None
    is_active: bool = False


@dataclass
class DecisionExplanation:
    """Human-interpretable nearest-labeled paper explanation for routing decisions."""
    paper_id: str
    method: str  # 'classifier' or 'threshold'
    score_or_prob: float
    cosine_score: float
    thresholds: dict[str, float]
    model_version_id: Optional[int] = None
    vector_version_id: Optional[int] = None
    nearest_liked: Optional[dict] = None
    nearest_rejected: Optional[dict] = None


def check_classifier_eligibility(
    user_id: str,
    session: Session,
    effective_labels: Optional[Sequence[TrainingLabel]] = None,
) -> ClassifierEligibility:
    """
    Evaluate eligibility for classifier training per Roadmap Section 7.5:
      - Initial training requires >= 30 effective labels, >= 8 positive, >= 8 negative.
      - Retraining requires >= 25 new effective labels since previous model's trained_on_max_event_id.
    """
    labels = (
        effective_labels
        if effective_labels is not None
        else get_effective_training_labels(session=session, user_id=user_id)
    )

    total_effective = len(labels)
    pos_count = sum(1 for l in labels if l.label == 1)
    neg_count = sum(1 for l in labels if l.label == 0)

    latest_model = (
        session.query(ModelVersion)
        .filter_by(user_id=user_id)
        .order_by(ModelVersion.version.desc())
        .first()
    )

    cutoff_id = latest_model.trained_on_max_event_id if latest_model else None
    current_ver = latest_model.version if latest_model else None

    # Initial training check
    if latest_model is None:
        if total_effective < MIN_EFFECTIVE_LABELS:
            return ClassifierEligibility(
                user_id=user_id,
                eligible=False,
                reason=(
                    f"Insufficient effective labels for initial classifier training: "
                    f"{total_effective}/{MIN_EFFECTIVE_LABELS} required."
                ),
                total_effective_labels=total_effective,
                positive_labels=pos_count,
                negative_labels=neg_count,
                new_effective_labels_since_cutoff=total_effective,
                current_model_version=None,
                cutoff_event_id=None,
            )

        if pos_count < MIN_POS_LABELS:
            return ClassifierEligibility(
                user_id=user_id,
                eligible=False,
                reason=(
                    f"Insufficient positive labels for classifier training: "
                    f"{pos_count}/{MIN_POS_LABELS} required."
                ),
                total_effective_labels=total_effective,
                positive_labels=pos_count,
                negative_labels=neg_count,
                new_effective_labels_since_cutoff=total_effective,
                current_model_version=None,
                cutoff_event_id=None,
            )

        if neg_count < MIN_NEG_LABELS:
            return ClassifierEligibility(
                user_id=user_id,
                eligible=False,
                reason=(
                    f"Insufficient negative labels for classifier training: "
                    f"{neg_count}/{MIN_NEG_LABELS} required."
                ),
                total_effective_labels=total_effective,
                positive_labels=pos_count,
                negative_labels=neg_count,
                new_effective_labels_since_cutoff=total_effective,
                current_model_version=None,
                cutoff_event_id=None,
            )

        return ClassifierEligibility(
            user_id=user_id,
            eligible=True,
            reason=(
                f"Eligible for initial classifier training ({total_effective} labels, "
                f"{pos_count} pos, {neg_count} neg)."
            ),
            total_effective_labels=total_effective,
            positive_labels=pos_count,
            negative_labels=neg_count,
            new_effective_labels_since_cutoff=total_effective,
            current_model_version=None,
            cutoff_event_id=None,
        )

    # Retraining check
    new_labels = [l for l in labels if cutoff_id is None or l.event_id > cutoff_id]
    new_count = len(new_labels)

    if new_count < RETRAIN_INTERVAL:
        return ClassifierEligibility(
            user_id=user_id,
            eligible=False,
            reason=(
                f"Insufficient new effective labels since model v{current_ver}: "
                f"{new_count}/{RETRAIN_INTERVAL} required (cutoff event {cutoff_id})."
            ),
            total_effective_labels=total_effective,
            positive_labels=pos_count,
            negative_labels=neg_count,
            new_effective_labels_since_cutoff=new_count,
            current_model_version=current_ver,
            cutoff_event_id=cutoff_id,
        )

    if pos_count < MIN_POS_LABELS or neg_count < MIN_NEG_LABELS:
        return ClassifierEligibility(
            user_id=user_id,
            eligible=False,
            reason=f"Insufficient class balance for retraining: {pos_count} pos, {neg_count} neg.",
            total_effective_labels=total_effective,
            positive_labels=pos_count,
            negative_labels=neg_count,
            new_effective_labels_since_cutoff=new_count,
            current_model_version=current_ver,
            cutoff_event_id=cutoff_id,
        )

    return ClassifierEligibility(
        user_id=user_id,
        eligible=True,
        reason=(
            f"Eligible for retraining (v{current_ver + 1}) with "
            f"{new_count} new effective labels since cutoff event {cutoff_id}."
        ),
        total_effective_labels=total_effective,
        positive_labels=pos_count,
        negative_labels=neg_count,
        new_effective_labels_since_cutoff=new_count,
        current_model_version=current_ver,
        cutoff_event_id=cutoff_id,
    )


def _select_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_candidates: Sequence[float] = C_CANDIDATES,
    random_state: int = 42,
) -> float:
    """
    Select hyperparameter C from candidates using inner 3-fold stratified cross-validation.
    Ensures zero data leakage from outer evaluation fold.
    """
    pos_tr = np.sum(y_train == 1)
    neg_tr = np.sum(y_train == 0)
    n_splits = min(3, pos_tr, neg_tr)

    if n_splits < 2:
        return 1.0

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_c = 1.0
    best_score = -1.0

    for c_val in c_candidates:
        fold_scores = []
        for in_tr_idx, in_val_idx in inner_cv.split(X_train, y_train):
            X_in_tr, y_in_tr = X_train[in_tr_idx], y_train[in_tr_idx]
            X_in_val, y_in_val = X_train[in_val_idx], y_train[in_val_idx]

            if len(np.unique(y_in_tr)) < 2 or len(np.unique(y_in_val)) < 2:
                continue

            model = LogisticRegression(
                C=c_val,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state,
            )
            model.fit(X_in_tr, y_in_tr)
            probs = model.predict_proba(X_in_val)[:, 1]
            try:
                score = roc_auc_score(y_in_val, probs)
                fold_scores.append(score)
            except Exception:
                pass

        avg_score = float(np.mean(fold_scores)) if fold_scores else 0.5
        if avg_score > best_score:
            best_score = avg_score
            best_c = c_val

    return best_c


def _evaluate_repeated_stratified_oof(
    X: np.ndarray,
    y: np.ndarray,
    active_centroid: np.ndarray,
    c_candidates: Sequence[float] = C_CANDIDATES,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> tuple[float, float, np.ndarray, float]:
    """
    Repeated stratified 5-fold cross-validation with out-of-fold probability estimation.
    Compares classifier OOF predictions against cosine-to-current-centroid baseline on the exact same folds.

    Returns:
        (clf_oof_auc, baseline_oof_auc, oof_probs, best_overall_C)
    """
    n_samples = len(y)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    oof_prob_matrix = np.zeros((n_samples, n_repeats), dtype=np.float32)
    fold_occurrences = np.zeros(n_samples, dtype=np.int32)

    split_count = 0
    for train_idx, val_idx in rskf.split(X, y):
        repeat_idx = split_count // n_splits
        split_count += 1

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Inner CV on training fold only (prevent data leakage)
        best_c_fold = _select_best_C(X_tr, y_tr, c_candidates, random_state=random_state + repeat_idx)

        model = LogisticRegression(
            C=best_c_fold,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state + repeat_idx,
        )
        model.fit(X_tr, y_tr)
        probs_val = model.predict_proba(X_val)[:, 1]

        oof_prob_matrix[val_idx, repeat_idx] = probs_val
        fold_occurrences[val_idx] += 1

    # Mean out-of-fold probability per sample across repeats
    oof_probs = np.mean(oof_prob_matrix, axis=1)

    # Cosine baseline on identical embeddings
    centroid = active_centroid / np.linalg.norm(active_centroid)
    cosine_scores = X @ centroid

    clf_auc = float(roc_auc_score(y, oof_probs))
    base_auc = float(roc_auc_score(y, cosine_scores))

    # Overall best C chosen across all data for final deployment
    best_overall_c = _select_best_C(X, y, c_candidates, random_state=random_state)

    return clf_auc, base_auc, oof_probs, best_overall_c


def calibrate_classifier_thresholds(
    oof_probs: np.ndarray,
    y_true: np.ndarray,
    dev_probs: Optional[np.ndarray] = None,
    dev_y: Optional[np.ndarray] = None,
) -> tuple[float, float, str]:
    """
    Calibrate p_high and p_low following Roadmap Section 7.2 & 7.5:
      - Objective:
          * Smallest p_high whose HIGH precision >= 0.80.
          * Largest p_low whose LOW miss rate <= 0.10 (aiming for <= 0.05).
          * Require p_low < p_high.
      - Uses development labels when available; otherwise out-of-fold predictions.
      - Falls back safely to provisional (0.70, 0.30) if targets are unattainable.
    """
    use_dev = dev_probs is not None and dev_y is not None and len(dev_y) >= 20 and np.sum(dev_y == 1) >= 5
    probs = dev_probs if use_dev else oof_probs
    y = dev_y if use_dev else y_true
    source_label = "dev_labels" if use_dev else "oof_predictions"

    total_pos = np.sum(y == 1)
    if total_pos < 3:
        return DEFAULT_PROVISIONAL_P_HIGH, DEFAULT_PROVISIONAL_P_LOW, "provisional"

    # Search candidate threshold pairs
    p_high_candidates = np.arange(0.50, 0.95, 0.02)
    p_low_candidates = np.arange(0.05, 0.50, 0.02)

    best_p_high: Optional[float] = None
    best_p_low: Optional[float] = None

    # Find smallest p_high with precision >= 0.80
    for ph in p_high_candidates:
        mask_high = probs >= ph
        if np.sum(mask_high) >= 2:
            prec = np.sum(y[mask_high] == 1) / np.sum(mask_high)
            if prec >= 0.80:
                best_p_high = float(ph)
                break

    # Find largest p_low with miss rate <= 0.10
    for pl in reversed(p_low_candidates):
        mask_low = probs <= pl
        miss_rate = np.sum(y[mask_low] == 1) / total_pos
        if miss_rate <= 0.10:
            best_p_low = float(pl)
            break

    if best_p_high is not None and best_p_low is not None and best_p_low < best_p_high:
        return best_p_high, best_p_low, source_label

    return DEFAULT_PROVISIONAL_P_HIGH, DEFAULT_PROVISIONAL_P_LOW, "provisional"


def train_and_evaluate_classifier(
    user_id: str,
    session: Session,
    random_state: int = 42,
) -> ClassifierTrainingResult:
    """
    Train, evaluate, and conditionally promote a logistic regression preference classifier (Phase 11).

    Parameters:
        user_id: Target user identifier.
        session: Active SQLAlchemy database session.
        random_state: Seed for cross-validation and optimization reproducibility.

    Returns:
        ClassifierTrainingResult reporting metrics, promotion decision, and active model status.
    """
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"User '{user_id}' does not exist in database.",
        )

    # 1. Query user-specific effective labels (never mixed across users)
    effective_labels = get_effective_training_labels(session=session, user_id=user_id)

    # 2. Check trigger eligibility
    eligibility = check_classifier_eligibility(
        user_id=user_id,
        session=session,
        effective_labels=effective_labels,
    )

    if not eligibility.eligible:
        logger.info("Classifier training skipped for user '%s': %s", user_id, eligibility.reason)
        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=False,
            status="skipped",
            reason=eligibility.reason,
            n_samples=eligibility.total_effective_labels,
            n_pos=eligibility.positive_labels,
            n_neg=eligibility.negative_labels,
        )

    # 3. Retrieve valid 384-dimensional paper embeddings
    paper_ids = [l.paper_id for l in effective_labels]
    paper_rows = (
        session.query(Paper.id, Paper.embedding)
        .filter(Paper.id.in_(paper_ids))
        .all()
    )
    paper_emb_map = {row[0]: row[1] for row in paper_rows if row[1] is not None}

    X_list = []
    y_list = []
    for l in effective_labels:
        raw_emb = paper_emb_map.get(l.paper_id)
        if raw_emb is None:
            continue
        arr = np.asarray(raw_emb, dtype=np.float32).ravel()
        if arr.size == 384 and np.all(np.isfinite(arr)):
            norm = np.linalg.norm(arr)
            if norm > 1e-12:
                X_list.append(arr / norm)
                y_list.append(int(l.label))

    if len(X_list) < MIN_EFFECTIVE_LABELS:
        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=False,
            status="skipped",
            reason=f"Insufficient valid paper embeddings found ({len(X_list)}/{MIN_EFFECTIVE_LABELS}).",
            n_samples=len(X_list),
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))

    if pos_count < MIN_POS_LABELS or neg_count < MIN_NEG_LABELS:
        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=False,
            status="skipped",
            reason=f"Insufficient class balance after embedding retrieval ({pos_count} pos, {neg_count} neg).",
            n_samples=len(y),
            n_pos=pos_count,
            n_neg=neg_count,
        )

    # 4. Resolve active interest vector for cosine baseline comparison
    v_active = get_active_vector_version(user_id=user_id, session=session)
    if v_active and v_active.vector is not None:
        active_centroid = np.asarray(v_active.vector, dtype=np.float32).ravel()
        vector_version_id = v_active.id
    else:
        active_centroid = get_user_seed_embedding(user_id=user_id, session=session)
        vector_version_id = None

    # 5. Out-of-fold cross-validation and promotion guard evaluation
    try:
        clf_auc, base_auc, oof_probs, best_C = _evaluate_repeated_stratified_oof(
            X=X,
            y=y,
            active_centroid=active_centroid,
            c_candidates=C_CANDIDATES,
            n_splits=5,
            n_repeats=3,
            random_state=random_state,
        )
    except Exception as exc:
        logger.error("Classifier CV evaluation failed for user '%s': %s", user_id, exc)
        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=True,
            status="error",
            reason=f"Cross-validation evaluation failed: {exc}",
        )

    # Promotion Guard check
    is_promoted = bool(
        (clf_auc >= MIN_PROMOTION_AUC)
        and (clf_auc >= base_auc + PROMOTION_AUC_MARGIN)
    )

    # 6. Fit final model on all effective feedback data
    final_model = LogisticRegression(
        C=best_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    final_model.fit(X, y)
    model_blob = pickle.dumps(final_model)

    # 7. Calibrate thresholds
    p_high, p_low, cal_source = calibrate_classifier_thresholds(
        oof_probs=oof_probs,
        y_true=y,
    )

    # 8. Persist ModelVersion with safe transaction boundaries
    latest_model = (
        session.query(ModelVersion)
        .filter_by(user_id=user_id)
        .order_by(ModelVersion.version.desc())
        .first()
    )
    next_ver = (latest_model.version + 1) if latest_model else 1
    max_event_id = max(l.event_id for l in effective_labels)

    try:
        # If promoted, deactivate previous models for this user
        if is_promoted:
            session.query(ModelVersion).filter_by(user_id=user_id).update({"is_active": False})

        new_model_record = ModelVersion(
            user_id=user_id,
            version=next_ver,
            trained_at=datetime.now(timezone.utc),
            n_samples=len(y),
            n_pos=pos_count,
            n_neg=neg_count,
            hyperparameters={"C": float(best_C), "penalty": "l2", "class_weight": "balanced"},
            oof_auc=clf_auc,
            oof_auc_cosine_baseline=base_auc,
            promoted=is_promoted,
            p_high=p_high,
            p_low=p_low,
            threshold_calibration_source=cal_source,
            trained_on_max_event_id=max_event_id,
            vector_version_id=vector_version_id,
            model_blob=model_blob,
            is_active=is_promoted,
        )
        session.add(new_model_record)
        session.commit()

        status_str = "promoted" if is_promoted else "not_promoted"
        guard_reason = (
            f"Classifier promoted to active model v{next_ver} (OOF ROC-AUC={clf_auc:.4f} vs baseline={base_auc:.4f}, margin={clf_auc - base_auc:+.4f} >= {PROMOTION_AUC_MARGIN})."
            if is_promoted
            else f"Classifier not promoted: OOF ROC-AUC={clf_auc:.4f} (baseline={base_auc:.4f}, delta={clf_auc - base_auc:+.4f}) did not satisfy guard (>= {MIN_PROMOTION_AUC:.2f} and >= baseline + {PROMOTION_AUC_MARGIN:.2f}). Previous routing method retained."
        )

        logger.info("[CLASSIFIER_%s] User '%s': %s", status_str.upper(), user_id, guard_reason)

        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=True,
            status=status_str,
            reason=guard_reason,
            version_created=next_ver,
            model_version_id=new_model_record.id,
            promoted=is_promoted,
            oof_auc=clf_auc,
            oof_auc_cosine_baseline=base_auc,
            best_C=best_C,
            n_samples=len(y),
            n_pos=pos_count,
            n_neg=neg_count,
            p_high=p_high,
            p_low=p_low,
            threshold_calibration_source=cal_source,
            trained_on_max_event_id=max_event_id,
            is_active=is_promoted,
        )

    except Exception as exc:
        session.rollback()
        logger.error("Failed to persist model version for user '%s': %s", user_id, exc)
        return ClassifierTrainingResult(
            user_id=user_id,
            eligible=True,
            status="error",
            reason=f"Model persistence transaction failed: {exc}",
        )


def find_nearest_feedback_papers(
    paper_embedding: Optional[Sequence[float] | np.ndarray],
    user_id: str,
    session: Session,
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Find nearest liked (positive) and nearest rejected (negative) papers for a candidate paper.
    Used for human-readable decision explanations in the review UI (Roadmap Section 7.8).
    Does NOT require PyTorch or SentenceTransformers (pure NumPy cosine dot products).

    Returns:
        (nearest_liked_dict, nearest_rejected_dict)
    """
    if paper_embedding is None:
        return None, None

    x = np.asarray(paper_embedding, dtype=np.float32).ravel()
    x_norm = np.linalg.norm(x)
    if x.size != 384 or x_norm < 1e-12:
        return None, None
    x = x / x_norm

    effective_labels = get_effective_training_labels(session=session, user_id=user_id)
    if not effective_labels:
        return None, None

    pos_ids = [l.paper_id for l in effective_labels if l.label == 1]
    neg_ids = [l.paper_id for l in effective_labels if l.label == 0]

    all_ids = pos_ids + neg_ids
    if not all_ids:
        return None, None

    paper_rows = (
        session.query(Paper.id, Paper.title, Paper.embedding)
        .filter(Paper.id.in_(all_ids))
        .all()
    )
    paper_dict = {row[0]: (row[1], row[2]) for row in paper_rows if row[2] is not None}

    nearest_liked: Optional[dict] = None
    max_pos_sim = -1.0

    for pid in pos_ids:
        if pid in paper_dict:
            title, raw_emb = paper_dict[pid]
            emb = np.asarray(raw_emb, dtype=np.float32).ravel()
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                sim = float(np.dot(x, emb / norm))
                if sim > max_pos_sim:
                    max_pos_sim = sim
                    nearest_liked = {"paper_id": pid, "title": title, "similarity": sim}

    nearest_rejected: Optional[dict] = None
    max_neg_sim = -1.0

    for pid in neg_ids:
        if pid in paper_dict:
            title, raw_emb = paper_dict[pid]
            emb = np.asarray(raw_emb, dtype=np.float32).ravel()
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                sim = float(np.dot(x, emb / norm))
                if sim > max_neg_sim:
                    max_neg_sim = sim
                    nearest_rejected = {"paper_id": pid, "title": title, "similarity": sim}

    return nearest_liked, nearest_rejected
