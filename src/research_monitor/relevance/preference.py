"""
preference.py
~~~~~~~~~~~~~
Phase 10 — Preference Memory: Interest Vector (Rocchio Centroid).

Follows Roadmap Section 6, 7.3, and 7.4:
  - Calculates updated interest vector via Rocchio relevance feedback:
      c = normalize(alpha * c0 + beta * mean(pos) - gamma * mean(neg))
    where alpha = 1.0, beta = 0.75, gamma = 0.25.
    Negative term is omitted when no negative examples exist.
  - Update triggers:
      * First centroid update: at least 10 effective labels and at least 3 positive labels.
      * Subsequent updates: after every 25 new effective labels since previous version's
        built_from_max_event_id.
  - Effective labels are sourced from the training_labels view (user likes/dislikes override reviewer).
  - Recalculates t_high and t_low from reference paper sample preserving stored percentile levels.
  - Versioned in interest_vector_versions (method='rocchio', is_active=True).
  - Safe transaction boundaries; previous versions deactivated but preserved.
  - Activated for the NEXT cycle; historical decisions keep their original vector_version_id.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Optional, Sequence

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    InterestProfile,
    InterestVectorVersion,
    Paper,
    TrainingLabel,
    User,
)
from research_monitor.relevance.thresholds import compute_vector_thresholds
from research_monitor.review.labels import get_effective_training_labels

logger = logging.getLogger(__name__)

# Rocchio hyper-parameters (Roadmap Section 7.4)
ROCCHIO_ALPHA = 1.0
ROCCHIO_BETA = 0.75
ROCCHIO_GAMMA = 0.25

FIRST_UPDATE_MIN_LABELS = 10
FIRST_UPDATE_MIN_POS = 3
SUBSEQUENT_UPDATE_INTERVAL = 25


@dataclass(frozen=True)
class VectorUpdateEligibility:
    """Eligibility telemetry report for interest vector updates."""
    user_id: str
    is_first_update: bool
    current_version: int
    eligible: bool
    reason: str
    total_effective_labels: int
    positive_labels: int
    negative_labels: int
    new_effective_labels_since_cutoff: int
    cutoff_event_id: Optional[int]


@dataclass
class VectorUpdateResult:
    """Outcome report for an interest vector update attempt."""
    user_id: str
    eligible: bool
    status: str  # "updated", "skipped", "error"
    reason: str
    version_created: Optional[int] = None
    vector_version_id: Optional[int] = None
    n_effective_labels_used: int = 0
    n_positive_labels_used: int = 0
    n_negative_labels_used: int = 0
    built_from_max_event_id: Optional[int] = None
    p_high: Optional[float] = None
    p_low: Optional[float] = None
    t_high: Optional[float] = None
    t_low: Optional[float] = None
    is_active_for_next_cycle: bool = False


def compute_rocchio_centroid(
    c0: np.ndarray,
    positive_embeddings: Sequence[np.ndarray],
    negative_embeddings: Sequence[np.ndarray] = (),
    alpha: float = ROCCHIO_ALPHA,
    beta: float = ROCCHIO_BETA,
    gamma: float = ROCCHIO_GAMMA,
) -> np.ndarray:
    """
    Compute normalized Rocchio centroid vector:
      c = normalize(alpha * c0 + beta * mean(pos) - gamma * mean(neg))

    Parameters
    ----------
    c0 : np.ndarray
        Original seed embedding (1D array of length 384).
    positive_embeddings : Sequence[np.ndarray]
        List or array of positive paper embeddings (length 384 each).
    negative_embeddings : Sequence[np.ndarray], optional
        List or array of negative paper embeddings (length 384 each).
        Negative term is omitted if empty.
    alpha : float
        Seed weight (default: 1.0).
    beta : float
        Positive feedback weight (default: 0.75).
    gamma : float
        Negative feedback weight (default: 0.25).

    Returns
    -------
    np.ndarray
        1D L2-normalized float32 numpy array of length 384.

    Raises
    ------
    ValueError
        If inputs are invalid or output centroid is degenerate (near-zero norm).
    """
    # 1. Validate seed vector c0
    seed = np.asarray(c0, dtype=np.float32).ravel()
    if seed.size != 384:
        raise ValueError(f"Seed vector must have dimension 384, got {seed.size}.")
    if not np.all(np.isfinite(seed)):
        raise ValueError("Seed vector contains non-finite values (NaN/Inf).")

    seed_norm = np.linalg.norm(seed)
    if seed_norm < 1e-12:
        raise ValueError("Seed vector has degenerate near-zero norm.")
    seed = seed / seed_norm

    # 2. Positive centroid term
    pos_term = np.zeros(384, dtype=np.float32)
    valid_pos = []
    for p in positive_embeddings:
        pv = np.asarray(p, dtype=np.float32).ravel()
        if pv.size == 384 and np.all(np.isfinite(pv)):
            p_norm = np.linalg.norm(pv)
            if p_norm > 1e-12:
                valid_pos.append(pv / p_norm)

    if len(valid_pos) > 0:
        pos_mean = np.mean(valid_pos, axis=0, dtype=np.float32)
        pos_term = beta * pos_mean

    # 3. Negative centroid term (omitted when empty)
    neg_term = np.zeros(384, dtype=np.float32)
    valid_neg = []
    if negative_embeddings is not None and len(negative_embeddings) > 0:
        for n in negative_embeddings:
            nv = np.asarray(n, dtype=np.float32).ravel()
            if nv.size == 384 and np.all(np.isfinite(nv)):
                n_norm = np.linalg.norm(nv)
                if n_norm > 1e-12:
                    valid_neg.append(nv / n_norm)

        if len(valid_neg) > 0:
            neg_mean = np.mean(valid_neg, axis=0, dtype=np.float32)
            neg_term = gamma * neg_mean

    # 4. Combine terms
    raw_centroid = (alpha * seed) + pos_term - neg_term

    if not np.all(np.isfinite(raw_centroid)):
        raise ValueError("Computed centroid vector contains non-finite values (NaN/Inf).")

    c_norm = np.linalg.norm(raw_centroid)
    if c_norm < 1e-12:
        raise ValueError(
            f"Degenerate near-zero centroid encountered (norm={c_norm:.4e}); update aborted."
        )

    updated_centroid = (raw_centroid / c_norm).astype(np.float32)
    return updated_centroid


def get_user_seed_embedding(user_id: str, session: Session) -> np.ndarray:
    """
    Retrieve and validate the original seed embedding for a user.
    Preserves the original seed vector; never replaces it.
    """
    # Try InterestProfile first
    profile = (
        session.query(InterestProfile)
        .filter_by(user_id=user_id)
        .order_by(InterestProfile.created_at.asc())
        .first()
    )
    if profile and profile.seed_embedding is not None:
        seed = np.asarray(profile.seed_embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(seed)
        if norm > 1e-12:
            return seed / norm

    # Fallback to vector version 0
    v0 = (
        session.query(InterestVectorVersion)
        .filter_by(user_id=user_id, version=0)
        .first()
    )
    if v0 and v0.vector is not None:
        seed = np.asarray(v0.vector, dtype=np.float32).ravel()
        norm = np.linalg.norm(seed)
        if norm > 1e-12:
            return seed / norm

    raise ValueError(f"User '{user_id}' has no original seed embedding available.")


def check_vector_update_eligibility(
    user_id: str,
    session: Session,
    effective_labels: Optional[list[TrainingLabel]] = None,
) -> VectorUpdateEligibility:
    """
    Determine whether a user is eligible for a Rocchio centroid update.

    Update Policy (Roadmap Section 7.4):
      - First update: >= 10 effective labels AND >= 3 positive labels.
      - Subsequent updates: >= 25 new effective labels since previous vector version's
        built_from_max_event_id.
    """
    labels = (
        effective_labels
        if effective_labels is not None
        else get_effective_training_labels(session=session, user_id=user_id)
    )

    total_effective = len(labels)
    pos_count = sum(1 for l in labels if l.label == 1)
    neg_count = sum(1 for l in labels if l.label == 0)

    # Determine latest vector version for user
    latest_v = (
        session.query(InterestVectorVersion)
        .filter_by(user_id=user_id)
        .order_by(InterestVectorVersion.version.desc())
        .first()
    )

    if not latest_v:
        return VectorUpdateEligibility(
            user_id=user_id,
            is_first_update=True,
            current_version=0,
            eligible=False,
            reason=f"User '{user_id}' has no existing vector version in database.",
            total_effective_labels=total_effective,
            positive_labels=pos_count,
            negative_labels=neg_count,
            new_effective_labels_since_cutoff=0,
            cutoff_event_id=None,
        )

    is_first = (latest_v.version == 0) or (latest_v.method == "seed") or (latest_v.built_from_max_event_id is None)
    cutoff_id = latest_v.built_from_max_event_id

    if is_first:
        # First centroid update requirements
        if total_effective < FIRST_UPDATE_MIN_LABELS:
            return VectorUpdateEligibility(
                user_id=user_id,
                is_first_update=True,
                current_version=latest_v.version,
                eligible=False,
                reason=(
                    f"Insufficient effective labels for first update: "
                    f"{total_effective}/{FIRST_UPDATE_MIN_LABELS} required."
                ),
                total_effective_labels=total_effective,
                positive_labels=pos_count,
                negative_labels=neg_count,
                new_effective_labels_since_cutoff=total_effective,
                cutoff_event_id=cutoff_id,
            )

        if pos_count < FIRST_UPDATE_MIN_POS:
            return VectorUpdateEligibility(
                user_id=user_id,
                is_first_update=True,
                current_version=latest_v.version,
                eligible=False,
                reason=(
                    f"Insufficient positive labels for first update: "
                    f"{pos_count}/{FIRST_UPDATE_MIN_POS} required."
                ),
                total_effective_labels=total_effective,
                positive_labels=pos_count,
                negative_labels=neg_count,
                new_effective_labels_since_cutoff=total_effective,
                cutoff_event_id=cutoff_id,
            )

        return VectorUpdateEligibility(
            user_id=user_id,
            is_first_update=True,
            current_version=latest_v.version,
            eligible=True,
            reason=(
                f"Eligible for first centroid update ({total_effective} effective labels, "
                f"{pos_count} positive)."
            ),
            total_effective_labels=total_effective,
            positive_labels=pos_count,
            negative_labels=neg_count,
            new_effective_labels_since_cutoff=total_effective,
            cutoff_event_id=cutoff_id,
        )

    # Subsequent update requirements: >= 25 new effective labels since cutoff_event_id
    if cutoff_id is not None:
        new_labels = [l for l in labels if l.event_id > cutoff_id]
    else:
        new_labels = labels

    new_count = len(new_labels)
    if new_count < SUBSEQUENT_UPDATE_INTERVAL:
        return VectorUpdateEligibility(
            user_id=user_id,
            is_first_update=False,
            current_version=latest_v.version,
            eligible=False,
            reason=(
                f"Insufficient new effective labels since vector v{latest_v.version}: "
                f"{new_count}/{SUBSEQUENT_UPDATE_INTERVAL} required (cutoff event {cutoff_id})."
            ),
            total_effective_labels=total_effective,
            positive_labels=pos_count,
            negative_labels=neg_count,
            new_effective_labels_since_cutoff=new_count,
            cutoff_event_id=cutoff_id,
        )

    return VectorUpdateEligibility(
        user_id=user_id,
        is_first_update=False,
        current_version=latest_v.version,
        eligible=True,
        reason=(
            f"Eligible for update to version {latest_v.version + 1} "
            f"({new_count} new effective labels since cutoff event {cutoff_id})."
        ),
        total_effective_labels=total_effective,
        positive_labels=pos_count,
        negative_labels=neg_count,
        new_effective_labels_since_cutoff=new_count,
        cutoff_event_id=cutoff_id,
    )


def update_user_interest_vector(
    user_id: str,
    session: Session,
    reference_embeddings: Optional[np.ndarray] = None,
    alpha: float = ROCCHIO_ALPHA,
    beta: float = ROCCHIO_BETA,
    gamma: float = ROCCHIO_GAMMA,
) -> VectorUpdateResult:
    """
    Evaluate eligibility and atomically persist an updated Rocchio interest vector version.

    Parameters
    ----------
    user_id : str
        Target user identifier.
    session : Session
        Active database session.
    reference_embeddings : Optional[np.ndarray]
        Pre-loaded reference paper embeddings for threshold computation.
    alpha, beta, gamma : float
        Rocchio hyper-parameters.

    Returns
    -------
    VectorUpdateResult
        Structured outcome describing eligibility, version created, thresholds, and activation state.
    """
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"User '{user_id}' does not exist in database.",
        )

    # 1. Query user-specific effective labels (never mix across users)
    effective_labels = get_effective_training_labels(session=session, user_id=user_id)

    # 2. Check update trigger eligibility
    eligibility = check_vector_update_eligibility(
        user_id=user_id,
        session=session,
        effective_labels=effective_labels,
    )

    if not eligibility.eligible:
        logger.info("Vector update skipped for user '%s': %s", user_id, eligibility.reason)
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="skipped",
            reason=eligibility.reason,
            n_effective_labels_used=eligibility.total_effective_labels,
            n_positive_labels_used=eligibility.positive_labels,
            n_negative_labels_used=eligibility.negative_labels,
            built_from_max_event_id=eligibility.cutoff_event_id,
        )

    # 3. Retrieve original seed vector c0
    try:
        c0 = get_user_seed_embedding(user_id=user_id, session=session)
    except Exception as exc:
        logger.error("Failed to load seed embedding for user '%s': %s", user_id, exc)
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"Seed embedding error: {exc}",
        )

    # 4. Fetch stored paper embeddings for effective labels
    paper_ids = [l.paper_id for l in effective_labels]
    paper_rows = (
        session.query(Paper.id, Paper.embedding)
        .filter(Paper.id.in_(paper_ids))
        .all()
    )
    paper_emb_map = {row[0]: row[1] for row in paper_rows if row[1] is not None}

    pos_embs = []
    neg_embs = []
    for l in effective_labels:
        raw_emb = paper_emb_map.get(l.paper_id)
        if raw_emb is None:
            continue
        emb_arr = np.asarray(raw_emb, dtype=np.float32).ravel()
        if emb_arr.size == 384 and np.all(np.isfinite(emb_arr)) and np.linalg.norm(emb_arr) > 1e-12:
            if l.label == 1:
                pos_embs.append(emb_arr)
            elif l.label == 0:
                neg_embs.append(emb_arr)

    if len(pos_embs) == 0:
        logger.warning("No valid positive embeddings available for user '%s'. Skipping update.", user_id)
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="skipped",
            reason="No valid positive paper embeddings found in database.",
            n_effective_labels_used=len(effective_labels),
        )

    # 5. Calculate Rocchio centroid
    try:
        new_centroid = compute_rocchio_centroid(
            c0=c0,
            positive_embeddings=pos_embs,
            negative_embeddings=neg_embs,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
    except Exception as exc:
        logger.error("Rocchio centroid computation failed for user '%s': %s", user_id, exc)
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"Centroid calculation error: {exc}",
        )

    # 6. Retrieve stored percentile levels from previous version
    latest_v = (
        session.query(InterestVectorVersion)
        .filter_by(user_id=user_id)
        .order_by(InterestVectorVersion.version.desc())
        .first()
    )
    p_high = latest_v.p_high_level if latest_v else None
    p_low = latest_v.p_low_level if latest_v else None

    if p_high is None or p_low is None:
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"User '{user_id}' has missing percentile levels (p_high={p_high}, p_low={p_low}).",
        )

    # 7. Recalculate thresholds on the fixed reference sample
    try:
        thresholds = compute_vector_thresholds(
            vector=new_centroid,
            reference_embeddings=reference_embeddings,
            p_high=p_high,
            p_low=p_low,
            session=session,
        )
    except Exception as exc:
        logger.error("Threshold recalculation failed for user '%s': %s", user_id, exc)
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"Threshold calculation error: {exc}",
        )

    # 8. Atomically deactivate old versions and persist new vector version
    next_version_num = (latest_v.version + 1) if latest_v else 1
    max_event_id = max(l.event_id for l in effective_labels) if effective_labels else None

    try:
        # Deactivate previous versions
        session.query(InterestVectorVersion).filter_by(user_id=user_id).update({"is_active": False})

        new_v = InterestVectorVersion(
            user_id=user_id,
            version=next_version_num,
            vector=new_centroid.tolist(),
            method="rocchio",
            n_effective_labels=len(effective_labels),
            built_from_max_event_id=max_event_id,
            p_high_level=thresholds.p_high,
            p_low_level=thresholds.p_low,
            t_high=thresholds.t_high,
            t_low=thresholds.t_low,
            created_at=datetime.now(timezone.utc),
            is_active=True,
        )
        session.add(new_v)
        session.commit()

        logger.info(
            "Successfully updated interest vector for user '%s' to version %d (method='rocchio', "
            "labels=%d, pos=%d, neg=%d, t_high=%.4f, t_low=%.4f). Active for next cycle.",
            user_id,
            new_v.version,
            len(effective_labels),
            len(pos_embs),
            len(neg_embs),
            thresholds.t_high,
            thresholds.t_low,
        )

        return VectorUpdateResult(
            user_id=user_id,
            eligible=True,
            status="updated",
            reason=f"Successfully created vector version {new_v.version}.",
            version_created=new_v.version,
            vector_version_id=new_v.id,
            n_effective_labels_used=len(effective_labels),
            n_positive_labels_used=len(pos_embs),
            n_negative_labels_used=len(neg_embs),
            built_from_max_event_id=max_event_id,
            p_high=thresholds.p_high,
            p_low=thresholds.p_low,
            t_high=thresholds.t_high,
            t_low=thresholds.t_low,
            is_active_for_next_cycle=True,
        )

    except Exception as exc:
        session.rollback()
        logger.error("Failed to commit vector version update for user '%s': %s", user_id, exc)
        return VectorUpdateResult(
            user_id=user_id,
            eligible=False,
            status="error",
            reason=f"Database transaction failure: {exc}",
        )
