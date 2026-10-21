"""
user_manager.py
~~~~~~~~~~~~~~~
User creation and seed interest vector initialization (Phase 5).

Tasks:
  - Create user with is_simulated flag.
  - Embed seed-interest paragraph and save InterestProfile.
  - Calculate reference percentile thresholds (P_high, P_low -> t_high, t_low).
  - Create InterestVectorVersion 0 (active, method='seed').
"""

from datetime import datetime, timezone
import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from research_monitor.config import config
from research_monitor.database.models import (
    InterestProfile,
    InterestVectorVersion,
    User,
)
from research_monitor.relevance.thresholds import compute_vector_thresholds

logger = logging.getLogger(__name__)


def _generate_seed_vector(text: str) -> np.ndarray:
    """
    Generate normalized seed interest vector with minimal memory footprint.
    Restricts PyTorch thread pools and immediately frees model memory.
    """
    import gc
    import torch
    from sentence_transformers import SentenceTransformer

    # Restrict PyTorch thread pools to prevent memory bloat in memory-constrained environments
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    with torch.no_grad():
        vec = embedder.encode(text.strip(), convert_to_numpy=True)

    # Immediately release embedder and trigger garbage collection
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def create_user(
    user_id: str,
    name: str,
    seed_text: str,
    session: Session,
    is_simulated: bool = False,
    access_token: Optional[str] = None,
    p_high: Optional[float] = None,
    p_low: Optional[float] = None,
    reference_embeddings: Optional[np.ndarray] = None,
) -> User:
    """
    Create a new user, embed their seed paragraph, and initialize version 0 vector.
    Safe for retries: checks for existing user and completes partial initializations.

    Args:
        user_id: Unique string identifier for user (e.g. 'developer' or 'user_001').
        name: Display name.
        seed_text: Descriptive seed paragraph of user research interests.
        session: Active SQLAlchemy DB session.
        is_simulated: Boolean flag for simulated users (Section 8.1 / 8.6).
        access_token: Optional API / web access token.
        p_high: Optional custom P_high percentile level (defaults to config.active_p_high).
        p_low: Optional custom P_low percentile level (defaults to config.active_p_low).
        reference_embeddings: Optional pre-loaded reference embeddings array.

    Returns:
        Persisted User instance.
    """
    now = datetime.now(timezone.utc)

    # 1. Check if user already exists
    existing = session.query(User).filter_by(id=user_id).first()
    if existing:
        logger.warning("User '%s' already exists in database; verifying profile and v0 vector.", user_id)
        profile = session.query(InterestProfile).filter_by(user_id=existing.id).first()
        v0 = (
            session.query(InterestVectorVersion)
            .filter_by(user_id=existing.id, version=0)
            .first()
        )

        if profile is not None and v0 is not None:
            logger.info("User '%s' is already fully initialized with profile and v0 vector.", user_id)
            return existing

        logger.info(
            "User '%s' exists but is missing profile (%s) or v0 vector (%s). Repairing partial initialization...",
            user_id,
            profile is not None,
            v0 is not None,
        )

        # Retrieve or compute seed vector
        if profile and profile.seed_embedding:
            seed_vec = np.array(profile.seed_embedding, dtype=np.float32)
        else:
            effective_seed_text = seed_text.strip() if seed_text.strip() else config.SEED_INTEREST
            logger.info("Encoding seed interest paragraph for existing user '%s'...", user_id)
            seed_vec = _generate_seed_vector(effective_seed_text)

        thresholds = compute_vector_thresholds(
            vector=seed_vec,
            reference_embeddings=reference_embeddings,
            p_high=p_high,
            p_low=p_low,
            session=session,
        )

        if profile is None:
            effective_seed_text = seed_text.strip() if seed_text.strip() else config.SEED_INTEREST
            profile = InterestProfile(
                user_id=existing.id,
                seed_text=effective_seed_text,
                seed_embedding=seed_vec.tolist(),
                created_at=now,
            )
            session.add(profile)

        if v0 is None:
            v0 = InterestVectorVersion(
                user_id=existing.id,
                version=0,
                vector=seed_vec.tolist(),
                method="seed",
                n_effective_labels=0,
                built_from_max_event_id=None,
                p_high_level=thresholds.p_high,
                p_low_level=thresholds.p_low,
                t_high=thresholds.t_high,
                t_low=thresholds.t_low,
                created_at=now,
                is_active=True,
            )
            session.add(v0)

        session.commit()
        logger.info("Repaired and completed initialization for user '%s'.", existing.id)
        return existing

    # 2. Embed the seed interest paragraph using memory-efficient routine
    logger.info("Encoding seed interest paragraph for user '%s'...", user_id)
    seed_vec = _generate_seed_vector(seed_text)

    # 3. Calculate reference percentile thresholds
    thresholds = compute_vector_thresholds(
        vector=seed_vec,
        reference_embeddings=reference_embeddings,
        p_high=p_high,
        p_low=p_low,
        session=session,
    )

    import secrets
    if not access_token:
        access_token = secrets.token_urlsafe(24)

    # 4. Create User entity
    user = User(
        id=str(user_id),
        name=str(name),
        access_token=access_token,
        is_simulated=is_simulated,
        created_at=now,
    )
    session.add(user)
    session.flush()

    # 5. Create InterestProfile entity
    profile = InterestProfile(
        user_id=user.id,
        seed_text=seed_text.strip(),
        seed_embedding=seed_vec.tolist(),
        created_at=now,
    )
    session.add(profile)

    # 6. Create InterestVectorVersion 0 entity
    v0 = InterestVectorVersion(
        user_id=user.id,
        version=0,
        vector=seed_vec.tolist(),
        method="seed",
        n_effective_labels=0,
        built_from_max_event_id=None,
        p_high_level=thresholds.p_high,
        p_low_level=thresholds.p_low,
        t_high=thresholds.t_high,
        t_low=thresholds.t_low,
        created_at=now,
        is_active=True,
    )
    session.add(v0)
    session.commit()

    logger.info(
        "Created user '%s' (%s) with seed vector v0: t_high=%.4f (P%.1f), t_low=%.4f (P%.1f)",
        user.id,
        user.name,
        thresholds.t_high,
        thresholds.p_high,
        thresholds.t_low,
        thresholds.p_low,
    )
    return user


def get_active_vector_version(
    user_id: str,
    session: Session,
) -> Optional[InterestVectorVersion]:
    """Retrieve the currently active InterestVectorVersion for a user."""
    return (
        session.query(InterestVectorVersion)
        .filter_by(user_id=user_id, is_active=True)
        .order_by(InterestVectorVersion.version.desc())
        .first()
    )
