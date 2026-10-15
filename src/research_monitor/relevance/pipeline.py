"""
pipeline.py
~~~~~~~~~~~
Explicit state machine implementing:
  discover -> preprocess -> embed -> score -> route

Follows Roadmap Section 9.1 and Phase 5 & 11 deliverables:
  - Cycle record management with mode (replay/live) and as_of_date.
  - Supports both threshold-based routing (against active vector centroid) and classifier-based routing (promoted model).
  - Persists auditable decisions per (user, paper), including LOW papers.
  - Always logs cosine similarity to active vector, even in classifier mode.
  - Re-running a cycle never creates duplicate decisions.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.config import config
from research_monitor.database.models import (
    Cycle,
    ModelVersion,
    Paper,
    RoutingDecision,
    User,
)
from research_monitor.ingestion.preprocessor import normalize_text
from research_monitor.relevance.router import route_score
from research_monitor.relevance.state import (
    DecisionStatus,
    PaperCandidate,
    PipelineState,
    Route,
)
from research_monitor.relevance.user_manager import get_active_vector_version

logger = logging.getLogger(__name__)


def discover_stage(
    state: PipelineState,
    session: Session,
    limit: Optional[int] = None,
    papers_override: Optional[list[dict]] = None,
    corpus_path: Path | str = "data/corpus.parquet",
) -> PipelineState:
    """
    Discover eligible candidate papers for the current monitoring cycle.
    Excludes any paper that already has an auditable routing decision for this user.
    """
    # 1. Fetch already-routed paper IDs for this user to prevent duplicates
    already_routed_pids = set(
        pid[0]
        for pid in session.query(RoutingDecision.paper_id)
        .filter_by(user_id=state.user_id)
        .all()
    )

    candidates: list[PaperCandidate] = []

    # 2. If papers_override is passed (e.g. from fixtures or test mocks)
    if papers_override is not None:
        for p in papers_override:
            pid = str(p["id" if "id" in p else "paper_id"])
            if pid in already_routed_pids:
                continue
            candidates.append(
                PaperCandidate(
                    paper_id=pid,
                    title=str(p.get("title", "")),
                    abstract=str(p.get("abstract", "")),
                    submitted_date=str(p.get("submitted_date", "")),
                    categories=p.get("categories", []),
                    is_reference=bool(p.get("is_reference", False)),
                    embedding=p.get("embedding", None),
                )
            )
            if limit and len(candidates) >= limit:
                break
        state.discovered = candidates
        state.counts["discovered"] = len(candidates)
        return state

    # 3. Replay mode: Query existing corpus
    if state.mode == "replay":
        db_queried = False
        try:
            total_in_db = session.query(Paper).count()
            if total_in_db > 0:
                db_queried = True
                query = session.query(Paper)
                if state.as_of_date:
                    query = query.filter(Paper.submitted_date <= state.as_of_date)
                if already_routed_pids:
                    query = query.filter(~Paper.id.in_(already_routed_pids))
                query = query.order_by(Paper.submitted_date.desc())
                if limit:
                    query = query.limit(limit)

                papers = query.all()
                for p in papers:
                    emb = np.array(p.embedding, dtype=np.float32) if p.embedding is not None else None
                    candidates.append(
                        PaperCandidate(
                            paper_id=p.id,
                            title=p.title,
                            abstract=p.abstract,
                            submitted_date=p.submitted_date,
                            categories=p.categories,
                            is_reference=p.is_reference,
                            embedding=emb,
                        )
                    )
        except Exception as e:
            logger.warning("Could not query papers from database in replay mode (%s). Falling back to Parquet.", e)

        if not db_queried:
            c_path = Path(corpus_path)
            if not c_path.exists():
                raise FileNotFoundError(f"Corpus file not found at {c_path}")
            df = pd.read_parquet(c_path)
            if state.as_of_date and "submitted_date" in df.columns:
                df = df[df["submitted_date"] <= state.as_of_date]
            if already_routed_pids:
                df = df[~df["paper_id"].isin(already_routed_pids)]
            if limit:
                df = df.head(limit)

            for _, row in df.iterrows():
                candidates.append(
                    PaperCandidate(
                        paper_id=str(row["paper_id"]),
                        title=str(row["title"]),
                        abstract=str(row["abstract"]),
                        submitted_date=str(row.get("submitted_date", "")),
                        categories=row.get("categories", []),
                        is_reference=bool(row.get("is_reference", False)),
                        embedding=None,
                    )
                )

    # 4. Live mode: Query arXiv or fetch newly ingested papers from database
    elif state.mode == "live":
        try:
            query = session.query(Paper)
            if state.as_of_date:
                query = query.filter(Paper.submitted_date <= state.as_of_date)
            if already_routed_pids:
                query = query.filter(~Paper.id.in_(already_routed_pids))
            query = query.order_by(Paper.submitted_date.desc())
            if limit:
                query = query.limit(limit)

            papers = query.all()
            for p in papers:
                emb = np.array(p.embedding, dtype=np.float32) if p.embedding is not None else None
                candidates.append(
                    PaperCandidate(
                        paper_id=p.id,
                        title=p.title,
                        abstract=p.abstract,
                        submitted_date=p.submitted_date,
                        categories=p.categories,
                        is_reference=p.is_reference,
                        embedding=emb,
                    )
                )
        except Exception as e:
            logger.error("Live discovery query failed: %s", e)
            raise

    state.discovered = candidates
    state.counts["discovered"] = len(candidates)
    logger.info("Discover stage completed: %d unrouted candidates found.", len(candidates))
    return state


def preprocess_stage(state: PipelineState) -> PipelineState:
    """
    Preprocess stage: Normalizes text encoding and whitespace for discovered candidates.
    """
    preprocessed: list[PaperCandidate] = []
    for c in state.discovered:
        c.title = normalize_text(c.title)
        c.abstract = normalize_text(c.abstract)
        preprocessed.append(c)

    state.preprocessed = preprocessed
    state.counts["preprocessed"] = len(preprocessed)
    return state


def embed_stage(
    state: PipelineState,
    session: Session,
    embeddings_path: Path | str = "data/embeddings.npy",
    corpus_path: Path | str = "data/corpus.parquet",
) -> PipelineState:
    """
    Embed stage: Attaches L2-normalized 384-dimensional embeddings.
    In replay mode, looks up existing embeddings from DB or precomputed arrays.
    In live mode, computes embeddings for un-embedded papers and persists them to DB.
    """
    embedded: list[PaperCandidate] = []

    # Check if embeddings are already populated
    missing_pids = [c.paper_id for c in state.preprocessed if c.embedding is None]

    if not missing_pids:
        # All embeddings already attached
        for c in state.preprocessed:
            v = np.asarray(c.embedding, dtype=np.float32).ravel()
            norm = np.linalg.norm(v)
            if norm > 0:
                c.embedding = v / norm
            embedded.append(c)
        state.embedded = embedded
        state.counts["embedded"] = len(embedded)
        return state

    # Lookup in DB first
    db_embeddings: dict[str, np.ndarray] = {}
    try:
        rows = (
            session.query(Paper.id, Paper.embedding)
            .filter(Paper.id.in_(missing_pids))
            .all()
        )
        for pid, emb in rows:
            if emb is not None:
                v = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(v)
                db_embeddings[pid] = v / norm if norm > 0 else v
    except Exception as e:
        logger.warning("DB embedding lookup error: %s", e)

    # If in live mode and still missing, generate dynamically
    still_missing = [c for c in state.preprocessed if c.paper_id not in db_embeddings and c.embedding is None]
    if still_missing and state.mode == "live":
        try:
            from research_monitor.embeddings import compute_embeddings
            texts = [f"{c.title}\n\n{c.abstract}" for c in still_missing]
            new_embs = compute_embeddings(texts)
            for c, emb in zip(still_missing, new_embs):
                v = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(v)
                normed = v / norm if norm > 0 else v
                db_embeddings[c.paper_id] = normed

                paper_row = session.query(Paper).filter_by(id=c.paper_id).first()
                if paper_row:
                    paper_row.embedding = normed.tolist()
                    paper_row.embedding_model = config.EMBEDDING_MODEL
            session.commit()
            logger.info("Dynamically embedded and updated %d papers in live mode.", len(still_missing))
        except Exception as e:
            logger.error("Failed to dynamically compute embeddings in live mode: %s", e)
            raise

    # Attach embeddings to candidates
    for c in state.preprocessed:
        if c.embedding is not None:
            v = np.asarray(c.embedding, dtype=np.float32).ravel()
            norm = np.linalg.norm(v)
            c.embedding = v / norm if norm > 0 else v
            embedded.append(c)
        elif c.paper_id in db_embeddings:
            c.embedding = db_embeddings[c.paper_id]
            embedded.append(c)

    state.embedded = embedded
    state.counts["embedded"] = len(embedded)
    return state


def score_stage(state: PipelineState) -> PipelineState:
    """
    Score stage:
      - Always calculates cosine similarity against active interest vector:
          cosine_score = dot(paper_embedding, active_vector).
      - In classifier mode, additionally calculates classifier probability:
          clf_prob = predict_proba(paper_embedding).
      - Sets candidate.score = clf_prob if classifier mode else cosine_score.
    """
    if state.active_vector is None:
        raise ValueError("Cannot score without an active interest vector in state.")

    v = np.asarray(state.active_vector, dtype=np.float32).ravel()
    scored: list[PaperCandidate] = []

    for c in state.embedded:
        if c.embedding is None:
            raise ValueError(f"Paper {c.paper_id} has no embedding in score stage.")
        emb = np.asarray(c.embedding, dtype=np.float32).ravel()
        cosine_sim = float(np.dot(emb, v))
        c.cosine_score = cosine_sim
        c.method = state.method
        c.model_version_id = state.model_version_id

        if state.method == "classifier" and state.active_model is not None:
            prob = float(state.active_model.predict_proba([emb])[0, 1])
            c.clf_prob = prob
            c.score = prob
        else:
            c.clf_prob = None
            c.score = cosine_sim

        scored.append(c)

    state.scored = scored
    state.counts["scored"] = len(scored)
    return state


def route_stage(state: PipelineState) -> PipelineState:
    """
    Route stage: Assigns HIGH, BORDERLINE, or LOW based on active thresholds and routing method.
    """
    routed: list[PaperCandidate] = []
    high_count = 0
    borderline_count = 0
    low_count = 0

    for c in state.scored:
        if c.score is None:
            raise ValueError(f"Paper {c.paper_id} has not been scored.")

        if state.method == "classifier":
            prob = float(c.score)
            th = float(state.t_high)
            tl = float(state.t_low)
            if prob >= th:
                route = Route.HIGH
                status = DecisionStatus.ROUTED_HIGH
                reason = (
                    f"Classifier probability {prob:.4f} >= threshold {th:.4f} "
                    f"(model v{state.model_version_num}). Cleared gate for automatic delivery."
                )
            elif prob <= tl:
                route = Route.LOW
                status = DecisionStatus.DISCARDED
                reason = (
                    f"Classifier probability {prob:.4f} <= threshold {tl:.4f} "
                    f"(model v{state.model_version_num}). Discarded paper below relevance floor."
                )
            else:
                route = Route.BORDERLINE
                status = DecisionStatus.ROUTED_BORDERLINE
                reason = (
                    f"Classifier probability {prob:.4f} in borderline band "
                    f"({tl:.4f} < prob < {th:.4f}, model v{state.model_version_num}). Queued for human review."
                )
            c.route = route
            c.status = status
            c.reason = reason
        else:
            res = route_score(
                score=c.score,
                t_high=state.t_high,
                t_low=state.t_low,
                p_high=state.p_high_level,
                p_low=state.p_low_level,
            )
            c.route = res.route
            c.status = res.status
            c.reason = res.reason

        routed.append(c)

        if c.route == Route.HIGH:
            high_count += 1
        elif c.route == Route.BORDERLINE:
            borderline_count += 1
        elif c.route == Route.LOW:
            low_count += 1

    state.routed = routed
    state.counts.update({
        "total_routed": len(routed),
        "high": high_count,
        "borderline": borderline_count,
        "low": low_count,
    })
    logger.info(
        "Route stage completed: %d total (HIGH=%d, BORDERLINE=%d, LOW=%d, method=%s)",
        len(routed),
        high_count,
        borderline_count,
        low_count,
        state.method,
    )
    return state


def persist_decisions(
    state: PipelineState,
    session: Session,
) -> None:
    """
    Persist one auditable routing decision per (user, paper), including LOW papers.
    Updates the Cycle record with completion timestamp and telemetry counts.
    """
    now = datetime.now(timezone.utc)
    for c in state.routed:
        if c.route is None or c.status is None or c.score is None or c.reason is None:
            continue

        decision = RoutingDecision(
            paper_id=c.paper_id,
            user_id=state.user_id,
            cycle_id=state.cycle_id,
            cosine_score=c.cosine_score if c.cosine_score is not None else c.score,
            vector_version_id=state.vector_version_id,
            clf_prob=c.clf_prob,
            method=state.method,
            model_version_id=state.model_version_id,
            thresholds_used=state.thresholds_dict,
            route=c.route.value,
            status=c.status.value,
            reason=c.reason,
            created_at=now,
        )
        session.add(decision)

    if state.cycle_id is not None:
        cycle = session.query(Cycle).filter_by(id=state.cycle_id).first()
        if cycle:
            cycle.finished_at = now
            cycle.counts = state.counts
            cycle.config_snapshot = {
                "method": state.method,
                "model_version_id": state.model_version_id,
                "model_version_num": state.model_version_num,
                "t_high": state.t_high,
                "t_low": state.t_low,
                "p_high": state.p_high_level,
                "p_low": state.p_low_level,
                "embedding_model": config.EMBEDDING_MODEL,
            }

    session.commit()
    logger.info("Persisted %d routing decisions to database.", len(state.routed))


def run_monitoring_cycle(
    user_id: str,
    session: Session,
    mode: str = "replay",
    as_of_date: Optional[str] = None,
    limit: Optional[int] = None,
    papers_override: Optional[list[dict]] = None,
    auto_build_queue: bool = True,
    auto_update_vector: bool = False,
    auto_train_model: bool = False,
) -> PipelineState:
    """
    Execute an autonomous monitoring cycle for a user through the explicit state machine:
    discover -> preprocess -> embed -> score -> route -> persist

    Args:
        user_id: Target user identifier.
        session: Active SQLAlchemy DB session.
        mode: Execution mode ('replay' or 'live').
        as_of_date: Historical cut-off date (ISO string 'YYYY-MM-DD').
        limit: Max candidates to process in this cycle.
        papers_override: Optional explicit candidate list for testing.
        auto_build_queue: Build human review queue for borderline candidates.
        auto_update_vector: Evaluate Rocchio vector update after cycle.
        auto_train_model: Evaluate preference classifier training after cycle.

    Returns:
        Final PipelineState object.
    """
    # 1. Resolve user and active preference vector
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        raise ValueError(f"User '{user_id}' does not exist in database.")

    v_version = get_active_vector_version(user_id=user_id, session=session)
    if not v_version:
        raise ValueError(f"User '{user_id}' has no active interest vector version.")

    active_vec = np.array(v_version.vector, dtype=np.float32)
    norm = np.linalg.norm(active_vec)
    if norm > 0:
        active_vec = active_vec / norm

    # Resolve active classifier model if promoted (Phase 11)
    active_model_ver = (
        session.query(ModelVersion)
        .filter_by(user_id=user.id, is_active=True)
        .order_by(ModelVersion.version.desc())
        .first()
    )

    if active_model_ver and active_model_ver.model_blob:
        method = "classifier"
        active_model = pickle.loads(active_model_ver.model_blob)
        model_version_id = active_model_ver.id
        model_version_num = active_model_ver.version
        t_high = float(active_model_ver.p_high if active_model_ver.p_high is not None else 0.70)
        t_low = float(active_model_ver.p_low if active_model_ver.p_low is not None else 0.30)
        p_high_level = 70.0
        p_low_level = 30.0
    else:
        method = "threshold"
        active_model = None
        model_version_id = None
        model_version_num = None
        t_high = v_version.t_high
        t_low = v_version.t_low
        p_high_level = v_version.p_high_level
        p_low_level = v_version.p_low_level

    # 2. Record new Cycle entity
    now = datetime.now(timezone.utc)
    date_str = as_of_date or now.strftime("%Y-%m-%d")

    cycle = Cycle(
        user_id=user.id,
        as_of_date=date_str,
        mode=mode,
        started_at=now,
        finished_at=None,
        counts={},
        config_snapshot={},
    )
    session.add(cycle)
    session.flush()

    # 3. Initialize PipelineState
    state = PipelineState(
        user_id=user.id,
        cycle_id=cycle.id,
        mode=mode,
        as_of_date=date_str,
        is_simulated=user.is_simulated,
        active_vector=active_vec,
        vector_version_id=v_version.id,
        vector_version_num=v_version.version,
        t_high=t_high,
        t_low=t_low,
        p_high_level=p_high_level,
        p_low_level=p_low_level,
        method=method,
        active_model=active_model,
        model_version_id=model_version_id,
        model_version_num=model_version_num,
    )

    logger.info(
        "Starting cycle %d for user '%s' (mode=%s, method=%s, as_of=%s, t_high=%.4f, t_low=%.4f)",
        cycle.id,
        user.id,
        mode,
        state.method,
        date_str,
        state.t_high,
        state.t_low,
    )

    # 4. State transitions: discover -> preprocess -> embed -> score -> route
    discover_stage(state=state, session=session, limit=limit, papers_override=papers_override)
    preprocess_stage(state=state)
    embed_stage(state=state, session=session)
    score_stage(state=state)
    route_stage(state=state)

    # 5. Persist auditable decisions and finish cycle
    persist_decisions(state=state, session=session)

    # 6. Queue borderline papers for human review (Phase 6)
    if auto_build_queue:
        try:
            from research_monitor.review.queue import build_review_queue
            build_review_queue(user_id=user.id, cycle_id=cycle.id, session=session)
        except Exception as e:
            logger.warning("Could not build review queue for cycle %d: %s", cycle.id, e)

    # 7. Optionally trigger preference vector update for next cycle (Phase 10)
    if auto_update_vector:
        try:
            from research_monitor.relevance.preference import update_user_interest_vector
            update_user_interest_vector(user_id=user.id, session=session)
        except Exception as e:
            logger.warning("Could not evaluate vector update for user '%s': %s", user.id, e)

    # 8. Optionally trigger preference model retraining & evaluation for next cycle (Phase 11)
    if auto_train_model:
        try:
            from research_monitor.relevance.classifier import train_and_evaluate_classifier
            train_and_evaluate_classifier(user_id=user.id, session=session)
        except Exception as e:
            logger.warning("Could not evaluate classifier update for user '%s': %s", user.id, e)

    return state
