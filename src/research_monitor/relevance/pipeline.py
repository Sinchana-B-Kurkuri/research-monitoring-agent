"""
pipeline.py
~~~~~~~~~~~
Explicit state machine implementing:
  discover -> preprocess -> embed -> score -> route

Follows Roadmap Section 9.1 and Phase 5 deliverables:
  - Cycle record management with mode (replay/live) and as_of_date.
  - Persists auditable decisions per (user, paper), including LOW papers.
  - Re-running a cycle never creates duplicate decisions.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.config import config
from research_monitor.database.models import (
    Cycle,
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
        # Check DB first
        try:
            total_in_db = session.query(Paper).count()
            if total_in_db > 0:
                db_queried = True
                query = session.query(Paper)
                if state.as_of_date:
                    query = query.filter(Paper.submitted_date <= state.as_of_date)
                # Filter unrouted
                if already_routed_pids:
                    query = query.filter(~Paper.id.in_(already_routed_pids))

                query = query.order_by(Paper.submitted_date.desc())
                if limit:
                    query = query.limit(limit)

                db_papers = query.all()
                for dp in db_papers:
                    candidates.append(
                        PaperCandidate(
                            paper_id=dp.id,
                            title=dp.title,
                            abstract=dp.abstract,
                            submitted_date=dp.submitted_date,
                            categories=dp.categories,
                            is_reference=dp.is_reference,
                            embedding=np.array(dp.embedding, dtype=np.float32) if dp.embedding else None,
                        )
                    )
        except Exception as e:
            logger.warning("Could not query papers from DB: %s; falling back to parquet artifact.", e)

        # Fallback to corpus.parquet only if DB was not populated or inaccessible
        if not db_queried:
            c_path = Path(corpus_path)
            if c_path.exists():
                df = pd.read_parquet(c_path)
                if state.as_of_date and "submitted_date" in df.columns:
                    df = df[df["submitted_date"] <= state.as_of_date]

                df = df[~df["paper_id"].astype(str).isin(already_routed_pids)]
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
                        )
                    )

    elif state.mode == "live":
        # Live mode: Discover unrouted papers from the database
        query = session.query(Paper)
        if already_routed_pids:
            query = query.filter(~Paper.id.in_(already_routed_pids))

        query = query.order_by(Paper.submitted_date.desc(), Paper.id.desc())
        if limit:
            query = query.limit(limit)

        db_papers = query.all()
        for dp in db_papers:
            candidates.append(
                PaperCandidate(
                    paper_id=dp.id,
                    title=dp.title,
                    abstract=dp.abstract,
                    submitted_date=dp.submitted_date,
                    categories=dp.categories,
                    is_reference=dp.is_reference,
                    embedding=np.array(dp.embedding, dtype=np.float32) if dp.embedding else None,
                )
            )

    state.discovered = candidates
    state.counts["discovered"] = len(candidates)
    logger.info("Discover stage completed: %d unrouted candidates identified.", len(candidates))
    return state


def preprocess_stage(state: PipelineState) -> PipelineState:
    """Preprocess candidates: Unicode NFC normalization, whitespace collapsing."""
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
    Embed stage:
      - In replay mode: looks up existing normalized embeddings.
      - In live mode (or if embedding missing): computes with SentenceTransformer.
    """
    embedded: list[PaperCandidate] = []
    needed_computation: list[PaperCandidate] = []

    # Try file lookup cache for replay mode if embedding not already set
    pid_to_file_emb: dict[str, np.ndarray] = {}
    e_path = Path(embeddings_path)
    c_path = Path(corpus_path)
    if state.mode == "replay" and e_path.exists() and c_path.exists():
        try:
            corpus_df = pd.read_parquet(c_path)
            all_embs = np.load(e_path)
            for idx, pid in enumerate(corpus_df["paper_id"].values):
                pid_to_file_emb[str(pid)] = all_embs[idx]
        except Exception as e:
            logger.debug("Could not pre-load file embeddings cache: %s", e)

    for c in state.preprocessed:
        if c.embedding is not None:
            c_emb = np.asarray(c.embedding, dtype=np.float32)
            norm = np.linalg.norm(c_emb)
            c.embedding = c_emb / norm if norm > 0 else c_emb
            embedded.append(c)
        elif c.paper_id in pid_to_file_emb:
            c_emb = np.asarray(pid_to_file_emb[c.paper_id], dtype=np.float32)
            norm = np.linalg.norm(c_emb)
            c.embedding = c_emb / norm if norm > 0 else c_emb
            embedded.append(c)
        else:
            needed_computation.append(c)

    # Compute missing embeddings (e.g. live mode or unindexed papers)
    if needed_computation:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        texts = [f"{c.title} {c.abstract}".strip() for c in needed_computation]
        vectors = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        for c, vec in zip(needed_computation, vectors):
            vec = vec.astype(np.float32)
            norm = np.linalg.norm(vec)
            c.embedding = vec / norm if norm > 0 else vec
            embedded.append(c)

    state.embedded = embedded
    state.counts["embedded"] = len(embedded)
    return state


def score_stage(state: PipelineState) -> PipelineState:
    """
    Score stage: Calculates cosine similarity s = dot(paper_embedding, active_vector).
    """
    if state.active_vector is None:
        raise ValueError("Cannot score without an active interest vector in state.")

    v = np.asarray(state.active_vector, dtype=np.float32).ravel()
    scored: list[PaperCandidate] = []

    for c in state.embedded:
        if c.embedding is None:
            raise ValueError(f"Paper {c.paper_id} has no embedding in score stage.")
        score = float(np.dot(c.embedding, v))
        c.score = score
        scored.append(c)

    state.scored = scored
    state.counts["scored"] = len(scored)
    return state


def route_stage(state: PipelineState) -> PipelineState:
    """
    Route stage: Assigns HIGH, BORDERLINE, or LOW based on active thresholds.
    """
    routed: list[PaperCandidate] = []
    high_count = 0
    borderline_count = 0
    low_count = 0

    for c in state.scored:
        if c.score is None:
            raise ValueError(f"Paper {c.paper_id} has not been scored.")

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

        if res.route == Route.HIGH:
            high_count += 1
        elif res.route == Route.BORDERLINE:
            borderline_count += 1
        elif res.route == Route.LOW:
            low_count += 1

    state.routed = routed
    state.counts.update({
        "total_routed": len(routed),
        "high": high_count,
        "borderline": borderline_count,
        "low": low_count,
    })
    logger.info(
        "Route stage completed: %d total (HIGH=%d, BORDERLINE=%d, LOW=%d)",
        len(routed),
        high_count,
        borderline_count,
        low_count,
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
            cosine_score=c.score,
            vector_version_id=state.vector_version_id,
            clf_prob=None,
            method="threshold",
            model_version_id=None,
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
        t_high=v_version.t_high,
        t_low=v_version.t_low,
        p_high_level=v_version.p_high_level,
        p_low_level=v_version.p_low_level,
    )

    logger.info(
        "Starting cycle %d for user '%s' (mode=%s, as_of=%s, t_high=%.4f, t_low=%.4f)",
        cycle.id,
        user.id,
        mode,
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

    return state
