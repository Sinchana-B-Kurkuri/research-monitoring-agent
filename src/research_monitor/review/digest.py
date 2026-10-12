"""
digest.py
~~~~~~~~~
Digest construction, resume workflow, and user feedback recording (Phase 6).

Follows Roadmap Section 6, 7.3, 7.6, and 9.1:
  - Idempotent resume_after_review operation.
  - Delivers HIGH papers (via = auto) and review-accepted borderline papers (via = review).
  - Maximum 20 digest items per user per cycle, ordered by score DESC.
  - Digest overflow logged as DEFERRED_OVERFLOW on routing_decisions.
  - Rejected and expired papers strictly excluded from digest.
  - User Like / Dislike appends exactly one feedback_event with source = user.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    Digest,
    DigestItem,
    FeedbackEvent,
    ReviewTask,
    RoutingDecision,
)
from research_monitor.relevance.state import DecisionStatus
from research_monitor.review.queue import expire_stale_tasks
from research_monitor.summary.service import summarize_decision
from research_monitor.summary.summarizer import BaseSummarizer

logger = logging.getLogger(__name__)

MAX_DIGEST_ITEMS_PER_CYCLE = 20


def resume_after_review(
    user_id: str,
    cycle_id: int,
    session: Session,
    max_digest_items: int = MAX_DIGEST_ITEMS_PER_CYCLE,
    now: Optional[datetime] = None,
    summarizer: Optional[BaseSummarizer] = None,
) -> tuple[Digest, bool]:
    """
    Idempotent operation to build and finalize the delivery digest for a monitoring cycle.

    1. Expires any undecided tasks older than 3 days.
    2. Returns existing digest if one was already generated for (user_id, cycle_id).
    3. Collects HIGH papers (via = auto) and review-accepted borderline papers (via = review).
    4. Orders candidate papers by cosine_score DESC.
    5. Delivers up to max_digest_items (default: 20) with status DELIVERED.
    6. Summarizes delivered papers if not already summarized (Phase 7).
    7. Logs overflow candidates as DEFERRED_OVERFLOW (never summarized).
    8. Returns tuple of (Digest, already_existed_boolean).
    """
    current_time = now or datetime.now(timezone.utc)

    # 1. Expire stale tasks
    expire_stale_tasks(user_id=user_id, session=session, now=current_time)

    # 2. Check for existing digest (idempotency guard)
    existing_digest = (
        session.query(Digest)
        .filter(Digest.user_id == user_id, Digest.cycle_id == cycle_id)
        .first()
    )
    if existing_digest:
        logger.info("Digest already exists for user '%s', cycle %d. Returning existing.", user_id, cycle_id)
        return existing_digest, True

    # 3. Gather eligible auto-cleared HIGH papers
    auto_high_decisions = (
        session.query(RoutingDecision)
        .filter(
            RoutingDecision.user_id == user_id,
            RoutingDecision.cycle_id == cycle_id,
            RoutingDecision.route == "HIGH",
            RoutingDecision.status.in_([
                DecisionStatus.ROUTED_HIGH.value,
                DecisionStatus.SUMMARIZE_PENDING.value,
                DecisionStatus.SUMMARIZED.value,
            ]),
        )
        .all()
    )

    # 4. Gather review-accepted borderline papers
    review_accepted_decisions = (
        session.query(RoutingDecision)
        .join(ReviewTask, ReviewTask.decision_id == RoutingDecision.id)
        .filter(
            RoutingDecision.user_id == user_id,
            RoutingDecision.cycle_id == cycle_id,
            RoutingDecision.route == "BORDERLINE",
            RoutingDecision.status.in_([
                DecisionStatus.SUMMARIZE_PENDING.value,
                DecisionStatus.SUMMARIZED.value,
            ]),
            ReviewTask.decision == "accept",
            ReviewTask.kind != "audit",  # Audit papers never enter digest
        )
        .all()
    )

    # Combine candidates with origin tags
    candidates: list[tuple[RoutingDecision, str]] = []
    for dec in auto_high_decisions:
        candidates.append((dec, "auto"))
    for dec in review_accepted_decisions:
        candidates.append((dec, "review"))

    # 5. Order all candidates by cosine_score DESC (highest first)
    candidates.sort(key=lambda pair: pair[0].cosine_score, reverse=True)

    # 6. Create digest container
    digest = Digest(
        user_id=user_id,
        cycle_id=cycle_id,
        created_at=current_time,
    )
    session.add(digest)
    session.flush()  # populate digest.id

    # 7. Deliver up to max_digest_items
    delivered = candidates[:max_digest_items]
    overflow = candidates[max_digest_items:]

    for position, (dec, via) in enumerate(delivered, start=1):
        # Phase 7: Summarize paper if not already summarized
        if dec.summary is None:
            try:
                summarize_decision(
                    decision_id=dec.id,
                    user_id=user_id,
                    session=session,
                    summarizer=summarizer,
                )
            except Exception as e:
                logger.warning(
                    "Summarization attempt failed for decision %d in digest: %s. Preserving fallback.",
                    dec.id,
                    e,
                )

        item = DigestItem(
            digest_id=digest.id,
            decision_id=dec.id,
            position=position,
            via=via,
            shown_at=current_time,
        )
        dec.status = DecisionStatus.DELIVERED.value
        session.add(item)

    # 8. Mark overflow beyond digest cap as DEFERRED_OVERFLOW (strictly never summarized)
    for dec, _ in overflow:
        dec.status = DecisionStatus.DEFERRED_OVERFLOW.value
        dec.reason = (
            f"{dec.reason} | Digest cap ({max_digest_items}) exceeded. Logged as DEFERRED_OVERFLOW."
        )

    session.commit()
    logger.info(
        "Created digest id %d for user '%s', cycle %d: %d delivered, %d deferred overflow.",
        digest.id,
        user_id,
        cycle_id,
        len(delivered),
        len(overflow),
    )
    return digest, False


def record_user_feedback(
    digest_item_id: int,
    reaction: str,
    user_id: str,
    session: Session,
) -> FeedbackEvent:
    """
    Record user Like/Dislike reaction to a delivered digest item.
    Enforces user access control and appends a feedback event with source = 'user'.
    """
    item = session.query(DigestItem).filter(DigestItem.id == digest_item_id).first()
    if not item:
        raise ValueError(f"DigestItem with id {digest_item_id} not found.")

    if not item.digest or item.digest.user_id != user_id:
        raise PermissionError(f"DigestItem {digest_item_id} does not belong to user '{user_id}'.")

    reaction_norm = reaction.lower().strip()
    if reaction_norm not in ["like", "dislike"]:
        raise ValueError(f"Invalid user reaction '{reaction}'. Must be 'like' or 'dislike'.")

    dec = item.routing_decision
    now = datetime.now(timezone.utc)
    label = 1 if reaction_norm == "like" else 0

    event = FeedbackEvent(
        user_id=user_id,
        paper_id=dec.paper_id,
        decision_id=dec.id,
        source="user",
        label=label,
        is_audit=False,
        created_at=now,
    )
    session.add(event)
    dec.status = DecisionStatus.REACTED.value

    session.commit()
    logger.info("Recorded user reaction '%s' (label=%d) on digest item %d (user '%s').", reaction_norm, label, digest_item_id, user_id)
    return event
