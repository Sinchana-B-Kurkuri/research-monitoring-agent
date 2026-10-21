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
    max_digest_items: Optional[int] = None,
    now: Optional[datetime] = None,
    summarizer: Optional[BaseSummarizer] = None,
) -> tuple[Digest, bool]:
    """Append later acceptances and retry missing summaries without duplicate items.

    Already delivered papers retain their slots. Remaining slots go to eligible
    papers ranked by the active scoring method. The cycle lock serializes resume
    requests, including summary generation, on PostgreSQL.
    """
    from research_monitor.config import config
    from research_monitor.database.models import Cycle

    if max_digest_items is None:
        max_digest_items = config.DIGEST_MAX_ITEMS
    if max_digest_items < 0:
        raise ValueError("Digest cap cannot be negative.")
    cycle = session.query(Cycle).filter_by(id=cycle_id).with_for_update().first()
    if cycle is None:
        raise ValueError("Cycle not found.")
    if cycle.user_id != user_id:
        raise PermissionError("Cycle does not belong to user.")
    current_time = now or datetime.now(timezone.utc)
    expire_stale_tasks(user_id=user_id, session=session, now=current_time, commit=False)
    digest = session.query(Digest).filter_by(user_id=user_id, cycle_id=cycle_id).first()
    existed = digest is not None
    if digest is None:
        digest = Digest(user_id=user_id, cycle_id=cycle_id, created_at=current_time)
        session.add(digest)
        session.flush()

    existing = {item.decision_id: item for item in digest.items}
    decisions = session.query(RoutingDecision).filter_by(user_id=user_id, cycle_id=cycle_id).all()
    candidates = []
    for dec in decisions:
        if dec.id in existing or dec.status not in {"ROUTED_HIGH", "SUMMARIZE_PENDING", "SUMMARIZED"}:
            continue
        accepted = any(t.decision == "accept" and t.kind != "audit" for t in dec.review_tasks)
        if dec.route == "HIGH" or (dec.route == "BORDERLINE" and accepted):
            candidates.append((dec, "auto" if dec.route == "HIGH" else "review"))

    def score(dec):
        return dec.clf_prob if dec.method == "classifier" and dec.clf_prob is not None else dec.cosine_score

    candidates.sort(key=lambda pair: (-score(pair[0]), pair[0].paper_id))
    available = max(0, max_digest_items - len(existing))
    for dec, via in candidates[:available]:
        item = DigestItem(decision_id=dec.id, position=len(existing) + 1, via=via)
        digest.items.append(item)
        existing[dec.id] = item
    for dec, _ in candidates[available:]:
        dec.status = DecisionStatus.DEFERRED_OVERFLOW.value
        dec.reason += f" | Digest cap ({max_digest_items}) exceeded. Logged as DEFERRED_OVERFLOW."
    session.flush()

    digest.items.sort(key=lambda item: (-score(item.routing_decision), item.routing_decision.paper_id))
    for position, item in enumerate(digest.items, start=1):
        item.position = position
        dec = item.routing_decision
        if dec.summary is None:
            summarize_decision(dec.id, user_id, session, summarizer=summarizer, commit=False)
        if dec.summary is not None and dec.status != DecisionStatus.REACTED.value:
            dec.status = DecisionStatus.DELIVERED.value
    session.commit()
    return digest, existed


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
