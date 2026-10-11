"""
queue.py
~~~~~~~~
Human review queue management, decision actions, and task expiry (Phase 6).

Follows Roadmap Section 6, 7.3, 7.6, and 9.1:
  - BORDERLINE papers queued up to 15 per user per cycle, ordered by score (highest first).
  - Overflow logged as DEFERRED_OVERFLOW on routing_decisions.
  - Accept -> SUMMARIZE_PENDING + 1 reviewer feedback event (label=1).
  - Reject -> REVIEW_REJECTED + 1 reviewer feedback event (label=0).
  - Skip -> recorded on task only, leaves task pending, no feedback event.
  - Tasks undecided after 3 days -> REVIEW_EXPIRED (terminal, no label).
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Optional
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    FeedbackEvent,
    ReviewTask,
    RoutingDecision,
)
from research_monitor.relevance.state import DecisionStatus

logger = logging.getLogger(__name__)

MAX_REVIEW_TASKS_PER_CYCLE = 15
REVIEW_EXPIRY_DAYS = 3


def build_review_queue(
    user_id: str,
    cycle_id: int,
    session: Session,
    max_tasks: int = MAX_REVIEW_TASKS_PER_CYCLE,
    enable_audit_sample: bool = False,
) -> list[ReviewTask]:
    """
    Populate human review queue for BORDERLINE papers from a monitoring cycle.
    Enforces maximum of 15 review tasks per user per cycle, ordered by score DESC.
    Logs overflow papers as DEFERRED_OVERFLOW.
    """
    # 1. Fetch unqueued borderline routing decisions for this cycle
    query = (
        session.query(RoutingDecision)
        .filter(
            RoutingDecision.user_id == user_id,
            RoutingDecision.cycle_id == cycle_id,
            RoutingDecision.route == "BORDERLINE",
            RoutingDecision.status == DecisionStatus.ROUTED_BORDERLINE.value,
        )
        .order_by(RoutingDecision.cosine_score.desc())
    )
    borderline_decisions = query.all()

    created_tasks: list[ReviewTask] = []
    now = datetime.now(timezone.utc)

    # 2. Queue up to max_tasks
    eligible_decisions = borderline_decisions[:max_tasks]
    overflow_decisions = borderline_decisions[max_tasks:]

    for dec in eligible_decisions:
        task = ReviewTask(
            decision_id=dec.id,
            kind="borderline",
            queued_at=now,
        )
        dec.status = DecisionStatus.PENDING_REVIEW.value
        session.add(task)
        created_tasks.append(task)

    # 3. Log overflow papers as DEFERRED_OVERFLOW
    for dec in overflow_decisions:
        dec.status = DecisionStatus.DEFERRED_OVERFLOW.value
        dec.reason = (
            f"{dec.reason} | Review queue cap ({max_tasks}) exceeded. Logged as DEFERRED_OVERFLOW."
        )

    # 4. Optional: audit sample (disabled by default)
    if enable_audit_sample:
        low_query = (
            session.query(RoutingDecision)
            .filter(
                RoutingDecision.user_id == user_id,
                RoutingDecision.cycle_id == cycle_id,
                RoutingDecision.route == "LOW",
                RoutingDecision.status == DecisionStatus.DISCARDED.value,
            )
            .limit(3)
        )
        for dec in low_query.all():
            task = ReviewTask(
                decision_id=dec.id,
                kind="audit",
                queued_at=now,
            )
            session.add(task)
            created_tasks.append(task)

    session.commit()
    logger.info(
        "Built review queue for user '%s', cycle %d: %d tasks queued, %d deferred overflow.",
        user_id,
        cycle_id,
        len(eligible_decisions),
        len(overflow_decisions),
    )
    return created_tasks


def record_review_action(
    task_id: int,
    action: str,
    user_id: str,
    session: Session,
) -> ReviewTask:
    """
    Apply a reviewer decision (accept, reject, skip) to a review task.
    Enforces user access control and records reviewer feedback events.
    """
    task = session.query(ReviewTask).filter(ReviewTask.id == task_id).first()
    if not task:
        raise ValueError(f"ReviewTask with id {task_id} not found.")

    dec = task.routing_decision
    if not dec or dec.user_id != user_id:
        raise PermissionError(f"ReviewTask {task_id} does not belong to user '{user_id}'.")

    action_norm = action.lower().strip()
    now = datetime.now(timezone.utc)

    if action_norm == "accept":
        task.decision = "accept"
        task.decided_at = now
        dec.status = DecisionStatus.SUMMARIZE_PENDING.value

        # Create exactly one reviewer feedback event
        event = FeedbackEvent(
            user_id=user_id,
            paper_id=dec.paper_id,
            decision_id=dec.id,
            source="reviewer",
            label=1,
            is_audit=(task.kind == "audit"),
            created_at=now,
        )
        session.add(event)

    elif action_norm == "reject":
        task.decision = "reject"
        task.decided_at = now
        dec.status = DecisionStatus.REVIEW_REJECTED.value

        # Create exactly one reviewer feedback event
        event = FeedbackEvent(
            user_id=user_id,
            paper_id=dec.paper_id,
            decision_id=dec.id,
            source="reviewer",
            label=0,
            is_audit=(task.kind == "audit"),
            created_at=now,
        )
        session.add(event)

    elif action_norm == "skip":
        task.decision = "skip"
        task.decided_at = now
        # Skip is recorded on task only; leaves decision pending, creates no feedback event
        dec.status = DecisionStatus.PENDING_REVIEW.value

    else:
        raise ValueError(f"Unsupported review action '{action}'. Must be 'accept', 'reject', or 'skip'.")

    session.commit()
    logger.info("Recorded review action '%s' for task %d (user '%s').", action_norm, task_id, user_id)
    return task


def expire_stale_tasks(
    user_id: str,
    session: Session,
    expiry_days: int = REVIEW_EXPIRY_DAYS,
    now: Optional[datetime] = None,
) -> list[ReviewTask]:
    """
    Expire undecided or skipped tasks older than expiry_days (default: 3 days).
    Sets task.decision='expired', task.expired_at=now, and decision.status='REVIEW_EXPIRED'.
    Creates no feedback event.
    """
    current_time = now or datetime.now(timezone.utc)
    cutoff = current_time - timedelta(days=expiry_days)

    stale_tasks = (
        session.query(ReviewTask)
        .join(RoutingDecision, ReviewTask.decision_id == RoutingDecision.id)
        .filter(
            RoutingDecision.user_id == user_id,
            ReviewTask.queued_at <= cutoff,
            ReviewTask.expired_at.is_(None),
            (ReviewTask.decision.is_(None) | (ReviewTask.decision == "skip")),
        )
        .all()
    )

    expired: list[ReviewTask] = []
    for task in stale_tasks:
        task.decision = "expired"
        task.expired_at = current_time
        if task.routing_decision:
            task.routing_decision.status = DecisionStatus.REVIEW_EXPIRED.value
        expired.append(task)

    if expired:
        session.commit()
        logger.info("Expired %d stale review tasks for user '%s'.", len(expired), user_id)

    return expired
