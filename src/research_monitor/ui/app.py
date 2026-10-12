"""
app.py
~~~~~~
FastAPI application for Phase 6 Human Review, Digest, and Feedback UI.

Follows Roadmap Section 6, 7.3, 7.6, and 9.1:
  - Access-token based authentication via URL parameters (/page?token=... or /page/{token}).
  - Human review queue for borderline candidate papers.
  - Digest presentation with auto-cleared and review-accepted papers.
  - Interactive Like / Dislike controls on delivered papers.
  - Transparent audit view of discarded LOW papers.
  - Strict user data isolation: 401 on invalid token, 403 on foreign resource access.
  - Tokens and sensitive credentials are never exposed in log output.
"""

from pathlib import Path
import logging
from typing import Generator, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    ReviewTask,
    RoutingDecision,
    User,
)
from research_monitor.database.session import SessionLocal
from research_monitor.relevance.state import DecisionStatus
from research_monitor.review.digest import (
    record_user_feedback,
    resume_after_review,
)
from research_monitor.review.queue import record_review_action
from research_monitor.summary.service import summarize_decision_background

logger = logging.getLogger(__name__)

# Template configuration
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Research Monitor UI", version="0.6.0")


def get_db() -> Generator[Session, None, None]:
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def authenticate_user(
    token: Optional[str] = Query(None),
    session: Session = Depends(get_db),
) -> tuple[User, str]:
    """
    Validate per-user URL access token without logging or leaking secrets.
    """
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied: URL access token is required.",
        )

    user = session.query(User).filter(User.access_token == token.strip()).first()
    if not user:
        # Generic unauthenticated error; never leak tokens
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied: Invalid access token.",
        )

    return user, token.strip()


# =========================================================================
# Human Review Queue Routes
# =========================================================================

@app.get("/review", response_class=HTMLResponse)
def review_page_query(
    request: Request,
    auth: tuple[User, str] = Depends(authenticate_user),
    session: Session = Depends(get_db),
):
    """Review queue page with query parameter token."""
    user, token = auth
    return _render_review_page(request, user, token, session)


@app.get("/review/{token_param}", response_class=HTMLResponse)
def review_page_path(
    request: Request,
    token_param: str,
    session: Session = Depends(get_db),
):
    """Review queue page with path token."""
    user, token = authenticate_user(token=token_param, session=session)
    return _render_review_page(request, user, token, session)


def _render_review_page(request: Request, user: User, token: str, session: Session) -> HTMLResponse:
    # Fetch pending review tasks for this user, ordered by score DESC
    tasks = (
        session.query(ReviewTask)
        .join(RoutingDecision, ReviewTask.decision_id == RoutingDecision.id)
        .filter(
            RoutingDecision.user_id == user.id,
            (ReviewTask.decision.is_(None) | (ReviewTask.decision == "skip")),
            ReviewTask.expired_at.is_(None),
        )
        .order_by(RoutingDecision.cosine_score.desc())
        .all()
    )

    # Get latest cycle id for convenience button
    latest_cycle = (
        session.query(Cycle)
        .filter(Cycle.user_id == user.id)
        .order_by(Cycle.id.desc())
        .first()
    )
    cycle_id = latest_cycle.id if latest_cycle else None

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "user": user,
            "token": token,
            "tasks": tasks,
            "cycle_id": cycle_id,
            "active_tab": "review",
        },
    )


@app.post("/review/{task_id}/action", response_class=HTMLResponse)
def handle_review_action(
    task_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    action: str = Query(...),
    token: Optional[str] = Query(None),
    auth: tuple[User, str] = Depends(authenticate_user),
    session: Session = Depends(get_db),
):
    """
    Handle HTMX Accept, Reject, or Skip button clicks.
    """
    user, verified_token = auth
    try:
        updated_task = record_review_action(
            task_id=task_id,
            action=action,
            user_id=user.id,
            session=session,
        )
        if action.lower().strip() == "accept":
            background_tasks.add_task(
                summarize_decision_background,
                decision_id=updated_task.decision_id,
                user_id=user.id,
            )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))

    return templates.TemplateResponse(
        request=request,
        name="_review_card.html",
        context={
            "task": updated_task,
            "token": verified_token,
        },
    )


# =========================================================================
# Research Digest Routes
# =========================================================================

@app.get("/digest", response_class=HTMLResponse)
def digest_page_query(
    request: Request,
    cycle_id: Optional[int] = Query(None),
    auth: tuple[User, str] = Depends(authenticate_user),
    session: Session = Depends(get_db),
):
    """Personalized research digest page with query parameter token."""
    user, token = auth
    return _render_digest_page(request, user, token, cycle_id, session)


@app.get("/digest/{token_param}", response_class=HTMLResponse)
def digest_page_path(
    request: Request,
    token_param: str,
    cycle_id: Optional[int] = Query(None),
    session: Session = Depends(get_db),
):
    """Personalized research digest page with path token."""
    user, token = authenticate_user(token=token_param, session=session)
    return _render_digest_page(request, user, token, cycle_id, session)


def _render_digest_page(
    request: Request,
    user: User,
    token: str,
    cycle_id: Optional[int],
    session: Session,
) -> HTMLResponse:
    query = session.query(Digest).filter(Digest.user_id == user.id)
    if cycle_id:
        query = query.filter(Digest.cycle_id == cycle_id)
    else:
        query = query.order_by(Digest.id.desc())

    digest = query.first()

    # Map latest user reactions on digest items
    reactions: dict[int, int] = {}
    if digest and digest.items:
        item_pids = [item.routing_decision.paper_id for item in digest.items]
        user_events = (
            session.query(FeedbackEvent)
            .filter(
                FeedbackEvent.user_id == user.id,
                FeedbackEvent.source == "user",
                FeedbackEvent.paper_id.in_(item_pids),
            )
            .order_by(FeedbackEvent.created_at.asc(), FeedbackEvent.id.asc())
            .all()
        )
        pid_to_label = {ev.paper_id: ev.label for ev in user_events}
        for item in digest.items:
            if item.routing_decision.paper_id in pid_to_label:
                reactions[item.id] = pid_to_label[item.routing_decision.paper_id]

    return templates.TemplateResponse(
        request=request,
        name="digest.html",
        context={
            "user": user,
            "token": token,
            "digest": digest,
            "reactions": reactions,
            "active_tab": "digest",
        },
    )


@app.post("/digest/{item_id}/feedback", response_class=HTMLResponse)
def handle_digest_feedback(
    item_id: int,
    request: Request,
    reaction: str = Query(...),
    auth: tuple[User, str] = Depends(authenticate_user),
    session: Session = Depends(get_db),
):
    """
    Handle HTMX Like or Dislike clicks on digest items.
    """
    user, verified_token = auth
    try:
        event = record_user_feedback(
            digest_item_id=item_id,
            reaction=reaction,
            user_id=user.id,
            session=session,
        )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))

    item = session.query(DigestItem).filter(DigestItem.id == item_id).first()
    return templates.TemplateResponse(
        request=request,
        name="_feedback_buttons.html",
        context={
            "item": item,
            "token": verified_token,
            "user_reaction": event.label,
        },
    )


@app.post("/cycle/{cycle_id}/resume", response_class=HTMLResponse)
def handle_resume_cycle(
    cycle_id: int,
    auth: tuple[User, str] = Depends(authenticate_user),
    session: Session = Depends(get_db),
):
    """
    Trigger idempotent resume_after_review operation for a cycle.
    """
    user, _ = auth
    digest, already_existed = resume_after_review(
        user_id=user.id,
        cycle_id=cycle_id,
        session=session,
    )
    status_text = "Retrieved existing digest" if already_existed else "Created new digest"
    return HTMLResponse(
        f"""<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent-green); padding: 0.75rem; border-radius: 6px; color: var(--accent-green);">
            &#10004; {status_text} #{digest.id} with {len(digest.items)} items. <a href="/digest?token={auth[1]}" style="color: var(--primary); text-decoration: underline;">View Digest &rarr;</a>
        </div>"""
    )


# =========================================================================
# Discarded Papers Audit Routes
# =========================================================================

@app.get("/audit", response_class=HTMLResponse)
def audit_page_query(
    request: Request,
    auth: tuple[User, str] = Depends(authenticate_user),
    session: Session = Depends(get_db),
):
    """Audit log page with query parameter token."""
    user, token = auth
    return _render_audit_page(request, user, token, session)


@app.get("/audit/{token_param}", response_class=HTMLResponse)
def audit_page_path(
    request: Request,
    token_param: str,
    session: Session = Depends(get_db),
):
    """Audit log page with path token."""
    user, token = authenticate_user(token=token_param, session=session)
    return _render_audit_page(request, user, token, session)


def _render_audit_page(request: Request, user: User, token: str, session: Session) -> HTMLResponse:
    # Fetch discarded LOW papers for this user
    decisions = (
        session.query(RoutingDecision)
        .filter(
            RoutingDecision.user_id == user.id,
            RoutingDecision.route == "LOW",
            RoutingDecision.status == DecisionStatus.DISCARDED.value,
        )
        .order_by(RoutingDecision.cosine_score.desc())
        .limit(100)
        .all()
    )

    return templates.TemplateResponse(
        request=request,
        name="audit.html",
        context={
            "user": user,
            "token": token,
            "decisions": decisions,
            "active_tab": "audit",
        },
    )
