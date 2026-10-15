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
import subprocess
import sys
from typing import Generator, Optional
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
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

# Template and script paths
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CREATE_USER_SCRIPT = PROJECT_ROOT / "scripts" / "create_user.py"

app = FastAPI(title="Research Monitor UI", version="0.9.0")


def get_db() -> Generator[Session, None, None]:
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================================================================
# Production Health Check & Root Portal
# =========================================================================

@app.get("/health")
def health_check(session: Session = Depends(get_db)):
    """
    Production health check endpoint verifying application and database connectivity.
    Returns 200 OK with connectivity telemetry; 503 SERVICE UNAVAILABLE on failure.
    """
    try:
        session.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "connected",
            "version": "0.9.0",
        }
    except Exception as exc:
        logger.error("Health check database probe failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "degraded",
                "database": "disconnected",
                "error": "Database connectivity check failed",
            },
        )


@app.get("/", response_class=HTMLResponse)
def root_page(
    request: Request,
    token: Optional[str] = Query(None),
    session: Session = Depends(get_db),
):
    """
    Root portal landing:
    - If valid user token provided in query, redirect to their digest.
    - Otherwise, redirect to cold-start pilot onboarding.
    """
    if token and token.strip():
        user = session.query(User).filter_by(access_token=token.strip()).first()
        if user:
            return RedirectResponse(
                url=f"/digest?token={token.strip()}",
                status_code=status.HTTP_302_FOUND,
            )
    return RedirectResponse(url="/onboard", status_code=status.HTTP_302_FOUND)


# =========================================================================
# Cold-Start Pilot Onboarding Routes
# =========================================================================

@app.get("/onboard", response_class=HTMLResponse)
def onboard_page(request: Request):
    """Render cold-start pilot researcher onboarding form."""
    return templates.TemplateResponse(
        request=request,
        name="onboard.html",
        context={
            "active_tab": "onboard",
            "token": None,
            "user": None,
        },
    )


@app.post("/onboard")
async def handle_onboard_user(
    request: Request,
    session: Session = Depends(get_db),
):
    """
    Onboard a new pilot user:
    - Parses user_id, name, and seed_text without requiring python-multipart.
    - Executes scripts/create_user.py in an isolated subprocess to keep the web
      service completely free of PyTorch / sentence_transformers.
    - Returns personalized token and links to their review queue and digest.
    """
    user_id = ""
    name = ""
    seed_text = ""

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
            user_id = str(data.get("user_id", "")).strip()
            name = str(data.get("name", "")).strip()
            seed_text = str(data.get("seed_text", "")).strip()
        except Exception:
            pass
    else:
        body_bytes = await request.body()
        parsed = parse_qs(body_bytes.decode("utf-8", errors="replace"))
        user_id = parsed.get("user_id", [""])[0].strip()
        name = parsed.get("name", [""])[0].strip()
        seed_text = parsed.get("seed_text", [""])[0].strip()

    # Query params fallback
    if not user_id:
        user_id = request.query_params.get("user_id", "").strip()
    if not name:
        name = request.query_params.get("name", "").strip()
    if not seed_text:
        seed_text = request.query_params.get("seed_text", "").strip()

    if not user_id or not name or not seed_text:
        error_msg = "User ID, Display Name, and Research Interests paragraph are all required."
        if "application/json" in content_type or request.headers.get("accept", "") == "application/json":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
        return templates.TemplateResponse(
            request=request,
            name="onboard.html",
            context={
                "error": error_msg,
                "form_user_id": user_id,
                "form_name": name,
                "form_seed_text": seed_text,
                "active_tab": "onboard",
                "token": None,
                "user": None,
            },
        )

    # Execute create_user.py in separate subprocess to preserve web process isolation
    proc = subprocess.run(
        [
            sys.executable,
            str(CREATE_USER_SCRIPT),
            "--user-id", user_id,
            "--name", name,
            "--seed-text", seed_text,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        logger.error("User initialization subprocess failed: %s", proc.stderr)
        error_msg = f"User initialization failed: {proc.stderr.strip() or 'Unknown subprocess error'}"
        if "application/json" in content_type or request.headers.get("accept", "") == "application/json":
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg)
        return templates.TemplateResponse(
            request=request,
            name="onboard.html",
            context={
                "error": error_msg,
                "form_user_id": user_id,
                "form_name": name,
                "form_seed_text": seed_text,
                "active_tab": "onboard",
                "token": None,
                "user": None,
            },
        )

    # Retrieve created or existing user from database session
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User was initialized by script but not found in active database.",
        )

    if "application/json" in content_type or request.headers.get("accept", "") == "application/json":
        return JSONResponse(
            content={
                "status": "ok",
                "user_id": user.id,
                "name": user.name,
                "access_token": user.access_token,
                "review_url": f"/review?token={user.access_token}",
                "digest_url": f"/digest?token={user.access_token}",
                "audit_url": f"/audit?token={user.access_token}",
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="onboard_success.html",
        context={
            "user": user,
            "token": user.access_token,
            "active_tab": "onboard",
        },
    )


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

    from research_monitor.relevance.classifier import find_nearest_feedback_papers
    for t in tasks:
        if t.routing_decision and t.routing_decision.paper:
            liked, rejected = find_nearest_feedback_papers(
                paper_embedding=t.routing_decision.paper.embedding,
                user_id=user.id,
                session=session,
            )
            t.nearest_liked = liked
            t.nearest_rejected = rejected

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

    from research_monitor.relevance.classifier import find_nearest_feedback_papers
    if updated_task.routing_decision and updated_task.routing_decision.paper:
        liked, rejected = find_nearest_feedback_papers(
            paper_embedding=updated_task.routing_decision.paper.embedding,
            user_id=user.id,
            session=session,
        )
        updated_task.nearest_liked = liked
        updated_task.nearest_rejected = rejected

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
