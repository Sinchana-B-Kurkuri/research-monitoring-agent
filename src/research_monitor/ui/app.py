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
import re
import secrets
import subprocess
import sys
import time
from typing import Generator, Optional
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.orm import Session, joinedload, selectinload

from research_monitor.config import config
from research_monitor.ui.jobs import processing_slot
from research_monitor.ui.workspace import active_token, accessible_profiles, is_operator, remember_profile

from research_monitor.database.models import (
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    InterestVectorVersion,
    ReviewTask,
    RoutingDecision,
    User,
)
from research_monitor.database.session import SessionLocal
from research_monitor.relevance.state import DecisionStatus
from research_monitor.relevance.thresholds import ReferenceEmbeddingsUnavailable
from research_monitor.review.digest import (
    record_user_feedback,
    resume_after_review,
)
from research_monitor.review.queue import record_review_action
from research_monitor.summary.service import summarize_decision_background

class AccessTokenLogFilter(logging.Filter):
    """Redact bearer links from Uvicorn's access-log arguments."""
    def filter(self, record):
        def redact(value):
            if not isinstance(value, str):
                return value
            value = re.sub(r"([?&](?:token|workspace_key)=)[^&\s]*", r"\1[REDACTED]", value)
            return re.sub(r"(/(?:review|digest|audit)/)[^/?\s]+(?=\?|$)", r"\1[REDACTED]", value)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(value) for value in record.args)
        record.msg = redact(record.msg)
        return True


logging.getLogger("uvicorn.access").addFilter(AccessTokenLogFilter())

logger = logging.getLogger("uvicorn.error").getChild(__name__)

# Template and script paths
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CREATE_USER_SCRIPT = PROJECT_ROOT / "scripts" / "create_user.py"

app = FastAPI(title="Research Monitor UI", version="0.9.0")
app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR.parent / "static")), name="static")


@app.middleware("http")
async def protect_personal_links(request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        expected_origin = str(request.base_url).rstrip("/")
        if request.headers.get("sec-fetch-site") == "cross-site" or (origin and origin != expected_origin):
            return JSONResponse({"detail": "Cross-site actions are not allowed."}, status_code=403)
    response = await call_next(request)
    # Let the native onboarding form send its actual Origin while keeping
    # paths and query strings out of the Referer header.
    is_onboarding_html = (
        request.url.path == "/onboard"
        and response.headers.get("content-type", "").startswith("text/html")
    )
    response.headers["Referrer-Policy"] = "strict-origin" if is_onboarding_html else "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    token = getattr(request.state, "profile_token", None)
    if token:
        remember_profile(response, request, token)
    return response


def get_db() -> Generator[Session, None, None]:
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================================================================
# Database Health Check & Root Portal
# =========================================================================

@app.get("/health")
def health_check(session: Session = Depends(get_db)):
    """
    Check application and database connectivity for local diagnostics.
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
    token = token or active_token(request)
    if token and token.strip():
        user = session.query(User).filter_by(access_token=token.strip()).first()
        if user:
            request.state.profile_token = token.strip()
            return RedirectResponse(
                url=f"/dashboard?token={token.strip()}",
                status_code=status.HTTP_302_FOUND,
            )
    return RedirectResponse(url="/workspace" if is_operator(request) else "/onboard", status_code=status.HTTP_302_FOUND)


# =========================================================================
# Cold-Start Pilot Onboarding Routes
# =========================================================================

@app.get("/onboard", response_class=HTMLResponse)
def onboard_page(request: Request, session: Session = Depends(get_db)):
    """Render cold-start pilot researcher onboarding form."""
    from research_monitor.ui.portal import profile_cards
    cards = profile_cards(request, session)
    return templates.TemplateResponse(
        request=request,
        name="onboard.html",
        context={
            "active_tab": "onboard",
            "token": None,
            "user": None,
            "cards": cards,
            "operator": is_operator(request),
        },
    )


def run_onboarding_worker(bind, command):
    with processing_slot(bind, wait=False) as acquired:
        if not acquired:
            return None
        return subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=90,
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
    inherit_from_user = ""

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
            user_id = str(data.get("user_id", "")).strip()
            name = str(data.get("name", "")).strip()
            seed_text = str(data.get("seed_text", "")).strip()
            inherit_from_user = str(data.get("inherit_from_user", "")).strip()
        except Exception:
            pass
    else:
        body_bytes = await request.body()
        parsed = parse_qs(body_bytes.decode("utf-8", errors="replace"))
        user_id = parsed.get("user_id", [""])[0].strip()
        name = parsed.get("name", [""])[0].strip()
        seed_text = parsed.get("seed_text", [""])[0].strip()
        inherit_from_user = parsed.get("inherit_from_user", [""])[0].strip()

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

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id) or len(name) > 200 or len(seed_text) > 10000:
        raise HTTPException(status_code=400, detail="Invalid user identifier or input too long.")
    existing_user = session.query(User).filter_by(id=user_id).first()
    if inherit_from_user and inherit_from_user not in {profile.id for profile in accessible_profiles(request, session) if not profile.is_simulated}:
        raise HTTPException(403, "Choose a source profile that you can access in this workspace.")
    if existing_user:
        supplied_token = request.query_params.get("token", "")
        if not existing_user.access_token or not secrets.compare_digest(supplied_token, existing_user.access_token):
            raise HTTPException(status_code=409, detail="User ID already registered. Use your existing personal access link.")
        active = session.query(InterestVectorVersion).filter_by(user_id=user_id, is_active=True).first()
        if active is not None:
            request.state.profile_token = existing_user.access_token
            return JSONResponse(content={
                "status": "ok", "user_id": existing_user.id, "name": existing_user.name,
                "access_token": existing_user.access_token,
                "review_url": f"/review?token={supplied_token}",
                "digest_url": f"/digest?token={supplied_token}",
                "audit_url": f"/audit?token={supplied_token}",
            })

    # The lookup above starts a transaction and holds a users-table read lock.
    # Release it before waiting for another process that writes to that database.
    bind = session.get_bind()
    session.rollback()

    # Execute create_user.py in separate subprocess to preserve web process isolation.
    # Schema migrations belong to setup, not an interactive signup request.
    logger.info("Starting isolated onboarding worker")
    start_time = time.perf_counter()
    try:
        proc = await run_in_threadpool(
            run_onboarding_worker, bind,
            [
                sys.executable,
                str(CREATE_USER_SCRIPT),
                "--user-id", user_id,
                "--name", name,
                "--seed-text", seed_text,
                "--skip-db-init",
                *(["--inherit-from-user", inherit_from_user] if inherit_from_user else []),
                *(["--require-new"] if existing_user is None else []),
            ],
        )
    except subprocess.TimeoutExpired:
        logger.error("Onboarding worker timed out after %.2fs", time.perf_counter() - start_time)
        error_msg = "User initialization timed out. Please try again."
        error_status = status.HTTP_504_GATEWAY_TIMEOUT
    except OSError as exc:
        logger.error("Unable to start onboarding worker (%s)", type(exc).__name__)
        error_msg = "Unable to start user initialization. Check the server configuration."
        error_status = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        error_msg = None
        if proc is None:
            logger.info("Onboarding worker deferred because another processing job is active")
            error_msg = "A processing session is already running. Try again when it finishes."
            error_status = status.HTTP_503_SERVICE_UNAVAILABLE
        elif proc.returncode != 0:
            logger.error("Onboarding worker exited with code %d after %.2fs",
                         proc.returncode, time.perf_counter() - start_time)
            error_msg = "User initialization failed. Check the server configuration or choose an unused user ID."
            error_status = status.HTTP_500_INTERNAL_SERVER_ERROR
            if proc.stderr.splitlines()[-1:] == [ReferenceEmbeddingsUnavailable.__name__]:
                logger.error("Onboarding requires reference embeddings in the database or corpus files")
                error_msg = "User initialization is unavailable until the reference paper embeddings are loaded."
                error_status = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            logger.info("Onboarding worker completed in %.2fs", time.perf_counter() - start_time)

    if error_msg:
        if "application/json" in content_type or request.headers.get("accept", "") == "application/json":
            raise HTTPException(status_code=error_status, detail=error_msg)
        return templates.TemplateResponse(
            request=request,
            name="onboard.html",
            status_code=error_status,
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

    request.state.profile_token = user.access_token

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
    request: Request = None,
) -> tuple[User, str]:
    """
    Validate per-user URL access token without logging or leaking secrets.
    """
    if token is None and request is not None:
        token = active_token(request)
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

    if request is not None:
        request.state.profile_token = token.strip()
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
    request.state.profile_token = token
    from research_monitor.review.queue import expire_stale_tasks
    expire_stale_tasks(user.id, session)
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
            "review_cycle_ids": sorted({t.routing_decision.cycle_id for t in tasks if t.routing_decision.cycle_id}),
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
        raise HTTPException(status_code=404 if "not found" in str(ve) else 409, detail=str(ve))

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
    request.state.profile_token = token
    query = session.query(Digest).options(
        selectinload(Digest.items).joinedload(DigestItem.routing_decision).joinedload(RoutingDecision.paper),
        selectinload(Digest.items).joinedload(DigestItem.routing_decision).joinedload(RoutingDecision.summary),
    ).filter(Digest.user_id == user.id)
    if cycle_id:
        query = query.filter(Digest.cycle_id == cycle_id)
    else:
        query = query.order_by(Digest.id.desc())

    digest = query.first()
    latest_cycle = session.query(Cycle).filter_by(user_id=user.id).order_by(Cycle.id.desc()).first()
    selected_cycle_id = digest.cycle_id if digest else (latest_cycle.id if latest_cycle else None)
    pending_summaries = sum(1 for item in digest.items if item.routing_decision.summary is None) if digest else 0

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
            "cycle_id": selected_cycle_id,
            "pending_summaries": pending_summaries,
            "gemini_configured": bool(config.GEMINI_API_KEY),
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
    try:
        digest, already_existed = resume_after_review(user_id=user.id, cycle_id=cycle_id, session=session)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    status_text = "Updated existing digest" if already_existed else "Created new digest"
    pending = sum(1 for item in digest.items if item.routing_decision.summary is None)
    pending_text = f" {pending} summaries pending; original abstracts are available." if pending else ""
    return HTMLResponse(
        f"""<div class="notice" role="status">
            &#10004; {status_text} #{digest.id} with {len(digest.items)} items.{pending_text} <a href="/digest?token={auth[1]}&amp;cycle_id={cycle_id}" style="color: var(--primary); text-decoration: underline;">View Digest &rarr;</a>
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
    request.state.profile_token = token
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


from research_monitor.ui.portal import create_portal_router
app.include_router(create_portal_router(templates, get_db, authenticate_user))
