"""Profile dashboard, operator directory, cycle history, and worker controls."""
import secrets
import re
from urllib.parse import parse_qs

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from research_monitor.config import config
from research_monitor.database.models import (BackgroundJob, Cycle, Digest, FeedbackEvent,
    InterestProfile, InterestVectorVersion, ModelVersion, RoutingDecision, Summary, TrainingLabel, User)
from research_monitor.ui.jobs import enqueue_job, expire_interrupted_jobs, run_background_job
from research_monitor.ui.workspace import ROOT, accessible_profiles, is_operator, workspace_key, OPERATOR_COOKIE

HIDDEN_PROFILES_PATH = ROOT / "data/local_runtime/hidden_pilots.txt"


def profile_cards(request, session):
    profiles = accessible_profiles(request, session)
    try:
        hidden_ids = set(HIDDEN_PROFILES_PATH.read_text(encoding="utf-8").split())
    except FileNotFoundError:
        hidden_ids = set()
    cycles = dict(session.query(Cycle.user_id, func.count(Cycle.id)).group_by(Cycle.user_id).all())
    digests = dict(session.query(Digest.user_id, func.count(Digest.id)).group_by(Digest.user_id).all())
    labels = dict(session.query(TrainingLabel.user_id, func.count(TrainingLabel.event_id)).group_by(TrainingLabel.user_id).all())
    return [{"user": user, "cycles": cycles.get(user.id, 0), "digests": digests.get(user.id, 0),
        "labels": labels.get(user.id, 0)} for user in profiles if user.id not in hidden_ids]


def personal_stats(user_id, session):
    labels = session.query(TrainingLabel).filter_by(user_id=user_id).all()
    events = session.query(FeedbackEvent).filter_by(user_id=user_id).count()
    return {"cycles": session.query(Cycle).filter_by(user_id=user_id).count(),
        "digests": session.query(Digest).filter_by(user_id=user_id).count(),
        "summaries": session.query(Summary).filter_by(user_id=user_id).count(),
        "labels": len(labels), "positive": sum(label.label == 1 for label in labels),
        "negative": sum(label.label == 0 for label in labels), "events": events,
        "inherited": sum(label.origin.startswith("inherited:") for label in labels)}


def job_payload(job):
    return {"id": job.id, "kind": job.kind, "status": job.status, "phase": job.phase,
        "cycle_id": job.cycle_id, "result": job.result, "error": job.error}


def create_portal_router(templates, get_db, authenticate_user):
    router = APIRouter()

    @router.get("/workspace", response_class=HTMLResponse)
    def workspace_page(request: Request, workspace_key_param: str = Query(None, alias="workspace_key"), session: Session = Depends(get_db)):
        if workspace_key_param:
            if not secrets.compare_digest(workspace_key_param, workspace_key()):
                raise HTTPException(401, "Invalid workspace access link.")
            response = RedirectResponse("/workspace", status_code=303)
            response.set_cookie(OPERATOR_COOKIE, workspace_key(), httponly=True, samesite="lax",
                secure=request.url.scheme == "https", max_age=60*60*24*365)
            return response
        cards = profile_cards(request, session)
        return templates.TemplateResponse(request=request, name="workspace.html", context={
            "cards": cards, "operator": is_operator(request), "user": None, "token": None,
            "active_tab": "workspace"})

    @router.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request, auth=Depends(authenticate_user), session: Session = Depends(get_db)):
        user, token = auth
        expire_interrupted_jobs(session, user.id)
        session.commit()
        jobs = session.query(BackgroundJob).filter_by(user_id=user.id).order_by(BackgroundJob.created_at.desc()).limit(12).all()
        recent_monitor = next((job for job in jobs if job.kind == "monitor"), None)
        profile = session.query(InterestProfile).filter_by(user_id=user.id).order_by(InterestProfile.id.desc()).first()
        vector = session.query(InterestVectorVersion).filter_by(user_id=user.id, is_active=True).first()
        return templates.TemplateResponse(request=request, name="dashboard.html", context={
            "user": user, "token": token, "stats": personal_stats(user.id, session), "profile": profile,
            "vector": vector, "jobs": jobs, "settings": recent_monitor.options if recent_monitor else {
                "categories": config.CORPUS_CATEGORIES.split(","), "days": 1, "max_results": 50},
            "active_tab": "dashboard"})

    @router.get("/history", response_class=HTMLResponse)
    def history(request: Request, page: int = Query(1, ge=1), feedback_page: int = Query(1, ge=1), jobs_page: int = Query(1, ge=1), auth=Depends(authenticate_user), session: Session = Depends(get_db)):
        user, token = auth
        cycles = session.query(Cycle).filter_by(user_id=user.id).order_by(Cycle.id.desc()).offset((page-1)*25).limit(26).all()
        jobs = session.query(BackgroundJob).filter_by(user_id=user.id).order_by(BackgroundJob.created_at.desc()).offset((jobs_page-1)*25).limit(26).all()
        events = session.query(FeedbackEvent).options(joinedload(FeedbackEvent.paper)).filter_by(user_id=user.id).order_by(FeedbackEvent.id.desc()).offset((feedback_page-1)*50).limit(51).all()
        return templates.TemplateResponse(request=request, name="history.html", context={
            "user": user, "token": token, "cycles": cycles[:25], "jobs": jobs[:25], "events": events[:50],
            "page": page, "has_more": len(cycles)>25, "active_tab": "history",
            "feedback_page": feedback_page, "feedback_more": len(events)>50,
            "jobs_page": jobs_page, "jobs_more": len(jobs)>25})

    @router.get("/history/{cycle_id}", response_class=HTMLResponse)
    def cycle_history(cycle_id: int, request: Request, page: int = Query(1, ge=1), auth=Depends(authenticate_user), session: Session = Depends(get_db)):
        user, token = auth
        cycle = session.get(Cycle, cycle_id)
        if not cycle or cycle.user_id != user.id:
            raise HTTPException(404, "Session not found for this profile.")
        decisions = session.query(RoutingDecision).options(joinedload(RoutingDecision.paper), joinedload(RoutingDecision.summary)).filter_by(user_id=user.id, cycle_id=cycle_id).order_by(RoutingDecision.id).offset((page-1)*50).limit(51).all()
        return templates.TemplateResponse(request=request, name="cycle_history.html", context={
            "user": user, "token": token, "cycle": cycle, "decisions": decisions[:50],
            "page": page, "has_more": len(decisions)>50, "active_tab": "history"})

    @router.get("/evaluation", response_class=HTMLResponse)
    def evaluation(request: Request, auth=Depends(authenticate_user), session: Session = Depends(get_db)):
        user, token = auth
        job = session.query(BackgroundJob).filter_by(user_id=user.id, kind="evaluation").order_by(BackgroundJob.created_at.desc()).first()
        return templates.TemplateResponse(request=request, name="evaluation.html", context={
            "user": user, "token": token, "stats": personal_stats(user.id, session),
            "job": job, "active_tab": "evaluation"})

    @router.post("/jobs")
    async def start_job(request: Request, background_tasks: BackgroundTasks, auth=Depends(authenticate_user), session: Session = Depends(get_db)):
        user, token = auth
        if "application/json" in request.headers.get("content-type", ""):
            values = await request.json()
        else:
            values = {key: value[0] for key, value in parse_qs((await request.body()).decode()).items()}
        kind = values.get("kind", "monitor")
        if kind not in {"monitor", "evaluation"}:
            raise HTTPException(400, "Unknown session type.")
        options = {}
        if kind == "monitor":
            categories = [value.strip() for value in str(values.get("categories", config.CORPUS_CATEGORIES)).split(",") if value.strip()]
            try:
                days, limit = int(values.get("days", 1)), int(values.get("max_results", 50))
            except (ValueError, TypeError):
                raise HTTPException(400, "Days and paper limit must be whole numbers.")
            if not categories or len(categories)>12 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,63}", value) for value in categories):
                raise HTTPException(400, "Enter up to 12 comma-separated arXiv category codes.")
            if not 1<=days<=30 or not 1<=limit<=50:
                raise HTTPException(400, "Use 1–30 days and 1–50 papers per session.")
            options = {"categories": categories, "days": days, "max_results": limit}
        job, created = enqueue_job(session, user.id, kind, options)
        payload = job_payload(job)
        session.rollback()
        if created:
            background_tasks.add_task(run_background_job, payload["id"])
        if request.headers.get("accept") == "application/json":
            return JSONResponse(payload, status_code=202)
        return RedirectResponse(f"/dashboard?token={token}", status_code=303)

    @router.get("/jobs/{job_id}")
    def job_status(job_id: str, auth=Depends(authenticate_user), session: Session = Depends(get_db)):
        user, _ = auth
        expire_interrupted_jobs(session, user.id)
        session.commit()
        job = session.get(BackgroundJob, job_id)
        if not job or job.user_id != user.id:
            raise HTTPException(404, "Processing session not found for this profile.")
        return job_payload(job)

    return router
