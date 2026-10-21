"""Remember profiles in this browser; protect the local operator directory."""
import base64
import json
import os
from pathlib import Path
import secrets

from research_monitor.database.models import User

ROOT = Path(__file__).resolve().parents[3]
PROFILE_COOKIE = "research_profiles"
OPERATOR_COOKIE = "research_workspace"


def workspace_key():
    path = Path(os.getenv("WORKSPACE_KEY_PATH", ROOT / "data/local_runtime/workspace.key"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(32))
    except FileExistsError:
        pass
    return path.read_text(encoding="utf-8").strip()


def is_operator(request):
    supplied = request.cookies.get(OPERATOR_COOKIE, "")
    return bool(supplied) and secrets.compare_digest(supplied, workspace_key())


def remembered_tokens(request):
    try:
        values = json.loads(base64.urlsafe_b64decode(request.cookies.get(PROFILE_COOKIE, "")).decode())
        return [v for v in values if isinstance(v, str) and 1 <= len(v) <= 200][:12] if isinstance(values, list) else []
    except (ValueError, UnicodeError):
        return []


def active_token(request):
    values = remembered_tokens(request)
    return values[0] if values else None


def remember_profile(response, request, token):
    tokens = [token] + [v for v in remembered_tokens(request) if v != token]
    encoded = base64.urlsafe_b64encode(json.dumps(tokens[:12]).encode()).decode()
    response.set_cookie(PROFILE_COOKIE, encoded, max_age=60*60*24*365,
        httponly=True, samesite="lax", secure=request.url.scheme == "https")


def accessible_profiles(request, session):
    query = session.query(User)
    if not is_operator(request):
        query = query.filter(User.access_token.in_(remembered_tokens(request)))
    return query.order_by(User.created_at.desc()).all()
