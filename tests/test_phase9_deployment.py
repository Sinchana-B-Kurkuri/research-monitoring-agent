"""
test_phase9_deployment.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit and integration tests for Phase 9 — Cold-Start Deployment & Pilot Launch:
  1. Health check endpoint (/health) under normal and degraded conditions.
  2. Root portal navigation (/) with and without access tokens.
  3. Web process lightweight startup (no PyTorch / sentence_transformers loaded).
  4. Cold-start pilot user onboarding web routes (/onboard).
  5. Multi-tenant privacy and token isolation.
  6. Daily monitoring runner cycle tagging ([CYCLE_SUCCESS], [CYCLE_EMPTY], [CYCLE_FAILURE]).
  7. Pilot progress tracking & consecutive daily streak calculation.
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_monitor.database.models import (
    Base,
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    InterestProfile,
    InterestVectorVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    User,
    create_training_labels_view,
)
from research_monitor.relevance.state import DecisionStatus, PipelineState
from research_monitor.ui.app import app, get_db


@pytest.fixture
def db_session():
    """In-memory SQLite session with full schema initialized."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    create_training_labels_view(engine)

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with overridden DB session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pilot_user(db_session):
    """Creates a sample pilot user with access token and v0 interest vector."""
    now = datetime.now(timezone.utc)
    user = User(
        id="pilot_alice",
        name="Dr. Alice Chen",
        access_token="secret_token_alice_123",
        is_simulated=False,
        created_at=now,
    )
    db_session.add(user)
    db_session.flush()

    profile = InterestProfile(
        user_id=user.id,
        seed_text="Reinforcement learning from human feedback and agentic workflows.",
        seed_embedding=[0.05] * 384,
        created_at=now,
    )
    db_session.add(profile)

    v0 = InterestVectorVersion(
        user_id=user.id,
        version=0,
        vector=[0.05] * 384,
        method="seed",
        n_effective_labels=0,
        p_high_level=98.0,
        p_low_level=90.0,
        t_high=0.75,
        t_low=0.55,
        created_at=now,
    )
    db_session.add(v0)
    db_session.commit()
    return user


# =========================================================================
# 1. Health Check Endpoint Tests
# =========================================================================

def test_health_check_healthy(client):
    """Verify GET /health returns 200 OK and connected telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "version" in data


def test_health_check_database_failure(client):
    """Verify GET /health returns 503 SERVICE UNAVAILABLE when database probe fails."""
    mock_db = MagicMock()
    mock_db.execute.side_effect = Exception("Connection terminated by host")

    def failing_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = failing_get_db
    try:
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["status"] == "degraded"
        assert data["detail"]["database"] == "disconnected"
    finally:
        app.dependency_overrides.clear()


# =========================================================================
# 2. Root Portal Navigation Tests
# =========================================================================

def test_root_redirect_without_token(client):
    """Navigating to / without a token redirects to /onboard."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/onboard"


def test_root_redirect_with_valid_token(client, sample_pilot_user):
    """Navigating to /?token=... with a valid user token redirects to their digest."""
    response = client.get(f"/?token={sample_pilot_user.access_token}", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == f"/digest?token={sample_pilot_user.access_token}"


# =========================================================================
# 3. Pilot Onboarding Web Flow Tests
# =========================================================================

def test_onboard_page_renders_form(client):
    """GET /onboard renders the researcher onboarding form."""
    response = client.get("/onboard")
    assert response.status_code == 200
    assert "Pilot Researcher Onboarding" in response.text
    assert "User Identifier" in response.text
    assert "Research Interests" in response.text


def test_onboard_post_missing_fields_validation(client):
    """POST /onboard with missing fields returns 400 Bad Request on JSON."""
    response = client.post(
        "/onboard",
        json={"user_id": "test_incomplete", "name": ""},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 400
    assert "required" in response.json()["detail"].lower()


def test_onboard_post_success(client, db_session):
    """POST /onboard successfully registers user via subprocess delegation and returns token."""
    with patch("subprocess.run") as mock_run:
        # Simulate successful create_user.py subprocess run
        def side_effect(cmd, **kwargs):
            # Simulate DB insertion that create_user.py would do
            user = User(
                id="new_pilot_bob",
                name="Dr. Bob Smith",
                access_token="bob_token_secure_999",
                is_simulated=False,
                created_at=datetime.now(timezone.utc),
            )
            db_session.add(user)
            db_session.commit()
            res = MagicMock()
            res.returncode = 0
            res.stderr = ""
            return res

        mock_run.side_effect = side_effect

        response = client.post(
            "/onboard",
            json={
                "user_id": "new_pilot_bob",
                "name": "Dr. Bob Smith",
                "seed_text": "Causal inference and agentic planning systems.",
            },
            headers={"Accept": "application/json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["user_id"] == "new_pilot_bob"
        assert data["access_token"] == "bob_token_secure_999"
        assert f"token={data['access_token']}" in data["review_url"]
        assert f"token={data['access_token']}" in data["digest_url"]


# =========================================================================
# 4. Web Process Isolation Test
# =========================================================================

def test_web_module_lightweight_imports():
    """
    Ensure research_monitor.ui.app does not import PyTorch or sentence_transformers.
    The web service process must remain lightweight.
    """
    from pathlib import Path
    app_file = Path(__file__).resolve().parent.parent / "src" / "research_monitor" / "ui" / "app.py"
    source = app_file.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
    assert "import sentence_transformers" not in source
    assert "from sentence_transformers" not in source


# =========================================================================
# 5. Daily Monitoring Runner Tests
# =========================================================================

def test_execute_daily_cycle_empty_results(db_session, sample_pilot_user):
    """A live cycle with 0 fetched papers produces [CYCLE_EMPTY] cleanly without error."""
    from scripts.run_daily_monitoring import execute_daily_cycle_for_user

    mock_state = MagicMock()
    mock_state.cycle_id = 101
    mock_state.counts = {"discovered": 0, "total_routed": 0, "high": 0, "borderline": 0, "low": 0}

    mock_ingestion = MagicMock()
    mock_ingestion.fetched = 0
    mock_ingestion.inserted = 0
    mock_ingestion.skipped_duplicates = 0
    mock_ingestion.embedded = 0

    with patch("scripts.run_daily_monitoring.run_live_cycle", return_value=(mock_state, mock_ingestion)):
        status_tag, cycle_id, metrics = execute_daily_cycle_for_user(
            user=sample_pilot_user,
            session=db_session,
            categories=["cs.AI"],
            max_results=50,
            days=1,
        )

        assert status_tag == "[CYCLE_EMPTY]"
        assert cycle_id == 101
        assert metrics["fetched"] == 0


def test_execute_daily_cycle_success_results(db_session, sample_pilot_user):
    """A live cycle with new fetched papers produces [CYCLE_SUCCESS] and compiles digest."""
    from scripts.run_daily_monitoring import execute_daily_cycle_for_user

    mock_state = MagicMock()
    mock_state.cycle_id = 102
    mock_state.counts = {"discovered": 10, "total_routed": 10, "high": 2, "borderline": 3, "low": 5}

    mock_ingestion = MagicMock()
    mock_ingestion.fetched = 10
    mock_ingestion.inserted = 8
    mock_ingestion.skipped_duplicates = 2
    mock_ingestion.embedded = 8

    mock_digest = MagicMock()
    mock_digest.items = [1, 2]

    with patch("scripts.run_daily_monitoring.run_live_cycle", return_value=(mock_state, mock_ingestion)), \
         patch("scripts.run_daily_monitoring.resume_after_review", return_value=(mock_digest, False)):
        status_tag, cycle_id, metrics = execute_daily_cycle_for_user(
            user=sample_pilot_user,
            session=db_session,
            categories=["cs.AI"],
            max_results=50,
            days=1,
        )

        assert status_tag == "[CYCLE_SUCCESS]"
        assert cycle_id == 102
        assert metrics["fetched"] == 10
        assert metrics["inserted"] == 8
        assert metrics["high"] == 2
        assert metrics["delivered"] == 2


def test_execute_daily_cycle_failure_handled(db_session, sample_pilot_user):
    """Exceptions during a cycle are caught, logged, and return [CYCLE_FAILURE]."""
    from scripts.run_daily_monitoring import execute_daily_cycle_for_user

    with patch("scripts.run_daily_monitoring.run_live_cycle", side_effect=RuntimeError("arXiv API timeout")):
        status_tag, cycle_id, metrics = execute_daily_cycle_for_user(
            user=sample_pilot_user,
            session=db_session,
            categories=["cs.AI"],
            max_results=50,
            days=1,
        )

        assert status_tag == "[CYCLE_FAILURE]"
        assert cycle_id is None
        assert "arXiv API timeout" in metrics["error"]


# =========================================================================
# 6. Pilot Tracking & Streak Calculation Tests
# =========================================================================

def test_calculate_consecutive_streak():
    """Test streak calculator with empty, single, consecutive, and broken date sets."""
    from scripts.track_pilot import calculate_consecutive_streak

    # Empty
    assert calculate_consecutive_streak([]) == 0

    # Single day
    d1 = date(2026, 10, 1)
    assert calculate_consecutive_streak([d1]) == 1

    # Duplicate runs on same day count as 1 day
    assert calculate_consecutive_streak([d1, d1]) == 1

    # 3 consecutive days
    d2 = date(2026, 10, 2)
    d3 = date(2026, 10, 3)
    assert calculate_consecutive_streak([d1, d2, d3]) == 3

    # Broken sequence (gap between Oct 3 and Oct 5) -> current streak from latest is 1
    d5 = date(2026, 10, 5)
    assert calculate_consecutive_streak([d1, d2, d3, d5]) == 1

    # Broken sequence followed by 2 consecutive days -> streak is 2
    d6 = date(2026, 10, 6)
    assert calculate_consecutive_streak([d1, d2, d3, d5, d6]) == 2


def test_track_pilot_inspects_real_db_without_fabrication(db_session, sample_pilot_user):
    """Verify track_pilot reads actual DB state and reports accurately without fabrication."""
    from scripts.track_pilot import inspect_user_pilot_progress

    # When no cycles have run
    progress = inspect_user_pilot_progress(sample_pilot_user, db_session)
    assert progress["total_cycles"] == 0
    assert progress["streak_days"] == 0
    assert progress["target_days"] == 7
    assert progress["routing"]["total_decisions"] == 0
    assert progress["review_tasks"]["pending"] == 0

    # Add 1 real cycle
    now = datetime.now(timezone.utc)
    c1 = Cycle(
        user_id=sample_pilot_user.id,
        as_of_date=now.strftime("%Y-%m-%d"),
        mode="live",
        started_at=now,
        finished_at=now,
        counts={"discovered": 5, "total_routed": 5, "high": 1, "borderline": 1, "low": 3},
    )
    db_session.add(c1)
    db_session.commit()

    progress_after = inspect_user_pilot_progress(sample_pilot_user, db_session)
    assert progress_after["total_cycles"] == 1
    assert progress_after["streak_days"] == 1
    assert len(progress_after["cycle_details"]) == 1
    assert progress_after["cycle_details"][0]["status"] == "[CYCLE_SUCCESS]"
