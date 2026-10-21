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
    assert response.headers["location"] == f"/dashboard?token={sample_pilot_user.access_token}"


# =========================================================================
# 3. Pilot Onboarding Web Flow Tests
# =========================================================================

def test_onboard_page_renders_form(client):
    """GET /onboard renders the researcher onboarding form."""
    response = client.get("/onboard")
    assert response.status_code == 200
    assert "Initialize a pilot" in response.text
    assert "User Identifier" in response.text
    assert "Research Interests" in response.text


def test_web_onboarding_cli_mode_does_not_run_schema_migrations(monkeypatch, capsys):
    from types import SimpleNamespace
    from scripts import create_user as cli

    monkeypatch.setattr("sys.argv", ["create_user.py", "--user-id", "cli_test",
        "--name", "CLI Test", "--seed-text", "Robotics", "--skip-db-init"])
    user = SimpleNamespace(id="cli_test", name="CLI Test", access_token="test-token",
        is_simulated=True, created_at=datetime.now(timezone.utc), vector_versions=[])
    with patch.object(cli, "init_db") as migrate, patch.object(cli, "SessionLocal") as sessions, patch.object(cli, "create_user", return_value=user):
        cli.main()
        migrate.assert_not_called()
        sessions.return_value.close.assert_called_once()
    capsys.readouterr()


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

    with patch("scripts.run_daily_monitoring.run_live_cycle", return_value=(mock_state, mock_ingestion)), patch("scripts.run_daily_monitoring.resume_after_review", return_value=(MagicMock(items=[]), False)):
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


# =========================================================================
# 7. Render 512 MB Onboarding Optimization & Resilience Tests
# =========================================================================

def test_onboard_retry_existing_user_short_circuits(client, sample_pilot_user):
    """
    POST /onboard with an already initialized user returns immediately without
    spawning the heavy create_user subprocess, preventing duplicate memory spikes.
    """
    with patch("subprocess.run") as mock_run:
        response = client.post(
            f"/onboard?token={sample_pilot_user.access_token}",
            json={
                "user_id": sample_pilot_user.id,
                "name": sample_pilot_user.name,
                "seed_text": "Sample interests paragraph.",
            },
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["user_id"] == sample_pilot_user.id
        assert data["access_token"] == sample_pilot_user.access_token
        # Assert subprocess was never called
        mock_run.assert_not_called()


def test_onboard_subprocess_timeout_handling(client):
    """POST /onboard handles subprocess.TimeoutExpired gracefully by returning 504."""
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="create_user.py", timeout=90)):
        response = client.post(
            "/onboard",
            json={
                "user_id": "slow_user",
                "name": "Slow Researcher",
                "seed_text": "AI and reinforcement learning.",
            },
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()


def test_create_user_partial_initialization_repair(db_session):
    """
    If a previous onboarding attempt created User but crashed (OOM) before creating
    InterestProfile or InterestVectorVersion, calling create_user repairs it.
    """
    from research_monitor.relevance.user_manager import create_user

    # Create orphaned User without profile or vector version
    now = datetime.now(timezone.utc)
    orphaned = User(
        id="crashed_pilot",
        name="Dr. Crashed",
        access_token="crashed_token_123",
        is_simulated=False,
        created_at=now,
    )
    db_session.add(orphaned)
    db_session.commit()

    # Pre-mock reference embeddings to avoid needing 5000 DB records in SQLite
    mock_ref_embs = [([0.05] * 384)] * 10
    import numpy as np

    with patch("research_monitor.relevance.user_manager._generate_seed_vector", return_value=np.array([0.05] * 384, dtype=np.float32)):
        repaired_user = create_user(
            user_id="crashed_pilot",
            name="Dr. Crashed",
            seed_text="Autonomous agents and reasoning.",
            session=db_session,
            reference_embeddings=np.array(mock_ref_embs, dtype=np.float32),
        )

        assert repaired_user.id == "crashed_pilot"
        # Profile must now exist
        profile = db_session.query(InterestProfile).filter_by(user_id="crashed_pilot").first()
        assert profile is not None
        # Version 0 must now exist
        v0 = db_session.query(InterestVectorVersion).filter_by(user_id="crashed_pilot", version=0).first()
        assert v0 is not None
        assert v0.method == "seed"
        assert v0.is_active is True


def test_in_database_pgvector_threshold_computation_branch():
    """Verify in-database percentile_cont SQL query execution on PostgreSQL sessions."""
    import numpy as np
    from research_monitor.relevance.thresholds import compute_vector_thresholds

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"
    # Return (t_high, t_low, ref_count)
    mock_session.execute.return_value.fetchone.return_value = (0.825, 0.612, 5000)

    vec = np.ones(384, dtype=np.float32)
    thresholds = compute_vector_thresholds(vector=vec, session=mock_session)

    assert thresholds.t_high == 0.825
    assert thresholds.t_low == 0.612
    # Verify execute was called with percentile_cont query and cosine distance operator
    mock_session.execute.assert_called_once()
    called_sql = str(mock_session.execute.call_args[0][0])
    assert "percentile_cont" in called_sql
    assert "is_reference = TRUE" in called_sql
    assert "<=>" in called_sql


# =========================================================================
# 8. Threshold Numerical Equivalence, Normalization & Fallback Tests
# =========================================================================

def test_percentile_cont_mathematical_formula_equivalence():
    """
    Mathematical formula equivalence test:
    Verify that PostgreSQL's percentile_cont linear interpolation formula
    and NumPy's np.percentile(..., method='linear') are mathematically identical
    across representative distributions, including 5,000 scores.
    """
    import gc
    import numpy as np

    rng = np.random.default_rng(2026)

    distributions = [
        ("Normal 5000", rng.standard_normal(5000, dtype=np.float32)),
        ("Uniform 5000", rng.uniform(-1.0, 1.0, 5000).astype(np.float32)),
        ("Exponential 5000", rng.exponential(1.0, 5000).astype(np.float32)),
        ("Small 10", rng.standard_normal(10, dtype=np.float32)),
    ]

    percentiles = [98.0, 90.0, 50.0, 10.0, 2.0]

    for dist_name, raw_scores in distributions:
        sorted_scores = np.sort(raw_scores)
        N = len(sorted_scores)

        for p_val in percentiles:
            p_frac = p_val / 100.0

            # PostgreSQL percentile_cont linear continuous interpolation formula:
            # index = (N - 1) * p; j = floor(index); f = index - j; val = (1 - f)*Y[j] + f*Y[j+1]
            idx = (N - 1) * p_frac
            j = int(np.floor(idx))
            f = idx - j
            if j >= N - 1:
                pg_cont_val = float(sorted_scores[-1])
            else:
                pg_cont_val = float((1.0 - f) * sorted_scores[j] + f * sorted_scores[j + 1])

            np_val = float(np.percentile(raw_scores, p_val, method="linear"))

            diff = abs(pg_cont_val - np_val)
            assert diff < 1e-6, (
                f"Percentile mismatch on {dist_name} at P={p_val}: "
                f"PostgreSQL formula={pg_cont_val:.8f}, NumPy={np_val:.8f}, diff={diff}"
            )

    del distributions
    gc.collect()


def test_real_postgresql_pgvector_percentile_integration():
    """
    Real PostgreSQL/pgvector database integration test.
    Executes actual SQL percentile_cont on PostgreSQL if available, comparing against NumPy.
    Skips cleanly if PostgreSQL database or pgvector is unavailable.
    """
    import gc
    import os
    import numpy as np
    from sqlalchemy import create_engine, text
    from research_monitor.config import DATABASE_URL

    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            has_vector = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).fetchone()
            if not has_vector:
                pytest.skip("pgvector extension not installed in PostgreSQL.")
    except Exception as e:
        pytest.skip(f"Real PostgreSQL database is unavailable for integration testing: {e}")

    # Generate 500 synthetic reference embeddings and compare actual SQL execution against NumPy
    rng = np.random.default_rng(42)
    raw_refs = rng.standard_normal((500, 384), dtype=np.float32)
    ref_matrix = raw_refs / np.linalg.norm(raw_refs, axis=1, keepdims=True)

    raw_query = rng.standard_normal(384, dtype=np.float32)
    query_vec = raw_query / np.linalg.norm(raw_query)

    # Compute NumPy baseline
    dot_products = np.dot(ref_matrix, query_vec)
    np_t_high = float(np.percentile(dot_products, 98.0))
    np_t_low = float(np.percentile(dot_products, 90.0))

    vec_str = "[" + ",".join(f"{x:.8f}" for x in query_vec) + "]"

    # Execute actual SQL query in a temporary table within a rolled-back transaction
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TEMP TABLE temp_test_ref_papers (embedding vector(384))"))
            for emb in ref_matrix:
                e_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                conn.execute(text("INSERT INTO temp_test_ref_papers VALUES (CAST(:e AS vector))"), {"e": e_str})

            sql = text(
                """
                SELECT
                    percentile_cont(0.98) WITHIN GROUP (ORDER BY (1.0 - (embedding <=> CAST(:vec_str AS vector)))) AS t_high,
                    percentile_cont(0.90) WITHIN GROUP (ORDER BY (1.0 - (embedding <=> CAST(:vec_str AS vector)))) AS t_low
                FROM temp_test_ref_papers
                """
            )
            row = conn.execute(sql, {"vec_str": vec_str}).fetchone()
            sql_t_high = float(row[0])
            sql_t_low = float(row[1])

            assert sql_t_high == pytest.approx(np_t_high, abs=1e-4)
            assert sql_t_low == pytest.approx(np_t_low, abs=1e-4)
    finally:
        del raw_refs, ref_matrix, raw_query, query_vec
        gc.collect()


def test_cosine_distance_unnormalized_vectors_mathematical_verification():
    """
    Verify that 1.0 - (u <=> v) yields exact cosine similarity even if vectors are unnormalized,
    whereas dot product (u <#> v) * -1.0 deviates from cosine similarity when norms != 1.0.
    """
    import gc
    import numpy as np

    rng = np.random.default_rng(99)
    # Create unnormalized vector with norm != 1.0 directly in float32
    u = rng.standard_normal(384, dtype=np.float32) * 3.5
    v = rng.standard_normal(384, dtype=np.float32) * 0.7

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    true_cosine_sim = float(np.dot(u, v) / (norm_u * norm_v))

    # pgvector <=> formula: 1.0 - (u . v) / (|u| * |v|)
    pg_cosine_distance = 1.0 - (float(np.dot(u, v)) / (norm_u * norm_v))
    recovered_cosine_sim = 1.0 - pg_cosine_distance

    # Inner product: u . v
    raw_inner_product = float(np.dot(u, v))

    assert recovered_cosine_sim == pytest.approx(true_cosine_sim, abs=1e-6)
    # Raw inner product deviates significantly because norms are 3.5 and 0.7
    assert abs(raw_inner_product - true_cosine_sim) > 0.5

    del u, v
    gc.collect()


def test_compute_thresholds_fallback_when_db_has_zero_reference_papers(monkeypatch):
    """
    When PostgreSQL has 0 reference papers, compute_vector_thresholds must
    not return None, but fall back to load_reference_embeddings and calculate valid thresholds.
    """
    import gc
    import numpy as np
    from research_monitor.config import config
    from research_monitor.relevance.thresholds import compute_vector_thresholds

    monkeypatch.setattr(config, "REFERENCE_SAMPLE_SIZE", 50)

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"
    # Database returns count = 0
    mock_session.execute.return_value.fetchone.return_value = (None, None, 0)

    # Provide compact mock reference embeddings to avoid large allocations
    rng = np.random.default_rng(42)
    mock_ref = rng.standard_normal((50, 384), dtype=np.float32)
    mock_ref = mock_ref / np.linalg.norm(mock_ref, axis=1, keepdims=True)

    vec = np.ones(384, dtype=np.float32)

    with patch("research_monitor.relevance.thresholds.load_reference_embeddings", return_value=mock_ref) as mock_load:
        thresholds = compute_vector_thresholds(vector=vec, session=mock_session)

        # Fallback loader must have been called
        mock_load.assert_called_once_with(session=mock_session)
        assert thresholds.t_high is not None
        assert thresholds.t_low is not None
        assert thresholds.t_high > thresholds.t_low

    del mock_ref, thresholds
    gc.collect()


def test_compute_thresholds_fallback_when_db_has_underpopulated_reference_papers(monkeypatch):
    """
    When PostgreSQL has fewer than REFERENCE_SAMPLE_SIZE papers,
    compute_vector_thresholds must reject the incomplete sample and fall back to the full reference set.
    """
    import gc
    import numpy as np
    from research_monitor.config import config
    from research_monitor.relevance.thresholds import compute_vector_thresholds

    monkeypatch.setattr(config, "REFERENCE_SAMPLE_SIZE", 50)

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"
    # Database returns only 10 papers (less than required 50)
    mock_session.execute.return_value.fetchone.return_value = (0.75, 0.55, 10)

    rng = np.random.default_rng(42)
    mock_ref = rng.standard_normal((50, 384), dtype=np.float32)
    mock_ref = mock_ref / np.linalg.norm(mock_ref, axis=1, keepdims=True)

    vec = np.ones(384, dtype=np.float32)

    with patch("research_monitor.relevance.thresholds.load_reference_embeddings", return_value=mock_ref) as mock_load:
        thresholds = compute_vector_thresholds(vector=vec, session=mock_session)

        # Must fall back to file artifacts instead of using the underpopulated DB result
        mock_load.assert_called_once_with(session=mock_session)
        assert thresholds.t_high is not None
        assert thresholds.t_low is not None

    del mock_ref, thresholds
    gc.collect()


def test_compute_thresholds_fallback_when_db_query_raises_exception(monkeypatch):
    """
    When PostgreSQL query raises an OperationalError (e.g. pgvector not installed or connection drops),
    compute_vector_thresholds must catch it, log a warning, and fall back without crashing.
    """
    import gc
    import numpy as np
    from research_monitor.config import config
    from research_monitor.relevance.thresholds import compute_vector_thresholds

    monkeypatch.setattr(config, "REFERENCE_SAMPLE_SIZE", 50)

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"
    mock_session.execute.side_effect = Exception("UndefinedFunction: operator does not exist: vector <=> vector")

    rng = np.random.default_rng(42)
    mock_ref = rng.standard_normal((50, 384), dtype=np.float32)
    mock_ref = mock_ref / np.linalg.norm(mock_ref, axis=1, keepdims=True)

    vec = np.ones(384, dtype=np.float32)

    with patch("research_monitor.relevance.thresholds.load_reference_embeddings", return_value=mock_ref) as mock_load:
        thresholds = compute_vector_thresholds(vector=vec, session=mock_session)

        mock_load.assert_called_once_with(session=mock_session)
        assert thresholds.t_high is not None
        assert thresholds.t_low is not None

    del mock_ref, thresholds
    gc.collect()


def test_load_reference_embeddings_missing_artifacts_raises_file_not_found(tmp_path):
    """
    When both DB and local file artifacts are missing, load_reference_embeddings
    must raise FileNotFoundError to prevent silent corruption.
    """
    from research_monitor.relevance.thresholds import load_reference_embeddings

    missing_corpus = tmp_path / "nonexistent_corpus.parquet"
    missing_embeddings = tmp_path / "nonexistent_embeddings.npy"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_reference_embeddings(
            session=None,
            corpus_path=missing_corpus,
            embeddings_path=missing_embeddings,
        )

    assert "missing" in str(exc_info.value).lower()


def test_load_reference_embeddings_loads_real_artifacts_correctly():
    """
    Verify that load_reference_embeddings loads the real project artifacts
    (data/corpus.parquet and data/embeddings.npy) with correct shape and L2 normalization.
    Avoids unnecessary matrix copies and cleans up memory immediately.
    """
    import gc
    import numpy as np
    from research_monitor.relevance.thresholds import load_reference_embeddings

    ref_embs = load_reference_embeddings(session=None)

    try:
        assert ref_embs.shape == (5000, 384)
        assert ref_embs.dtype == np.float32

        # Check normalization on representative sampled rows across the matrix
        # without copying or allocating 5,000 float64 norms
        sample_indices = [0, 1, 10, 100, 500, 1000, 2500, 4999]
        sample_norms = np.linalg.norm(ref_embs[sample_indices], axis=1)
        assert np.allclose(sample_norms, 1.0, atol=1e-5)

        # Verify norm over the first 100 rows
        slice_norms = np.linalg.norm(ref_embs[:100], axis=1)
        assert np.allclose(slice_norms, 1.0, atol=1e-5)
    finally:
        del ref_embs
        gc.collect()
