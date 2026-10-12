"""
test_live_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~
Comprehensive automated test suite for Phase 8 Live arXiv Monitoring.

Verifies:
  1. Correct category-restricted query construction.
  2. Correct parsing and normalization of arXiv paper IDs.
  3. Cross-listed papers are not duplicated.
  4. Existing papers are skipped without overwriting their embeddings.
  5. Repeated live runs do not create duplicate records.
  6. New papers receive embeddings with expected dimensions (384) and L2 normalization.
  7. Existing papers are not re-embedded.
  8. New papers enter the existing relevance-routing pipeline.
  9. Live cycles are tagged mode = 'live'.
  10. An empty arXiv result set completes successfully.
  11. Transient network failures are retried with bounded backoff.
  12. Permanent failures terminate cleanly and are logged.
  13. Partial ingestion or embedding failures do not corrupt previously stored data.
  14. Replay-mode behavior remains intact.
  15. Web service imports without loading PyTorch or sentence-transformers at startup.
"""

from datetime import datetime, timezone
import sys
from typing import Any
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_monitor.database.models import (
    Base,
    Cycle,
    InterestProfile,
    InterestVectorVersion,
    Paper,
    RoutingDecision,
    User,
    create_training_labels_view,
)
from research_monitor.ingestion.live import (
    LiveIngestionResult,
    build_live_query,
    fetch_live_papers,
    ingest_live_papers,
    run_live_cycle,
)
from research_monitor.ingestion.preprocessor import strip_version
from research_monitor.relevance.pipeline import run_monitoring_cycle
from research_monitor.relevance.state import DecisionStatus


class MockArxivResult:
    """Mock item representing an arXiv API search result."""
    def __init__(
        self,
        entry_id: str,
        title: str,
        summary: str,
        published: Optional[datetime] = None,
        categories: Optional[list[str]] = None,
    ):
        self.entry_id = entry_id
        self.title = title
        self.summary = summary
        self.published = published or datetime(2024, 3, 15, tzinfo=timezone.utc)
        self.categories = categories or ["cs.CL", "cs.AI"]

    def get_short_id(self) -> str:
        return self.entry_id


class MockArxivClient:
    """Mock arXiv.Client simulating network streams, transient drops, and permanent errors."""
    def __init__(
        self,
        results_list: Optional[list[Any]] = None,
        fail_times: int = 0,
        permanent_fail: bool = False,
    ):
        self.results_list = results_list or []
        self.fail_times = fail_times
        self.permanent_fail = permanent_fail
        self.call_count = 0

    def results(self, search: Any) -> Any:
        self.call_count += 1
        if self.permanent_fail:
            raise RuntimeError("HTTP 400: Bad Request - invalid query syntax")
        if self.call_count <= self.fail_times:
            raise ConnectionError("Transient network drop (503 Service Unavailable)")
        return iter(self.results_list)


@pytest.fixture
def db_session():
    """Isolated SQLite in-memory database with pgvector shim and training_labels view."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    create_training_labels_view(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_user(db_session):
    user = User(
        id="developer",
        name="Developer Pilot",
        access_token="tok_dev_pilot_123",
        is_simulated=False,
    )
    db_session.add(user)
    db_session.flush()

    profile = InterestProfile(
        user_id=user.id,
        seed_text="Parameter-efficient fine-tuning and retrieval-augmented generation for LLMs",
        seed_embedding=[0.0] * 384,
    )
    db_session.add(profile)

    ivv = InterestVectorVersion(
        user_id=user.id,
        version=1,
        vector=[0.0] * 384,
        method="seed",
        n_effective_labels=0,
        t_high=0.75,
        t_low=0.45,
        p_high_level=98.0,
        p_low_level=90.0,
    )
    db_session.add(ivv)
    db_session.commit()
    return user


# =========================================================================
# Requirement 1: Correct category-restricted query construction
# =========================================================================

def test_query_construction_categories_only():
    q = build_live_query(categories=["cs.CL", "cs.LG", "cs.AI"])
    assert q == "(cat:cs.CL OR cat:cs.LG OR cat:cs.AI)"


def test_query_construction_with_date_range():
    q = build_live_query(
        categories=["cs.CL", "cs.AI"],
        start_date="2024-03-01",
        end_date="2024-03-07",
    )
    assert "(cat:cs.CL OR cat:cs.AI)" in q
    assert "submittedDate:[202403010000 TO 202403072359]" in q


# =========================================================================
# Requirement 2: Correct parsing and normalization of arXiv paper IDs
# =========================================================================

def test_paper_id_normalization():
    assert strip_version("https://arxiv.org/abs/2401.12345v2") == "2401.12345"
    assert strip_version("2401.12345v3") == "2401.12345"
    assert strip_version("cs/0612007v1") == "cs/0612007"
    assert strip_version("2403.99999") == "2403.99999"


# =========================================================================
# Requirement 3: Cross-listed papers are not duplicated
# =========================================================================

def test_cross_listed_papers_deduplication():
    # Same paper returned twice under different category listings
    res1 = MockArxivResult("2403.01001v1", "Title 1", "Abstract 1", categories=["cs.CL"])
    res2 = MockArxivResult("2403.01001v2", "Title 1", "Abstract 1", categories=["cs.AI"])
    res3 = MockArxivResult("2403.01002v1", "Title 2", "Abstract 2", categories=["cs.LG"])

    client = MockArxivClient(results_list=[res1, res2, res3])
    papers = fetch_live_papers(client_override=client)

    # Must contain only 2 unique papers
    assert len(papers) == 2
    paper_ids = [p["paper_id"] for p in papers]
    assert paper_ids == ["2403.01001", "2403.01002"]


# =========================================================================
# Requirement 4 & 5: Existing papers skipped without overwriting embeddings
# =========================================================================

def test_existing_papers_skipped_and_not_duplicated(db_session):
    # Insert existing paper with a known specific embedding
    orig_emb = [0.5] * 384
    existing = Paper(
        id="2403.existing",
        title="Existing Paper Title",
        abstract="Existing abstract.",
        submitted_date="2024-03-01",
        categories=["cs.CL"],
        embedding=orig_emb,
        embedding_model="all-MiniLM-L6-v2",
        is_reference=False,
    )
    db_session.add(existing)
    db_session.commit()

    candidates = [
        {
            "paper_id": "2403.existingv2",  # Should be normalized to 2403.existing and skipped
            "title": "New Title That Should Not Overwrite",
            "abstract": "New abstract that should not overwrite.",
            "submitted_date": "2024-03-10",
            "categories": ["cs.CL"],
            "embedding": [0.99] * 384,
        },
        {
            "paper_id": "2403.brand_newv1",
            "title": "Brand New Paper",
            "abstract": "Fresh abstract.",
            "submitted_date": "2024-03-10",
            "categories": ["cs.AI"],
            "embedding": [0.1] * 384,
        },
    ]

    result = ingest_live_papers(session=db_session, papers_override=candidates)

    assert result.fetched == 2
    assert result.skipped_duplicates == 1
    assert result.inserted == 1
    assert result.embedded == 1
    assert result.new_paper_ids == ["2403.brand_new"]

    # Verify existing record was untouched
    refreshed = db_session.query(Paper).filter_by(id="2403.existing").first()
    assert refreshed.title == "Existing Paper Title"
    assert refreshed.embedding == orig_emb


# =========================================================================
# Requirement 6: New papers receive embeddings with expected dimensions & normalization
# =========================================================================

def test_new_papers_receive_l2_normalized_embeddings(db_session):
    candidates = [
        {
            "paper_id": "2403.embed_testv1",
            "title": "Testing L2 Normalized Embeddings",
            "abstract": "This paper explores representation learning in NLP.",
            "submitted_date": "2024-03-10",
            "categories": ["cs.CL"],
        }
    ]

    result = ingest_live_papers(session=db_session, papers_override=candidates)
    assert result.inserted == 1
    assert result.embedded == 1

    paper = db_session.query(Paper).filter_by(id="2403.embed_test").first()
    assert paper is not None
    assert len(paper.embedding) == 384

    # Check L2 normalization: norm should equal 1.0
    vec = np.array(paper.embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-4) == 1.0


# =========================================================================
# Requirement 7: Existing papers are not re-embedded
# =========================================================================

def test_existing_papers_not_reembedded(db_session):
    paper = Paper(
        id="2403.already_there",
        title="Already There",
        abstract="Abstract.",
        submitted_date="2024-03-01",
        categories=["cs.CL"],
        embedding=[0.0] * 384,
        embedding_model="all-MiniLM-L6-v2",
        is_reference=False,
    )
    db_session.add(paper)
    db_session.commit()

    candidates = [
        {
            "paper_id": "2403.already_therev3",
            "title": "Already There",
            "abstract": "Abstract.",
            "submitted_date": "2024-03-01",
            "categories": ["cs.CL"],
        }
    ]

    # Ingestion should skip without calling embedder
    result = ingest_live_papers(session=db_session, papers_override=candidates)
    assert result.inserted == 0
    assert result.skipped_duplicates == 1
    assert result.embedded == 0


# =========================================================================
# Requirement 8 & 9: New papers enter relevance pipeline & tagged mode = 'live'
# =========================================================================

def test_new_papers_enter_relevance_pipeline_with_mode_live(db_session, sample_user):
    candidates = [
        {
            "paper_id": "2403.pipeline_01v1",
            "title": "Parameter-Efficient Fine-Tuning with LoRA",
            "abstract": "We adapt language models efficiently.",
            "submitted_date": "2024-03-15",
            "categories": ["cs.CL"],
            "embedding": [0.0] * 384,
        },
        {
            "paper_id": "2403.pipeline_02v1",
            "title": "Unrelated Cooking Recipes",
            "abstract": "How to bake sourdough bread.",
            "submitted_date": "2024-03-15",
            "categories": ["cs.AI"],
            "embedding": [0.0] * 384,
        },
    ]

    state, ingestion = run_live_cycle(
        user_id=sample_user.id,
        session=db_session,
        papers_override=candidates,
    )

    assert ingestion.inserted == 2
    assert state.mode == "live"
    assert state.cycle_id is not None

    # Check cycle in database
    cycle = db_session.query(Cycle).filter_by(id=state.cycle_id).first()
    assert cycle is not None
    assert cycle.mode == "live"

    # Decisions persisted for both papers
    decisions = db_session.query(RoutingDecision).filter_by(cycle_id=cycle.id).all()
    assert len(decisions) == 2
    dec_pids = [d.paper_id for d in decisions]
    assert "2403.pipeline_01" in dec_pids
    assert "2403.pipeline_02" in dec_pids


# =========================================================================
# Requirement 10: Empty arXiv result set completes successfully
# =========================================================================

def test_empty_arxiv_result_set_completes_successfully(db_session, sample_user):
    client = MockArxivClient(results_list=[])

    state, ingestion = run_live_cycle(
        user_id=sample_user.id,
        session=db_session,
        client_override=client,
    )

    assert ingestion.fetched == 0
    assert ingestion.inserted == 0
    assert ingestion.skipped_duplicates == 0
    assert ingestion.embedded == 0
    assert state.mode == "live"
    assert state.counts.get("discovered", 0) == 0
    assert state.counts.get("total_routed", 0) == 0


# =========================================================================
# Requirement 11: Transient network failures retried with bounded backoff
# =========================================================================

def test_transient_network_failures_retried_with_backoff():
    res = MockArxivResult("2403.transient_01", "Title", "Abstract")
    # Fails 2 times with transient network timeout, succeeds on 3rd attempt
    flaky_client = MockArxivClient(results_list=[res], fail_times=2)

    papers = fetch_live_papers(
        client_override=flaky_client,
        max_retries=3,
        initial_backoff=0.01,  # Fast backoff for test speed
    )

    assert len(papers) == 1
    assert papers[0]["paper_id"] == "2403.transient_01"
    assert flaky_client.call_count == 3


# =========================================================================
# Requirement 12: Permanent failures terminate cleanly and are logged
# =========================================================================

def test_permanent_failures_raise_immediately():
    bad_client = MockArxivClient(permanent_fail=True)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        fetch_live_papers(
            client_override=bad_client,
            max_retries=3,
            initial_backoff=0.01,
        )

    # Must fail immediately without useless retries
    assert bad_client.call_count == 1


# =========================================================================
# Requirement 13: Partial ingestion or embedding failures do not corrupt DB
# =========================================================================

def test_embedding_failure_rolls_back_safely(db_session):
    # Insert existing control paper
    control_paper = Paper(
        id="2403.control",
        title="Control Paper",
        abstract="Abstract.",
        submitted_date="2024-03-01",
        categories=["cs.CL"],
        embedding=[0.0] * 384,
        embedding_model="all-MiniLM-L6-v2",
        is_reference=False,
    )
    db_session.add(control_paper)
    db_session.commit()

    # Pass candidate with missing embedding, but monkeypatch embedder to fail
    candidates = [
        {
            "paper_id": "2403.will_failv1",
            "title": "Will Fail Paper",
            "abstract": "Abstract text.",
            "submitted_date": "2024-03-10",
            "categories": ["cs.AI"],
        }
    ]

    import research_monitor.ingestion.embedder as emb_module
    orig_embed_texts = emb_module.embed_texts

    def failing_embed_texts(*args, **kwargs):
        raise RuntimeError("Simulated OOM or CUDA embedding failure")

    emb_module.embed_texts = failing_embed_texts
    try:
        with pytest.raises(RuntimeError, match="Simulated OOM"):
            ingest_live_papers(session=db_session, papers_override=candidates)
    finally:
        emb_module.embed_texts = orig_embed_texts

    # Control paper is intact
    assert db_session.query(Paper).count() == 1
    assert db_session.query(Paper).filter_by(id="2403.control").first() is not None
    # Failed paper was not partially persisted
    assert db_session.query(Paper).filter_by(id="2403.will_fail").first() is None


# =========================================================================
# Requirement 14: Replay-mode behavior remains intact
# =========================================================================

def test_replay_mode_behavior_intact(db_session, sample_user):
    paper = Paper(
        id="2403.replay_p1",
        title="Replay Paper 1",
        abstract="Replay abstract.",
        submitted_date="2024-02-01",
        categories=["cs.CL"],
        embedding=[0.0] * 384,
        embedding_model="all-MiniLM-L6-v2",
        is_reference=False,
    )
    db_session.add(paper)
    db_session.commit()

    state = run_monitoring_cycle(
        user_id=sample_user.id,
        session=db_session,
        mode="replay",
        as_of_date="2024-03-01",
    )

    assert state.mode == "replay"
    assert state.counts.get("discovered", 0) >= 1
    dec = db_session.query(RoutingDecision).filter_by(paper_id="2403.replay_p1", user_id=sample_user.id).first()
    assert dec is not None


# =========================================================================
# Requirement 15: Web service imports without loading PyTorch or SentenceTransformer
# =========================================================================

def test_web_service_imports_without_torch():
    # Verify in a clean process that ui/app does not import PyTorch or SentenceTransformer
    import os
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        "import sys; from research_monitor.ui.app import app; assert app is not None; assert 'torch' not in sys.modules; assert 'sentence_transformers' not in sys.modules; print('ISOLATED')",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"Import failed with stderr: {proc.stderr}"
    assert "ISOLATED" in proc.stdout
