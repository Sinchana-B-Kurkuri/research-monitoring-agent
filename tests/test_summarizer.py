"""
test_summarizer.py
~~~~~~~~~~~~~~~~~~
Comprehensive test suite for Phase 7 Gated LLM Summarization.

Verifies:
  1. A HIGH paper eligible for the digest gets summarized.
  2. A HIGH paper excluded by the digest cap does not get summarized.
  3. A reviewer-accepted BORDERLINE paper becomes eligible for background summarization.
  4. Pending, rejected, and expired BORDERLINE papers are not summarized.
  5. LOW and DEFERRED_OVERFLOW papers are never summarized.
  6. Repeated runs reuse the cached summary without another LLM call.
  7. Different prompt versions do not incorrectly reuse an incompatible cached summary.
  8. Failed generation preserves a retryable state and does not create a successful summary record.
  9. Retrying after failure can succeed without duplicate successful records.
  10. Concurrent or repeated requests do not generate duplicate summaries unnecessarily.
  11. Digest pages display summaries correctly and preserve existing functionality.
  12. Prompt rendering, faithfulness instructions, and credential security.
"""

from datetime import datetime, timezone
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
    InterestProfile,
    InterestVectorVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    Summary,
    User,
    create_training_labels_view,
)
from research_monitor.relevance.state import DecisionStatus
from research_monitor.review.digest import resume_after_review
from research_monitor.review.queue import record_review_action
from research_monitor.summary.service import (
    is_eligible_for_summarization,
    summarize_decision,
    summarize_decision_background,
)
from research_monitor.summary.summarizer import (
    GeminiSummarizer,
    MockSummarizer,
    PROMPT_TEMPLATES,
    get_summarizer,
)
from research_monitor.ui.app import app, get_db


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
        id="alice",
        name="Alice Researcher",
        access_token="tok_alice_secret_123",
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

    cycle = Cycle(
        user_id=user.id,
        mode="replay",
        as_of_date="2024-03-01",
    )
    db_session.add(cycle)
    db_session.commit()
    return user


def create_paper_and_decision(
    db_session,
    user_id: str,
    cycle_id: int,
    paper_id: str,
    title: str,
    abstract: str,
    route: str,
    status: str,
    score: float = 0.85,
) -> tuple[Paper, RoutingDecision]:
    paper = Paper(
        id=paper_id,
        title=title,
        abstract=abstract,
        submitted_date="2024-03-01",
        categories=["cs.CL", "cs.AI"],
        embedding=[0.0] * 384,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        is_reference=False,
    )
    db_session.add(paper)
    db_session.flush()

    ivv = db_session.query(InterestVectorVersion).filter_by(user_id=user_id).first()
    dec = RoutingDecision(
        paper_id=paper.id,
        user_id=user_id,
        cycle_id=cycle_id,
        cosine_score=score,
        vector_version_id=ivv.id,
        thresholds_used={"t_high": 0.75, "t_low": 0.45},
        route=route,
        status=status,
        reason=f"Route: {route} based on score {score:.4f}",
    )
    db_session.add(dec)
    db_session.commit()
    return paper, dec


# =========================================================================
# Requirement 1: HIGH paper eligible for digest gets summarized
# =========================================================================

def test_high_paper_eligible_for_digest_gets_summarized(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    paper, dec = create_paper_and_decision(
        db_session,
        sample_user.id,
        cycle.id,
        paper_id="2403.0001",
        title="LoRA: Low-Rank Adaptation of Large Models",
        abstract="We propose Low-Rank Adaptation (LoRA) to freeze pretrained model weights and inject trainable rank decomposition matrices.",
        route="HIGH",
        status=DecisionStatus.ROUTED_HIGH.value,
        score=0.92,
    )

    mock_llm = MockSummarizer(model="mock-gemini-flash")
    digest, existed = resume_after_review(
        user_id=sample_user.id,
        cycle_id=cycle.id,
        session=db_session,
        summarizer=mock_llm,
    )

    assert existed is False
    assert len(digest.items) == 1
    assert mock_llm.call_count == 1

    db_session.refresh(dec)
    assert dec.status == DecisionStatus.DELIVERED.value
    assert dec.summary is not None
    assert "LoRA" in dec.summary.text
    assert dec.summary.llm_model == "mock-gemini-flash"
    assert dec.summary.prompt_version == "v1.0"
    assert dec.summary.tokens_in > 0
    assert dec.summary.tokens_out > 0


# =========================================================================
# Requirement 2: HIGH paper excluded by digest cap does NOT get summarized
# =========================================================================

def test_high_paper_excluded_by_digest_cap_not_summarized(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    decisions = []
    for i in range(25):
        _, dec = create_paper_and_decision(
            db_session,
            sample_user.id,
            cycle.id,
            paper_id=f"2403.cap_{i:02d}",
            title=f"Paper Title {i}",
            abstract=f"Abstract content for paper {i}.",
            route="HIGH",
            status=DecisionStatus.ROUTED_HIGH.value,
            score=0.95 - (i * 0.01),
        )
        decisions.append(dec)

    mock_llm = MockSummarizer(model="mock-gemini-flash")
    digest, _ = resume_after_review(
        user_id=sample_user.id,
        cycle_id=cycle.id,
        session=db_session,
        max_digest_items=20,
        summarizer=mock_llm,
    )

    # Exactly 20 delivered and summarized
    assert len(digest.items) == 20
    assert mock_llm.call_count == 20

    # Remaining 5 must be DEFERRED_OVERFLOW and NEVER summarized
    overflow_decs = decisions[20:]
    for dec in overflow_decs:
        db_session.refresh(dec)
        assert dec.status == DecisionStatus.DEFERRED_OVERFLOW.value
        summary_in_db = db_session.query(Summary).filter_by(decision_id=dec.id).first()
        assert summary_in_db is None

        # Verify gating explicitly blocks summarizing overflow papers
        with pytest.raises(PermissionError, match="DEFERRED_OVERFLOW"):
            summarize_decision(
                decision_id=dec.id,
                user_id=sample_user.id,
                session=db_session,
                summarizer=mock_llm,
            )

    # Call count should not have increased
    assert mock_llm.call_count == 20


# =========================================================================
# Requirement 3: Reviewer-accepted BORDERLINE paper becomes eligible for background summarization
# =========================================================================

def test_reviewer_accepted_borderline_eligible_for_background_summarization(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    paper, dec = create_paper_and_decision(
        db_session,
        sample_user.id,
        cycle.id,
        paper_id="2403.border_01",
        title="Sparse Retrieval for Knowledge Grounding",
        abstract="A study on hybrid sparse-dense retrieval.",
        route="BORDERLINE",
        status=DecisionStatus.ROUTED_BORDERLINE.value,
        score=0.62,
    )

    task = ReviewTask(
        decision_id=dec.id,
        kind="borderline",
        queued_at=datetime.now(timezone.utc),
    )
    dec.status = DecisionStatus.PENDING_REVIEW.value
    db_session.add(task)
    db_session.commit()

    # Before acceptance: not eligible
    eligible, _ = is_eligible_for_summarization(dec)
    assert eligible is False

    # Reviewer accepts task
    record_review_action(task.id, "accept", sample_user.id, db_session)
    db_session.refresh(dec)
    assert dec.status == DecisionStatus.SUMMARIZE_PENDING.value

    # After acceptance: eligible
    eligible, _ = is_eligible_for_summarization(dec)
    assert eligible is True

    # Run background summarization
    mock_llm = MockSummarizer(model="mock-gemini-flash")
    summary = summarize_decision(
        decision_id=dec.id,
        user_id=sample_user.id,
        session=db_session,
        summarizer=mock_llm,
    )

    assert summary is not None
    assert summary.text != ""
    assert mock_llm.call_count == 1
    db_session.refresh(dec)
    assert dec.status == DecisionStatus.SUMMARIZED.value


# =========================================================================
# Requirement 4: Pending, rejected, and expired BORDERLINE papers are NOT summarized
# =========================================================================

def test_pending_rejected_expired_borderline_not_summarized(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    mock_llm = MockSummarizer()

    # 1. PENDING_REVIEW
    _, dec_pending = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.p1", "Pending Paper", "Abs 1",
        route="BORDERLINE", status=DecisionStatus.PENDING_REVIEW.value,
    )
    with pytest.raises(PermissionError, match="require reviewer acceptance"):
        summarize_decision(dec_pending.id, sample_user.id, db_session, summarizer=mock_llm)

    # 2. REVIEW_REJECTED
    _, dec_rejected = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.p2", "Rejected Paper", "Abs 2",
        route="BORDERLINE", status=DecisionStatus.REVIEW_REJECTED.value,
    )
    with pytest.raises(PermissionError, match="rejected"):
        summarize_decision(dec_rejected.id, sample_user.id, db_session, summarizer=mock_llm)

    # 3. REVIEW_EXPIRED
    _, dec_expired = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.p3", "Expired Paper", "Abs 3",
        route="BORDERLINE", status=DecisionStatus.REVIEW_EXPIRED.value,
    )
    with pytest.raises(PermissionError, match="expired"):
        summarize_decision(dec_expired.id, sample_user.id, db_session, summarizer=mock_llm)

    assert mock_llm.call_count == 0
    assert db_session.query(Summary).count() == 0


# =========================================================================
# Requirement 5: LOW and DEFERRED_OVERFLOW papers are NEVER summarized
# =========================================================================

def test_low_and_deferred_overflow_never_summarized(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    mock_llm = MockSummarizer()

    # LOW / DISCARDED
    _, dec_low = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.low", "Low Paper", "Irrelevant abstract",
        route="LOW", status=DecisionStatus.DISCARDED.value, score=0.20,
    )
    with pytest.raises(PermissionError, match="LOW papers"):
        summarize_decision(dec_low.id, sample_user.id, db_session, summarizer=mock_llm)

    # DEFERRED_OVERFLOW
    _, dec_overflow = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.over", "Overflow Paper", "Abstract",
        route="HIGH", status=DecisionStatus.DEFERRED_OVERFLOW.value, score=0.88,
    )
    with pytest.raises(PermissionError, match="DEFERRED_OVERFLOW"):
        summarize_decision(dec_overflow.id, sample_user.id, db_session, summarizer=mock_llm)

    assert mock_llm.call_count == 0
    assert db_session.query(Summary).count() == 0


# =========================================================================
# Requirement 6: Repeated runs reuse cached summary without another LLM call
# =========================================================================

def test_caching_repeated_runs_reuse_cached_summary(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    _, dec = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.cache_01", "Caching Test Paper", "Abstract text.",
        route="HIGH", status=DecisionStatus.ROUTED_HIGH.value, score=0.90,
    )

    mock_llm = MockSummarizer(model="mock-gemini-flash")

    # First call: triggers LLM
    s1 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=mock_llm, prompt_version="v1.0")
    assert s1 is not None
    assert mock_llm.call_count == 1

    # Second call for the same decision: reuses cache without LLM call
    s2 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=mock_llm, prompt_version="v1.0")
    assert s2.id == s1.id
    assert s2.text == s1.text
    assert mock_llm.call_count == 1  # Unchanged!

    # Third call: still reuses cache
    s3 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=mock_llm, prompt_version="v1.0")
    assert s3.id == s1.id
    assert mock_llm.call_count == 1  # Still unchanged!


# =========================================================================
# Requirement 7: Different prompt versions do not incorrectly reuse an incompatible cached summary
# =========================================================================

def test_different_prompt_versions_do_not_reuse_incompatible_cached_summary(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    _, dec = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.prompt_ver", "Prompt Version Test", "Abstract text.",
        route="HIGH", status=DecisionStatus.ROUTED_HIGH.value, score=0.90,
    )

    mock_llm = MockSummarizer(model="mock-gemini-flash")

    # Call with prompt version v1.0
    s_v1 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=mock_llm, prompt_version="v1.0")
    assert s_v1.prompt_version == "v1.0"
    assert mock_llm.call_count == 1

    # Call with a DIFFERENT prompt version v2.0
    s_v2 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=mock_llm, prompt_version="v2.0")
    assert s_v2.prompt_version == "v2.0"
    assert mock_llm.call_count == 2  # Must invoke LLM with the new prompt version!
    assert mock_llm.calls[1]["prompt_version"] == "v2.0"


# =========================================================================
# Requirement 8: Failed generation preserves retryable state and does NOT create a successful summary record
# =========================================================================

def test_failed_generation_preserves_retryable_state(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    _, dec = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.fail_01", "Fail Test Paper", "Abstract text.",
        route="HIGH", status=DecisionStatus.ROUTED_HIGH.value, score=0.88,
    )

    failing_llm = MockSummarizer(should_fail=True, failure_exception=TimeoutError("API call timed out"))
    result = summarize_decision(dec.id, sample_user.id, db_session, summarizer=failing_llm)

    assert result is None
    db_session.refresh(dec)
    # Status is preserved in a retryable SUMMARIZE_PENDING state
    assert dec.status == DecisionStatus.SUMMARIZE_PENDING.value
    # No summary record created
    assert db_session.query(Summary).filter_by(decision_id=dec.id).count() == 0


# =========================================================================
# Requirement 9: Retrying after failure can succeed without duplicate successful records
# =========================================================================

def test_retrying_after_failure_succeeds_without_duplicates(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    _, dec = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.retry_01", "Retry Test Paper", "Abstract text.",
        route="HIGH", status=DecisionStatus.SUMMARIZE_PENDING.value, score=0.89,
    )

    # Fails on 1st attempt, succeeds on 2nd attempt
    flaky_llm = MockSummarizer(fail_times=1, failure_exception=RuntimeError("Rate limit exceeded"))

    # Attempt 1: Fails
    res1 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=flaky_llm)
    assert res1 is None
    assert flaky_llm.call_count == 1
    assert db_session.query(Summary).filter_by(decision_id=dec.id).count() == 0

    # Attempt 2 (Retry): Succeeds
    res2 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=flaky_llm)
    assert res2 is not None
    assert flaky_llm.call_count == 2
    assert db_session.query(Summary).filter_by(decision_id=dec.id).count() == 1

    # Attempt 3: Caching ensures no additional summary or LLM calls
    res3 = summarize_decision(dec.id, sample_user.id, db_session, summarizer=flaky_llm)
    assert res3.id == res2.id
    assert flaky_llm.call_count == 2
    assert db_session.query(Summary).filter_by(decision_id=dec.id).count() == 1


# =========================================================================
# Requirement 10: Repeated requests do not generate duplicate summaries
# =========================================================================

def test_repeated_requests_prevent_duplicates(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()
    _, dec = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.dup_01", "Deduplication Paper", "Abstract.",
        route="HIGH", status=DecisionStatus.ROUTED_HIGH.value, score=0.91,
    )

    mock_llm = MockSummarizer()
    for _ in range(5):
        s = summarize_decision(dec.id, sample_user.id, db_session, summarizer=mock_llm)
        assert s is not None

    assert mock_llm.call_count == 1
    summaries = db_session.query(Summary).filter_by(decision_id=dec.id).all()
    assert len(summaries) == 1


# =========================================================================
# Requirement 11: Digest pages display summaries correctly and preserve existing functionality
# =========================================================================

def test_digest_pages_display_summaries_and_fallbacks(db_session, sample_user):
    cycle = db_session.query(Cycle).filter_by(user_id=sample_user.id).first()

    # Paper 1: Has a completed summary
    _, dec1 = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.ui_sum", "Summarized Paper", "Raw abstract 1.",
        route="HIGH", status=DecisionStatus.SUMMARIZED.value, score=0.93,
    )
    s1 = Summary(
        decision_id=dec1.id,
        paper_id=dec1.paper_id,
        user_id=sample_user.id,
        text="This interest-framed summary explains key findings concisely.",
        llm_model="gemini-2.5-flash",
        prompt_version="v1.0",
        tokens_in=120,
        tokens_out=45,
    )
    db_session.add(s1)

    # Paper 2: Summary pending (fallback state)
    _, dec2 = create_paper_and_decision(
        db_session, sample_user.id, cycle.id, "2403.ui_pend", "Pending Summary Paper", "Raw abstract 2.",
        route="HIGH", status=DecisionStatus.SUMMARIZE_PENDING.value, score=0.88,
    )

    # Build digest
    digest = Digest(user_id=sample_user.id, cycle_id=cycle.id)
    db_session.add(digest)
    db_session.flush()

    item1 = DigestItem(digest_id=digest.id, decision_id=dec1.id, position=1, via="auto")
    item2 = DigestItem(digest_id=digest.id, decision_id=dec2.id, position=2, via="auto")
    db_session.add_all([item1, item2])
    db_session.commit()

    # Query digest page via FastAPI TestClient
    app.dependency_overrides[get_db] = lambda: db_session
    client = TestClient(app)

    try:
        response = client.get(f"/digest?token={sample_user.access_token}")
        assert response.status_code == 200
        html = response.text

        # Paper 1 assertions: summary rendered, model metadata shown, accordion present
        assert "This interest-framed summary explains key findings concisely." in html
        assert "AI Summary" in html
        assert "gemini-2.5-flash" in html
        assert "v1.0" in html
        assert "Show original paper abstract" in html
        assert "Raw abstract 1." in html

        # Paper 2 assertions: pending notice rendered, raw abstract shown without claiming to be an AI summary
        assert "Summary Pending" in html
        assert "Raw abstract 2." in html

        # Existing UI elements preserved: score badge, feedback buttons, title
        assert "Score: 0.9300" in html
        assert "Score: 0.8800" in html
        assert "Summarized Paper" in html
        assert "Pending Summary Paper" in html
        assert "btn-feedback" in html
    finally:
        app.dependency_overrides.clear()


# =========================================================================
# Requirement 12: Prompt faithfulness and credential security
# =========================================================================

def test_prompt_rendering_and_faithfulness_instructions():
    mock_llm = MockSummarizer()
    prompt = mock_llm.render_prompt(
        title="Attention Is All You Need",
        abstract="We introduce the Transformer model.",
        user_interests="Neural machine translation",
        prompt_version="v1.0",
    )

    # Must contain title, abstract, and user interests
    assert "Attention Is All You Need" in prompt
    assert "We introduce the Transformer model." in prompt
    assert "Neural machine translation" in prompt

    # Faithfulness instructions must be explicit
    assert "Remain strictly faithful to the paper abstract" in prompt
    assert "Never invent methods, results, claims, or findings absent from the source" in prompt


def test_missing_api_key_raises_cleanly_without_leaks():
    summarizer = GeminiSummarizer(api_key=None)
    summarizer.api_key = None  # Ensure no key
    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        summarizer._get_client()


def test_get_summarizer_returns_mock_when_unconfigured():
    summarizer = get_summarizer(mock=True)
    assert isinstance(summarizer, MockSummarizer)
