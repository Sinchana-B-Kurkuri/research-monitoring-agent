"""
service.py
~~~~~~~~~~
Summary generation service, caching layer, and relevance gate enforcement (Phase 7).

Follows Roadmap Section 6, 7.7, and 9.1:
  - Enforces strict relevance gate before summarization:
      HIGH (within digest cap): eligible.
      BORDERLINE accepted by reviewer: eligible.
      BORDERLINE pending / rejected / expired: NOT eligible.
      LOW / DISCARDED: NOT eligible.
      DEFERRED_OVERFLOW: NOT eligible.
  - Transparent caching by (user_id, paper_id, llm_model, prompt_version).
  - Preserves SUMMARIZE_PENDING on generation failure; never caches failures.
  - Idempotent and thread-safe background execution for reviewer acceptance.
"""

from datetime import datetime, timezone
import logging
from typing import Optional
from sqlalchemy.orm import Session

from research_monitor import config
from research_monitor.database.models import (
    InterestProfile,
    RoutingDecision,
    Summary,
    User,
)
from research_monitor.relevance.state import DecisionStatus
from research_monitor.summary.summarizer import (
    BaseSummarizer,
    DEFAULT_PROMPT_VERSION,
    get_summarizer,
)

logger = logging.getLogger(__name__)


def is_eligible_for_summarization(decision: RoutingDecision) -> tuple[bool, str]:
    """
    Check if a routing decision is strictly eligible for LLM summarization.

    Returns
    -------
    (is_eligible, reason)
    """
    # 1. Reject LOW papers unconditionally
    if decision.route == "LOW" or decision.status == DecisionStatus.DISCARDED.value:
        return False, "LOW papers are filtered out by the relevance gate and must never be summarized."

    # 2. Reject overflow papers unconditionally
    if decision.status == DecisionStatus.DEFERRED_OVERFLOW.value:
        return False, "DEFERRED_OVERFLOW papers exceeded the digest/review cap and must not be summarized."

    # 3. Check BORDERLINE papers
    if decision.route == "BORDERLINE":
        if decision.status in (
            DecisionStatus.PENDING_REVIEW.value,
            DecisionStatus.ROUTED_BORDERLINE.value,
        ):
            return False, "Pending BORDERLINE papers require reviewer acceptance before summarization."
        if decision.status == DecisionStatus.REVIEW_REJECTED.value:
            return False, "Reviewer-rejected BORDERLINE papers must not be summarized."
        if decision.status == DecisionStatus.REVIEW_EXPIRED.value:
            return False, "Review-expired BORDERLINE papers must not be summarized."
        if decision.status in (
            DecisionStatus.SUMMARIZE_PENDING.value,
            DecisionStatus.SUMMARIZED.value,
            DecisionStatus.DELIVERED.value,
        ):
            return True, "Accepted BORDERLINE paper is eligible for summarization."
        return False, f"Borderline paper with status '{decision.status}' is not eligible."

    # 4. Check HIGH papers
    if decision.route == "HIGH":
        if decision.status in (
            DecisionStatus.ROUTED_HIGH.value,
            DecisionStatus.SUMMARIZE_PENDING.value,
            DecisionStatus.SUMMARIZED.value,
            DecisionStatus.DELIVERED.value,
        ):
            return True, "HIGH paper is eligible for summarization."
        return False, f"HIGH paper with status '{decision.status}' is not eligible."

    return False, f"Paper with route '{decision.route}' and status '{decision.status}' is not eligible."


def summarize_decision(
    decision_id: int,
    user_id: str,
    session: Session,
    summarizer: Optional[BaseSummarizer] = None,
    prompt_version: Optional[str] = None,
) -> Optional[Summary]:
    """
    Summarize a single routing decision, respecting the relevance gate and summary cache.

    Parameters
    ----------
    decision_id : int
        Primary key of the RoutingDecision.
    user_id : str
        Owner user ID for isolation and interest framing.
    session : Session
        Active database session.
    summarizer : Optional[BaseSummarizer]
        Summarizer instance (defaults to get_summarizer()).
    prompt_version : Optional[str]
        Prompt template version (defaults to config.SUMMARY_PROMPT_VERSION).

    Returns
    -------
    Optional[Summary]
        The created or cached Summary, or None if generation failed.
    """
    decision = session.query(RoutingDecision).filter(RoutingDecision.id == decision_id).first()
    if not decision:
        raise ValueError(f"RoutingDecision with id {decision_id} not found.")

    if decision.user_id != user_id:
        raise PermissionError(f"Decision {decision_id} does not belong to user '{user_id}'.")

    # 1. Enforce strict relevance gate
    eligible, reason = is_eligible_for_summarization(decision)
    if not eligible:
        logger.info("Gating skipped summarization for decision %d: %s", decision_id, reason)
        raise PermissionError(reason)

    active_prompt_version = prompt_version or config.SUMMARY_PROMPT_VERSION or DEFAULT_PROMPT_VERSION
    active_summarizer = summarizer or get_summarizer()
    model_name = getattr(active_summarizer, "model", config.LLM_MODEL)

    # 2. Check cache on decision itself
    if decision.summary:
        if (
            decision.summary.prompt_version == active_prompt_version
            and decision.summary.llm_model == model_name
        ):
            logger.info("Reusing existing summary on decision %d (model=%s, prompt=%s)", decision_id, model_name, active_prompt_version)
            return decision.summary
        # If prompt version or model changed, we will regenerate below

    # 3. Check cache across decisions for same (user_id, paper_id, model, prompt_version)
    cached = (
        session.query(Summary)
        .filter(
            Summary.user_id == user_id,
            Summary.paper_id == decision.paper_id,
            Summary.llm_model == model_name,
            Summary.prompt_version == active_prompt_version,
        )
        .first()
    )
    if cached:
        logger.info(
            "Found cached summary for paper '%s', user '%s' (model=%s, prompt=%s). Reusing without LLM call.",
            decision.paper_id,
            user_id,
            model_name,
            active_prompt_version,
        )
        if decision.summary is None:
            # Create decision-level summary link with same text/metadata
            summary = Summary(
                decision_id=decision.id,
                paper_id=decision.paper_id,
                user_id=user_id,
                text=cached.text,
                llm_model=cached.llm_model,
                prompt_version=cached.prompt_version,
                tokens_in=cached.tokens_in,
                tokens_out=cached.tokens_out,
            )
            session.add(summary)
            if decision.status == DecisionStatus.SUMMARIZE_PENDING.value:
                decision.status = DecisionStatus.SUMMARIZED.value
            session.commit()
            return summary
        return decision.summary

    # 4. Fetch user seed interests
    profile = (
        session.query(InterestProfile)
        .filter(InterestProfile.user_id == user_id)
        .order_by(InterestProfile.id.desc())
        .first()
    )
    user_interests = profile.seed_text if profile else "General artificial intelligence and machine learning"

    # 5. Fetch paper details
    paper = decision.paper
    if not paper:
        raise ValueError(f"Paper '{decision.paper_id}' associated with decision {decision_id} not found.")

    # 6. Execute LLM generation
    try:
        result = active_summarizer.generate_summary(
            title=paper.title,
            abstract=paper.abstract,
            user_interests=user_interests,
            prompt_version=active_prompt_version,
        )
    except Exception as e:
        logger.warning(
            "Summarization failed for decision %d (paper %s): %s: %s. Preserving SUMMARIZE_PENDING for retry.",
            decision_id,
            decision.paper_id,
            type(e).__name__,
            str(e),
        )
        # Ensure retryable state: preserve or set SUMMARIZE_PENDING
        decision.status = DecisionStatus.SUMMARIZE_PENDING.value
        session.commit()
        return None

    # 7. Persist successful summary record
    if decision.summary:
        # Overwrite existing summary on this decision if prompt version changed
        decision.summary.text = result.text
        decision.summary.llm_model = result.model
        decision.summary.prompt_version = result.prompt_version
        decision.summary.tokens_in = result.tokens_in
        decision.summary.tokens_out = result.tokens_out
        summary = decision.summary
    else:
        summary = Summary(
            decision_id=decision.id,
            paper_id=decision.paper_id,
            user_id=user_id,
            text=result.text,
            llm_model=result.model,
            prompt_version=result.prompt_version,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )
        session.add(summary)

    if decision.status in (DecisionStatus.SUMMARIZE_PENDING.value, DecisionStatus.ROUTED_HIGH.value):
        decision.status = DecisionStatus.SUMMARIZED.value

    session.commit()
    logger.info(
        "Persisted summary id %d for decision %d, paper '%s' (tokens_in=%s, tokens_out=%s)",
        summary.id,
        decision.id,
        decision.paper_id,
        summary.tokens_in,
        summary.tokens_out,
    )
    return summary


def summarize_decision_background(
    decision_id: int,
    user_id: str,
    summarizer: Optional[BaseSummarizer] = None,
    prompt_version: Optional[str] = None,
) -> None:
    """
    Background worker task to summarize an accepted paper asynchronously.
    Creates and closes its own database session.
    """
    from research_monitor.database.session import SessionLocal

    db = SessionLocal()
    try:
        summarize_decision(
            decision_id=decision_id,
            user_id=user_id,
            session=db,
            summarizer=summarizer,
            prompt_version=prompt_version,
        )
    except Exception as exc:
        logger.error("Error in summarize_decision_background for decision %d: %s", decision_id, exc)
    finally:
        db.close()
