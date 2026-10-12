"""
live.py
~~~~~~~
Live arXiv paper retrieval, database deduplication, and selective embedding (Phase 8).

Follows Roadmap Section 6, 7.8, and 9.1:
  - Fetches latest research papers from arXiv matching research categories (cs.CL, cs.LG, cs.AI).
  - Version-strips and normalizes paper IDs (e.g. '2401.12345v2' -> '2401.12345').
  - Filters withdrawn papers and empty abstracts.
  - Deduplicates cross-listed papers and checks against existing DB papers.
  - Generates L2-normalized 384-d embeddings strictly for genuinely new papers.
  - Bounded retries with exponential backoff for transient arXiv network/rate-limit errors.
  - Treats empty result sets as successful cycles.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Iterator, Optional

from sqlalchemy.orm import Session

from research_monitor.config import config
from research_monitor.database.models import Paper
from research_monitor.ingestion.preprocessor import (
    is_withdrawn,
    normalize_text,
    strip_version,
)

logger = logging.getLogger(__name__)


@dataclass
class LiveIngestionResult:
    """Telemetry report for a live arXiv ingestion run."""
    fetched: int = 0
    inserted: int = 0
    skipped_duplicates: int = 0
    embedded: int = 0
    failed_embeddings: int = 0
    new_paper_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return self.inserted + self.skipped_duplicates


def build_live_query(
    categories: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Construct category-restricted arXiv query with optional submitted date window.

    Parameters
    ----------
    categories : Optional[list[str]]
        Target arXiv categories (e.g. ['cs.CL', 'cs.LG', 'cs.AI']).
    start_date : Optional[str]
        ISO date 'YYYY-MM-DD' (inclusive).
    end_date : Optional[str]
        ISO date 'YYYY-MM-DD' (inclusive).

    Returns
    -------
    str
        Formatted query string for arXiv API search.
    """
    cats = categories or [c.strip() for c in config.CORPUS_CATEGORIES.split(",") if c.strip()]
    cat_terms = [f"cat:{c}" for c in cats]
    cat_part = " OR ".join(cat_terms)

    if start_date and end_date:
        start = start_date.replace("-", "") + "0000"
        end = end_date.replace("-", "") + "2359"
        return f"({cat_part}) AND submittedDate:[{start} TO {end}]"
    elif start_date:
        start = start_date.replace("-", "") + "0000"
        return f"({cat_part}) AND submittedDate:[{start} TO 999912312359]"
    elif end_date:
        end = end_date.replace("-", "") + "2359"
        return f"({cat_part}) AND submittedDate:[000001010000 TO {end}]"

    return f"({cat_part})"


def fetch_live_papers(
    categories: Optional[list[str]] = None,
    max_results: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    client_override: Optional[Any] = None,
    sleep_between_pages: float = 3.0,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Fetch and clean recent papers from arXiv with bounded retries and exponential backoff.

    Parameters
    ----------
    categories : Optional[list[str]]
        Categories to include.
    max_results : int
        Maximum papers to retrieve.
    start_date : Optional[str]
        Optional start date filter.
    end_date : Optional[str]
        Optional end date filter.
    client_override : Optional[Any]
        Mock client for automated testing.
    sleep_between_pages : float
        Delay seconds between requests to respect arXiv rate limits (default: 3.0s).
    max_retries : int
        Number of retry attempts for transient errors.
    initial_backoff : float
        Initial backoff delay in seconds.

    Returns
    -------
    list[dict[str, Any]]
        List of cleaned, version-stripped paper dicts without duplicates.
    """
    import arxiv

    query = build_live_query(categories=categories, start_date=start_date, end_date=end_date)
    logger.info("Executing live arXiv query: %s (max_results=%d)", query, max_results)

    client = client_override or arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=sleep_between_pages,
        num_retries=max_retries,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results_generator = None
    for attempt in range(max_retries):
        try:
            results_generator = client.results(search)
            # Evaluate generator to trigger any immediate connection/HTTP errors
            raw_results = list(results_generator)
            break
        except Exception as exc:
            # Check for permanent errors (e.g. 400 Bad Request, malformed syntax)
            err_str = str(exc).lower()
            if "400" in err_str or "bad request" in err_str or "invalid query" in err_str:
                logger.error("Permanent arXiv error encountered: %s: %s", type(exc).__name__, exc)
                raise

            if attempt == max_retries - 1:
                logger.error("Live arXiv retrieval failed after %d attempts: %s", max_retries, exc)
                raise

            wait_time = initial_backoff * (2 ** attempt)
            logger.warning(
                "Transient arXiv fetch failure (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt + 1,
                max_retries,
                exc,
                wait_time,
            )
            time.sleep(wait_time)

    cleaned_papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in raw_results:
        # Extract and version-strip paper ID
        entry_id = getattr(item, "entry_id", None) or getattr(item, "get_short_id", lambda: "")()
        clean_id = strip_version(str(entry_id))
        if not clean_id:
            continue

        # Prevent duplicate entries for cross-listed papers in the same run
        if clean_id in seen_ids:
            continue
        seen_ids.add(clean_id)

        # Normalize text content
        raw_title = getattr(item, "title", "") or ""
        raw_abstract = getattr(item, "summary", "") or ""
        title = normalize_text(raw_title)
        abstract = normalize_text(raw_abstract)

        # Filter withdrawn and empty-abstract papers
        if is_withdrawn(abstract) or not abstract:
            logger.debug("Filtered paper %s (withdrawn or empty abstract).", clean_id)
            continue

        # Extract submitted date
        pub = getattr(item, "published", None)
        submitted_date = str(pub.date()) if pub and hasattr(pub, "date") else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Categories
        cats = getattr(item, "categories", [])
        if isinstance(cats, (list, tuple)):
            categories_list = list(cats)
        elif isinstance(cats, str):
            categories_list = [c.strip() for c in cats.split(",") if c.strip()]
        else:
            categories_list = []

        cleaned_papers.append({
            "paper_id": clean_id,
            "title": title,
            "abstract": abstract,
            "submitted_date": submitted_date,
            "categories": categories_list,
        })

    logger.info("Retrieved and preprocessed %d unique live papers from arXiv.", len(cleaned_papers))
    return cleaned_papers


def ingest_live_papers(
    session: Session,
    categories: Optional[list[str]] = None,
    max_results: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    papers_override: Optional[list[dict[str, Any]]] = None,
    client_override: Optional[Any] = None,
) -> LiveIngestionResult:
    """
    Fetch, deduplicate against the database, embed, and persist new arXiv papers.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy database session.
    categories : Optional[list[str]]
        arXiv categories to fetch.
    max_results : int
        Maximum papers to fetch.
    start_date : Optional[str]
        Optional start date filter.
    end_date : Optional[str]
        Optional end date filter.
    papers_override : Optional[list[dict]]
        Optional explicit paper dicts (used for unit tests and mocks).
    client_override : Optional[Any]
        Optional mock arXiv client.

    Returns
    -------
    LiveIngestionResult
        Detailed telemetry counts (fetched, inserted, skipped_duplicates, embedded).
    """
    result = LiveIngestionResult()

    # 1. Fetch papers or use provided override
    try:
        if papers_override is not None:
            raw_papers = papers_override
        else:
            raw_papers = fetch_live_papers(
                categories=categories,
                max_results=max_results,
                start_date=start_date,
                end_date=end_date,
                client_override=client_override,
            )
    except Exception as exc:
        result.errors.append(f"Fetch failure: {exc}")
        logger.error("Error during live paper fetching: %s", exc)
        raise

    result.fetched = len(raw_papers)
    if not raw_papers:
        logger.info("No live papers found matching query. Returning successful empty result.")
        return result

    # 2. Extract candidate IDs and normalize
    normalized_candidates: list[dict[str, Any]] = []
    seen_in_batch: set[str] = set()

    for p in raw_papers:
        raw_id = str(p.get("paper_id") or p.get("id") or "")
        clean_id = strip_version(raw_id)
        if not clean_id or clean_id in seen_in_batch:
            continue
        seen_in_batch.add(clean_id)

        title = normalize_text(p.get("title", ""))
        abstract = normalize_text(p.get("abstract", ""))
        if is_withdrawn(abstract) or not abstract:
            continue

        normalized_candidates.append({
            "paper_id": clean_id,
            "title": title,
            "abstract": abstract,
            "submitted_date": str(p.get("submitted_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))),
            "categories": p.get("categories", []),
            "embedding": p.get("embedding", None),
        })

    candidate_ids = [p["paper_id"] for p in normalized_candidates]

    # 3. Deduplicate against existing database records
    existing_records = (
        session.query(Paper.id)
        .filter(Paper.id.in_(candidate_ids))
        .all()
    )
    existing_ids = set(r[0] for r in existing_records)

    new_papers: list[dict[str, Any]] = []
    for p in normalized_candidates:
        if p["paper_id"] in existing_ids:
            result.skipped_duplicates += 1
        else:
            new_papers.append(p)

    logger.info(
        "Deduplication summary: %d candidate papers, %d existing in DB (skipped), %d genuinely new.",
        len(normalized_candidates),
        result.skipped_duplicates,
        len(new_papers),
    )

    if not new_papers:
        logger.info("No genuinely new papers to insert or embed.")
        return result

    # 4. Generate embeddings strictly for new papers
    try:
        # Check if embeddings are already supplied (e.g. in test fixtures)
        missing_embeddings = [p for p in new_papers if p.get("embedding") is None]
        if missing_embeddings:
            # Lazy import to ensure web service startup never imports SentenceTransformer
            from research_monitor.ingestion.embedder import embed_texts

            texts = [f"{p['title']} {p['abstract']}".strip() for p in missing_embeddings]
            computed_vectors = embed_texts(texts, model_name=config.EMBEDDING_MODEL)

            for p, vec in zip(missing_embeddings, computed_vectors):
                p["embedding"] = vec.tolist()

        result.embedded = len(new_papers)

    except Exception as exc:
        result.failed_embeddings = len(new_papers)
        result.errors.append(f"Embedding failure: {exc}")
        logger.error("Failed to generate embeddings for new papers: %s", exc)
        session.rollback()
        raise

    # 5. Persist new paper records to database
    try:
        for p in new_papers:
            emb_vector = p.get("embedding")
            paper_entity = Paper(
                id=p["paper_id"],
                title=p["title"],
                abstract=p["abstract"],
                submitted_date=p["submitted_date"],
                categories=p["categories"],
                embedding=emb_vector,
                embedding_model=config.EMBEDDING_MODEL,
                is_reference=False,
            )
            session.add(paper_entity)
            result.inserted += 1
            result.new_paper_ids.append(p["paper_id"])

        session.commit()
        logger.info("Successfully persisted %d new papers with embeddings.", result.inserted)

    except Exception as exc:
        session.rollback()
        result.errors.append(f"Database insertion failure: {exc}")
        logger.error("Database insertion failed, transaction rolled back safely: %s", exc)
        raise

    return result


def run_live_cycle(
    user_id: str,
    session: Session,
    categories: Optional[list[str]] = None,
    max_results: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    papers_override: Optional[list[dict[str, Any]]] = None,
    client_override: Optional[Any] = None,
) -> tuple[Any, LiveIngestionResult]:
    """
    Execute an autonomous live monitoring cycle:
      1. Ingests new arXiv papers (fetch, deduplicate, embed, persist).
      2. Runs the Relevance Agent pipeline state machine with mode = 'live'.
      3. Queues borderline papers for human review.

    Returns
    -------
    (pipeline_state, ingestion_result)
    """
    from research_monitor.relevance.pipeline import run_monitoring_cycle

    # 1. Ingest live papers
    ingestion_result = ingest_live_papers(
        session=session,
        categories=categories,
        max_results=max_results,
        start_date=start_date,
        end_date=end_date,
        papers_override=papers_override,
        client_override=client_override,
    )

    # 2. Run monitoring cycle in live mode
    state = run_monitoring_cycle(
        user_id=user_id,
        session=session,
        mode="live",
        as_of_date=end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        limit=max_results,
    )

    return state, ingestion_result
