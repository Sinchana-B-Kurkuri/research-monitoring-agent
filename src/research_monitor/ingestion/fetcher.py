"""
fetcher.py
~~~~~~~~~~
Fetches arXiv paper metadata via the official arXiv API client.

Streams results incrementally — never loads the full corpus into memory at once.
Returns an iterator of raw paper dicts suitable for preprocessing.
"""

import time
import logging
from typing import Iterator

import arxiv

logger = logging.getLogger(__name__)


def _build_query(categories: list[str], start_date: str, end_date: str) -> str:
    """
    Build an arXiv API search query.

    Uses the submittedDate field for date filtering and combines categories
    with OR logic.

    Args:
        categories:  List of arXiv category strings, e.g. ["cs.CL", "cs.LG"].
        start_date:  ISO date string "YYYY-MM-DD" (inclusive).
        end_date:    ISO date string "YYYY-MM-DD" (inclusive).

    Returns:
        A query string ready for the arXiv API.
    """
    # arXiv date format for submittedDate: YYYYMMDD0000 TO YYYYMMDD2359
    start = start_date.replace("-", "") + "0000"
    end = end_date.replace("-", "") + "2359"
    date_filter = f"submittedDate:[{start} TO {end}]"

    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cat_filter}) AND {date_filter}"


def fetch_papers(
    categories: list[str],
    start_date: str,
    end_date: str,
    max_results: int = 10_000,
    page_size: int = 500,
    sleep_between_pages: float = 3.0,
) -> Iterator[dict]:
    """
    Stream arXiv paper metadata from the API.

    Yields raw paper dicts, one at a time, without buffering everything.
    Respects arXiv's rate-limit guidelines (3s between page requests).

    Args:
        categories:            arXiv categories to include.
        start_date:            Corpus start date "YYYY-MM-DD".
        end_date:              Corpus end date "YYYY-MM-DD".
        max_results:           Maximum papers to retrieve in total.
        page_size:             Papers per API page request.
        sleep_between_pages:   Seconds to wait between page fetches.

    Yields:
        Dict with keys: paper_id, title, abstract, categories, submitted_date.
    """
    query = _build_query(categories, start_date, end_date)
    logger.info("arXiv query: %s  (max_results=%d)", query, max_results)

    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=sleep_between_pages,
        num_retries=5,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    fetched = 0
    for result in client.results(search):
        paper = {
            "paper_id": result.entry_id,          # full URL, version-stripped in preprocessor
            "title": result.title or "",
            "abstract": result.summary or "",
            "categories": ",".join(result.categories),
            "submitted_date": str(result.published.date()) if result.published else "",
        }
        yield paper
        fetched += 1
        if fetched % 500 == 0:
            logger.info("  fetched %d papers so far…", fetched)

    logger.info("Fetch complete. Total yielded: %d", fetched)
