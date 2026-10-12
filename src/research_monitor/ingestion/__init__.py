"""
research_monitor.ingestion
~~~~~~~~~~~~~~~~~~~~~~~~~~
Corpus ingestion, preprocessing, embedding, and live arXiv monitoring (Phase 2 & Phase 8).
"""

from research_monitor.ingestion.fetcher import fetch_papers
from research_monitor.ingestion.preprocessor import (
    is_withdrawn,
    normalize_text,
    preprocess,
    strip_version,
)
from research_monitor.ingestion.live import (
    LiveIngestionResult,
    build_live_query,
    fetch_live_papers,
    ingest_live_papers,
    run_live_cycle,
)

__all__ = [
    "fetch_papers",
    "is_withdrawn",
    "normalize_text",
    "preprocess",
    "strip_version",
    "LiveIngestionResult",
    "build_live_query",
    "fetch_live_papers",
    "ingest_live_papers",
    "run_live_cycle",
]
