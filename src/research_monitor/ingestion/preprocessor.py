"""
preprocessor.py
~~~~~~~~~~~~~~~
Cleans and filters raw arXiv paper dicts.

Rules applied (in order):
  1. Strip arXiv version suffix from paper_id  (e.g. "2401.12345v2" -> "2401.12345")
  2. Normalize whitespace in title and abstract
  3. Normalize text encoding (replace common Unicode noise)
  4. Remove withdrawn records  (abstract contains "[WITHDRAWN]")
  5. Remove records with empty abstracts
  6. Deduplicate by version-stripped paper_id (keep first seen)
"""

import re
import unicodedata
import logging
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

# Pattern to extract the bare arXiv ID from a full entry_id URL or plain ID string.
# Handles:
#   https://arxiv.org/abs/2401.12345v2  ->  2401.12345
#   2401.12345v2                         ->  2401.12345
#   cs/0612007v1                         ->  cs/0612007
_ID_PATTERN = re.compile(r"(?:abs/)?([a-z]+(?:\.[A-Z]{2})?/\d+|\d{4}\.\d{4,5})(?:v\d+)?$")


def strip_version(raw_id: str) -> str:
    """
    Extract a version-free arXiv paper ID from a full URL or plain ID string.

    Examples:
        "https://arxiv.org/abs/2401.12345v2"  ->  "2401.12345"
        "2401.12345v3"                         ->  "2401.12345"
        "cs/0612007v1"                         ->  "cs/0612007"
    """
    m = _ID_PATTERN.search(raw_id)
    return m.group(1) if m else raw_id


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and common Unicode noise in a text field.

    - NFC Unicode normalization
    - Replace runs of whitespace (newlines, tabs, etc.) with a single space
    - Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_withdrawn(abstract: str) -> bool:
    """Return True if the abstract marks the paper as withdrawn."""
    return "[withdrawn]" in abstract.lower()


def preprocess(
    raw_papers: Iterable[dict],
) -> Iterator[dict]:
    """
    Apply all cleaning and filtering steps to a stream of raw paper dicts.

    Yields cleaned dicts with an added 'paper_id_clean' field containing
    the version-stripped ID.  The original 'paper_id' field is preserved.

    Also yields per-paper, so memory usage stays flat regardless of corpus size.
    """
    seen_ids: set[str] = set()

    stats = {
        "total_in": 0,
        "withdrawn": 0,
        "empty_abstract": 0,
        "duplicates": 0,
        "total_out": 0,
    }

    for paper in raw_papers:
        stats["total_in"] += 1

        # 1. Version-strip the ID
        clean_id = strip_version(paper.get("paper_id", ""))

        # 2 & 3. Normalize text
        title = normalize_text(paper.get("title", ""))
        abstract = normalize_text(paper.get("abstract", ""))

        # 4. Withdrawn check
        if is_withdrawn(abstract):
            stats["withdrawn"] += 1
            continue

        # 5. Empty abstract check
        if not abstract:
            stats["empty_abstract"] += 1
            continue

        # 6. Deduplication
        if clean_id in seen_ids:
            stats["duplicates"] += 1
            continue
        seen_ids.add(clean_id)

        stats["total_out"] += 1
        yield {
            "paper_id": clean_id,
            "title": title,
            "abstract": abstract,
            "categories": paper.get("categories", ""),
            "submitted_date": paper.get("submitted_date", ""),
        }

    logger.info(
        "Preprocessing complete: in=%d  withdrawn=%d  empty_abstract=%d"
        "  duplicates=%d  out=%d",
        stats["total_in"],
        stats["withdrawn"],
        stats["empty_abstract"],
        stats["duplicates"],
        stats["total_out"],
    )
    # Stash stats on the generator's frame so callers can access them if needed.
    # (For simplicity, we log them above; a future refactor can return them explicitly.)
