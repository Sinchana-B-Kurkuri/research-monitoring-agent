"""
summary package
~~~~~~~~~~~~~~~
Interest-framed paper summarization module and services (Phase 7).
"""

from research_monitor.summary.summarizer import (
    BaseSummarizer,
    GeminiSummarizer,
    MockSummarizer,
    SummaryResult,
    get_summarizer,
)
from research_monitor.summary.service import (
    summarize_decision,
    summarize_decision_background,
)

__all__ = [
    "BaseSummarizer",
    "GeminiSummarizer",
    "MockSummarizer",
    "SummaryResult",
    "get_summarizer",
    "summarize_decision",
    "summarize_decision_background",
]
