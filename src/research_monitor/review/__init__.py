"""
Human review, digest, and feedback package (Phase 6).
"""

from .queue import (
    build_review_queue,
    record_review_action,
    expire_stale_tasks,
    MAX_REVIEW_TASKS_PER_CYCLE,
    REVIEW_EXPIRY_DAYS,
)
from .labels import get_effective_training_labels


def __getattr__(name):
    # Loading labels/queue must not also initialize summary services: those
    # services depend on relevance, which itself imports effective labels.
    if name in {"resume_after_review", "record_user_feedback", "MAX_DIGEST_ITEMS_PER_CYCLE"}:
        from importlib import import_module
        value = getattr(import_module(".digest", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "build_review_queue",
    "record_review_action",
    "expire_stale_tasks",
    "resume_after_review",
    "record_user_feedback",
    "get_effective_training_labels",
    "MAX_REVIEW_TASKS_PER_CYCLE",
    "MAX_DIGEST_ITEMS_PER_CYCLE",
    "REVIEW_EXPIRY_DAYS",
]
