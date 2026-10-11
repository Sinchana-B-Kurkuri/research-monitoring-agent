"""
labels.py
~~~~~~~~~
Query and extract effective training labels derived from user and reviewer feedback (Phase 6).

Follows Roadmap Section 6 and 7.3:
  - training_labels view maps one effective label per (user, paper).
  - Latest user reaction takes precedence over reviewer decision.
  - Skips and expired tasks produce no feedback events or training labels.
"""

from typing import Optional
from sqlalchemy.orm import Session

from research_monitor.database.models import TrainingLabel


def get_effective_training_labels(
    session: Session,
    user_id: Optional[str] = None,
) -> list[TrainingLabel]:
    """
    Retrieve effective training labels from the training_labels database view.
    Optionally filter by user_id.
    """
    query = session.query(TrainingLabel)
    if user_id:
        query = query.filter(TrainingLabel.user_id == user_id)
    return query.all()
