"""
state.py
~~~~~~~~
Pipeline state object, candidate representation, and state-machine statuses (Phase 5).

Follows Roadmap Section 9.1:
  - Gate decisions:
      HIGH       -> ROUTED_HIGH
      BORDERLINE -> ROUTED_BORDERLINE
      LOW        -> DISCARDED
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


class Route(str, Enum):
    HIGH = "HIGH"
    BORDERLINE = "BORDERLINE"
    LOW = "LOW"


class DecisionStatus(str, Enum):
    # Relevance Agent cold start statuses (Roadmap Section 9.1)
    ROUTED_HIGH = "ROUTED_HIGH"
    ROUTED_BORDERLINE = "ROUTED_BORDERLINE"
    DISCARDED = "DISCARDED"

    # Subsequent phase statuses (Section 9.1)
    PENDING_REVIEW = "PENDING_REVIEW"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    REVIEW_EXPIRED = "REVIEW_EXPIRED"
    DEFERRED_OVERFLOW = "DEFERRED_OVERFLOW"
    SUMMARIZE_PENDING = "SUMMARIZE_PENDING"
    SUMMARIZED = "SUMMARIZED"
    DELIVERED = "DELIVERED"
    REACTED = "REACTED"


@dataclass
class PaperCandidate:
    """Represents a paper candidate flowing through the pipeline nodes."""
    paper_id: str
    title: str
    abstract: str
    submitted_date: str
    categories: list[str] | str = field(default_factory=list)
    is_reference: bool = False
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None
    route: Optional[Route] = None
    status: Optional[DecisionStatus] = None
    reason: Optional[str] = None


@dataclass
class PipelineState:
    """
    Explicit pipeline state object carrying execution context between stages:
    discover -> preprocess -> embed -> score -> route
    """
    user_id: str
    cycle_id: Optional[int] = None
    mode: str = "replay"  # 'replay' or 'live'
    as_of_date: str = ""
    is_simulated: bool = False

    # Active user preference memory & thresholds
    active_vector: Optional[np.ndarray] = None
    vector_version_id: Optional[int] = None
    vector_version_num: int = 0
    t_high: float = 0.0
    t_low: float = 0.0
    p_high_level: float = 98.0
    p_low_level: float = 90.0

    # Node progression containers
    discovered: list[PaperCandidate] = field(default_factory=list)
    preprocessed: list[PaperCandidate] = field(default_factory=list)
    embedded: list[PaperCandidate] = field(default_factory=list)
    scored: list[PaperCandidate] = field(default_factory=list)
    routed: list[PaperCandidate] = field(default_factory=list)

    # Execution telemetry
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def thresholds_dict(self) -> dict[str, float]:
        return {
            "t_high": float(self.t_high),
            "t_low": float(self.t_low),
            "p_high": float(self.p_high_level),
            "p_low": float(self.p_low_level),
        }
