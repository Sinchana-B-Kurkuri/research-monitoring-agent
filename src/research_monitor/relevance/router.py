"""
router.py
~~~~~~~~~
Routing logic, boundary rules, and decision reason generation (Phase 5).

Follows Roadmap Section 7.2 & 9.1:
  - s >= t_high -> HIGH (Status: ROUTED_HIGH)
  - s <= t_low  -> LOW (Status: DISCARDED)
  - otherwise   -> BORDERLINE (Status: ROUTED_BORDERLINE)
"""

from dataclasses import dataclass
from typing import Optional
from research_monitor.relevance.state import DecisionStatus, Route


@dataclass(frozen=True)
class RoutingResult:
    route: Route
    status: DecisionStatus
    reason: str
    cosine_score: float
    t_high: float
    t_low: float
    p_high: float
    p_low: float


def route_score(
    score: float,
    t_high: float,
    t_low: float,
    p_high: float = 98.0,
    p_low: float = 90.0,
) -> RoutingResult:
    """
    Apply the exact threshold-boundary rules from Roadmap Section 7.2.

    Rules:
      - score >= t_high: route = Route.HIGH, status = DecisionStatus.ROUTED_HIGH
      - score <= t_low:  route = Route.LOW, status = DecisionStatus.DISCARDED
      - otherwise:       route = Route.BORDERLINE, status = DecisionStatus.ROUTED_BORDERLINE

    Boundary checks:
      - An exact score of t_high routes to HIGH.
      - An exact score of t_low routes to LOW.

    Returns:
      RoutingResult with route, status, and descriptive reason.
    """
    s = float(score)
    th = float(t_high)
    tl = float(t_low)

    if s >= th:
        route = Route.HIGH
        status = DecisionStatus.ROUTED_HIGH
        reason = (
            f"Score {s:.4f} >= threshold {th:.4f} (P_high={p_high:.1f}th percentile). "
            f"Cleared gate for automatic delivery."
        )
    elif s <= tl:
        route = Route.LOW
        status = DecisionStatus.DISCARDED
        reason = (
            f"Score {s:.4f} <= threshold {tl:.4f} (P_low={p_low:.1f}th percentile). "
            f"Discarded paper below relevance floor."
        )
    else:
        route = Route.BORDERLINE
        status = DecisionStatus.ROUTED_BORDERLINE
        reason = (
            f"Score {s:.4f} in borderline band ({tl:.4f} < score < {th:.4f}). "
            f"Queued for human review."
        )

    return RoutingResult(
        route=route,
        status=status,
        reason=reason,
        cosine_score=s,
        t_high=th,
        t_low=tl,
        p_high=float(p_high),
        p_low=float(p_low),
    )
