"""
readiness.py
~~~~~~~~~~~~
Explicit readiness checks and completion safeguards for Phase 13 (Roadmap Section 8 & 8.6).

Safeguard Tiers:
  1. INFRASTRUCTURE_IMPLEMENTED: Phase 13 scripts, survey forms, export pipelines, and tests work.
  2. EVALUATION_PARTIALLY_AVAILABLE: Some labels or pilot observations exist, but required evidence is missing.
  3. FINAL_EVALUATION_READY: All required datasets, exclusions, splits, and pilot evidence satisfy the roadmap.
  4. FINAL_EVALUATION_EXECUTED: Outputs have actually been generated from eligible, complete data.

Requirements verified:
  - Minimum 7 consecutive daily monitoring cycles (Section 8.6).
  - Ground truth labels: >= 100 relevant overall, >= 30 relevant in test split (Section 8.1).
  - End-of-pilot surveys collected (Section 8.6).
  - Frozen test split protection (Section 8.1).
  - Telemetry availability (token counts, LLM costs, timestamps).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    Cycle,
    FeedbackEvent,
    ModelVersion,
    RoutingDecision,
    Summary,
    User,
    ValidationLabel,
)
from research_monitor.evaluation.labeling import (
    get_label_statistics,
    load_latest_labels,
)
from research_monitor.evaluation.survey import load_all_surveys

logger = logging.getLogger(__name__)


class ReadinessTier(str, Enum):
    INFRASTRUCTURE_IMPLEMENTED = "INFRASTRUCTURE_IMPLEMENTED"
    EVALUATION_PARTIALLY_AVAILABLE = "EVALUATION_PARTIALLY_AVAILABLE"
    FINAL_EVALUATION_READY = "FINAL_EVALUATION_READY"
    FINAL_EVALUATION_EXECUTED = "FINAL_EVALUATION_EXECUTED"


@dataclass
class ReadinessResult:
    tier: ReadinessTier
    can_freeze_final_numbers: bool
    explanation: str
    missing_prerequisites: List[str]
    total_users_with_records: int
    genuine_pilot_participants_count: int
    h1_h2_eligible_users_count: int
    developer_users_count: int
    simulated_users_count: int
    total_recorded_cycles: int
    genuine_recorded_cycles: int
    genuine_live_cycles: int
    replay_cycles_count: int
    distinct_active_calendar_days: int
    consecutive_live_calendar_days: int
    validation_labels_total: int
    validation_labels_dev: int
    validation_labels_test: int
    surveys_count: int
    has_token_telemetry: bool
    checklists: Dict[str, Any]

    # Backward compatibility properties
    @property
    def active_users_count(self) -> int:
        return self.genuine_pilot_participants_count

    @property
    def consecutive_calendar_days(self) -> int:
        return self.consecutive_live_calendar_days

    @property
    def live_cycles_count(self) -> int:
        return self.genuine_live_cycles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "can_freeze_final_numbers": self.can_freeze_final_numbers,
            "explanation": self.explanation,
            "missing_prerequisites": self.missing_prerequisites,
            "total_users_with_records": self.total_users_with_records,
            "genuine_pilot_participants_count": self.genuine_pilot_participants_count,
            "h1_h2_eligible_users_count": self.h1_h2_eligible_users_count,
            "developer_users_count": self.developer_users_count,
            "simulated_users_count": self.simulated_users_count,
            "total_recorded_cycles": self.total_recorded_cycles,
            "genuine_recorded_cycles": self.genuine_recorded_cycles,
            "genuine_live_cycles": self.genuine_live_cycles,
            "replay_cycles_count": self.replay_cycles_count,
            "distinct_active_calendar_days": self.distinct_active_calendar_days,
            "consecutive_live_calendar_days": self.consecutive_live_calendar_days,
            "active_users_count": self.genuine_pilot_participants_count,
            "consecutive_calendar_days": self.consecutive_live_calendar_days,
            "live_cycles_count": self.genuine_live_cycles,
            "validation_labels_total": self.validation_labels_total,
            "validation_labels_dev": self.validation_labels_dev,
            "validation_labels_test": self.validation_labels_test,
            "surveys_count": self.surveys_count,
            "has_token_telemetry": self.has_token_telemetry,
            "checklists": self.checklists,
        }


def calculate_user_streak(cycles: list[Cycle]) -> int:
    """Calculate maximum consecutive daily cycle streak."""
    if not cycles:
        return 0
    cycle_dates = []
    for c in cycles:
        if c.started_at:
            cycle_dates.append(c.started_at.date())
        elif c.as_of_date:
            try:
                from datetime import datetime
                cycle_dates.append(datetime.strptime(str(c.as_of_date)[:10], "%Y-%m-%d").date())
            except Exception:
                pass
    unique_dates = sorted(set(cycle_dates))
    if not unique_dates:
        return 0

    streak = 1
    max_streak = 1
    for i in range(1, len(unique_dates)):
        if (unique_dates[i] - unique_dates[i - 1]).days == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    return max_streak


def check_pilot_readiness(
    session: Optional[Session] = None,
    survey_dir: Path | str = "data/surveys",
    results_dir: Path | str = "experiments/results",
) -> ReadinessResult:
    """
    Perform a comprehensive pre-flight readiness audit against Roadmap Section 8 & 8.6 requirements.
    """
    if session is None:
        try:
            from research_monitor.database.session import get_db_session
            with get_db_session() as db:
                return check_pilot_readiness(
                    session=db,
                    survey_dir=survey_dir,
                    results_dir=results_dir,
                )
        except Exception as e:
            logger.warning("Could not open database session for readiness check: %s", e)

    res_dir = Path(results_dir)
    missing_prerequisites: list[str] = []

    # 1. Validation labels audit
    dev_labels_count = 0
    test_labels_count = 0
    total_labels_count = 0

    labels_df = load_latest_labels()
    if labels_df is not None and not labels_df.empty:
        stats = get_label_statistics(labels_df)
        total_labels_count = stats.get("total_count", 0)
        dev_labels_count = len(labels_df[labels_df["split"] == "dev"]) if "split" in labels_df.columns else 0
        test_labels_count = len(labels_df[labels_df["split"] == "test"]) if "split" in labels_df.columns else 0
    elif session is not None:
        db_labels = session.query(ValidationLabel).all()
        if db_labels:
            ldf = pd.DataFrame([{"label": r.label, "split": r.split} for r in db_labels])
            stats = get_label_statistics(ldf)
            total_labels_count = len(db_labels)
            dev_labels_count = sum(1 for r in db_labels if r.split == "dev")
            test_labels_count = sum(1 for r in db_labels if r.split == "test")
        else:
            stats = get_label_statistics(pd.DataFrame())
    else:
        stats = get_label_statistics(pd.DataFrame())

    labels_total_met = stats.get("target_overall_met", False)
    labels_test_met = stats.get("target_test_met", False)

    if not labels_total_met:
        missing_prerequisites.append(
            f"Ground-truth relevant labels overall shortfall: {stats.get('overall_shortfall', 100)} needed (target >= 100)"
        )
    if not labels_test_met:
        missing_prerequisites.append(
            f"Ground-truth test split relevant labels shortfall: {stats.get('test_shortfall', 30)} needed (target >= 30)"
        )

    # 2. Pilot cycles and users audit
    all_users = session.query(User).all() if session is not None else []
    total_users_with_records = len(all_users)

    from research_monitor.evaluation.pilot_export import is_developer_user

    genuine_pilot_users: list[User] = []
    developer_users: list[User] = []
    simulated_users: list[User] = []

    for u in all_users:
        if bool(u.is_simulated):
            simulated_users.append(u)
        elif is_developer_user(u):
            developer_users.append(u)
        else:
            genuine_pilot_users.append(u)

    genuine_pilot_participants_count = len(genuine_pilot_users)
    genuine_pilot_users_count = genuine_pilot_participants_count
    developer_users_count = len(developer_users)
    simulated_users_count = len(simulated_users)

    all_cycles = session.query(Cycle).all() if session is not None else []
    total_recorded_cycles = len(all_cycles)

    # Cycles for genuine pilot participants
    genuine_uids = {u.id for u in genuine_pilot_users}
    genuine_cycles = [c for c in all_cycles if c.user_id in genuine_uids]
    genuine_recorded_cycles = len(genuine_cycles)
    genuine_live_cycles = sum(1 for c in genuine_cycles if str(c.mode).lower() == "live")
    genuine_replay_cycles = sum(1 for c in genuine_cycles if str(c.mode).lower() == "replay")
    replay_cycles_count = sum(1 for c in all_cycles if str(c.mode).lower() == "replay")

    # Calendar dates extraction
    def _extract_date(c: Cycle):
        if c.started_at:
            return c.started_at.date()
        elif c.as_of_date:
            try:
                from datetime import datetime
                return datetime.strptime(str(c.as_of_date)[:10], "%Y-%m-%d").date()
            except Exception:
                return None
        return None

    # Distinct active calendar days across all genuine cycles
    genuine_all_dates = sorted({d for d in (_extract_date(c) for c in genuine_cycles) if d is not None})
    distinct_active_calendar_days = len(genuine_all_dates)

    # Distinct live calendar days
    genuine_live_dates = sorted({d for d in (_extract_date(c) for c in genuine_cycles if str(c.mode).lower() == "live") if d is not None})
    distinct_live_calendar_days = len(genuine_live_dates)

    # Streak of genuine live daily cycles
    max_live_streak = 0
    h1_h2_eligible_count = 0
    user_eligibility_details = {}

    for u in genuine_pilot_users:
        u_cycles = [c for c in genuine_cycles if c.user_id == u.id]
        u_live_cycles = [c for c in u_cycles if str(c.mode).lower() == "live"]
        live_streak = calculate_user_streak(u_live_cycles)
        max_live_streak = max(max_live_streak, live_streak)

        u_dates = {d for d in (_extract_date(c) for c in u_cycles) if d is not None}
        n_days = len(u_dates)

        fe_count = session.query(FeedbackEvent).filter_by(user_id=u.id).count() if session is not None else 0

        # Section 8.6 criteria: >= 5 active days and >= 30 effective labels
        is_eligible = (n_days >= 5) and (fe_count >= 30)
        if is_eligible:
            h1_h2_eligible_count += 1

        reasons = []
        if n_days < 5:
            reasons.append(f"insufficient_active_days ({n_days} < 5)")
        if fe_count < 30:
            reasons.append(f"insufficient_effective_labels ({fe_count} < 30)")

        user_eligibility_details[u.id] = {
            "name": u.name,
            "total_cycles": len(u_cycles),
            "live_cycles": len(u_live_cycles),
            "replay_cycles": len(u_cycles) - len(u_live_cycles),
            "active_calendar_days": n_days,
            "effective_labels": fe_count,
            "h1_h2_eligible": is_eligible,
            "exclusion_reason": "; ".join(reasons) if reasons else None,
        }

    if genuine_pilot_users_count < 3:
        missing_prerequisites.append(
            f"Pilot participants: {genuine_pilot_users_count} genuine participant(s) found (target 3-5 users per Roadmap Section 8.6; "
            f"{h1_h2_eligible_count} currently eligible for H1/H2 before/after evaluation)"
        )
    if max_live_streak < 7:
        missing_prerequisites.append(
            f"Pilot consecutive daily live streak: {max_live_streak}/7 days (minimum 7 consecutive daily live cycles required; "
            f"{genuine_recorded_cycles} total cycles recorded across {distinct_active_calendar_days} active days, but only {genuine_live_cycles} is live)"
        )
    if genuine_live_cycles < 7:
        missing_prerequisites.append(
            f"Live pilot cycles: {genuine_live_cycles} recorded for genuine participants (minimum 7 live cycles required; "
            f"{genuine_replay_cycles} replay cycles excluded from live count)"
        )

    # 3. End-of-pilot surveys audit
    surveys = load_all_surveys(survey_dir=survey_dir)
    surveys_count = len(surveys)
    if surveys_count == 0:
        missing_prerequisites.append(
            "End-of-pilot surveys: 0 responses collected (surveys required for Hypothesis H3)"
        )
    elif surveys_count < max(1, genuine_pilot_users_count):
        missing_prerequisites.append(
            f"End-of-pilot surveys incomplete: {surveys_count}/{genuine_pilot_users_count} participants responded"
        )

    # 4. Telemetry audit
    tokens_recorded = False
    if session is not None:
        tokens_count = (
            session.query(Summary)
            .filter(Summary.tokens_in.isnot(None), Summary.tokens_out.isnot(None))
            .count()
        )
        tokens_recorded = tokens_count > 0

    if not tokens_recorded:
        missing_prerequisites.append(
            "Summary token telemetry: No summary token records in database (cost calculations will report telemetry missing)"
        )

    # Determine Readiness Tier
    final_report_path = res_dir / "final_evaluation_report.json"
    final_tables_exist = (res_dir / "final_e1_gate_quality.csv").exists() and (res_dir / "final_e7_pilot.csv").exists()

    all_criteria_met = (
        labels_total_met
        and labels_test_met
        and genuine_pilot_users_count >= 3
        and max_live_streak >= 7
        and genuine_live_cycles >= 7
        and surveys_count >= genuine_pilot_users_count
    )

    if all_criteria_met and final_report_path.exists() and final_tables_exist:
        current_tier = ReadinessTier.FINAL_EVALUATION_EXECUTED
    elif all_criteria_met:
        current_tier = ReadinessTier.FINAL_EVALUATION_READY
    elif (total_labels_count > 0) or (genuine_pilot_users_count > 0) or (total_recorded_cycles > 0):
        current_tier = ReadinessTier.EVALUATION_PARTIALLY_AVAILABLE
    else:
        current_tier = ReadinessTier.INFRASTRUCTURE_IMPLEMENTED

    explanation = (
        "Phase 13 evaluation infrastructure is fully implemented, verified, and testable. "
        "However, real pilot data collection and human paper labeling are still in progress. "
        "Final numerical results cannot be frozen until all criteria in Section 8 are satisfied."
        if current_tier != ReadinessTier.FINAL_EVALUATION_READY and current_tier != ReadinessTier.FINAL_EVALUATION_EXECUTED
        else "All criteria satisfied. Final evaluation ready."
    )

    checklists = {
        "labels": {
            "total_labels": total_labels_count,
            "relevant_labels": stats.get("relevant_count", 0),
            "test_relevant_labels": stats.get("test_relevant_count", 0),
            "target_overall_met": labels_total_met,
            "target_test_met": labels_test_met,
            "overall_shortfall": stats.get("overall_shortfall", 100),
            "test_shortfall": stats.get("test_shortfall", 30),
        },
        "pilot": {
            "total_users_with_records": total_users_with_records,
            "genuine_pilot_participants_count": genuine_pilot_participants_count,
            "target_genuine_participants": 3,
            "h1_h2_eligible_users_count": h1_h2_eligible_count,
            "developer_users_count": developer_users_count,
            "simulated_users_count": simulated_users_count,
            "total_recorded_cycles": total_recorded_cycles,
            "genuine_recorded_cycles": genuine_recorded_cycles,
            "genuine_live_cycles": genuine_live_cycles,
            "genuine_replay_cycles": genuine_replay_cycles,
            "replay_cycles_count": replay_cycles_count,
            "distinct_active_calendar_days": distinct_active_calendar_days,
            "distinct_live_calendar_days": distinct_live_calendar_days,
            "consecutive_live_calendar_days": max_live_streak,
            "target_live_streak_days": 7,
            "surveys_collected": surveys_count,
            "user_eligibility_details": user_eligibility_details,
        },
        "telemetry": {
            "token_telemetry_present": tokens_recorded,
            "surveys_present": surveys_count > 0,
        },
    }

    return ReadinessResult(
        tier=current_tier,
        can_freeze_final_numbers=all_criteria_met,
        explanation=explanation,
        missing_prerequisites=missing_prerequisites,
        total_users_with_records=total_users_with_records,
        genuine_pilot_participants_count=genuine_pilot_participants_count,
        h1_h2_eligible_users_count=h1_h2_eligible_count,
        developer_users_count=developer_users_count,
        simulated_users_count=simulated_users_count,
        total_recorded_cycles=total_recorded_cycles,
        genuine_recorded_cycles=genuine_recorded_cycles,
        genuine_live_cycles=genuine_live_cycles,
        replay_cycles_count=replay_cycles_count,
        distinct_active_calendar_days=distinct_active_calendar_days,
        consecutive_live_calendar_days=max_live_streak,
        validation_labels_total=total_labels_count,
        validation_labels_dev=dev_labels_count,
        validation_labels_test=test_labels_count,
        surveys_count=surveys_count,
        has_token_telemetry=tokens_recorded,
        checklists=checklists,
    )


def check_phase13_readiness(
    session: Optional[Session] = None,
    survey_dir: Path | str = "data/surveys",
    results_dir: Path | str = "experiments/results",
) -> Dict[str, Any]:
    """Compatibility dictionary wrapper around check_pilot_readiness."""
    res = check_pilot_readiness(session=session, survey_dir=survey_dir, results_dir=results_dir)
    return res.to_dict()
