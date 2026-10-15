"""
Efficiency and Operational Analysis (Phase 13, Section 8.5 E5 & Section 2.E).

Calculates:
- LLM-call reduction: 1.0 - (summaries / papers_scored) against summarize-everything baseline.
- Token and estimated cost savings (using actual records; reports missing telemetry if absent).
- Review queue workload and overflow (DEFERRED_OVERFLOW, caps, expiry).
- Operational metrics over monitoring cycles.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    Cycle,
    DigestItem,
    FeedbackEvent,
    ReviewTask,
    RoutingDecision,
    Summary,
    User,
)

logger = logging.getLogger(__name__)

# Standard pricing baseline (Gemini 1.5 Flash default pricing: $0.075 / 1M in, $0.30 / 1M out)
DEFAULT_COST_PER_1M_INPUT_USD = 0.075
DEFAULT_COST_PER_1M_OUTPUT_USD = 0.30
ESTIMATED_AVG_INPUT_TOKENS_PER_ABSTRACT = 250
ESTIMATED_AVG_OUTPUT_TOKENS_PER_SUMMARY = 150


@dataclass
class CycleEfficiencyMetric:
    cycle_id: Optional[int]
    as_of_date: Optional[str]
    user_id: str
    is_developer: bool
    mode: str
    papers_scored: int
    routed_high: int
    routed_borderline: int
    routed_low: int
    summaries_generated: int
    review_tasks_queued: int
    review_tasks_decided: int
    review_tasks_expired: int
    review_overflow: int
    digest_overflow: int
    llm_call_reduction: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class EfficiencySummary:
    total_papers_scored: int
    total_summaries_generated: int
    overall_llm_call_reduction: Optional[float]
    baseline_description: str
    has_token_telemetry: bool
    total_tokens_in: Optional[int]
    total_tokens_out: Optional[int]
    actual_cost_usd: Optional[float]
    counterfactual_cost_usd: Optional[float]
    estimated_cost_savings_usd: Optional[float]
    cost_savings_pct: Optional[float]
    missing_telemetry_reasons: List[str]
    total_review_tasks_queued: int
    total_review_tasks_decided: int
    total_review_tasks_expired: int
    total_review_overflow: int
    total_digest_overflow: int
    mean_review_tasks_per_cycle: float
    max_review_tasks_per_cycle: int
    cycles_analyzed: int
    cycle_metrics: List[CycleEfficiencyMetric] = field(default_factory=list)
    measured_savings_status: str = "NOT_MEASURABLE"
    theoretical_call_reduction: float = 0.0
    measured_llm_call_reduction: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def calculate_operational_efficiency(
    session: Optional[Session] = None,
    export_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    user_id: Optional[str] = None,
    exclude_developer: bool = True,
    cost_per_1m_input_usd: float = DEFAULT_COST_PER_1M_INPUT_USD,
    cost_per_1m_output_usd: float = DEFAULT_COST_PER_1M_OUTPUT_USD,
) -> EfficiencySummary:
    """
    Computes operational efficiency metrics from database or exported DataFrames.

    Honesty guarantee: If token telemetry (tokens_in, tokens_out) is not recorded,
    this function reports missing data rather than fabricating token counts or costs.
    """
    missing_telemetry_reasons: List[str] = []

    # 1. Load data from DB or export dict
    if session is not None:
        # Load via SQLAlchemy
        q_decisions = session.query(
            RoutingDecision.id,
            RoutingDecision.user_id,
            RoutingDecision.cycle_id,
            RoutingDecision.route,
            RoutingDecision.status,
            RoutingDecision.created_at,
        )
        if user_id:
            q_decisions = q_decisions.filter(RoutingDecision.user_id == user_id)
        decisions_df = pd.DataFrame([
            {
                "id": d.id,
                "user_id": d.user_id,
                "cycle_id": d.cycle_id,
                "route": d.route,
                "status": d.status,
                "created_at": d.created_at,
            }
            for d in q_decisions.all()
        ])

        q_summaries = session.query(
            Summary.id,
            Summary.decision_id,
            Summary.user_id,
            Summary.tokens_in,
            Summary.tokens_out,
            Summary.llm_model,
            Summary.prompt_version,
            Summary.created_at,
        )
        if user_id:
            q_summaries = q_summaries.filter(Summary.user_id == user_id)
        summaries_df = pd.DataFrame([
            {
                "id": s.id,
                "decision_id": s.decision_id,
                "user_id": s.user_id,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "llm_model": s.llm_model,
                "prompt_version": s.prompt_version,
                "created_at": s.created_at,
            }
            for s in q_summaries.all()
        ])

        q_tasks = session.query(
            ReviewTask.id,
            ReviewTask.decision_id,
            ReviewTask.kind,
            ReviewTask.decision,
            ReviewTask.queued_at,
            ReviewTask.decided_at,
            ReviewTask.expired_at,
        )
        tasks_df = pd.DataFrame([
            {
                "id": t.id,
                "decision_id": t.decision_id,
                "kind": t.kind,
                "decision": t.decision,
                "queued_at": t.queued_at,
                "decided_at": t.decided_at,
                "expired_at": t.expired_at,
            }
            for t in q_tasks.all()
        ])

        q_cycles = session.query(Cycle.id, Cycle.user_id, Cycle.as_of_date, Cycle.mode)
        cycles_df = pd.DataFrame([
            {"id": c.id, "user_id": c.user_id, "as_of_date": c.as_of_date, "mode": c.mode}
            for c in q_cycles.all()
        ])

        q_users = session.query(User.id, User.is_simulated)
        users_df = pd.DataFrame([{"id": u.id, "is_simulated": u.is_simulated} for u in q_users.all()])

    elif export_dfs is not None:
        decisions_df = export_dfs.get("routing_decisions", pd.DataFrame())
        summaries_df = export_dfs.get("summaries", pd.DataFrame())
        tasks_df = export_dfs.get("review_tasks", pd.DataFrame())
        cycles_df = export_dfs.get("cycles", pd.DataFrame())
        users_df = export_dfs.get("users", pd.DataFrame())
    else:
        raise ValueError("Must provide either 'session' or 'export_dfs'.")

    # Developer user filtering
    developer_ids = {"developer", "dev_phase5", "dev"}
    if exclude_developer and not decisions_df.empty and "user_id" in decisions_df.columns:
        decisions_df = decisions_df[~decisions_df["user_id"].isin(developer_ids)]
        if not summaries_df.empty and "user_id" in summaries_df.columns:
            summaries_df = summaries_df[~summaries_df["user_id"].isin(developer_ids)]

    # Check for empty data
    if decisions_df.empty:
        missing_telemetry_reasons.append("No routing decisions found.")
        return EfficiencySummary(
            total_papers_scored=0,
            total_summaries_generated=0,
            overall_llm_call_reduction=0.0,
            baseline_description="Summarize-everything (every paper discovered and scored receives an LLM call)",
            has_token_telemetry=False,
            total_tokens_in=None,
            total_tokens_out=None,
            actual_cost_usd=None,
            counterfactual_cost_usd=None,
            estimated_cost_savings_usd=None,
            cost_savings_pct=None,
            missing_telemetry_reasons=missing_telemetry_reasons,
            total_review_tasks_queued=0,
            total_review_tasks_decided=0,
            total_review_tasks_expired=0,
            total_review_overflow=0,
            total_digest_overflow=0,
            mean_review_tasks_per_cycle=0.0,
            max_review_tasks_per_cycle=0,
            cycles_analyzed=0,
            cycle_metrics=[],
        )

    # Calculate overall volume
    total_papers_scored = len(decisions_df)
    total_summaries = len(summaries_df)

    # Theoretical gating call reduction vs summarize-everything counterfactual
    if total_papers_scored > 0:
        theoretical_call_reduction = 1.0 - (total_summaries / total_papers_scored)
    else:
        theoretical_call_reduction = 0.0

    # Token telemetry check
    has_token_telemetry = False
    total_tokens_in: Optional[int] = None
    total_tokens_out: Optional[int] = None
    actual_cost_usd: Optional[float] = None
    counterfactual_cost_usd: Optional[float] = None
    estimated_cost_savings_usd: Optional[float] = None
    cost_savings_pct: Optional[float] = None

    if summaries_df.empty or total_summaries == 0:
        missing_telemetry_reasons.append(
            "Zero LLM summaries generated in evaluated cycles and token telemetry is absent. "
            "Measured LLM usage and cost savings are NOT_MEASURABLE. "
            "Theoretical gating reduction vs summarize-everything counterfactual is reported separately."
        )
        measured_savings_status = "NOT_MEASURABLE"
        measured_llm_call_reduction = None
        overall_llm_call_reduction = None
    elif "tokens_in" not in summaries_df.columns or "tokens_out" not in summaries_df.columns:
        missing_telemetry_reasons.append(
            "Columns 'tokens_in' or 'tokens_out' absent from summary dataset. "
            "Cost savings are NOT_MEASURABLE without token records."
        )
        measured_savings_status = "PARTIAL_CALLS_ONLY_NO_TOKENS"
        measured_llm_call_reduction = theoretical_call_reduction
        overall_llm_call_reduction = theoretical_call_reduction
    elif summaries_df["tokens_in"].dropna().empty or (summaries_df["tokens_in"].fillna(0) == 0).all():
        missing_telemetry_reasons.append(
            "Token counts (tokens_in, tokens_out) are unrecorded (all null or zero). "
            "Measured cost savings are NOT_MEASURABLE without fabrication."
        )
        measured_savings_status = "PARTIAL_CALLS_ONLY_NO_TOKENS"
        measured_llm_call_reduction = theoretical_call_reduction
        overall_llm_call_reduction = theoretical_call_reduction
    else:
        # We have actual token records!
        has_token_telemetry = True
        measured_savings_status = "MEASURED"
        measured_llm_call_reduction = theoretical_call_reduction
        overall_llm_call_reduction = theoretical_call_reduction
        total_tokens_in = int(summaries_df["tokens_in"].fillna(0).sum())
        total_tokens_out = int(summaries_df["tokens_out"].fillna(0).sum())
        actual_cost_usd = (total_tokens_in / 1_000_000 * cost_per_1m_input_usd) + (
            total_tokens_out / 1_000_000 * cost_per_1m_output_usd
        )

        # Counterfactual: if all scored papers were summarized with average token lengths
        avg_in = (
            total_tokens_in / total_summaries
            if total_summaries > 0
            else ESTIMATED_AVG_INPUT_TOKENS_PER_ABSTRACT
        )
        avg_out = (
            total_tokens_out / total_summaries
            if total_summaries > 0
            else ESTIMATED_AVG_OUTPUT_TOKENS_PER_SUMMARY
        )
        counterfactual_tokens_in = total_papers_scored * avg_in
        counterfactual_tokens_out = total_papers_scored * avg_out
        counterfactual_cost_usd = (counterfactual_tokens_in / 1_000_000 * cost_per_1m_input_usd) + (
            counterfactual_tokens_out / 1_000_000 * cost_per_1m_output_usd
        )
        estimated_cost_savings_usd = counterfactual_cost_usd - actual_cost_usd
        cost_savings_pct = (
            (estimated_cost_savings_usd / counterfactual_cost_usd)
            if counterfactual_cost_usd > 0
            else 0.0
        )

    # Review queue and overflow analysis
    total_review_tasks_queued = len(tasks_df[tasks_df["kind"] == "borderline"]) if not tasks_df.empty else 0
    total_review_tasks_decided = (
        len(tasks_df[tasks_df["decision"].isin(["accept", "reject", "skip"])]) if not tasks_df.empty else 0
    )
    total_review_tasks_expired = (
        len(tasks_df[tasks_df["decision"] == "expired"]) if not tasks_df.empty else 0
    )

    # Overflow counts from status or reason
    if "status" in decisions_df.columns:
        total_overflow = len(decisions_df[decisions_df["status"] == "DEFERRED_OVERFLOW"])
    else:
        total_overflow = 0

    # Distinguish queue overflow vs digest overflow if reason exists
    total_review_overflow = 0
    total_digest_overflow = 0
    if "reason" in decisions_df.columns:
        overflow_df = decisions_df[decisions_df["status"] == "DEFERRED_OVERFLOW"]
        for _, row in overflow_df.iterrows():
            reason_str = str(row.get("reason", "")).lower()
            if "queue" in reason_str or "review" in reason_str:
                total_review_overflow += 1
            else:
                total_digest_overflow += 1
    else:
        total_review_overflow = total_overflow

    # Cycle-level breakdown
    cycle_metrics: List[CycleEfficiencyMetric] = []
    cycle_ids = decisions_df["cycle_id"].unique()
    valid_cycle_ids = [c for c in cycle_ids if pd.notna(c)]

    tasks_by_cycle: Dict[int, int] = {}
    if not tasks_df.empty and "decision_id" in tasks_df.columns and not decisions_df.empty:
        merged_tasks = tasks_df.merge(decisions_df[["id", "cycle_id"]], left_on="decision_id", right_on="id")
        for cid, grp in merged_tasks.groupby("cycle_id"):
            tasks_by_cycle[int(cid)] = len(grp)

    for cid in valid_cycle_ids:
        c_decisions = decisions_df[decisions_df["cycle_id"] == cid]
        u_id = c_decisions["user_id"].iloc[0] if not c_decisions.empty else "unknown"
        is_dev = u_id in developer_ids

        # Look up cycle metadata
        as_of = None
        mode = "live"
        if not cycles_df.empty and "id" in cycles_df.columns:
            c_meta = cycles_df[cycles_df["id"] == cid]
            if not c_meta.empty:
                as_of = str(c_meta["as_of_date"].iloc[0])
                mode = str(c_meta["mode"].iloc[0])

        c_scored = len(c_decisions)
        c_high = len(c_decisions[c_decisions["route"] == "HIGH"])
        c_border = len(c_decisions[c_decisions["route"] == "BORDERLINE"])
        c_low = len(c_decisions[c_decisions["route"] == "LOW"])

        # Cycle summaries
        c_summary_count = 0
        c_tokens_in = None
        c_tokens_out = None
        c_cost = None
        if not summaries_df.empty and "decision_id" in summaries_df.columns:
            c_summaries = summaries_df[summaries_df["decision_id"].isin(c_decisions["id"])]
            c_summary_count = len(c_summaries)
            if has_token_telemetry:
                c_tokens_in = int(c_summaries["tokens_in"].fillna(0).sum())
                c_tokens_out = int(c_summaries["tokens_out"].fillna(0).sum())
                c_cost = (c_tokens_in / 1_000_000 * cost_per_1m_input_usd) + (
                    c_tokens_out / 1_000_000 * cost_per_1m_output_usd
                )

        c_tasks_q = tasks_by_cycle.get(int(cid), 0)
        c_overflow = len(c_decisions[c_decisions["status"] == "DEFERRED_OVERFLOW"])
        c_reduction = 1.0 - (c_summary_count / c_scored) if c_scored > 0 else 0.0

        cycle_metrics.append(
            CycleEfficiencyMetric(
                cycle_id=int(cid),
                as_of_date=as_of,
                user_id=u_id,
                is_developer=is_dev,
                mode=mode,
                papers_scored=c_scored,
                routed_high=c_high,
                routed_borderline=c_border,
                routed_low=c_low,
                summaries_generated=c_summary_count,
                review_tasks_queued=c_tasks_q,
                review_tasks_decided=0,  # cycle specific decisions resolved via tasks
                review_tasks_expired=0,
                review_overflow=c_overflow,
                digest_overflow=0,
                llm_call_reduction=c_reduction,
                tokens_in=c_tokens_in,
                tokens_out=c_tokens_out,
                cost_usd=c_cost,
            )
        )

    # Review burden averages
    tasks_per_cycle_list = [cm.review_tasks_queued for cm in cycle_metrics]
    mean_tasks = float(np.mean(tasks_per_cycle_list)) if tasks_per_cycle_list else 0.0
    max_tasks = int(np.max(tasks_per_cycle_list)) if tasks_per_cycle_list else 0

    return EfficiencySummary(
        total_papers_scored=total_papers_scored,
        total_summaries_generated=total_summaries,
        overall_llm_call_reduction=overall_llm_call_reduction,
        baseline_description="Summarize-everything baseline (all scored papers receive an interest-framed summary)",
        has_token_telemetry=has_token_telemetry,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        actual_cost_usd=actual_cost_usd,
        counterfactual_cost_usd=counterfactual_cost_usd,
        estimated_cost_savings_usd=estimated_cost_savings_usd,
        cost_savings_pct=cost_savings_pct,
        missing_telemetry_reasons=missing_telemetry_reasons,
        total_review_tasks_queued=total_review_tasks_queued,
        total_review_tasks_decided=total_review_tasks_decided,
        total_review_tasks_expired=total_review_tasks_expired,
        total_review_overflow=total_review_overflow,
        total_digest_overflow=total_digest_overflow,
        mean_review_tasks_per_cycle=mean_tasks,
        max_review_tasks_per_cycle=max_tasks,
        cycles_analyzed=len(cycle_metrics),
        cycle_metrics=cycle_metrics,
        measured_savings_status=measured_savings_status,
        theoretical_call_reduction=theoretical_call_reduction,
        measured_llm_call_reduction=measured_llm_call_reduction,
    )


def export_efficiency_report(
    summary: EfficiencySummary,
    output_dir: Path,
) -> Path:
    """Exports efficiency metrics table to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "final_efficiency.csv"

    # Summary row + cycle rows
    summary_dict = {
        "metric_scope": "overall",
        "measured_savings_status": summary.measured_savings_status,
        "measured_llm_call_reduction": summary.measured_llm_call_reduction,
        "theoretical_call_reduction": summary.theoretical_call_reduction,
        "cycles_analyzed": summary.cycles_analyzed,
        "total_papers_scored": summary.total_papers_scored,
        "total_summaries_generated": summary.total_summaries_generated,
        "llm_call_reduction": summary.overall_llm_call_reduction,
        "has_token_telemetry": summary.has_token_telemetry,
        "total_tokens_in": summary.total_tokens_in,
        "total_tokens_out": summary.total_tokens_out,
        "actual_cost_usd": summary.actual_cost_usd,
        "counterfactual_cost_usd": summary.counterfactual_cost_usd,
        "estimated_cost_savings_usd": summary.estimated_cost_savings_usd,
        "cost_savings_pct": summary.cost_savings_pct,
        "total_review_tasks_queued": summary.total_review_tasks_queued,
        "total_review_overflow": summary.total_review_overflow,
        "total_digest_overflow": summary.total_digest_overflow,
        "mean_review_tasks_per_cycle": summary.mean_review_tasks_per_cycle,
        "max_review_tasks_per_cycle": summary.max_review_tasks_per_cycle,
        "notes": "; ".join(summary.missing_telemetry_reasons) if summary.missing_telemetry_reasons else "Complete",
    }

    df_summary = pd.DataFrame([summary_dict])
    df_summary.to_csv(out_file, index=False)
    logger.info("Saved efficiency report to %s", out_file)

    # Also save cycle breakdown
    if summary.cycle_metrics:
        cycle_file = output_dir / "final_efficiency_by_cycle.csv"
        cycle_df = pd.DataFrame([asdict(cm) for cm in summary.cycle_metrics])
        cycle_df.to_csv(cycle_file, index=False)
        logger.info("Saved cycle efficiency breakdown to %s", cycle_file)

    return out_file
