"""
pilot_export.py
~~~~~~~~~~~~~~~
Reproducible pilot data extraction, exclusion audit, and dataset preparation (Roadmap Section 8.6 & Phase 13).

Features:
  - Extracts all pilot telemetry from database (routing, review, digest, feedback, vector/model history).
  - Preserves distinct signal sources ('user' reactions vs 'reviewer' decisions).
  - Applies canonical effective-label rule (user like/dislike takes precedence over reviewer decision).
  - Evaluates Roadmap Section 8.6 exclusion rules:
      * User activity threshold for H1/H2 (< 5 active days or < 30 effective labels).
      * Outage day logging and exclusion.
      * Reactions before onboarding completion.
      * Developer-user isolation (reported separately).
      * Data categorisation (LIVE_PILOT, DEVELOPER_PILOT, REPLAY, SIMULATED).
  - Produces an audit trail of included/excluded records with exact exclusion reasons.
  - Deterministic and safe to rerun without modifying database state.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.database.models import (
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    InterestProfile,
    InterestVectorVersion,
    ModelVersion,
    Paper,
    ReviewTask,
    RoutingDecision,
    Summary,
    User,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_DIR = Path("data/pilot_export")


def is_developer_user(user: User) -> bool:
    """Check if a user is the developer user per Roadmap Section 8.6."""
    uid = str(user.id).lower()
    uname = str(user.name or "").lower()
    return (
        uid in {"developer", "dev", "dev_phase5"}
        or uid.startswith("dev_")
        or uid.startswith("developer")
        or "developer" in uname
        or "dev phase" in uname
        or uname in {"dev", "developer"}
    )


def categorize_record(
    is_sim: bool,
    is_dev: bool,
    cycle_mode: str,
) -> str:
    """Categorize a record into mutually exclusive data partitions."""
    if is_sim:
        return "SIMULATED"
    if is_dev:
        return "DEVELOPER_PILOT"
    if str(cycle_mode).lower() == "replay":
        return "REPLAY"
    return "LIVE_PILOT"


@dataclass
class PilotExportSummary:
    total_decisions_exported: int
    clean_pilot_records_count: int
    excluded_decisions_count: int
    developer_decisions_count: int
    simulated_replay_decisions_count: int
    audit_log_path: str
    output_dir: str
    genuine_pilot_replay_count: int = 0
    simulated_decisions_count: int = 0
    is_reconciled: bool = True
    user_eligibility: Dict[str, Any] = field(default_factory=dict)
    mutually_exclusive_partition: Dict[str, Any] = field(default_factory=dict)
    overlapping_flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_routing_decisions_exported(self) -> int:
        return self.total_decisions_exported

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_routing_decisions_exported"] = self.total_routing_decisions_exported
        return d


def export_pilot_dataset(
    session: Session,
    export_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    user_id: Optional[str] = None,
) -> PilotExportSummary:
    """
    Extract and export all pilot data from database, applying Section 8.6 exclusion rules.

    Args:
        session: SQLAlchemy DB session.
        export_dir: Directory where analysis-ready dataset and audit logs are saved.
        output_dir: Alias for export_dir.
        user_id: Optional user to filter on.

    Returns:
        PilotExportSummary instance.
    """
    target_path = export_dir or output_dir or DEFAULT_EXPORT_DIR
    out_dir = Path(target_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = session.query(User).order_by(User.id).all()

    # Pre-fetch paper titles
    papers_map = {p.id: p.title for p in session.query(Paper.id, Paper.title).all()}

    all_records: list[dict[str, Any]] = []
    user_audit: dict[str, Any] = {}

    for user in users:
        is_dev = is_developer_user(user)
        is_sim = bool(user.is_simulated)

        # 1. Onboarding timestamp (when initial profile was created)
        profile = (
            session.query(InterestProfile)
            .filter_by(user_id=user.id)
            .order_by(InterestProfile.created_at.asc())
            .first()
        )
        onboarding_ts = profile.created_at if profile else user.created_at

        # 2. Vector activation dates
        vector_versions = (
            session.query(InterestVectorVersion)
            .filter_by(user_id=user.id)
            .order_by(InterestVectorVersion.version.asc())
            .all()
        )
        vector_map = {v.id: v for v in vector_versions}
        # First adapted vector version (v1 or higher, rocchio)
        adapted_vectors = [v for v in vector_versions if v.version > 0 and v.method == "rocchio"]
        vector_activation_dt = adapted_vectors[0].created_at if adapted_vectors else None

        # 3. Model activation dates
        model_versions = (
            session.query(ModelVersion)
            .filter_by(user_id=user.id)
            .order_by(ModelVersion.version.asc())
            .all()
        )
        promoted_models = [m for m in model_versions if m.promoted and m.is_active]
        model_activation_dt = promoted_models[0].trained_at if promoted_models else None

        # 4. Cycles
        cycles = (
            session.query(Cycle)
            .filter_by(user_id=user.id)
            .order_by(Cycle.started_at.asc(), Cycle.id.asc())
            .all()
        )
        cycle_map = {c.id: c for c in cycles}

        # 5. Feedback events (index by paper_id and source)
        feedback_events = (
            session.query(FeedbackEvent)
            .filter_by(user_id=user.id)
            .order_by(FeedbackEvent.created_at.asc(), FeedbackEvent.id.asc())
            .all()
        )
        # Group feedback events by paper_id
        user_feedback_by_pid: dict[str, list[FeedbackEvent]] = {}
        for fe in feedback_events:
            user_feedback_by_pid.setdefault(fe.paper_id, []).append(fe)

        # 6. Review tasks
        review_tasks = (
            session.query(ReviewTask)
            .join(RoutingDecision, ReviewTask.decision_id == RoutingDecision.id)
            .filter(RoutingDecision.user_id == user.id)
            .all()
        )
        review_by_decision_id = {rt.decision_id: rt for rt in review_tasks}

        # 7. Digest items
        digest_items = (
            session.query(DigestItem)
            .join(Digest, DigestItem.digest_id == Digest.id)
            .filter(Digest.user_id == user.id)
            .all()
        )
        digest_by_decision_id = {di.decision_id: di for di in digest_items}

        # 8. Summaries
        summaries = (
            session.query(Summary)
            .filter_by(user_id=user.id)
            .all()
        )
        summary_by_decision_id = {s.decision_id: s for s in summaries}

        # 9. Compute active days and effective labels
        active_dates = set()
        for c in cycles:
            if c.started_at:
                active_dates.add(c.started_at.date())
        for fe in feedback_events:
            if fe.created_at:
                active_dates.add(fe.created_at.date())
        for rt in review_tasks:
            if rt.decided_at:
                active_dates.add(rt.decided_at.date())

        # Effective labels calculation (Section 7.3)
        effective_labels_dict: dict[str, tuple[int, str]] = {}
        all_pids = set(user_feedback_by_pid.keys()) | {
            rt.routing_decision.paper_id for rt in review_tasks if rt.routing_decision
        }

        for pid in all_pids:
            fe_list = user_feedback_by_pid.get(pid, [])
            user_fe = [e for e in fe_list if e.source == "user"]
            reviewer_fe = [e for e in fe_list if e.source == "reviewer"]

            if user_fe:
                # Latest user like/dislike takes precedence
                effective_labels_dict[pid] = (int(user_fe[-1].label), "user")
            elif reviewer_fe:
                effective_labels_dict[pid] = (int(reviewer_fe[-1].label), "reviewer")
            else:
                # Check review task decision directly if no feedback_event was logged
                matching_task = next(
                    (rt for rt in review_tasks if rt.routing_decision and rt.routing_decision.paper_id == pid),
                    None,
                )
                if matching_task and matching_task.decision in {"accept", "reject"}:
                    lbl = 1 if matching_task.decision == "accept" else 0
                    effective_labels_dict[pid] = (lbl, "reviewer")

        n_effective_labels = len(effective_labels_dict)
        n_active_days = len(active_dates)

        # Section 8.6 Exclusion Rule: users with < 5 active days or < 30 effective labels
        # are excluded from H1/H2 (but kept for H3 survey analysis)
        meets_h1_h2 = (n_active_days >= 5) and (n_effective_labels >= 30)
        h1_h2_exclusions = []
        if n_active_days < 5:
            h1_h2_exclusions.append(f"insufficient_active_days ({n_active_days} < 5)")
        if n_effective_labels < 30:
            h1_h2_exclusions.append(f"insufficient_effective_labels ({n_effective_labels} < 30)")

        user_audit[user.id] = {
            "name": user.name,
            "is_developer": is_dev,
            "is_simulated": is_sim,
            "total_cycles": len(cycles),
            "active_days_count": n_active_days,
            "effective_labels_count": n_effective_labels,
            "meets_h1_h2_criteria": meets_h1_h2,
            "h1_h2_exclusion_reasons": h1_h2_exclusions,
            "vector_activation_date": vector_activation_dt.isoformat() if vector_activation_dt else None,
            "model_activation_date": model_activation_dt.isoformat() if model_activation_dt else None,
        }

        # 10. Extract routing decisions
        decisions = (
            session.query(RoutingDecision)
            .filter_by(user_id=user.id)
            .order_by(RoutingDecision.created_at.asc(), RoutingDecision.id.asc())
            .all()
        )

        for dec in decisions:
            pid = str(dec.paper_id)
            cycle = cycle_map.get(dec.cycle_id)
            cycle_mode = cycle.mode if cycle else "unknown"
            cycle_date = cycle.as_of_date if cycle else None

            # Outage detection: cycle error or 0 discovered when mode is live
            is_outage = False
            if cycle and cycle.counts and isinstance(cycle.counts, dict):
                discovered = cycle.counts.get("discovered", 0)
                total_routed = cycle.counts.get(
                    "total_routed",
                    sum(v for v in cycle.counts.values() if isinstance(v, (int, float))),
                )
                if cycle.counts.get("error") or (discovered == 0 and total_routed == 0):
                    is_outage = True

            # Pre-onboarding detection
            is_pre_onboard = False
            if onboarding_ts and dec.created_at and dec.created_at < onboarding_ts:
                is_pre_onboard = True

            # Review details
            rt = review_by_decision_id.get(dec.id)
            review_decision = rt.decision if rt else None
            review_kind = rt.kind if rt else None
            review_queued_at = rt.queued_at.isoformat() if rt and rt.queued_at else None
            review_decided_at = rt.decided_at.isoformat() if rt and rt.decided_at else None

            # Digest details
            di = digest_by_decision_id.get(dec.id)
            in_digest = di is not None
            digest_pos = di.position if di else None
            digest_via = di.via if di else None

            # User reaction details
            fe_list = user_feedback_by_pid.get(pid, [])
            user_fe = [e for e in fe_list if e.source == "user"]
            user_reaction = int(user_fe[-1].label) if user_fe else None
            reaction_ts = user_fe[-1].created_at.isoformat() if user_fe else None

            # Effective label
            eff_tuple = effective_labels_dict.get(pid)
            eff_label = eff_tuple[0] if eff_tuple else None
            eff_source = eff_tuple[1] if eff_tuple else None

            # Summary details
            sum_rec = summary_by_decision_id.get(dec.id)
            has_summary = sum_rec is not None
            tokens_in = sum_rec.tokens_in if sum_rec else None
            tokens_out = sum_rec.tokens_out if sum_rec else None
            summary_model = sum_rec.llm_model if sum_rec else None
            summary_prompt = sum_rec.prompt_version if sum_rec else None

            # Vector & Model versions at time of routing
            v_ver = dec.vector_version.version if dec.vector_version else 0
            m_ver = dec.model_version.version if dec.model_version else None

            # Category
            data_cat = categorize_record(is_sim=is_sim, is_dev=is_dev, cycle_mode=cycle_mode)

            # Record exclusion tracking
            rec_exclusions = []
            if not meets_h1_h2:
                rec_exclusions.extend(h1_h2_exclusions)
            if is_outage:
                rec_exclusions.append("outage_day")
            if is_pre_onboard:
                rec_exclusions.append("pre_onboarding")
            if is_dev:
                rec_exclusions.append("developer_user (reported separately)")
            if is_sim:
                rec_exclusions.append("simulated_user (reported separately)")
            if cycle_mode == "replay":
                rec_exclusions.append("replay_mode (not live pilot)")

            record = {
                "decision_id": dec.id,
                "user_id": user.id,
                "user_name": user.name,
                "is_developer": is_dev,
                "is_simulated": is_sim,
                "data_category": data_cat,
                "paper_id": pid,
                "title": papers_map.get(pid, "No Title"),
                "cycle_id": dec.cycle_id,
                "cycle_date": cycle_date,
                "cycle_mode": cycle_mode,
                "cosine_score": dec.cosine_score,
                "clf_prob": dec.clf_prob,
                "route": dec.route,
                "status": dec.status,
                "reason": dec.reason,
                "method": dec.method,
                "t_high": dec.thresholds_used.get("t_high") if isinstance(dec.thresholds_used, dict) else None,
                "t_low": dec.thresholds_used.get("t_low") if isinstance(dec.thresholds_used, dict) else None,
                "in_review_queue": rt is not None,
                "review_decision": review_decision,
                "review_kind": review_kind,
                "review_queued_at": review_queued_at,
                "review_decided_at": review_decided_at,
                "delivered_in_digest": in_digest,
                "digest_position": digest_pos,
                "digest_via": digest_via,
                "user_reaction": user_reaction,
                "reaction_timestamp": reaction_ts,
                "effective_label": eff_label,
                "effective_source": eff_source,
                "has_summary": has_summary,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "summary_model": summary_model,
                "summary_prompt": summary_prompt,
                "vector_version": v_ver,
                "vector_activation_date": vector_activation_dt.isoformat() if vector_activation_dt else None,
                "model_version": m_ver,
                "model_activation_date": model_activation_dt.isoformat() if model_activation_dt else None,
                "is_outage": is_outage,
                "is_pre_onboard": is_pre_onboard,
                "excluded_from_h1_h2": not meets_h1_h2 or is_outage or is_pre_onboard or is_dev or is_sim,
                "exclusion_reasons": "; ".join(rec_exclusions) if rec_exclusions else "none",
                "created_at": dec.created_at.isoformat() if dec.created_at else None,
            }
            all_records.append(record)

    df_records = pd.DataFrame(all_records)
    records_csv = out_dir / "pilot_records.csv"
    clean_csv = out_dir / "pilot_records_clean.csv"
    dev_csv = out_dir / "developer_records.csv"
    audit_json = out_dir / "audit_log.json"
    legacy_audit_json = out_dir / "pilot_audit_log.json"
    summary_json = out_dir / "pilot_summary.json"

    # Deterministic sorting
    if not df_records.empty:
        df_records = df_records.sort_values(
            by=["user_id", "cycle_id", "decision_id"],
            na_position="first",
        ).reset_index(drop=True)
        df_records.to_csv(records_csv, index=False)

        # Clean pilot dataset (live pilot, non-developer, non-simulated, non-outage, non-pre-onboard)
        clean_mask = (
            (df_records["data_category"] == "LIVE_PILOT")
            & (~df_records["is_developer"])
            & (~df_records["is_simulated"])
            & (~df_records["is_outage"])
            & (~df_records["is_pre_onboard"])
        )
        df_clean = df_records[clean_mask].copy().reset_index(drop=True)
        df_clean.to_csv(clean_csv, index=False)

        # Developer records
        df_dev = df_records[df_records["is_developer"]].copy().reset_index(drop=True)
        df_dev.to_csv(dev_csv, index=False)
    else:
        # Write empty templates
        empty_cols = ["decision_id", "user_id", "paper_id", "route", "data_category", "created_at"]
        pd.DataFrame(columns=empty_cols).to_csv(records_csv, index=False)
        pd.DataFrame(columns=empty_cols).to_csv(clean_csv, index=False)
        pd.DataFrame(columns=empty_cols).to_csv(dev_csv, index=False)
        df_clean = pd.DataFrame()
        df_dev = pd.DataFrame()

    clean_count = len(df_clean)
    dev_count = len(df_dev)
    sim_count = int((df_records["is_simulated"]).sum()) if not df_records.empty else 0
    genuine_replay_count = int(
        ((df_records["data_category"] == "REPLAY") & (~df_records["is_developer"]) & (~df_records["is_simulated"])).sum()
    ) if not df_records.empty else 0
    outage_count = int(
        (df_records["is_outage"] & (~df_records["is_developer"]) & (~df_records["is_simulated"])).sum()
    ) if not df_records.empty else 0
    pre_onboard_count = int(
        (df_records["is_pre_onboard"] & (~df_records["is_developer"]) & (~df_records["is_simulated"])).sum()
    ) if not df_records.empty else 0

    sim_replay_count = int(
        (df_records["is_simulated"] | (df_records["cycle_mode"] == "replay")).sum()
    ) if not df_records.empty else 0
    excluded_count = int(df_records["excluded_from_h1_h2"].sum()) if not df_records.empty else 0

    partition_sum = clean_count + genuine_replay_count + dev_count + sim_count + outage_count + pre_onboard_count
    is_reconciled = bool(partition_sum == len(df_records))

    # User eligibility dictionary
    user_eligibility = {
        uid: {
            "name": u.get("name"),
            "is_developer": u.get("is_developer", False),
            "is_simulated": u.get("is_simulated", False),
            "total_cycles": u.get("total_cycles", 0),
            "active_days": u.get("active_days_count", 0),
            "effective_labels": u.get("effective_labels_count", 0),
            "h1_h2_eligible": u.get("meets_h1_h2_criteria", False),
            "h1_h2_exclusion_reason": "; ".join(u.get("h1_h2_exclusion_reasons", [])) if u.get("h1_h2_exclusion_reasons") else None,
        }
        for uid, u in user_audit.items()
    }

    # Full audit log
    audit_data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_records_exported": len(df_records),
        "clean_records_count": clean_count,
        "developer_records_count": dev_count,
        "user_audits": user_audit,
        "user_eligibility": user_eligibility,
        "simulated_users": [u.id for u in users if u.is_simulated],
        "developer_users": [u.id for u in users if is_developer_user(u)],
        "reconciliation": {
            "total_records_exported": len(df_records),
            "sum_of_mutually_exclusive_partitions": partition_sum,
            "is_reconciled": is_reconciled,
            "mutually_exclusive_partitions": {
                "clean_live_pilot_records": clean_count,
                "genuine_pilot_replay_records": genuine_replay_count,
                "developer_records": dev_count,
                "simulated_records": sim_count,
                "outage_records": outage_count,
                "pre_onboarding_records": pre_onboard_count,
            },
            "reconciliation_statement": (
                f"{len(df_records)} total records = {clean_count} clean live pilot + "
                f"{genuine_replay_count} genuine pilot replay + {dev_count} developer + "
                f"{sim_count} simulated (Exact partition match: {partition_sum}/{len(df_records)})."
            ),
        },
        "export_reconciliation": {
            "total_records_exported": len(df_records),
            "partition_sum": partition_sum,
            "is_reconciled": is_reconciled,
        },
        "overlapping_flag_counts": {
            "all_replay_records": int((df_records["cycle_mode"] == "replay").sum()) if not df_records.empty else 0,
            "all_live_records": int((df_records["cycle_mode"] == "live").sum()) if not df_records.empty else 0,
            "all_developer_records": dev_count,
            "all_simulated_records": sim_count,
        },
        "category_intersections": {
            "developer_replay_records": int((df_records["is_developer"] & (df_records["cycle_mode"] == "replay")).sum()) if not df_records.empty else 0,
            "developer_live_records": int((df_records["is_developer"] & (df_records["cycle_mode"] == "live")).sum()) if not df_records.empty else 0,
            "simulated_replay_records": int((df_records["is_simulated"] & (df_records["cycle_mode"] == "replay")).sum()) if not df_records.empty else 0,
            "simulated_live_records": int((df_records["is_simulated"] & (df_records["cycle_mode"] == "live")).sum()) if not df_records.empty else 0,
            "genuine_pilot_replay_records": genuine_replay_count,
            "genuine_pilot_live_records": clean_count,
        },
        "exclusions_summary": {
            "total_records": len(df_records),
            "clean_live_records": clean_count,
            "eligible_for_h1_h2": int((~df_records["excluded_from_h1_h2"]).sum()) if not df_records.empty else 0,
            "excluded_from_h1_h2": excluded_count,
            "developer_records": dev_count,
            "simulated_records": sim_count,
            "replay_records": int((df_records["cycle_mode"] == "replay").sum()) if not df_records.empty else 0,
        },
    }
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    with open(legacy_audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # Summary statistics
    summary_data = {
        "exported_at": audit_data["exported_at"],
        "total_records": len(df_records),
        "clean_records": clean_count,
        "unique_users": int(df_records["user_id"].nunique()) if not df_records.empty else 0,
        "unique_papers": int(df_records["paper_id"].nunique()) if not df_records.empty else 0,
        "category_breakdown": df_records["data_category"].value_counts().to_dict() if not df_records.empty else {},
        "route_breakdown": df_records["route"].value_counts().to_dict() if not df_records.empty else {},
        "digest_deliveries": int(df_records["delivered_in_digest"].sum()) if not df_records.empty else 0,
        "summaries_generated": int(df_records["has_summary"].sum()) if not df_records.empty else 0,
        "effective_labels_recorded": int(df_records["effective_label"].notna().sum()) if not df_records.empty else 0,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Exported %d records (%d clean, %d dev) to %s", len(df_records), clean_count, dev_count, out_dir)

    mutually_exclusive_partition = {
        "clean_live_records": clean_count,
        "genuine_replay_records": genuine_replay_count,
        "developer_records": dev_count,
        "simulated_records": sim_count,
        "partition_sum": partition_sum,
        "is_reconciled": is_reconciled,
    }
    overlapping_flags = {
        "clean_records_count": clean_count,
        "replay_decisions_count": int((df_records["cycle_mode"] == "replay").sum()) if not df_records.empty else 0,
        "developer_decisions_count": dev_count,
        "simulated_decisions_count": sim_count,
    }

    return PilotExportSummary(
        total_decisions_exported=len(df_records),
        clean_pilot_records_count=clean_count,
        excluded_decisions_count=excluded_count,
        developer_decisions_count=dev_count,
        simulated_replay_decisions_count=sim_replay_count,
        audit_log_path=str(audit_json),
        output_dir=str(out_dir),
        genuine_pilot_replay_count=genuine_replay_count,
        simulated_decisions_count=sim_count,
        is_reconciled=is_reconciled,
        user_eligibility=user_eligibility,
        mutually_exclusive_partition=mutually_exclusive_partition,
        overlapping_flags=overlapping_flags,
    )

