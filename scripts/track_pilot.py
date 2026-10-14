"""
track_pilot.py
~~~~~~~~~~~~~~
Phase 9 Pilot Progress & Consecutive Daily Cycle Tracker.

Inspects actual database state to report:
  - Registered pilot users and active vector versions.
  - Cycle history, outcomes ([CYCLE_SUCCESS], [CYCLE_EMPTY], [CYCLE_FAILURE]).
  - Consecutive daily streak toward the 7-day monitoring pilot objective.
  - Human review queue activity (pending, accepted, rejected, expired).
  - Digest delivery and user feedback reactions (likes, dislikes).

CRITICAL RULE:
  Never fabricates mock users, fake cycles, or simulated completions.
  Accurately reflects active database state at all times.

Usage:
  python scripts/track_pilot.py
  python scripts/track_pilot.py --user-id developer
  python scripts/track_pilot.py --export-json experiments/pilot_status.json
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
import sys
from typing import Optional

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.database.models import (
    Cycle,
    Digest,
    DigestItem,
    FeedbackEvent,
    ReviewTask,
    RoutingDecision,
    User,
)
from research_monitor.database.session import SessionLocal, init_db
from research_monitor.relevance.user_manager import get_active_vector_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("pilot_tracker")


def calculate_consecutive_streak(cycle_dates: list[date]) -> int:
    """
    Calculate the length of consecutive daily cycles from sorted unique dates.
    """
    if not cycle_dates:
        return 0

    unique_dates = sorted(set(cycle_dates))
    if not unique_dates:
        return 0

    # Walk backwards from the most recent cycle date
    streak = 1
    for i in range(len(unique_dates) - 1, 0, -1):
        diff = (unique_dates[i] - unique_dates[i - 1]).days
        if diff == 1:
            streak += 1
        else:
            break
    return streak


def inspect_user_pilot_progress(user: User, session) -> dict:
    """
    Query and summarize actual database activity for a single pilot user.
    """
    # 1. Vector version
    v_active = get_active_vector_version(user_id=user.id, session=session)

    # 2. Cycles
    cycles = (
        session.query(Cycle)
        .filter_by(user_id=user.id)
        .order_by(Cycle.started_at.asc(), Cycle.id.asc())
        .all()
    )

    cycle_dates: list[date] = []
    cycle_details = []
    for c in cycles:
        c_date = c.started_at.date() if c.started_at else None
        if not c_date and c.as_of_date:
            try:
                c_date = datetime.strptime(c.as_of_date, "%Y-%m-%d").date()
            except Exception:
                pass
        if c_date:
            cycle_dates.append(c_date)

        counts = c.counts or {}
        discovered = counts.get("discovered", 0)
        high = counts.get("high", 0)
        borderline = counts.get("borderline", 0)
        low = counts.get("low", 0)

        # Infer status
        if discovered == 0 and counts.get("total_routed", 0) == 0:
            status_tag = "[CYCLE_EMPTY]"
        else:
            status_tag = "[CYCLE_SUCCESS]"

        cycle_details.append({
            "cycle_id": c.id,
            "as_of_date": c.as_of_date,
            "mode": c.mode,
            "started_at": c.started_at.isoformat() if c.started_at else None,
            "finished_at": c.finished_at.isoformat() if c.finished_at else None,
            "discovered": discovered,
            "high": high,
            "borderline": borderline,
            "low": low,
            "status": status_tag,
        })

    streak = calculate_consecutive_streak(cycle_dates)

    # 3. Decisions breakdown
    total_decisions = session.query(RoutingDecision).filter_by(user_id=user.id).count()
    high_decisions = session.query(RoutingDecision).filter_by(user_id=user.id, route="HIGH").count()
    borderline_decisions = session.query(RoutingDecision).filter_by(user_id=user.id, route="BORDERLINE").count()
    low_decisions = session.query(RoutingDecision).filter_by(user_id=user.id, route="LOW").count()

    # 4. Review tasks
    tasks_query = (
        session.query(ReviewTask)
        .join(RoutingDecision, ReviewTask.decision_id == RoutingDecision.id)
        .filter(RoutingDecision.user_id == user.id)
    )
    pending_tasks = tasks_query.filter(ReviewTask.decision.is_(None)).count()
    accepted_tasks = tasks_query.filter(ReviewTask.decision == "accept").count()
    rejected_tasks = tasks_query.filter(ReviewTask.decision == "reject").count()
    expired_tasks = tasks_query.filter(ReviewTask.decision == "expired").count()

    # 5. Digests & Feedback
    digests_count = session.query(Digest).filter_by(user_id=user.id).count()
    delivered_items_count = (
        session.query(DigestItem)
        .join(Digest, DigestItem.digest_id == Digest.id)
        .filter(Digest.user_id == user.id)
        .count()
    )
    likes_count = (
        session.query(FeedbackEvent)
        .filter_by(user_id=user.id, source="user", label=1)
        .count()
    )
    dislikes_count = (
        session.query(FeedbackEvent)
        .filter_by(user_id=user.id, source="user", label=-1)
        .count()
    )

    return {
        "user_id": user.id,
        "name": user.name,
        "is_simulated": user.is_simulated,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "access_token": user.access_token,
        "active_vector": {
            "version": v_active.version if v_active else None,
            "method": v_active.method if v_active else None,
            "n_effective_labels": v_active.n_effective_labels if v_active else 0,
            "built_from_max_event_id": v_active.built_from_max_event_id if v_active else None,
            "p_high": v_active.p_high_level if v_active else None,
            "p_low": v_active.p_low_level if v_active else None,
            "t_high": v_active.t_high if v_active else None,
            "t_low": v_active.t_low if v_active else None,
        } if v_active else None,
        "streak_days": streak,
        "target_days": 7,
        "total_cycles": len(cycles),
        "cycle_details": cycle_details,
        "routing": {
            "total_decisions": total_decisions,
            "high": high_decisions,
            "borderline": borderline_decisions,
            "low": low_decisions,
        },
        "review_tasks": {
            "pending": pending_tasks,
            "accepted": accepted_tasks,
            "rejected": rejected_tasks,
            "expired": expired_tasks,
        },
        "digest_and_feedback": {
            "digests_created": digests_count,
            "items_delivered": delivered_items_count,
            "likes": likes_count,
            "dislikes": dislikes_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 9 Pilot Progress & Consecutive Daily Cycle Tracker."
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Filter tracking to a specific user ID.",
    )
    parser.add_argument(
        "--include-simulated",
        action="store_true",
        help="Include simulated users in tracker output (default: false).",
    )
    parser.add_argument(
        "--export-json",
        default=None,
        help="Path to export tracking summary in JSON format.",
    )
    args = parser.parse_args()

    # Ensure schema migrations are up to date
    try:
        init_db()
    except Exception as e:
        logger.warning("DB init check returned: %s. Proceeding with active session.", e)

    session = SessionLocal()
    try:
        query = session.query(User)
        if not args.include_simulated:
            query = query.filter_by(is_simulated=False)
        if args.user_id:
            query = query.filter_by(id=args.user_id)

        users = query.all()

        if not users:
            print("\n" + "=" * 65)
            print("  PILOT TRACKING AUDIT: NO USERS REGISTERED")
            print("=" * 65)
            print("  No matching users found in the active database.")
            print("  Run 'python scripts/create_user.py --user-id <id> --name <name>'")
            print("  or visit http://localhost:8000/onboard to register a pilot user.")
            print("=" * 65 + "\n")
            return 0

        print("\n" + "=" * 70)
        print("  PILOT PROGRESS & SEVEN-DAY CYCLE TRACKER (PHASE 9)")
        print(f"  Audit Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"  Pilot Users:     {len(users)}")
        print("=" * 70)

        results = []

        for u in users:
            data = inspect_user_pilot_progress(u, session)
            results.append(data)

            streak = data["streak_days"]
            target = data["target_days"]
            progress_bar = "[" + "#" * streak + "-" * max(0, target - streak) + "]"

            print(f"\n[USER: {u.name} (id='{u.id}')]")
            token_display = u.access_token or "(None - visit /onboard to initialize token)"
            print(f"  Access Token:       {token_display}")
            if data["active_vector"]:
                v = data["active_vector"]
                method_info = f" (method='{v['method']}', labels={v['n_effective_labels']})" if v.get("method") else ""
                print(f"  Active Vector:      v{v['version']}{method_info} (t_high={v['t_high']:.4f}, t_low={v['t_low']:.4f})")
            print(f"  Daily Streak:       {progress_bar} {streak} / {target} consecutive days")
            print(f"  Total Cycles Run:   {data['total_cycles']}")
            
            rt = data["review_tasks"]
            print(f"  Review Queue Tasks: {rt['pending']} pending | {rt['accepted']} accepted | {rt['rejected']} rejected | {rt['expired']} expired")

            df = data["digest_and_feedback"]
            print(f"  Digest Delivery:    {df['digests_created']} digest(s) | {df['items_delivered']} delivered papers | Likes: {df['likes']} | Dislikes: {df['dislikes']}")

            # Cycle history table
            cycles = data["cycle_details"]
            if cycles:
                print("\n  Cycle History (Most Recent First):")
                print(f"    {'Cycle':<7} {'Date':<12} {'Discovered':<12} {'High':<6} {'Border':<8} {'Low':<6} {'Status'}")
                print("    " + "-" * 62)
                for c in reversed(cycles[-10:]):  # Show up to 10 most recent
                    print(f"    #{c['cycle_id']:<6} {str(c['as_of_date']):<12} {c['discovered']:<12} {c['high']:<6} {c['borderline']:<8} {c['low']:<6} {c['status']}")
            else:
                print("\n  Cycle History:      0 cycles run yet.")
                print(f"  Next Action:        Run Day 1 cycle:")
                print(f"                      python scripts/run_daily_monitoring.py --user-id {u.id} --once")

            print("\n  Portal URLs:")
            if u.access_token:
                print(f"    Review: http://localhost:8000/review?token={u.access_token}")
                print(f"    Digest: http://localhost:8000/digest?token={u.access_token}")
            else:
                print(f"    Portal: http://localhost:8000/onboard (register new token)")

        print("\n" + "=" * 70 + "\n")

        if args.export_json:
            export_path = os.path.abspath(args.export_json)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Pilot progress exported to: {export_path}\n")

        return 0

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
