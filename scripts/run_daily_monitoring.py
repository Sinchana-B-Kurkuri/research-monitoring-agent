"""
run_daily_monitoring.py
~~~~~~~~~~~~~~~~~~~~~~~
Phase 9 Production Daily Monitoring Runner & Scheduler.

Executes autonomous daily arXiv monitoring cycles with:
  - Configurable conservative limits (max 50 papers ingested, 15 review tasks, 20 digest items).
  - Explicit cycle status tagging: [CYCLE_SUCCESS], [CYCLE_EMPTY], [CYCLE_FAILURE].
  - Multi-user support (--user-id specific or --user-id all).
  - Simple, lightweight daily scheduling (--once or --daemon --interval-hours 24).
  - Automatic digest compilation & summary caching for auto-cleared HIGH papers.
  - Zero heavyweight orchestration frameworks (no Celery, Redis, or Kubernetes).

Usage:
  python scripts/run_daily_monitoring.py --user-id developer --once
  python scripts/run_daily_monitoring.py --user-id all --once
  python scripts/run_daily_monitoring.py --user-id developer --daemon --interval-hours 24
"""

import argparse
from datetime import datetime, timedelta, timezone
import logging
import os
import signal
import sys
import time
from typing import Optional

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.config import config
from research_monitor.database.models import User
from research_monitor.database.session import SessionLocal, init_db
from research_monitor.ingestion.live import run_live_cycle
from research_monitor.review.digest import resume_after_review

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("daily_monitor")


def execute_daily_cycle_for_user(
    user: User,
    session,
    categories: list[str],
    max_results: int,
    days: int,
    max_digest_items: int = 20,
) -> tuple[str, Optional[int], dict]:
    """
    Execute a single daily monitoring cycle for a given user.

    Returns:
        tuple of (status_tag, cycle_id, metrics_dict)
        where status_tag is one of '[CYCLE_SUCCESS]', '[CYCLE_EMPTY]', '[CYCLE_FAILURE]'.
    """
    # 1. Compute date window
    now = datetime.now(timezone.utc)
    start_date = None
    end_date = now.strftime("%Y-%m-%d")
    if days > 0:
        start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    logger.info(
        "Initiating daily cycle for user '%s' (%s) [days=%d, max_results=%d]...",
        user.id,
        user.name,
        days,
        max_results,
    )

    try:
        # 2. Run live retrieval, deduplication, embedding, and relevance state machine
        state, ingestion = run_live_cycle(
            user_id=user.id,
            session=session,
            categories=categories,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date,
        )

        counts = state.counts
        metrics = {
            "fetched": ingestion.fetched,
            "inserted": ingestion.inserted,
            "skipped_duplicates": ingestion.skipped_duplicates,
            "embedded": ingestion.embedded,
            "discovered": counts.get("discovered", 0),
            "total_routed": counts.get("total_routed", 0),
            "high": counts.get("high", 0),
            "borderline": counts.get("borderline", 0),
            "low": counts.get("low", 0),
            "delivered": 0,
        }

        # 3. Determine status tag
        if ingestion.fetched == 0:
            status_tag = "[CYCLE_EMPTY]"
            logger.info(
                "%s 0 papers fetched from arXiv for user '%s' (weekend or lull). Cycle %d completed cleanly.",
                status_tag,
                user.id,
                state.cycle_id,
            )
        else:
            status_tag = "[CYCLE_SUCCESS]"
            logger.info(
                "%s Cycle %d completed for user '%s': %d fetched, %d new, %d high, %d borderline, %d low.",
                status_tag,
                state.cycle_id,
                user.id,
                ingestion.fetched,
                ingestion.inserted,
                counts.get("high", 0),
                counts.get("borderline", 0),
                counts.get("low", 0),
            )

        # 4. Automatically compile digest for auto-cleared HIGH papers & trigger summaries
        try:
            digest, _ = resume_after_review(
                user_id=user.id,
                cycle_id=state.cycle_id,
                session=session,
                max_digest_items=max_digest_items,
            )
            metrics["delivered"] = len(digest.items)
        except Exception as digest_err:
            logger.warning("Could not pre-build digest for cycle %d: %s", state.cycle_id, digest_err)

        # 5. Evaluate preference vector update for next cycle (Phase 10)
        try:
            from research_monitor.relevance.preference import update_user_interest_vector
            vec_res = update_user_interest_vector(user_id=user.id, session=session)
            if vec_res.status == "updated":
                logger.info(
                    "[VECTOR_UPDATE] Updated interest vector for '%s' to v%d (method='rocchio', t_high=%.4f, t_low=%.4f). Active for next cycle.",
                    user.id,
                    vec_res.version_created,
                    vec_res.t_high,
                    vec_res.t_low,
                )
                metrics["vector_updated"] = vec_res.version_created
            else:
                metrics["vector_status"] = vec_res.reason
        except Exception as vec_err:
            logger.warning("Could not evaluate vector update for user '%s': %s", user.id, vec_err)

        return status_tag, state.cycle_id, metrics

    except Exception as exc:
        status_tag = "[CYCLE_FAILURE]"
        logger.error("%s Daily monitoring cycle failed for user '%s': %s", status_tag, user.id, exc, exc_info=True)
        session.rollback()
        return status_tag, None, {"error": str(exc)}


def run_monitoring_round(args) -> int:
    """
    Execute one round of daily monitoring across target users.
    """
    categories_list = [c.strip() for c in args.categories.split(",") if c.strip()]
    days = args.days if args.days > 0 else 1  # Default to 1-day daily rolling window

    session = SessionLocal()
    overall_exit_code = 0

    try:
        # Resolve target users
        if args.user_id.lower() == "all":
            users = session.query(User).filter_by(is_simulated=False).all()
            if not users:
                logger.warning("No non-simulated users found in database.")
                users = session.query(User).all()
        else:
            target = session.query(User).filter_by(id=args.user_id).first()
            if not target:
                # Check first available
                first_u = session.query(User).first()
                if first_u:
                    logger.warning("User '%s' not found. Falling back to '%s'.", args.user_id, first_u.id)
                    users = [first_u]
                else:
                    logger.error("No users found in database. Run 'python scripts/create_user.py' first.")
                    return 1
            else:
                users = [target]

        print("\n" + "=" * 70)
        print("  DAILY RESEARCH MONITORING RUN (PHASE 9)")
        print(f"  Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
        print(f"  Target Users:    {', '.join(u.id for u in users)}")
        print(f"  Categories:      {args.categories}")
        print(f"  Ingestion Cap:   {args.max_results} papers")
        print(f"  Rolling Window:  {days} day(s)")
        print("=" * 70)

        for u in users:
            status_tag, cycle_id, metrics = execute_daily_cycle_for_user(
                user=u,
                session=session,
                categories=categories_list,
                max_results=args.max_results,
                days=days,
                max_digest_items=args.max_digest,
            )

            if status_tag == "[CYCLE_FAILURE]":
                overall_exit_code = 1

            print(f"\n--- User: {u.id} ({u.name}) ---")
            print(f"  Outcome:            {status_tag}")
            if cycle_id:
                print(f"  Cycle ID:           #{cycle_id}")
                print(f"  Papers Ingested:    {metrics.get('inserted', 0)} new (fetched {metrics.get('fetched', 0)}, skipped {metrics.get('skipped_duplicates', 0)} duplicates)")
                print(f"  Relevance Routing:  HIGH: {metrics.get('high', 0)} | BORDERLINE: {metrics.get('borderline', 0)} | LOW: {metrics.get('low', 0)}")
                print(f"  Delivered to Digest:{metrics.get('delivered', 0)} paper(s)")
                if metrics.get("vector_updated"):
                    print(f"  Preference Memory:  [UPDATED] Vector v{metrics['vector_updated']} (Rocchio centroid active for next cycle)")
                elif metrics.get("vector_status"):
                    print(f"  Preference Memory:  [STABLE] {metrics['vector_status']}")
                print(f"  Review Queue URL:   http://localhost:8000/review?token={u.access_token}")
                print(f"  Daily Digest URL:   http://localhost:8000/digest?token={u.access_token}")
            else:
                print(f"  Error Detail:       {metrics.get('error', 'Execution halted')}")

        print("\n" + "=" * 70 + "\n")
        return overall_exit_code

    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 9 Production Daily Monitoring Runner & Scheduler."
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("USER_ID", "all"),
        help="Target user ID or 'all' for all registered pilot users (default: 'all').",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=config.DAILY_INGESTION_MAX_PAPERS,
        help=f"Maximum papers to ingest per daily cycle (default: {config.DAILY_INGESTION_MAX_PAPERS}).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Rolling window in days (default: 1 for daily runs).",
    )
    parser.add_argument(
        "--categories",
        default=config.CORPUS_CATEGORIES,
        help=f"Comma-separated arXiv categories (default: '{config.CORPUS_CATEGORIES}').",
    )
    parser.add_argument(
        "--max-digest",
        type=int,
        default=config.DIGEST_MAX_ITEMS,
        help=f"Maximum items to deliver in daily digest (default: {config.DIGEST_MAX_ITEMS}).",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run in background daemon scheduling loop instead of one-shot execution.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=24.0,
        help="Interval in hours between daemon monitoring cycles (default: 24.0).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Explicit flag to run exactly once and exit (default behavior).",
    )

    args = parser.parse_args()

    # Ensure DB schema is ready
    try:
        init_db()
    except Exception as e:
        logger.warning("DB init check returned: %s. Proceeding with active session.", e)

    if not args.daemon:
        # Run one-shot cycle
        return run_monitoring_round(args)

    # Daemon scheduler mode
    logger.info(
        "Starting daily monitoring daemon (interval=%.1f hours). Press Ctrl+C to terminate.",
        args.interval_hours,
    )

    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received. Stopping daemon loop gracefully...")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    sleep_seconds = int(args.interval_hours * 3600)

    while running:
        try:
            logger.info("Triggering scheduled daily monitoring cycle...")
            run_monitoring_round(args)
        except Exception as e:
            logger.error("Daemon cycle encountered error: %s", e, exc_info=True)

        if not running:
            break

        logger.info("Next daily cycle scheduled in %.1f hours. Sleeping...", args.interval_hours)
        # Sleep in short increments to allow responsive shutdown
        slept = 0
        while running and slept < sleep_seconds:
            time.sleep(min(5, sleep_seconds - slept))
            slept += 5

    logger.info("Daily monitoring daemon stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
