"""
run_phase8_live.py
~~~~~~~~~~~~~~~~~~
Headless CLI command to execute a live arXiv monitoring cycle (Phase 8).

Usage:
  python scripts/run_phase8_live.py --user-id developer
  python scripts/run_phase8_live.py --user-id developer --max-results 20
  python scripts/run_phase8_live.py --user-id developer --days 2 --categories cs.CL,cs.AI
"""

import argparse
from datetime import datetime, timedelta, timezone
import logging
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.config import config
from research_monitor.database.models import User
from research_monitor.database.session import SessionLocal
from research_monitor.ingestion.live import run_live_cycle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a live arXiv monitoring cycle through the Relevance Agent pipeline (Phase 8)."
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("USER_ID", "developer"),
        help="Target user ID (default: 'developer' or $USER_ID).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=config.LIVE_MAX_RESULTS,
        help=f"Maximum papers to retrieve from arXiv (default: {config.LIVE_MAX_RESULTS}).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=config.LIVE_DAYS,
        help="Fetch papers submitted in the last N days (default: 0 for all recent).",
    )
    parser.add_argument(
        "--categories",
        default=config.CORPUS_CATEGORIES,
        help=f"Comma-separated arXiv categories (default: '{config.CORPUS_CATEGORIES}').",
    )
    args = parser.parse_args()

    # Determine date filter if --days is provided
    start_date = None
    end_date = None
    if args.days > 0:
        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

    categories_list = [c.strip() for c in args.categories.split(",") if c.strip()]

    session = SessionLocal()
    try:
        user = session.query(User).filter_by(id=args.user_id).first()
        if not user:
            # Check if any user exists
            first_user = session.query(User).first()
            if first_user:
                logger.warning("User '%s' not found; defaulting to first available user '%s'.", args.user_id, first_user.id)
                user = first_user
            else:
                logger.error("No users found in database. Run 'python scripts/create_user.py' first.")
                return 1

        logger.info(
            "Starting live monitoring cycle for user '%s' (max_results=%d, days=%d, categories=%s)...",
            user.id,
            args.max_results,
            args.days,
            categories_list,
        )

        state, ingestion = run_live_cycle(
            user_id=user.id,
            session=session,
            categories=categories_list,
            max_results=args.max_results,
            start_date=start_date,
            end_date=end_date,
        )

        counts = state.counts
        print("\n" + "=" * 65)
        print("  LIVE MONITORING CYCLE COMPLETED (PHASE 8)")
        print("=" * 65)
        print("  INGESTION & DEDUPLICATION:")
        print(f"  Papers Fetched:       {ingestion.fetched}")
        print(f"  Duplicates in DB:     {ingestion.skipped_duplicates} (skipped without re-embedding)")
        print(f"  New Papers Inserted:  {ingestion.inserted}")
        print(f"  New Embeddings:       {ingestion.embedded}")
        if ingestion.failed_embeddings > 0:
            print(f"  Failed Embeddings:    {ingestion.failed_embeddings}")
        print("-" * 65)
        print("  RELEVANCE PIPELINE (MODE = LIVE):")
        print(f"  Cycle ID:             {state.cycle_id}")
        print(f"  User ID:              {state.user_id}")
        print(f"  Execution Mode:       {state.mode}")
        print(f"  Active Vector:        v{state.vector_version_num} (ID {state.vector_version_id})")
        print(f"  Thresholds:           t_high={state.t_high:.4f}, t_low={state.t_low:.4f}")
        print(f"  Discovered:           {counts.get('discovered', 0)}")
        print(f"  Total Routed:         {counts.get('total_routed', 0)}")
        print(f"    - HIGH:             {counts.get('high', 0)} (Status: ROUTED_HIGH)")
        print(f"    - BORDERLINE:       {counts.get('borderline', 0)} (Status: ROUTED_BORDERLINE)")
        print(f"    - LOW:              {counts.get('low', 0)} (Status: DISCARDED)")
        print("-" * 65)
        print(f"  Auditable Decisions Persisted: {len(state.routed)}")
        if ingestion.errors:
            print(f"  Ingestion Warnings/Errors:     {len(ingestion.errors)}")
        print("=" * 65 + "\n")

        return 0

    except Exception as exc:
        logger.error("Live cycle execution failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        return 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
