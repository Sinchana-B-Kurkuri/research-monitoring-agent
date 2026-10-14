"""
update_vector.py
~~~~~~~~~~~~~~~~
CLI command to inspect eligibility and update user interest vectors via Rocchio centroid (Phase 10).

Usage:
  python scripts/update_vector.py --user-id developer
  python scripts/update_vector.py --user-id developer --check-only
  python scripts/update_vector.py --user-id all
"""

import argparse
import logging
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.database.models import User
from research_monitor.database.session import SessionLocal, init_db
from research_monitor.relevance.preference import (
    check_vector_update_eligibility,
    update_user_interest_vector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("update_vector")


def process_user_vector_update(user: User, session, check_only: bool = False) -> None:
    print(f"\n[USER: {user.name} (id='{user.id}')]")

    if check_only:
        eligibility = check_vector_update_eligibility(user_id=user.id, session=session)
        print(f"  Current Version:    v{eligibility.current_version}")
        print(f"  Update Eligible:    {eligibility.eligible}")
        print(f"  Status / Reason:    {eligibility.reason}")
        print(f"  Effective Labels:   {eligibility.total_effective_labels} (Pos: {eligibility.positive_labels}, Neg: {eligibility.negative_labels})")
        print(f"  New Since Cutoff:   {eligibility.new_effective_labels_since_cutoff} (Cutoff event ID: {eligibility.cutoff_event_id})")
        return

    result = update_user_interest_vector(user_id=user.id, session=session)
    print(f"  Outcome Status:     {result.status.upper()}")
    print(f"  Eligible:           {result.eligible}")
    print(f"  Detail / Reason:    {result.reason}")
    print(f"  Effective Labels:   {result.n_effective_labels_used} used (Pos: {result.n_positive_labels_used}, Neg: {result.n_negative_labels_used})")

    if result.status == "updated":
        print(f"  Created Version:    v{result.version_created} (ID {result.vector_version_id}, method='rocchio')")
        print(f"  Cutoff Event ID:    {result.built_from_max_event_id}")
        print(f"  Percentile Levels:  P_high={result.p_high:.1f}, P_low={result.p_low:.1f}")
        print(f"  Recalculated Gate:  t_high={result.t_high:.4f}, t_low={result.t_low:.4f}")
        print(f"  Next Cycle State:   Active for next monitoring cycle ({result.is_active_for_next_cycle})")
    else:
        print("  Next Cycle State:   Retaining current active vector version")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 10 Rocchio Preference Memory: Interest Vector Updater."
    )
    parser.add_argument(
        "--user-id",
        default="all",
        help="Target user ID or 'all' for all non-simulated pilot users (default: 'all').",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check eligibility without applying updates.",
    )
    parser.add_argument(
        "--include-simulated",
        action="store_true",
        help="Include simulated users in update processing.",
    )
    args = parser.parse_args()

    try:
        init_db()
    except Exception as e:
        logger.warning("DB init check returned: %s. Proceeding with active session.", e)

    session = SessionLocal()
    try:
        if args.user_id.lower() == "all":
            query = session.query(User)
            if not args.include_simulated:
                query = query.filter_by(is_simulated=False)
            users = query.all()
        else:
            u = session.query(User).filter_by(id=args.user_id).first()
            users = [u] if u else []

        if not users:
            print(f"No matching users found for user-id='{args.user_id}'.")
            return 1

        mode_str = "ELIGIBILITY CHECK" if args.check_only else "VECTOR UPDATE"
        print("\n" + "=" * 68)
        print(f"  PREFERENCE MEMORY: ROCCHIO CENTROID ({mode_str})")
        print("=" * 68)

        for user in users:
            process_user_vector_update(user, session, check_only=args.check_only)

        print("\n" + "=" * 68 + "\n")
        return 0

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
