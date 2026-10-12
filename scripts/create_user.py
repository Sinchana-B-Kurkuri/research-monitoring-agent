"""
create_user.py
~~~~~~~~~~~~~~
CLI command to create a user and initialize interest vector version 0 (Phase 5).

Usage:
  python scripts/create_user.py --user-id developer --name "Primary Developer"
  python scripts/create_user.py --user-id sim_user_1 --name "Simulated User 1" --is-simulated
"""

import argparse
import logging
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.config import config
from research_monitor.database.session import SessionLocal, init_db
from research_monitor.relevance.user_manager import create_user

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a user and initialize interest vector version 0 (Phase 5)."
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Unique string identifier for user (e.g. 'developer', 'user_01').",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Human display name for user.",
    )
    parser.add_argument(
        "--seed-text",
        default=None,
        help="Seed-interest paragraph. Defaults to config.SEED_INTEREST if not specified.",
    )
    parser.add_argument(
        "--is-simulated",
        action="store_true",
        help="Flag user as a simulated user (Section 8.1 / 8.6).",
    )
    args = parser.parse_args()

    seed_text = args.seed_text or config.SEED_INTEREST

    # Ensure DB schema is ready
    try:
        init_db()
    except Exception as e:
        logger.warning("DB init check returned: %s. Proceeding with active session.", e)

    session = SessionLocal()
    try:
        user = create_user(
            user_id=args.user_id,
            name=args.name,
            seed_text=seed_text,
            session=session,
            is_simulated=args.is_simulated,
        )

        v0 = user.vector_versions[0] if user.vector_versions else None

        print("\n" + "=" * 65)
        print("  USER CREATED & INITIALIZED (PHASE 5 / PHASE 9 COLD START)")
        print("=" * 65)
        print(f"  User ID:             {user.id}")
        print(f"  Name:                {user.name}")
        print(f"  Access Token:        {user.access_token}")
        print(f"  Simulated User:      {user.is_simulated}")
        print(f"  Created At (UTC):    {user.created_at}")
        if v0:
            print("-" * 65)
            print(f"  Vector Version:      {v0.version} (method='{v0.method}')")
            print(f"  P_high Level:        {v0.p_high_level:.1f}th percentile")
            print(f"  P_low Level:         {v0.p_low_level:.1f}th percentile")
            print(f"  Score Threshold t_h: {v0.t_high:.4f}")
            print(f"  Score Threshold t_l: {v0.t_low:.4f}")
        print("-" * 65)
        print("  PILOT USER PORTAL URLS:")
        print(f"  Review Queue:        http://localhost:8000/review?token={user.access_token}")
        print(f"  Research Digest:     http://localhost:8000/digest?token={user.access_token}")
        print(f"  Audit (Low papers):  http://localhost:8000/audit?token={user.access_token}")
        print("-" * 65)
        print(f"  Daily Run Command:   python scripts/run_daily_monitoring.py --user-id {user.id} --once")
        print("=" * 65 + "\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
