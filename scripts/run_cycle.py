"""
run_cycle.py
~~~~~~~~~~~~
CLI command to execute a monitoring cycle for a user (Phase 5).

Usage:
  python scripts/run_cycle.py --user-id developer --mode replay --limit 50
  python scripts/run_cycle.py --user-id developer --mode replay --as-of-date 2024-06-30
"""

import argparse
import logging
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.database.session import SessionLocal
from research_monitor.relevance.pipeline import run_monitoring_cycle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute a monitoring cycle through the Relevance Agent pipeline (Phase 5)."
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Target user ID.",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
        help="Cycle execution mode ('replay' or 'live'). Default: replay.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Historical cut-off date (ISO 'YYYY-MM-DD').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum candidate papers to process in this cycle.",
    )
    parser.add_argument(
        "--update-vector",
        action="store_true",
        help="Evaluate and update interest vector after cycle completion (Phase 10).",
    )
    args = parser.parse_args()

    session = SessionLocal()
    try:
        state = run_monitoring_cycle(
            user_id=args.user_id,
            session=session,
            mode=args.mode,
            as_of_date=args.as_of_date,
            limit=args.limit,
            auto_update_vector=args.update_vector,
        )

        counts = state.counts
        print("\n" + "=" * 65)
        print("  MONITORING CYCLE COMPLETED (PHASE 5 STATE MACHINE)")
        print("=" * 65)
        print(f"  Cycle ID:          {state.cycle_id}")
        print(f"  User ID:           {state.user_id} (simulated={state.is_simulated})")
        print(f"  Execution Mode:    {state.mode}")
        print(f"  As of Date:        {state.as_of_date}")
        print(f"  Active Vector v:   v{state.vector_version_num} (ID {state.vector_version_id})")
        print(f"  Threshold t_high:  {state.t_high:.4f} (P_high={state.p_high_level:.1f})")
        print(f"  Threshold t_low:   {state.t_low:.4f} (P_low={state.p_low_level:.1f})")
        print("-" * 65)
        print("  CYCLE DECISION SUMMARY:")
        print(f"  Discovered:        {counts.get('discovered', 0)}")
        print(f"  Preprocessed:      {counts.get('preprocessed', 0)}")
        print(f"  Embedded:          {counts.get('embedded', 0)}")
        print(f"  Scored:            {counts.get('scored', 0)}")
        print(f"  Total Routed:      {counts.get('total_routed', 0)}")
        print(f"    - HIGH:          {counts.get('high', 0)} (Status: ROUTED_HIGH)")
        print(f"    - BORDERLINE:    {counts.get('borderline', 0)} (Status: ROUTED_BORDERLINE)")
        print(f"    - LOW:           {counts.get('low', 0)} (Status: DISCARDED)")
        print("-" * 65)
        print(f"  Auditable Decisions Persisted: {len(state.routed)}")
        print("=" * 65 + "\n")

    finally:
        session.close()


if __name__ == "__main__":
    main()
