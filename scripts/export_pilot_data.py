"""
export_pilot_data.py
~~~~~~~~~~~~~~~~~~~~
CLI command to export pilot data from PostgreSQL and produce an analysis-ready dataset (Phase 13 Requirement A).

Features:
- Applies Section 8.6 exclusion rules (outage cycles, pre-onboarding decisions, <5 days or <30 labels).
- Reports developer-user data separately.
- Keeps live pilot, replay, and simulated data distinct.
- Generates a full audit log without overwriting raw database data.
- Deterministic and safe to rerun.

Usage:
    python scripts/export_pilot_data.py
    python scripts/export_pilot_data.py --output-dir data/pilot_export
    python scripts/export_pilot_data.py --user-id phase5_verify
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

# Ensure src is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from research_monitor.database.session import get_db_session
from research_monitor.evaluation.pilot_export import export_pilot_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export and clean pilot telemetry for Phase 13 evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pilot_export",
        help="Target directory for exported CSVs and audit log.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Optional specific user to export.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    print("\n" + "=" * 70)
    print("  PHASE 13: PILOT DATA EXPORT")
    print(f"  Target Directory: {out_dir}")
    print(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    try:
        with get_db_session() as session:
            summary = export_pilot_dataset(
                session=session,
                export_dir=out_dir,
                user_id=args.user_id,
            )

        print("\nExport Summary:")
        print(f"  Total Decisions Exported:      {summary.total_decisions_exported}")
        print(f"  Clean Pilot Records:           {summary.clean_pilot_records_count}")
        print(f"  Excluded Decisions:            {summary.excluded_decisions_count}")
        print(f"  Developer Decisions (separate):{summary.developer_decisions_count}")
        print(f"  Simulated/Replay Decisions:    {summary.simulated_replay_decisions_count}")
        print(f"  Audit Log:                     {summary.audit_log_path}")

        print("\nExport Completed Successfully.")
    except Exception as e:
        logger.exception("Failed to export pilot dataset: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
