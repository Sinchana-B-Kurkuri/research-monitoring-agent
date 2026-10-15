"""
run_phase13_evaluation.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Master CLI Runner for Phase 13: Pilot Completion and Final Evaluation.

Coordinates:
  1. Pre-flight pilot readiness audit and honesty safeguards.
  2. Reproducible pilot dataset export applying Section 8.6 exclusions.
  3. Final E1 Gate Quality on the frozen test split.
  4. Final E7 Pilot evaluation (Hypotheses H1-H3, paired comparisons, survey stats).
  5. Before-and-after adaptation analysis (vector activation & model promotion).
  6. Operational efficiency (LLM-call reduction, tokens/costs, queue workload).
  7. Generation of reproducible evaluation tables, reports, and PNG figures.

Usage:
    python scripts/run_phase13_evaluation.py
    python scripts/run_phase13_evaluation.py --figures
    python scripts/run_phase13_evaluation.py --results-dir experiments/results
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

# Ensure src and root are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from research_monitor.evaluation.final_evaluation import run_final_evaluation_suite
from scripts.generate_figures import regenerate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def print_dashboard(report: dict):
    """Pretty prints evaluation status and safeguards dashboard to stdout."""
    readiness = report.get("readiness", {})
    tier = readiness.get("tier", "UNKNOWN")
    can_freeze = report.get("numbers_frozen", False)
    streak = readiness.get("consecutive_calendar_days", 0)
    users_n = readiness.get("active_users_count", 0)
    genuine_users_n = readiness.get("genuine_pilot_participants_count", users_n)
    total_users_n = readiness.get("total_users_with_records", users_n)
    h1_h2_eligible_n = readiness.get("h1_h2_eligible_users_count", 0)
    total_cycles = readiness.get("total_recorded_cycles", 0)
    genuine_cycles = readiness.get("genuine_recorded_cycles", 0)
    live_cycles = readiness.get("genuine_live_cycles", 0)
    distinct_days = readiness.get("distinct_active_calendar_days", 0)
    dev_labels = readiness.get("validation_labels_dev", 0)
    test_labels = readiness.get("validation_labels_test", 0)
    surveys_n = readiness.get("surveys_count", 0)
    missing = readiness.get("missing_prerequisites", [])

    print("\n" + "=" * 78)
    print("   PHASE 13: PILOT COMPLETION & FINAL EVALUATION DASHBOARD")
    print(f"   Execution Timestamp: {report.get('generated_at')}")
    print("=" * 78)

    print(f"\n[1] READINESS & COMPLETION SAFEGUARDS")
    print(f"    Current Readiness Tier:      {tier}")
    print(f"    Final Numbers Frozen:        {'YES (Frozen)' if can_freeze else 'NO (Safeguard Active - Held)'}")
    print(f"    Total Users with Records:    {total_users_n} (Genuine: {genuine_users_n}, H1/H2 Eligible: {h1_h2_eligible_n})")
    print(f"    Recorded Cycles:             {total_cycles} total ({genuine_cycles} genuine: {live_cycles} live, {genuine_cycles - live_cycles} replay)")
    print(f"    Calendar Days Active:        {distinct_days} distinct active days")
    print(f"    Live Consecutive Streak:     {streak} / 7 required live days")
    print(f"    Validation Labels (Dev):     {dev_labels} / 70 target")
    print(f"    Validation Labels (Test):    {test_labels} / 30 target (Frozen Split)")
    print(f"    Survey Responses Filed:      {surveys_n} / {max(1, genuine_users_n)} genuine participants")

    if missing:
        print("\n    >>> MISSING PREREQUISITES FOR FINAL FREEZE <<<")
        for m in missing:
            print(f"      * {m}")

    print(f"\n[2] E1: GATE QUALITY (Frozen Test Split)")
    e1 = report.get("e1_gate_quality", {})
    print(f"    Status:                      {e1.get('status', 'NOT RUN')}")
    if e1.get("status") == "COMPLETED":
        print(f"    Test Set Overlap:            {e1.get('labeled_overlap')} papers")
        print(f"    Output Table:                {e1.get('output_csv')}")
    elif e1.get("message"):
        print(f"    Notice:                      {e1.get('message')}")

    print(f"\n[3] E7: PILOT EVALUATION (Hypotheses H1-H3)")
    e7 = report.get("e7_pilot_evaluation", {})
    hypotheses = e7.get("hypotheses", {})

    h1 = hypotheses.get("h1") or {}
    print(f"    H1 (Digest Like Rate):       {h1.get('status', 'N/A')} "
          f"({h1.get('users_supporting_h1', 0)}/{h1.get('eligible_pilot_users_tested', 0)} users)")

    h2 = hypotheses.get("h2") or {}
    print(f"    H2 (Review Workload):        {h2.get('status', 'N/A')} "
          f"({h2.get('users_supporting_h2', 0)}/{h2.get('eligible_pilot_users_tested', 0)} users)")

    h3 = hypotheses.get("h3") or {}
    print(f"    H3 (Usefulness & Summary):   {h3.get('status', 'N/A')} "
          f"(Median Useful: {h3.get('median_usefulness', 'N/A')}/5, "
          f"Helpful: {h3.get('median_summary_helpfulness', 'N/A')}/5)")

    print(f"\n[4] BEFORE-AND-AFTER ADAPTATION ANALYSIS")
    adapt = report.get("adaptation_analysis", {})
    print(f"    Status:                      {adapt.get('status', 'N/A')}")
    print(f"    Participants Analyzed:       {adapt.get('users_analyzed', 0)}")
    if adapt.get("output_csv"):
        print(f"    Output Table:                {adapt.get('output_csv')}")

    print(f"\n[5] OPERATIONAL EFFICIENCY & LLM WORKLOAD")
    eff = report.get("efficiency_analysis", {})
    meas_status = eff.get("measured_savings_status", "MEASURED")
    meas_red = eff.get("measured_llm_call_reduction")
    theo_red = eff.get("theoretical_call_reduction")
    if meas_status == "NOT_MEASURABLE" or meas_red is None:
        print(f"    Measured LLM Savings:        NOT_MEASURABLE (0 summaries generated & no token telemetry)")
        if theo_red is not None:
            print(f"    Theoretical Call Reduction:  {theo_red * 100:.1f}% (Counterfactual comparison only)")
    else:
        print(f"    Measured LLM-Call Reduction: {meas_red * 100:.1f}%")
        if theo_red is not None:
            print(f"    Theoretical Call Reduction:  {theo_red * 100:.1f}%")
    print(f"    Token Telemetry Present:     {'YES' if eff.get('has_token_telemetry') else 'NO (Missing Telemetry Recorded)'}")
    if eff.get("actual_cost_usd") is not None:
        print(f"    Actual LLM Cost:             ${eff.get('actual_cost_usd'):.4f}")
    if eff.get("estimated_cost_savings_usd") is not None:
        print(f"    Estimated Savings:           ${eff.get('estimated_cost_savings_usd'):.4f}")
    if eff.get("notes"):
        for note in eff.get("notes"):
            print(f"      * {note}")

    print(f"\n[6] DELIVERABLE ARTIFACTS GENERATED")
    print(f"    Master Report:               experiments/results/final_evaluation_report.json")
    print(f"    Readiness Audit:             experiments/results/readiness_report.json")
    print(f"    Pilot Telemetry Export:      data/pilot_export/pilot_records_clean.csv")
    print(f"    Audit Log:                   data/pilot_export/audit_log.json")
    print(f"    E7 Pilot Table:              experiments/results/final_e7_pilot.csv")
    print(f"    Before/After Table:          experiments/results/final_pilot_before_after.csv")
    print(f"    Efficiency Table:            experiments/results/final_efficiency.csv")

    print("\n" + "=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 13 Final Evaluation Suite and Deliverables."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory to save final tables, reports, and figures.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="data/pilot_export",
        help="Directory to export cleaned pilot datasets and audit log.",
    )
    parser.add_argument(
        "--survey-dir",
        type=str,
        default="data/surveys",
        help="Directory where participant surveys are located.",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Regenerate all publication-ready PNG figures.",
    )
    parser.add_argument(
        "--skip-e1",
        action="store_true",
        help="Skip E1 Gate Quality offline calculation.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    export_dir = Path(args.export_dir)
    survey_dir = Path(args.survey_dir)

    print("\nExecuting Phase 13 Final Evaluation Suite...")
    report = run_final_evaluation_suite(
        pilot_export_dir=export_dir,
        survey_dir=survey_dir,
        results_dir=results_dir,
        run_offline_e1=not args.skip_e1,
    )

    if args.figures:
        print("\nRegenerating publication PNG figures...")
        figs = regenerate_all_figures()
        print(f"Generated {len(figs)} figures under {results_dir}")

    print_dashboard(report)


if __name__ == "__main__":
    main()
