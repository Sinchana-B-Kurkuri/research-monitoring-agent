"""
record_pilot_survey.py
~~~~~~~~~~~~~~~~~~~~~~
CLI for managing end-of-pilot surveys and follow-up interviews (Phase 13 Requirement B).

Features:
- Generates reusable survey templates.
- Records participant responses on a 5-point Likert scale.
- Stores survey responses separately from model-training labels in data/surveys/.
- Computes aggregate survey statistics and tests Hypothesis H3.

Usage:
    python scripts/record_pilot_survey.py --template
    python scripts/record_pilot_survey.py --user-id user_alice --usefulness 4 --helpfulness 5 --triage-time 12.5 --feedback "Digest saved significant search time."
    python scripts/record_pilot_survey.py --list
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

from research_monitor.evaluation.survey import (
    DEFAULT_SURVEY_DIR,
    compute_survey_statistics,
    create_blank_survey_template,
    load_all_surveys,
    save_survey,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="End-of-Pilot Survey and Interview CLI (Phase 13)."
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Generate a blank survey template under data/surveys/template_survey.json.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Pilot participant user identifier.",
    )
    parser.add_argument(
        "--usefulness",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Perceived usefulness for triage (1-5 Likert scale).",
    )
    parser.add_argument(
        "--helpfulness",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Summary helpfulness (1-5 Likert scale).",
    )
    parser.add_argument(
        "--triage-time",
        type=float,
        help="Estimated time spent triaging papers (minutes/day).",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="",
        help="Free-text feedback about the monitoring workflow.",
    )
    parser.add_argument(
        "--interview-json",
        type=str,
        help="Optional path to a JSON file containing interview responses.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all recorded surveys and compute H3 statistics.",
    )
    parser.add_argument(
        "--survey-dir",
        type=str,
        default=str(DEFAULT_SURVEY_DIR),
        help="Directory where surveys are stored.",
    )

    args = parser.parse_args()
    survey_dir = Path(args.survey_dir)

    if args.template:
        t_path = create_blank_survey_template(survey_dir / "template_survey.json")
        print(f"Created blank survey template at: {t_path}")
        return

    if args.list:
        surveys = load_all_surveys(survey_dir)
        print(f"\nFound {len(surveys)} survey responses in {survey_dir}:")
        for s in surveys:
            uid = s.get("user_id")
            u = s.get("usefulness_score")
            h = s.get("summary_helpfulness_score")
            t = s.get("triage_time_minutes")
            print(f"  - User: {uid:15s} | Usefulness: {u}/5 | Helpfulness: {h}/5 | Triage Time: {t} min")

        stats = compute_survey_statistics(surveys)
        print("\nHypothesis H3 Test Results:")
        print(f"  Participants Responded:    {stats['n_responses']}")
        print(f"  Median Usefulness:         {stats['median_usefulness']} (Target >= 4.0)")
        print(f"  Median Helpfulness:        {stats['median_summary_helpfulness']} (Target >= 4.0)")
        print(f"  Mean Triage Time:          {stats['mean_triage_time_minutes']} min")
        print(f"  H3 Supported:              {stats['h3_supported']}")
        print(f"  Status:                    {stats['status']}")
        return

    if args.user_id:
        if args.usefulness is None or args.helpfulness is None:
            print("Error: Both --usefulness and --helpfulness (1-5) are required when recording a survey.")
            sys.exit(1)

        interview_data = {}
        if args.interview_json:
            with open(args.interview_json, "r", encoding="utf-8") as f:
                interview_data = json.load(f)

        survey_record = {
            "user_id": args.user_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "usefulness_score": args.usefulness,
            "summary_helpfulness_score": args.helpfulness,
            "triage_time_minutes": args.triage_time,
            "free_text_feedback": args.feedback,
            "interview": interview_data,
        }

        saved_path = save_survey(survey_record, survey_dir=survey_dir)
        print(f"Successfully recorded survey for user '{args.user_id}' at: {saved_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
