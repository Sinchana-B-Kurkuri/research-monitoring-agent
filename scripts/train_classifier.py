"""
train_classifier.py
~~~~~~~~~~~~~~~~~~~
CLI command to inspect eligibility, train, evaluate, and conditionally promote
the logistic regression preference classifier (Phase 11).

Usage:
  python scripts/train_classifier.py --user-id developer
  python scripts/train_classifier.py --user-id developer --check-only
  python scripts/train_classifier.py --user-id all
"""

import argparse
import logging
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from research_monitor.database.models import ModelVersion, User
from research_monitor.database.session import SessionLocal, init_db
from research_monitor.relevance.classifier import (
    check_classifier_eligibility,
    train_and_evaluate_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("train_classifier")


def process_user_classifier_training(user: User, session, check_only: bool = False) -> None:
    print(f"\n[USER: {user.name} (id='{user.id}')]")

    # Check active model version
    active_model = (
        session.query(ModelVersion)
        .filter_by(user_id=user.id, is_active=True)
        .order_by(ModelVersion.version.desc())
        .first()
    )
    if active_model:
        print(f"  Current Method:     CLASSIFIER (v{active_model.version}, p_high={active_model.p_high:.4f}, p_low={active_model.p_low:.4f})")
    else:
        print("  Current Method:     THRESHOLD (cold-start / Rocchio centroid)")

    if check_only:
        eligibility = check_classifier_eligibility(user_id=user.id, session=session)
        print(f"  Training Eligible:  {eligibility.eligible}")
        print(f"  Status / Reason:    {eligibility.reason}")
        print(f"  Effective Labels:   {eligibility.total_effective_labels} (Pos: {eligibility.positive_labels}, Neg: {eligibility.negative_labels})")
        print(f"  New Since Cutoff:   {eligibility.new_effective_labels_since_cutoff} (Cutoff event ID: {eligibility.cutoff_event_id})")
        return

    result = train_and_evaluate_classifier(user_id=user.id, session=session)
    print(f"  Outcome Status:     {result.status.upper()}")
    print(f"  Eligible:           {result.eligible}")
    print(f"  Detail / Reason:    {result.reason}")

    if result.status in ("promoted", "not_promoted"):
        print(f"  Model Version:      v{result.version_created} (ID {result.model_version_id})")
        print(f"  Training Samples:   {result.n_samples} total ({result.n_pos} pos, {result.n_neg} neg)")
        print(f"  Best Regularizer C: {result.best_C}")
        print(f"  OOF ROC-AUC:        {result.oof_auc:.4f}")
        print(f"  Baseline ROC-AUC:   {result.oof_auc_cosine_baseline:.4f} (Margin: {result.oof_auc - result.oof_auc_cosine_baseline:+.4f})")
        print(f"  Promoted to Active: {result.promoted}")
        print(f"  Thresholds:         p_high={result.p_high:.4f}, p_low={result.p_low:.4f} (source='{result.threshold_calibration_source}')")
        print(f"  Cutoff Event ID:    {result.trained_on_max_event_id}")
        if result.is_active:
            print("  Next Cycle State:   Active routing model switched to CLASSIFIER")
        else:
            print("  Next Cycle State:   Retaining previous routing method")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11 Preference Memory: Logistic Regression Classifier Trainer."
    )
    parser.add_argument(
        "--user-id",
        default="all",
        help="Target user ID or 'all' for all non-simulated pilot users (default: 'all').",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check eligibility without training.",
    )
    parser.add_argument(
        "--include-simulated",
        action="store_true",
        help="Include simulated users in processing.",
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

        print("=" * 65)
        print("  PHASE 11 PREFERENCE CLASSIFIER TRAINING / EVALUATION")
        print("=" * 65)
        for u in users:
            process_user_classifier_training(u, session, check_only=args.check_only)
        print("\n" + "=" * 65 + "\n")
        return 0

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
