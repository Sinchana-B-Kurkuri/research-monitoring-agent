"""
label_papers.py
~~~~~~~~~~~~~~~
Interactive CLI tool for Phase 4 human ground-truth labeling.

Features:
  - Prompts evaluator for: [1] Relevant, [2] Somewhat relevant, [3] Not relevant.
  - Persists labels to database (if available) and exports timestamped CSV backups in `data/labels/`.
  - Supports resume: skips already labeled papers automatically.
  - Tracks targets: >= 100 relevant overall, >= 30 relevant in test split.
  - Shows progress, current stratum, title, and abstract.
  - Supports `--status` flag to inspect current progress.
  - Supports `--import-csv` flag to import pre-annotated CSV files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd

from research_monitor.database.session import SessionLocal, engine
from research_monitor.evaluation.labeling import (
    export_labels_to_csv,
    get_label_statistics,
    load_latest_labels,
    normalize_label,
    record_validation_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def get_db_session():
    """Attempt to get an active DB session, returning None if database is down."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return SessionLocal()
    except Exception:
        return None


def print_status_report(labels_df: pd.DataFrame) -> None:
    """Print current labeling statistics and target status."""
    stats = get_label_statistics(labels_df)
    print("\n" + "=" * 60)
    print("  PHASE 4 GROUND TRUTH LABELING STATUS")
    print("=" * 60)
    print(f"  Total papers labeled:        {stats['total_count']}")
    print(f"  Relevant:                    {stats['relevant_count']}")
    print(f"  Somewhat relevant:           {stats['somewhat_count']}")
    print(f"  Not relevant:                {stats['not_relevant_count']}")
    print(f"  Dev split labeled:           {stats['dev_count']}")
    print(f"  Test split labeled:          {stats['test_count']}")
    print(f"  Test relevant count:         {stats['test_relevant_count']}")
    print("-" * 60)
    print(f"  Target Overall (>=100 rel):  {'MET' if stats['target_overall_met'] else f'SHORTFALL: {stats['overall_shortfall']} needed'}")
    print(f"  Target Test (>=30 rel):      {'MET' if stats['target_test_met'] else f'SHORTFALL: {stats['test_shortfall']} needed'}")
    print("=" * 60 + "\n")


def import_csv_labels(csv_path: str | Path, sample_df: pd.DataFrame) -> None:
    """Import externally annotated CSV file into labels backup and DB."""
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV file %s does not exist.", path)
        return

    import_df = pd.read_csv(path)
    required_cols = {"paper_id", "label"}
    if not required_cols.issubset(import_df.columns):
        logger.error("Import CSV must contain at least 'paper_id' and 'label' columns.")
        return

    db_session = get_db_session()
    sample_lookup = sample_df.set_index("paper_id").to_dict(orient="index")

    # Load existing labels so we do not overwrite previously labeled papers
    existing_df = load_latest_labels()
    records_by_pid: dict[str, dict] = {}
    if existing_df is not None and not existing_df.empty:
        for r in existing_df.to_dict(orient="records"):
            records_by_pid[str(r["paper_id"])] = r

    imported_count = 0
    for _, row in import_df.iterrows():
        pid = str(row["paper_id"])
        if pid not in sample_lookup:
            logger.warning("Paper %s not found in evaluation sample; skipping.", pid)
            continue

        meta = sample_lookup[pid]
        stratum = row.get("sampling_stratum", row.get("stratum", meta.get("stratum", "S1")))
        weight = float(row.get("sampling_weight", meta.get("sampling_weight", 1.0)))
        split = row.get("split", meta.get("split", "dev"))
        notes = row.get("notes", None)
        user_id = str(row.get("user_id", "developer"))

        try:
            rec = record_validation_label(
                paper_id=pid,
                label=row["label"],
                stratum=stratum,
                sampling_weight=weight,
                split=split,
                notes=notes,
                session=db_session,
                user_id=user_id,
            )
            records_by_pid[pid] = rec
            imported_count += 1
        except Exception as e:
            logger.warning("Failed to import label for %s: %s", pid, e)

    if records_by_pid:
        merged_records = list(records_by_pid.values())
        export_labels_to_csv(merged_records)
        logger.info(
            "Successfully imported %d labels from %s (total accumulated: %d).",
            imported_count,
            path,
            len(merged_records),
        )
        print_status_report(pd.DataFrame(merged_records))
    else:
        logger.warning("No valid records imported.")

    if db_session:
        db_session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive human ground truth labeling tool for Phase 4."
    )
    parser.add_argument(
        "--sample-path",
        default="data/evaluation_sample.parquet",
        help="Path to evaluation sample parquet artifact.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current labeling status and targets, then exit.",
    )
    parser.add_argument(
        "--import-csv",
        type=str,
        default=None,
        help="Import ground truth labels from an external CSV file.",
    )
    args = parser.parse_args()

    # Load evaluation sample
    sample_path = Path(args.sample_path)
    if not sample_path.exists():
        csv_fallback = sample_path.with_suffix(".csv")
        if csv_fallback.exists():
            sample_df = pd.read_csv(csv_fallback)
        else:
            logger.error(
                "Evaluation sample %s not found. Run scripts/prepare_evaluation_sample.py first.",
                sample_path,
            )
            sys.exit(1)
    else:
        sample_df = pd.read_parquet(sample_path)

    # Load existing labels
    existing_labels_df = load_latest_labels()
    if existing_labels_df is None:
        existing_labels_df = pd.DataFrame(columns=[
            "paper_id", "label", "stratum", "sampling_weight", "split", "notes", "created_at"
        ])

    if args.status:
        print_status_report(existing_labels_df)
        return

    if args.import_csv:
        import_csv_labels(args.import_csv, sample_df)
        return

    labeled_pids = set(existing_labels_df["paper_id"].astype(str).tolist())
    unlabeled_df = sample_df[~sample_df["paper_id"].astype(str).isin(labeled_pids)].copy()

    print("\n" + "=" * 65)
    print("  RESEARCH MONITORING AGENT - PHASE 4 HUMAN LABELING CLI")
    print("=" * 65)
    print(f"  Total Evaluation Sample: {len(sample_df)}")
    print(f"  Already Labeled:         {len(labeled_pids)}")
    print(f"  Remaining to Label:      {len(unlabeled_df)}")
    print("  Options: [1] Relevant | [2] Somewhat relevant | [3] Not relevant")
    print("           [s] Skip     | [q] Quit and save")
    print("=" * 65)

    if len(unlabeled_df) == 0:
        print("\nAll 500 evaluation sample papers have already been labeled!")
        print_status_report(existing_labels_df)
        return

    db_session = get_db_session()
    current_records = existing_labels_df.to_dict(orient="records")

    try:
        for idx, (_, row) in enumerate(unlabeled_df.iterrows(), start=1):
            pid = str(row["paper_id"])
            title = row.get("title", "No Title")
            abstract = row.get("abstract", "No Abstract")
            stratum = row.get("stratum", "Unknown")
            split = row.get("split", "Unknown")
            weight = float(row.get("sampling_weight", 1.0))
            score = float(row.get("score", 0.0))
            cats = row.get("categories", "")

            print(f"\n[{idx}/{len(unlabeled_df)}] Paper ID: {pid}")
            print(f"Stratum: {stratum} | Split: {split} | Score: {score:.4f} | Weight: {weight:.2f}")
            print(f"Categories: {cats}")
            print(f"Title: {title}")
            print("-" * 65)
            # Wrap abstract lines
            print(f"Abstract: {abstract[:400]}..." if len(str(abstract)) > 400 else f"Abstract: {abstract}")
            print("-" * 65)

            while True:
                choice = input("Label ([1] Rel, [2] Some, [3] Not, [s] Skip, [q] Quit): ").strip()
                if choice.lower() == "q":
                    print("\nExiting labeling tool. Saving current progress...")
                    return
                elif choice.lower() == "s":
                    print(f"Skipped paper {pid}.")
                    break
                elif choice.lower() in {"1", "2", "3", "relevant", "somewhat", "not", "not_relevant", "not relevant"}:
                    try:
                        canonical = normalize_label(choice)
                        rec = record_validation_label(
                            paper_id=pid,
                            label=canonical,
                            stratum=stratum,
                            sampling_weight=weight,
                            split=split,
                            notes=None,
                            session=db_session,
                            user_id="developer",
                        )
                        current_records.append(rec)
                        # Save backup immediately after each label
                        export_labels_to_csv(current_records)
                        print(f"Recorded '{canonical}' for {pid}.")
                        break
                    except Exception as err:
                        print(f"Error recording label: {err}")
                else:
                    print("Invalid input. Enter 1, 2, 3, s, or q.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved.")
    finally:
        if db_session:
            db_session.close()
        latest_df = pd.DataFrame(current_records)
        print_status_report(latest_df)


if __name__ == "__main__":
    main()
