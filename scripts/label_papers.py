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

from research_monitor.database.models import ValidationLabel
from research_monitor.database.session import SessionLocal, engine, init_db
from research_monitor.evaluation.labeling import (
    export_labels_to_csv,
    get_label_statistics,
    get_labeled_paper_ids,
    load_latest_labels,
    normalize_label,
    normalize_paper_id,
    reconcile_csv_and_db_labels,
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
        init_db()
        return SessionLocal()
    except Exception as e:
        logger.warning("Database unavailable (%s); operating in CSV-only mode.", e)
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


def handle_relabel(paper_id: str, sample_df: pd.DataFrame, session: Any, user_id: str = "developer") -> None:
    """Deliberately relabel a specific paper ID."""
    norm_pid = normalize_paper_id(paper_id)
    matches = sample_df[sample_df["paper_id"].astype(str).apply(normalize_paper_id) == norm_pid]
    if matches.empty:
        print(f"Paper '{paper_id}' not found in evaluation sample.")
        return
    row = matches.iloc[0]
    title = row.get("title", "No Title")
    abstract = row.get("abstract", "No Abstract")
    stratum = row.get("stratum", "S1")
    split = row.get("split", "dev")
    weight = float(row.get("sampling_weight", 1.0))
    score = float(row.get("score", 0.0))

    existing_lbl = None
    if session is not None:
        db_rec = (
            session.query(ValidationLabel)
            .filter(ValidationLabel.user_id == user_id, ValidationLabel.paper_id == norm_pid)
            .first()
        )
        if db_rec:
            existing_lbl = db_rec.label

    print("\n" + "=" * 65)
    print(f"  RELABEL PAPER: {norm_pid} (User: {user_id})")
    print("=" * 65)
    print(f"Title: {title}")
    print(f"Stratum: {stratum} | Split: {split} | Current Label: {existing_lbl or 'None'}")
    print(f"Abstract: {abstract[:400]}..." if len(str(abstract)) > 400 else f"Abstract: {abstract}")
    print("-" * 65)

    choice = input("Enter new label ([1] Rel, [2] Some, [3] Not, [c] Cancel): ").strip()
    if choice.lower() in {"c", "cancel"}:
        print("Relabel cancelled.")
        return
    try:
        canonical = normalize_label(choice)
        rec = record_validation_label(
            paper_id=norm_pid,
            label=canonical,
            stratum=stratum,
            sampling_weight=weight,
            split=split,
            notes="Deliberately relabeled",
            session=session,
            user_id=user_id,
        )
        existing_df = load_latest_labels()
        records_map = {}
        if existing_df is not None:
            for r in existing_df.to_dict(orient="records"):
                records_map[(r.get("user_id", "developer"), normalize_paper_id(r["paper_id"]))] = r
        records_map[(user_id, norm_pid)] = rec
        export_labels_to_csv(list(records_map.values()))
        print(f"Successfully relabeled {norm_pid} as '{canonical}'.")
    except ValueError as e:
        print(f"Error: {e}")


def import_csv_labels(csv_path: str | Path, sample_df: pd.DataFrame, user_id: str = "developer") -> None:
    """Import externally annotated CSV file into labels backup and DB."""
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV file %s does not exist.", path)
        return

    import_df = pd.read_csv(path, dtype={"paper_id": str, "user_id": str})
    required_cols = {"paper_id", "label"}
    if not required_cols.issubset(import_df.columns):
        logger.error("Import CSV must contain at least 'paper_id' and 'label' columns.")
        return

    db_session = get_db_session()
    sample_df["_norm_pid"] = sample_df["paper_id"].astype(str).apply(normalize_paper_id)
    sample_lookup = sample_df.set_index("_norm_pid").to_dict(orient="index")

    existing_df = load_latest_labels()
    records_by_key: dict[tuple[str, str], dict] = {}
    if existing_df is not None and not existing_df.empty:
        for r in existing_df.to_dict(orient="records"):
            u = str(r.get("user_id") or "developer").strip()
            p = normalize_paper_id(r["paper_id"])
            records_by_key[(u, p)] = r

    imported_count = 0
    for _, row in import_df.iterrows():
        raw_pid = str(row["paper_id"])
        pid = normalize_paper_id(raw_pid)
        if pid not in sample_lookup:
            logger.warning("Paper %s not found in evaluation sample; skipping.", raw_pid)
            continue

        meta = sample_lookup[pid]
        stratum = row.get("sampling_stratum", row.get("stratum", meta.get("stratum", "S1")))
        weight = float(row.get("sampling_weight", meta.get("sampling_weight", 1.0)))
        split = row.get("split", meta.get("split", "dev"))
        notes = row.get("notes", None)
        rec_user = str(row.get("user_id") or user_id).strip()

        try:
            rec = record_validation_label(
                paper_id=pid,
                label=row["label"],
                stratum=stratum,
                sampling_weight=weight,
                split=split,
                notes=notes,
                session=db_session,
                user_id=rec_user,
            )
            records_by_key[(rec_user, pid)] = rec
            imported_count += 1
        except Exception as e:
            logger.warning("Failed to import label for %s: %s", pid, e)

    if records_by_key:
        merged_records = list(records_by_key.values())
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
    parser.add_argument(
        "--user-id",
        type=str,
        default="developer",
        help="Evaluator user ID (default: developer).",
    )
    parser.add_argument(
        "--relabel",
        type=str,
        default=None,
        help="Paper ID to deliberately relabel.",
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

    sample_df["paper_id"] = sample_df["paper_id"].astype(str).str.strip()

    db_session = get_db_session()

    # Safely reconcile CSV records with DB if available
    if db_session is not None:
        reconcile_csv_and_db_labels(session=db_session, user_id=None, sample_df=sample_df)

    # Load latest labels (deduplicated by (user_id, paper_id))
    existing_labels_df = load_latest_labels()
    if existing_labels_df is None:
        existing_labels_df = pd.DataFrame(columns=[
            "user_id", "paper_id", "label", "sampling_stratum", "stratum",
            "sampling_weight", "split", "notes", "labelled_at", "created_at"
        ])

    if args.status:
        user_labels = (
            existing_labels_df[existing_labels_df["user_id"] == args.user_id]
            if "user_id" in existing_labels_df.columns
            else existing_labels_df
        )
        print_status_report(user_labels)
        if db_session:
            db_session.close()
        return

    if args.import_csv:
        import_csv_labels(args.import_csv, sample_df, user_id=args.user_id)
        if db_session:
            db_session.close()
        return

    if args.relabel:
        handle_relabel(args.relabel, sample_df, db_session, user_id=args.user_id)
        if db_session:
            db_session.close()
        return

    # Determine already-labeled papers for this user
    labeled_pids = get_labeled_paper_ids(session=db_session, user_id=args.user_id)
    sample_df["_norm_pid"] = sample_df["paper_id"].apply(normalize_paper_id)
    unlabeled_df = sample_df[~sample_df["_norm_pid"].isin(labeled_pids)].copy()
    sample_df.drop(columns=["_norm_pid"], inplace=True)
    unlabeled_df.drop(columns=["_norm_pid"], inplace=True)

    print("\n" + "=" * 65)
    print("  RESEARCH MONITORING AGENT - PHASE 4 HUMAN LABELING CLI")
    print("=" * 65)
    print(f"  Evaluator User:          {args.user_id}")
    print(f"  Total Evaluation Sample: {len(sample_df)}")
    print(f"  Already Labeled:         {len(labeled_pids)}")
    print(f"  Remaining to Label:      {len(unlabeled_df)}")
    print("  Options: [1] Relevant | [2] Somewhat relevant | [3] Not relevant")
    print("           [s] Skip     | [q] Quit and save")
    print("=" * 65)

    if len(unlabeled_df) == 0:
        print(f"\nAll {len(sample_df)} evaluation sample papers have already been labeled for user '{args.user_id}'.")
        print("No unlabeled papers remain in the validation queue.")
        user_labels = (
            existing_labels_df[existing_labels_df["user_id"] == args.user_id]
            if "user_id" in existing_labels_df.columns
            else existing_labels_df
        )
        print_status_report(user_labels)
        if db_session:
            db_session.close()
        return

    # Track presented papers to ensure no paper is presented twice in one session
    session_presented = set(labeled_pids)
    current_records_map: dict[tuple[str, str], dict] = {}
    if not existing_labels_df.empty:
        for r in existing_labels_df.to_dict(orient="records"):
            u = str(r.get("user_id") or "developer").strip()
            p = normalize_paper_id(r["paper_id"])
            current_records_map[(u, p)] = r

    try:
        for idx, (_, row) in enumerate(unlabeled_df.iterrows(), start=1):
            pid = normalize_paper_id(row["paper_id"])
            if pid in session_presented:
                continue

            # Double check database before presenting each paper
            if db_session is not None:
                existing_in_db = (
                    db_session.query(ValidationLabel)
                    .filter(
                        ValidationLabel.user_id == args.user_id,
                        ValidationLabel.paper_id == pid,
                    )
                    .first()
                )
                if existing_in_db:
                    session_presented.add(pid)
                    logger.info("Paper %s already labeled in database; skipping.", pid)
                    continue

            # Mark paper as presented in this session
            session_presented.add(pid)

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
                            user_id=args.user_id,
                        )
                        current_records_map[(args.user_id, pid)] = rec
                        # Save backup immediately after each label
                        export_labels_to_csv(list(current_records_map.values()))
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
        export_labels_to_csv(list(current_records_map.values()))
        user_labels = pd.DataFrame([
            r for (u, p), r in current_records_map.items() if u == args.user_id
        ])
        print_status_report(user_labels)


if __name__ == "__main__":
    main()
