"""
labeling.py
~~~~~~~~~~~
Labeling storage, validation, CSV backup, and target tracking (Phase 4).

Rules:
  - Three ground-truth labels: 'relevant', 'somewhat', 'not_relevant'.
  - Labels persist to database table `validation_labels` when available.
  - Timestamped CSV backups are written under `data/labels/`.
  - Targets: >= 100 relevant labels overall, >= 30 relevant in test split.
  - Shortfall is computed and reported honestly; never fabricate labels.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional, Any

import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.database.models import ValidationLabel

logger = logging.getLogger(__name__)

VALID_CANONICAL_LABELS = {"relevant", "somewhat", "not_relevant", "not"}

LABEL_NORMALIZATION = {
    "relevant": "relevant",
    "1": "relevant",
    "somewhat": "somewhat",
    "somewhat relevant": "somewhat",
    "2": "somewhat",
    "not_relevant": "not_relevant",
    "not relevant": "not_relevant",
    "not": "not_relevant",
    "3": "not_relevant",
}

LABEL_DIR = Path("data/labels")


def normalize_paper_id(paper_id: Any, valid_pids: Optional[set[str]] = None) -> str:
    """
    Normalize paper ID string and recover from float truncation (e.g. '2411.1151' -> '2411.11510').

    ArXiv IDs from 2015 onwards follow YYMM.NNNNN (5 digits after dot).
    If parsed as float, trailing zeros are lost.
    """
    if paper_id is None or (isinstance(paper_id, float) and pd.isna(paper_id)):
        return ""
    pid_str = str(paper_id).strip()
    if pid_str.endswith(".0") and len(pid_str) > 2 and pid_str[:-2].isdigit():
        pid_str = pid_str[:-2]

    if valid_pids is not None:
        if pid_str in valid_pids:
            return pid_str
        if f"{pid_str}0" in valid_pids:
            return f"{pid_str}0"
        if f"{pid_str}00" in valid_pids:
            return f"{pid_str}00"

    # Standard arXiv heuristic for post-2015 identifiers (YYMM.NNNNN)
    if "." in pid_str:
        parts = pid_str.split(".")
        if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit():
            yymm = int(parts[0])
            if yymm >= 1501 and len(parts[1]) == 4 and parts[1].isdigit():
                return f"{parts[0]}.{parts[1]}0"
    return pid_str


def parse_label_timestamp(ts: Any) -> Optional[datetime]:
    """Parse various timestamp representations into a timezone-aware UTC datetime."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    ts_str = str(ts).strip()
    if not ts_str:
        return None
    clean_str = ts_str.replace(" UTC", "+00:00").replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(clean_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        pdt = pd.to_datetime(ts_str)
        if pdt.tz is None:
            pdt = pdt.tz_localize("UTC")
        else:
            pdt = pdt.tz_convert("UTC")
        return pdt.to_pydatetime()
    except Exception:
        return None


def normalize_label(label: str) -> str:
    """
    Normalize human or CLI input into canonical label.

    Returns:
        One of 'relevant', 'somewhat', 'not_relevant'.

    Raises:
        ValueError if the label is unrecognized.
    """
    cleaned = str(label).strip().lower()
    if cleaned in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[cleaned]
    raise ValueError(
        f"Invalid label '{label}'. Must be one of: relevant (1), somewhat (2), not (3)."
    )


def record_validation_label(
    paper_id: str,
    label: str,
    stratum: Optional[str] = None,
    sampling_weight: float = 1.0,
    split: str = "dev",
    notes: Optional[str] = None,
    session: Optional[Session] = None,
    user_id: str = "developer",
    sampling_stratum: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate and record a ground truth label into a dictionary (and DB if session given).

    Args:
        paper_id: arXiv/corpus paper ID.
        label: One of 'relevant', 'somewhat', 'not' / 'not_relevant' (or accepted alias).
        stratum: S1-S5 (legacy alias for sampling_stratum).
        sampling_weight: Stratum population / sample size.
        split: 'dev' or 'test'.
        notes: Optional comments.
        session: Optional SQLAlchemy DB session.
        user_id: Evaluator user ID (default: 'developer').
        sampling_stratum: S1-S5 per Roadmap Section 6.

    Returns:
        Dict representing the created record.
    """
    eff_stratum = sampling_stratum or stratum
    if not eff_stratum or eff_stratum not in {"S1", "S2", "S3", "S4", "S5"}:
        raise ValueError(f"Invalid stratum '{eff_stratum}'. Must be S1..S5.")
    if split not in {"dev", "test", "calibration", "pilot_test"}:
        raise ValueError(f"Invalid split '{split}'. Must be 'dev', 'test', 'calibration', or 'pilot_test'.")

    canonical_label = normalize_label(label)
    norm_pid = normalize_paper_id(paper_id)
    eff_user_id = str(user_id or "developer").strip()
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "user_id": eff_user_id,
        "paper_id": norm_pid,
        "label": canonical_label,
        "sampling_stratum": str(eff_stratum),
        "stratum": str(eff_stratum),  # Backwards compatibility alias
        "sampling_weight": float(sampling_weight),
        "split": str(split),
        "notes": notes,
        "labelled_at": now_str,
        "created_at": now_str,  # Backwards compatibility alias
    }

    if session is not None:
        try:
            # Check for existing label for this user + paper
            existing = (
                session.query(ValidationLabel)
                .filter(
                    ValidationLabel.user_id == eff_user_id,
                    ValidationLabel.paper_id == norm_pid,
                )
                .first()
            )
            if existing:
                existing.label = canonical_label
                existing.sampling_stratum = eff_stratum
                existing.sampling_weight = float(sampling_weight)
                existing.split = split
                existing.notes = notes
                existing.labelled_at = now
                existing.updated_at = now
            else:
                db_record = ValidationLabel(
                    user_id=eff_user_id,
                    paper_id=norm_pid,
                    label=canonical_label,
                    sampling_stratum=str(eff_stratum),
                    sampling_weight=float(sampling_weight),
                    split=split,
                    notes=notes,
                    labelled_at=now,
                    updated_at=now,
                )
                session.add(db_record)
            session.commit()
            logger.info("Persisted validation label for paper %s (user %s) to database.", norm_pid, eff_user_id)
        except Exception as e:
            session.rollback()
            logger.warning("Could not persist validation label to database: %s", e)

    return record


def export_labels_to_csv(
    labels: list[dict[str, Any]] | pd.DataFrame,
    output_dir: Path | str = LABEL_DIR,
) -> tuple[Path, Path]:
    """
    Export labels to both a timestamped CSV backup and latest_labels.csv.

    Deduplicates records by (user_id, paper_id) keeping the latest.

    Args:
        labels: List of label dicts or DataFrame.
        output_dir: Directory where backups are written (default: data/labels).

    Returns:
        (timestamped_path, latest_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(labels) if isinstance(labels, list) else labels.copy()
    if not df.empty:
        if "user_id" not in df.columns:
            df["user_id"] = "developer"
        else:
            df["user_id"] = df["user_id"].fillna("developer").astype(str).str.strip()
        df["paper_id"] = df["paper_id"].apply(normalize_paper_id)

        ts_col = "labelled_at" if "labelled_at" in df.columns else ("created_at" if "created_at" in df.columns else None)
        if ts_col:
            df["_parsed_ts"] = df[ts_col].apply(parse_label_timestamp)
            df = df.sort_values(by=["_parsed_ts"], na_position="first")
            df = df.drop_duplicates(subset=["user_id", "paper_id"], keep="last")
            df = df.drop(columns=["_parsed_ts"])
        else:
            df = df.drop_duplicates(subset=["user_id", "paper_id"], keep="last")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamped_path = out_dir / f"validation_labels_{timestamp}.csv"
    latest_path = out_dir / "latest_labels.csv"

    df.to_csv(timestamped_path, index=False)
    df.to_csv(latest_path, index=False)

    logger.info("Exported %d labels to %s and %s", len(df), timestamped_path, latest_path)
    return timestamped_path, latest_path


def load_latest_labels(
    output_dir: Path | str = LABEL_DIR,
    deduplicate: bool = True,
    valid_pids: Optional[set[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load latest labels from data/labels/latest_labels.csv if present.

    Ensures:
      - String types for paper_id and user_id.
      - Normalized paper_id without float-truncation artifacts.
      - Deduplication on (user_id, paper_id) keeping newest record.
    """
    latest_path = Path(output_dir) / "latest_labels.csv"
    if not latest_path.exists():
        return None
    try:
        df = pd.read_csv(latest_path, dtype={"paper_id": str, "user_id": str})
        if df.empty:
            return df
        if "user_id" not in df.columns:
            df["user_id"] = "developer"
        else:
            df["user_id"] = df["user_id"].fillna("developer").astype(str).str.strip()

        df["paper_id"] = df["paper_id"].apply(lambda p: normalize_paper_id(p, valid_pids))

        if deduplicate and not df.empty:
            ts_col = "labelled_at" if "labelled_at" in df.columns else ("created_at" if "created_at" in df.columns else None)
            if ts_col:
                df["_parsed_ts"] = df[ts_col].apply(parse_label_timestamp)
                df = df.sort_values(by=["_parsed_ts"], na_position="first")
                df = df.drop_duplicates(subset=["user_id", "paper_id"], keep="last")
                df = df.drop(columns=["_parsed_ts"])
            else:
                df = df.drop_duplicates(subset=["user_id", "paper_id"], keep="last")

        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning("Could not read latest labels from %s: %s", latest_path, e)
        return None


def get_labeled_paper_ids(
    session: Optional[Session] = None,
    user_id: str = "developer",
    output_dir: Path | str = LABEL_DIR,
) -> set[str]:
    """
    Get all paper IDs that have already been labeled for user_id,
    checking both PostgreSQL and CSV backup.
    """
    labeled_pids = set()
    eff_user = str(user_id or "developer").strip()

    # 1. Check PostgreSQL if session provided
    if session is not None:
        try:
            db_rows = (
                session.query(ValidationLabel.paper_id)
                .filter(ValidationLabel.user_id == eff_user)
                .all()
            )
            for (pid,) in db_rows:
                norm_p = normalize_paper_id(pid)
                if norm_p:
                    labeled_pids.add(norm_p)
        except Exception as e:
            logger.warning("Error querying validation labels from database: %s", e)

    # 2. Check CSV backup
    latest_df = load_latest_labels(output_dir=output_dir, deduplicate=True)
    if latest_df is not None and not latest_df.empty:
        user_rows = latest_df[latest_df["user_id"] == eff_user]
        for pid in user_rows["paper_id"]:
            norm_p = normalize_paper_id(pid)
            if norm_p:
                labeled_pids.add(norm_p)

    return labeled_pids


def reconcile_csv_and_db_labels(
    session: Session,
    user_id: Optional[str] = None,
    output_dir: Path | str = LABEL_DIR,
    sample_df: Optional[pd.DataFrame] = None,
) -> dict[str, int]:
    """
    Safely reconcile CSV records with PostgreSQL validation_labels.

    Requirements satisfied:
      - Valid CSV-only labels (e.g. from when DB persistence failed) are inserted into DB.
      - Never overwrite a newer database label with an older CSV label.
      - If CSV label is strictly newer than DB label, update DB.
      - If DB label is newer or equal, preserve DB label.
      - Preserves multi-user separation.
      - Updates latest_labels.csv with the unified, reconciled state.
      - Does NOT delete or recreate any database tables or label files.
    """
    stats = {
        "inserted": 0,
        "updated": 0,
        "preserved_db": 0,
        "total_csv": 0,
    }

    # Load deduplicated CSV records
    valid_pids = set(sample_df["paper_id"].astype(str)) if sample_df is not None and "paper_id" in sample_df.columns else None
    csv_df = load_latest_labels(output_dir=output_dir, deduplicate=True, valid_pids=valid_pids)
    if csv_df is not None:
        stats["total_csv"] = len(csv_df)

    sample_lookup = {}
    if sample_df is not None and "paper_id" in sample_df.columns:
        sample_lookup = sample_df.set_index("paper_id").to_dict(orient="index")

    # Fetch existing DB records
    query = session.query(ValidationLabel)
    if user_id:
        query = query.filter(ValidationLabel.user_id == user_id)
    db_records = query.all()
    db_map = {(r.user_id, normalize_paper_id(r.paper_id)): r for r in db_records}

    if csv_df is not None and not csv_df.empty:
        for _, row in csv_df.iterrows():
            rec_user = str(row.get("user_id") or "developer").strip()
            if user_id and rec_user != user_id:
                continue

            pid = normalize_paper_id(row["paper_id"], valid_pids)
            if not pid:
                continue

            try:
                canonical_label = normalize_label(row["label"])
            except ValueError:
                logger.warning("Skipping CSV row with invalid label: %s", row.get("label"))
                continue

            csv_ts = parse_label_timestamp(row.get("labelled_at") or row.get("created_at"))

            meta = sample_lookup.get(pid, {})
            stratum = str(row.get("sampling_stratum") or row.get("stratum") or meta.get("stratum") or "S1")
            if stratum not in {"S1", "S2", "S3", "S4", "S5"}:
                stratum = "S1"
            weight = float(row.get("sampling_weight", meta.get("sampling_weight", 1.0)))
            split = str(row.get("split") or meta.get("split") or "dev")
            if split not in {"dev", "test", "calibration", "pilot_test"}:
                split = "dev"
            notes = row.get("notes")
            if pd.isna(notes):
                notes = None

            key = (rec_user, pid)
            existing = db_map.get(key)

            if existing is None:
                # CSV-only label: insert into DB safely
                now = csv_ts or datetime.now(timezone.utc)
                new_lbl = ValidationLabel(
                    user_id=rec_user,
                    paper_id=pid,
                    label=canonical_label,
                    sampling_stratum=stratum,
                    sampling_weight=weight,
                    split=split,
                    notes=notes,
                    labelled_at=now,
                    updated_at=now,
                )
                session.add(new_lbl)
                db_map[key] = new_lbl
                stats["inserted"] += 1
            else:
                # Existing in DB: check timestamps
                db_ts = parse_label_timestamp(existing.labelled_at or existing.updated_at)
                if csv_ts is not None and db_ts is not None and csv_ts > db_ts:
                    # CSV is strictly newer: update DB
                    existing.label = canonical_label
                    existing.sampling_stratum = stratum
                    existing.sampling_weight = weight
                    existing.split = split
                    existing.notes = notes
                    existing.labelled_at = csv_ts
                    existing.updated_at = csv_ts
                    stats["updated"] += 1
                else:
                    # DB is newer or equal: preserve DB! Do not overwrite.
                    stats["preserved_db"] += 1

        try:
            session.commit()
            logger.info("Reconciled CSV labels with DB: %s", stats)
        except Exception as e:
            session.rollback()
            logger.error("Failed to commit reconciled labels to DB: %s", e)

    # Sync reconciled state into latest_labels.csv (without generating an extra timestamped backup)
    try:
        all_db_labels = session.query(ValidationLabel).all()
        if all_db_labels:
            reconciled_rows = []
            for r in all_db_labels:
                ts_str = r.labelled_at.strftime("%Y-%m-%d %H:%M:%S UTC") if r.labelled_at else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                reconciled_rows.append({
                    "user_id": r.user_id,
                    "paper_id": normalize_paper_id(r.paper_id),
                    "label": r.label,
                    "sampling_stratum": r.sampling_stratum,
                    "stratum": r.sampling_stratum,
                    "sampling_weight": r.sampling_weight,
                    "split": r.split,
                    "notes": r.notes,
                    "labelled_at": ts_str,
                    "created_at": ts_str,
                })
            out_df = pd.DataFrame(reconciled_rows)
            out_df = out_df.sort_values(by=["labelled_at"]).drop_duplicates(
                subset=["user_id", "paper_id"], keep="last"
            )
            latest_path = Path(output_dir) / "latest_labels.csv"
            out_df.to_csv(latest_path, index=False)
            logger.info("Updated %s with %d reconciled labels.", latest_path, len(out_df))
    except Exception as e:
        logger.warning("Could not sync reconciled labels to %s: %s", output_dir, e)

    return stats


def get_label_statistics(labels_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute labeling progress, breakdown by label and split, and shortfall against targets.

    Targets:
      - At least 100 relevant labels overall
      - At least 30 relevant labels in the test split
    """
    if len(labels_df) == 0:
        return {
            "total_count": 0,
            "relevant_count": 0,
            "somewhat_count": 0,
            "not_relevant_count": 0,
            "dev_count": 0,
            "test_count": 0,
            "test_relevant_count": 0,
            "target_overall_met": False,
            "target_test_met": False,
            "overall_shortfall": 100,
            "test_shortfall": 30,
        }

    total = len(labels_df)
    counts_by_label = labels_df["label"].value_counts().to_dict()
    relevant_total = counts_by_label.get("relevant", 0)
    somewhat_total = counts_by_label.get("somewhat", 0)
    not_rel_total = counts_by_label.get("not_relevant", 0)

    dev_mask = labels_df["split"] == "dev"
    test_mask = labels_df["split"] == "test"

    dev_count = int(dev_mask.sum())
    test_count = int(test_mask.sum())

    test_relevant = int(
        ((labels_df["split"] == "test") & (labels_df["label"] == "relevant")).sum()
    )

    overall_shortfall = max(0, 100 - relevant_total)
    test_shortfall = max(0, 30 - test_relevant)

    return {
        "total_count": total,
        "relevant_count": relevant_total,
        "somewhat_count": somewhat_total,
        "not_relevant_count": not_rel_total,
        "dev_count": dev_count,
        "test_count": test_count,
        "test_relevant_count": test_relevant,
        "target_overall_met": overall_shortfall == 0,
        "target_test_met": test_shortfall == 0,
        "overall_shortfall": overall_shortfall,
        "test_shortfall": test_shortfall,
    }
