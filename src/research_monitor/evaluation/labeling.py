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
    now = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "user_id": str(user_id),
        "paper_id": str(paper_id),
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
                    ValidationLabel.user_id == user_id,
                    ValidationLabel.paper_id == paper_id,
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
                    user_id=str(user_id),
                    paper_id=str(paper_id),
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
            logger.info("Persisted validation label for paper %s to database.", paper_id)
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

    Args:
        labels: List of label dicts or DataFrame.
        output_dir: Directory where backups are written (default: data/labels).

    Returns:
        (timestamped_path, latest_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(labels) if isinstance(labels, list) else labels.copy()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamped_path = out_dir / f"validation_labels_{timestamp}.csv"
    latest_path = out_dir / "latest_labels.csv"

    df.to_csv(timestamped_path, index=False)
    df.to_csv(latest_path, index=False)

    logger.info("Exported %d labels to %s and %s", len(df), timestamped_path, latest_path)
    return timestamped_path, latest_path


def load_latest_labels(output_dir: Path | str = LABEL_DIR) -> Optional[pd.DataFrame]:
    """
    Load latest labels from data/labels/latest_labels.csv if present.
    """
    latest_path = Path(output_dir) / "latest_labels.csv"
    if latest_path.exists():
        return pd.read_csv(latest_path)
    return None


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
