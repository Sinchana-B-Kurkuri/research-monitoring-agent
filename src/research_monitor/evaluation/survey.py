"""
survey.py
~~~~~~~~~
End-of-pilot survey and follow-up interview collection and analysis (Roadmap Section 8.6 & Phase 13).

Measures (Hypothesis H3):
  - Perceived usefulness for triage (1-5 Likert scale)
  - Summary helpfulness (1-5 Likert scale)
  - Estimated triage time (minutes/day)
  - Free-text feedback
  - Structured follow-up interview responses

H3 criterion:
  Median usefulness >= 4.0 and Median summary helpfulness >= 4.0.
  Responses are stored separately from model-training labels.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SURVEY_DIR = Path("data/surveys")

# Standard interview prompt keys
INTERVIEW_QUESTIONS = {
    "q1_workflow_integration": "How well did the daily digest and review queue integrate into your research workflow?",
    "q2_borderline_escalation": "How manageable was the borderline review workload (approx. papers per cycle)?",
    "q3_summary_accuracy": "Did the interest-framed summaries accurately capture relevance to your stated topics?",
    "q4_limitations_and_friction": "What were the primary limitations, friction points, or missing capabilities?",
}


def validate_survey_dict(data: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that survey dictionary conforms to Phase 13 specifications.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Survey data must be a dictionary."

    user_id = data.get("user_id")
    if not user_id or not str(user_id).strip():
        return False, "Missing or empty 'user_id'."

    # Usefulness: 1 to 5 integer
    usefulness = data.get("usefulness_score")
    if usefulness is None:
        return False, "Missing 'usefulness_score'."
    try:
        u_val = int(usefulness)
        if not (1 <= u_val <= 5):
            return False, f"usefulness_score must be between 1 and 5 (got {u_val})."
    except (ValueError, TypeError):
        return False, f"usefulness_score must be an integer between 1 and 5 (got {usefulness})."

    # Summary helpfulness: 1 to 5 integer
    helpfulness = data.get("summary_helpfulness_score")
    if helpfulness is None:
        return False, "Missing 'summary_helpfulness_score'."
    try:
        h_val = int(helpfulness)
        if not (1 <= h_val <= 5):
            return False, f"summary_helpfulness_score must be between 1 and 5 (got {h_val})."
    except (ValueError, TypeError):
        return False, f"summary_helpfulness_score must be an integer between 1 and 5 (got {helpfulness})."

    # Triage time: non-negative float
    triage_time = data.get("triage_time_minutes")
    if triage_time is not None:
        try:
            t_val = float(triage_time)
            if t_val < 0:
                return False, f"triage_time_minutes cannot be negative (got {t_val})."
        except (ValueError, TypeError):
            return False, f"triage_time_minutes must be a number (got {triage_time})."

    return True, None


def get_survey_path(user_id: str, survey_dir: Path | str = DEFAULT_SURVEY_DIR) -> Path:
    """Return path to a specific user's survey JSON file."""
    s_dir = Path(survey_dir)
    return s_dir / f"{str(user_id).strip()}_survey.json"


def save_survey(
    survey_data: dict[str, Any],
    survey_dir: Path | str = DEFAULT_SURVEY_DIR,
) -> Path:
    """
    Validate and save survey data to disk.

    Raises:
        ValueError if survey validation fails.
    """
    is_valid, err_msg = validate_survey_dict(survey_data)
    if not is_valid:
        raise ValueError(f"Invalid survey data: {err_msg}")

    s_dir = Path(survey_dir)
    s_dir.mkdir(parents=True, exist_ok=True)

    user_id = str(survey_data["user_id"]).strip()
    out_path = get_survey_path(user_id, survey_dir=s_dir)

    record = dict(survey_data)
    if "submitted_at" not in record or not record["submitted_at"]:
        record["submitted_at"] = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    logger.info("Saved survey for user '%s' to %s", user_id, out_path)
    return out_path


def load_survey(
    user_id: str,
    survey_dir: Path | str = DEFAULT_SURVEY_DIR,
) -> Optional[dict[str, Any]]:
    """Load survey data for a user if file exists."""
    path = get_survey_path(user_id, survey_dir=survey_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read survey file %s: %s", path, e)
        return None


def load_all_surveys(
    survey_dir: Path | str = DEFAULT_SURVEY_DIR,
) -> list[dict[str, Any]]:
    """Load all valid survey responses found in survey directory."""
    s_dir = Path(survey_dir)
    if not s_dir.exists():
        return []

    surveys = []
    for p in sorted(s_dir.glob("*_survey.json")):
        if p.name == "template_survey.json":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            is_valid, _ = validate_survey_dict(data)
            if is_valid:
                surveys.append(data)
            else:
                logger.warning("Skipping invalid survey in %s", p)
        except Exception as e:
            logger.warning("Error reading %s: %s", p, e)

    return surveys


def create_blank_survey_template(
    output_path: Path | str = DEFAULT_SURVEY_DIR / "template_survey.json",
) -> Path:
    """Create a blank survey template for pilot participants."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    template = {
        "user_id": "REPLACE_WITH_USER_ID",
        "submitted_at": None,
        "usefulness_score": 5,
        "summary_helpfulness_score": 5,
        "triage_time_minutes": 15.0,
        "free_text_feedback": "Explain how the system influenced your daily research monitoring...",
        "interview": {
            "interviewer": "evaluator",
            "interviewed_at": None,
            "q1_workflow_integration": "Notes on workflow fit...",
            "q2_borderline_escalation": "Notes on review queue volume and quality...",
            "q3_summary_accuracy": "Notes on summary accuracy against interest seed...",
            "q4_limitations_and_friction": "Notes on limitations or feature requests...",
        },
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    return out_p


def compute_survey_statistics(surveys: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate survey metrics and test Hypothesis H3 (Roadmap Section 8.6).

    H3: Users rate the digest as useful for triage and summaries as helpful
        (median >= 4 of 5 for both).
    """
    if not surveys:
        return {
            "total_surveys": 0,
            "usefulness_median": None,
            "usefulness_mean": None,
            "summary_helpfulness_median": None,
            "summary_helpfulness_mean": None,
            "triage_time_median": None,
            "triage_time_mean": None,
            "h3_supported": False,
            "h3_notes": "No survey responses collected yet.",
            "free_text_responses": [],
            "interview_summaries": [],
        }

    usefulness_vals = [float(s["usefulness_score"]) for s in surveys if "usefulness_score" in s]
    helpfulness_vals = [float(s["summary_helpfulness_score"]) for s in surveys if "summary_helpfulness_score" in s]
    triage_times = [float(s["triage_time_minutes"]) for s in surveys if s.get("triage_time_minutes") is not None]

    u_med = float(np.median(usefulness_vals)) if usefulness_vals else 0.0
    u_mean = float(np.mean(usefulness_vals)) if usefulness_vals else 0.0
    h_med = float(np.median(helpfulness_vals)) if helpfulness_vals else 0.0
    h_mean = float(np.mean(helpfulness_vals)) if helpfulness_vals else 0.0

    t_med = float(np.median(triage_times)) if triage_times else 0.0
    t_mean = float(np.mean(triage_times)) if triage_times else 0.0

    h3_supported = (u_med >= 4.0) and (h_med >= 4.0)

    free_texts = [
        {"user_id": s["user_id"], "feedback": s.get("free_text_feedback", "")}
        for s in surveys
        if s.get("free_text_feedback")
    ]

    interviews = [
        {"user_id": s["user_id"], "interview": s.get("interview", {})}
        for s in surveys
        if s.get("interview")
    ]

    return {
        "total_surveys": len(surveys),
        "n_responses": len(surveys),
        "usefulness_median": u_med,
        "median_usefulness": u_med,
        "usefulness_mean": round(u_mean, 2),
        "summary_helpfulness_median": h_med,
        "median_summary_helpfulness": h_med,
        "summary_helpfulness_mean": round(h_mean, 2),
        "triage_time_median": t_med,
        "triage_time_mean": round(t_mean, 2),
        "mean_triage_time_minutes": round(t_mean, 2),
        "h3_supported": h3_supported,
        "status": "SUPPORTED" if h3_supported else "NOT_SUPPORTED",
        "h3_notes": f"Median usefulness={u_med:.1f}, helpfulness={h_med:.1f} (target >= 4.0)",
        "free_text_responses": free_texts,
        "interview_summaries": interviews,
    }

