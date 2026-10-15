"""
adaptation.py
~~~~~~~~~~~~~
Before-and-after adaptation analysis for pilot users (Roadmap Section 8.6 & Phase 13).

Evaluates:
  - Routing distributions (HIGH, BORDERLINE, LOW share) before vs after preference vector activation.
  - Review queue workload (borderline tasks per cycle) before vs after adaptation (Hypothesis H2).
  - Digest like rate before vs after adaptation (Hypothesis H1).
  - Classifier model activation impact if a model was promoted.
  - Explicit handling for users with no activation or insufficient observations.
  - Mandatory disclaimer: Before-and-after comparison is descriptive and does not prove causation.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CAUSATION_DISCLAIMER = (
    "Before-and-after comparisons are descriptive and paired within user. "
    "They do not prove causation due to potential temporal shifts in arXiv submission streams "
    "and small participant sample sizes (Roadmap Section 8.6)."
)


def _compute_window_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute operational and routing metrics for a subset of pilot records."""
    if df.empty:
        return {
            "record_count": 0,
            "cycles_count": 0,
            "high_count": 0,
            "high_share": 0.0,
            "borderline_count": 0,
            "borderline_share": 0.0,
            "low_count": 0,
            "low_share": 0.0,
            "delivered_count": 0,
            "user_likes": 0,
            "user_dislikes": 0,
            "like_rate": None,
            "review_tasks_count": 0,
            "review_burden_per_cycle": 0.0,
            "review_accept_rate": None,
        }

    n_total = len(df)
    n_cycles = max(1, df["cycle_id"].nunique())

    routes = df["route"].value_counts().to_dict()
    n_high = routes.get("HIGH", 0)
    n_border = routes.get("BORDERLINE", 0)
    n_low = routes.get("LOW", 0)

    delivered = int(df["delivered_in_digest"].sum()) if "delivered_in_digest" in df.columns else 0

    # User reactions
    user_likes = int((df["user_reaction"] == 1).sum()) if "user_reaction" in df.columns else 0
    user_dislikes = int((df["user_reaction"] == 0).sum()) if "user_reaction" in df.columns else 0
    total_reactions = user_likes + user_dislikes
    like_rate = round(user_likes / total_reactions, 4) if total_reactions > 0 else None

    # Review tasks
    in_review = int(df["in_review_queue"].sum()) if "in_review_queue" in df.columns else n_border
    review_burden = round(in_review / n_cycles, 2)

    # Review outcomes
    accepted = int((df["review_decision"] == "accept").sum()) if "review_decision" in df.columns else 0
    rejected = int((df["review_decision"] == "reject").sum()) if "review_decision" in df.columns else 0
    decided = accepted + rejected
    review_accept_rate = round(accepted / decided, 4) if decided > 0 else None

    return {
        "record_count": n_total,
        "cycles_count": n_cycles,
        "high_count": n_high,
        "high_share": round(n_high / n_total, 4) if n_total > 0 else 0.0,
        "borderline_count": n_border,
        "borderline_share": round(n_border / n_total, 4) if n_total > 0 else 0.0,
        "low_count": n_low,
        "low_share": round(n_low / n_total, 4) if n_total > 0 else 0.0,
        "delivered_count": delivered,
        "user_likes": user_likes,
        "user_dislikes": user_dislikes,
        "like_rate": like_rate,
        "review_tasks_count": in_review,
        "review_burden_per_cycle": review_burden,
        "review_accept_rate": review_accept_rate,
    }


def analyze_user_adaptation(
    user_id: str,
    pilot_records_df: pd.DataFrame,
    vector_activation_ts: Optional[str | datetime] = None,
    model_activation_ts: Optional[str | datetime] = None,
    user_eligibility: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Analyze before vs after adaptation behaviour for a specific user.

    Args:
        user_id: Target user identifier.
        pilot_records_df: Exported pilot dataset.
        vector_activation_ts: Timestamp when interest vector v1+ was activated.
        model_activation_ts: Timestamp when logistic regression model was promoted.
        user_eligibility: Optional dictionary containing user eligibility under Section 8.6.

    Returns:
        Structured before-and-after comparison dictionary.
    """
    u_elig = user_eligibility or {}
    is_eligible = u_elig.get("h1_h2_eligible", False) if user_eligibility else True
    excl_reason = u_elig.get("h1_h2_exclusion_reason")

    user_df = pilot_records_df[pilot_records_df["user_id"] == user_id].copy()
    if user_df.empty:
        return {
            "user_id": user_id,
            "status": "NO_RECORDS_FOR_USER",
            "is_h1_h2_eligible": False,
            "exclusion_reason": "No pilot records found in clean export.",
            "has_vector_activation": False,
            "has_model_activation": False,
            "pre_vector_metrics": None,
            "post_vector_metrics": None,
            "post_model_metrics": None,
            "h1_like_rate_test": None,
            "h2_review_burden_test": None,
            "disclaimer": CAUSATION_DISCLAIMER,
        }

    # Resolve activation timestamps from dataframe columns if not provided directly
    if vector_activation_ts is None and "vector_activation_date" in user_df.columns:
        valid_v = user_df["vector_activation_date"].dropna().unique()
        if len(valid_v) > 0 and valid_v[0]:
            vector_activation_ts = str(valid_v[0])

    if model_activation_ts is None and "model_activation_date" in user_df.columns:
        valid_m = user_df["model_activation_date"].dropna().unique()
        if len(valid_m) > 0 and valid_m[0]:
            model_activation_ts = str(valid_m[0])

    has_v_act = vector_activation_ts is not None and str(vector_activation_ts).strip() != ""
    has_m_act = model_activation_ts is not None and str(model_activation_ts).strip() != ""

    if not has_v_act:
        status_str = (
            "EXCLUDED_SECTION_8_6_NO_ACTIVATION"
            if not is_eligible
            else "INSUFFICIENT_OBSERVATIONS_NO_VECTOR_ACTIVATION"
        )
        reason_str = (
            f"User remained on cold-start seed vector; {excl_reason}"
            if excl_reason
            else "User remained on cold-start seed vector (no vector update)."
        )
        return {
            "user_id": user_id,
            "status": status_str,
            "is_h1_h2_eligible": is_eligible,
            "exclusion_reason": excl_reason or "No vector update recorded.",
            "has_vector_activation": False,
            "has_model_activation": False,
            "vector_activation_date": None,
            "model_activation_date": None,
            "all_observations_metrics": _compute_window_metrics(user_df),
            "pre_vector_metrics": _compute_window_metrics(user_df),
            "post_vector_metrics": None,
            "post_model_metrics": None,
            "h1_like_rate_test": {
                "supported": None,
                "note": f"Cannot evaluate H1: {reason_str}",
            },
            "h2_review_burden_test": {
                "supported": None,
                "note": f"Cannot evaluate H2: {reason_str}",
            },
            "disclaimer": CAUSATION_DISCLAIMER,
        }

    # Convert timestamps for splitting
    v_ts_str = str(vector_activation_ts)
    user_df["_parsed_ts"] = pd.to_datetime(user_df["created_at"], utc=True, errors="coerce")
    v_dt = pd.to_datetime(v_ts_str, utc=True)

    pre_mask = user_df["_parsed_ts"] < v_dt
    post_mask = user_df["_parsed_ts"] >= v_dt

    pre_metrics = _compute_window_metrics(user_df[pre_mask])
    post_metrics = _compute_window_metrics(user_df[post_mask])

    # Model window (if model was promoted)
    post_model_metrics = None
    if has_m_act:
        m_dt = pd.to_datetime(str(model_activation_ts), utc=True)
        post_model_metrics = _compute_window_metrics(user_df[user_df["_parsed_ts"] >= m_dt])

    # Test Hypothesis H1: Digest like rate after activation is at least as high as before
    pre_like = pre_metrics["like_rate"]
    post_like = post_metrics["like_rate"]
    h1_supported = None
    h1_note = "Insufficient feedback reactions to calculate like rate in one or both windows."
    if pre_like is not None and post_like is not None:
        h1_supported = bool(post_like >= pre_like)
        h1_note = (
            f"Pre-activation like rate={pre_like:.1%}, Post-activation like rate={post_like:.1%} "
            f"({'Observed: rate maintained/improved' if h1_supported else 'Not observed: rate decreased'})."
        )
    elif post_like is not None and pre_like is None:
        h1_note = f"Post-activation like rate={post_like:.1%}; pre-activation had zero reactions."

    # Test Hypothesis H2: Review burden (borderline tasks/cycle) does not increase after activation
    pre_burden = pre_metrics["review_burden_per_cycle"]
    post_burden = post_metrics["review_burden_per_cycle"]
    h2_supported = bool(post_burden <= pre_burden)
    h2_note = (
        f"Pre-activation burden={pre_burden:.2f} tasks/cycle, Post-activation burden={post_burden:.2f} tasks/cycle "
        f"({'Observed: burden did not increase' if h2_supported else 'Not observed: burden increased'})."
    )

    return {
        "user_id": user_id,
        "status": "ADAPTATION_ANALYZED" if is_eligible else "INSUFFICIENT_OBSERVATIONS_EXCLUDED",
        "is_h1_h2_eligible": is_eligible,
        "exclusion_reason": excl_reason,
        "has_vector_activation": True,
        "has_model_activation": has_m_act,
        "vector_activation_date": v_ts_str,
        "model_activation_date": str(model_activation_ts) if has_m_act else None,
        "pre_vector_metrics": pre_metrics,
        "post_vector_metrics": post_metrics,
        "post_model_metrics": post_model_metrics,
        "h1_like_rate_test": {
            "supported": h1_supported if is_eligible else None,
            "pre_like_rate": pre_like,
            "post_like_rate": post_like,
            "note": h1_note if is_eligible else f"Excluded: {excl_reason}",
        },
        "h2_review_burden_test": {
            "supported": h2_supported if is_eligible else None,
            "pre_burden_per_cycle": pre_burden,
            "post_burden_per_cycle": post_burden,
            "note": h2_note if is_eligible else f"Excluded: {excl_reason}",
        },
        "disclaimer": CAUSATION_DISCLAIMER,
    }


def analyze_all_pilot_adaptations(
    pilot_records_df: pd.DataFrame,
    user_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run before-and-after adaptation analysis across all unique users in pilot records.
    """
    if pilot_records_df.empty:
        return {"users": {}, "summary": "No pilot records to analyze."}

    user_ids = sorted(pilot_records_df["user_id"].unique())
    user_results = {}
    h1_outcomes = []
    h2_outcomes = []

    for uid in user_ids:
        u_elig = user_eligibility.get(uid) if user_eligibility else None
        res = analyze_user_adaptation(uid, pilot_records_df, user_eligibility=u_elig)
        user_results[uid] = res
        if res.get("h1_like_rate_test", {}).get("supported") is not None:
            h1_outcomes.append(res["h1_like_rate_test"]["supported"])
        if res.get("h2_review_burden_test", {}).get("supported") is not None:
            h2_outcomes.append(res["h2_review_burden_test"]["supported"])

    return {
        "users": user_results,
        "summary": {
            "total_users_analyzed": len(user_ids),
            "users_with_vector_activation": sum(1 for r in user_results.values() if r.get("has_vector_activation")),
            "users_with_model_activation": sum(1 for r in user_results.values() if r.get("has_model_activation")),
            "h1_tested_count": len(h1_outcomes),
            "h1_supported_count": sum(1 for x in h1_outcomes if x is True),
            "h2_tested_count": len(h2_outcomes),
            "h2_supported_count": sum(1 for x in h2_outcomes if x is True),
        },
        "disclaimer": CAUSATION_DISCLAIMER,
    }


# Convenient alias
analyze_all_pilot_adaptation = analyze_all_pilot_adaptations


def export_adaptation_table(
    adaptation_res: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    """
    Exports summary before-and-after table to CSV (final_pilot_before_after.csv).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "final_pilot_before_after.csv"

    users_data = adaptation_res.get("users", {})
    if isinstance(users_data, dict):
        user_list = list(users_data.values())
    elif isinstance(users_data, list):
        user_list = users_data
    else:
        user_list = []

    rows = []
    for u in user_list:
        uid = u.get("user_id")
        pre = u.get("pre_vector_metrics") or {}
        post = u.get("post_vector_metrics") or {}
        h1 = u.get("h1_like_rate_test") or {}
        h2 = u.get("h2_review_burden_test") or {}

        rows.append({
            "user_id": uid,
            "status": u.get("status"),
            "is_h1_h2_eligible": u.get("is_h1_h2_eligible", False),
            "exclusion_reason": u.get("exclusion_reason"),
            "has_vector_activation": u.get("has_vector_activation"),
            "vector_activation_date": u.get("vector_activation_date"),
            "has_model_activation": u.get("has_model_activation"),
            "model_activation_date": u.get("model_activation_date"),
            "pre_record_count": pre.get("record_count", 0),
            "pre_high_share": pre.get("high_share", 0.0),
            "pre_borderline_share": pre.get("borderline_share", 0.0),
            "pre_low_share": pre.get("low_share", 0.0),
            "pre_like_rate": pre.get("like_rate"),
            "pre_review_burden": pre.get("review_burden_per_cycle", 0.0),
            "post_record_count": post.get("record_count", 0),
            "post_high_share": post.get("high_share", 0.0),
            "post_borderline_share": post.get("borderline_share", 0.0),
            "post_low_share": post.get("low_share", 0.0),
            "post_like_rate": post.get("like_rate"),
            "post_review_burden": post.get("review_burden_per_cycle", 0.0),
            "h1_supported": h1.get("supported"),
            "h1_note": h1.get("note"),
            "h2_supported": h2.get("supported"),
            "h2_note": h2.get("note"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info("Saved Before/After Adaptation table to %s", out_csv)
    return out_csv

