"""
generate_figures.py
~~~~~~~~~~~~~~~~~~~
Regenerates all publication-ready PNG figures from experimental result CSVs (Roadmap Section 8.5).

Produces:
  - experiments/results/e2_learning_curves.png
  - experiments/results/e3_signal_ablation.png
  - experiments/results/e4_centroid_update.png
  - experiments/results/e5_efficiency.png
  - experiments/results/e6_human_review_value.png

Requirements:
  - Completely reproducible from CSV files alone.
  - Clear labels, axes, legends, and classifier switch-point indicators.
"""

import logging
import os
from pathlib import Path
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")


def plot_e2_learning_curves(csv_path: Path = RESULTS_DIR / "e2_learning_curves.csv") -> Optional[Path]:
    """Generates e2_learning_curves.png showing PR-AUC and F1 vs feedback size."""
    if not csv_path.exists():
        logger.warning("Cannot generate E2 figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    series_styles = {
        "System (Protocol A - Random)": {"color": "#1f77b4", "marker": "o", "label": "System (Protocol A - Random)"},
        "System (Protocol B - Gate Selected)": {"color": "#2ca02c", "marker": "s", "label": "System (Protocol B - Gate Selected)"},
        "B4: TF-IDF + LogReg (Protocol A)": {"color": "#ff7f0e", "marker": "^", "linestyle": "--", "label": "B4: TF-IDF + LogReg"},
    }

    # B2 baseline value (from n=0 point)
    b2_pr_auc = df[df["n_feedback"] == 0]["mean_pr_auc"].values[0] if len(df[df["n_feedback"] == 0]) > 0 else None
    b2_f1 = df[df["n_feedback"] == 0]["mean_f1"].values[0] if len(df[df["n_feedback"] == 0]) > 0 else None

    for series_name, style in series_styles.items():
        sub = df[df["series"] == series_name].sort_values("n_feedback")
        if len(sub) == 0:
            continue

        # Plot PR-AUC
        ax1.plot(
            sub["n_feedback"],
            sub["mean_pr_auc"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style.get("linestyle", "-"),
            linewidth=2,
        )
        if "ci_lower_pr_auc" in sub.columns and "ci_upper_pr_auc" in sub.columns:
            ax1.fill_between(
                sub["n_feedback"],
                sub["ci_lower_pr_auc"],
                sub["ci_upper_pr_auc"],
                color=style["color"],
                alpha=0.15,
            )

        # Plot F1
        ax2.plot(
            sub["n_feedback"],
            sub["mean_f1"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style.get("linestyle", "-"),
            linewidth=2,
        )

    # Add B2 baseline horizontal lines
    if b2_pr_auc is not None:
        ax1.axhline(b2_pr_auc, color="#7f7f7f", linestyle=":", linewidth=1.5, label="B2: Fixed Seed Vector")
    if b2_f1 is not None:
        ax2.axhline(b2_f1, color="#7f7f7f", linestyle=":", linewidth=1.5, label="B2: Fixed Seed Vector")

    # Mark classifier switch point at n=30
    ax1.axvline(30, color="#d62728", linestyle="--", alpha=0.7, label="Classifier Switch Point (n=30)")
    ax2.axvline(30, color="#d62728", linestyle="--", alpha=0.7, label="Classifier Switch Point (n=30)")

    ax1.set_title("E2: PR-AUC vs. Feedback Sample Size", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Effective Feedback Sample Size (n)", fontsize=11)
    ax1.set_ylabel("PR-AUC (Weighted)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_title("E2: F1 Score vs. Feedback Sample Size", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Effective Feedback Sample Size (n)", fontsize=11)
    ax2.set_ylabel("F1 Score (Weighted)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    plt.suptitle("E2 — Learning Curves under Simulation Protocols A and B [SIMULATED]", fontsize=14, y=1.02)
    plt.tight_layout()

    out_png = RESULTS_DIR / "e2_learning_curves.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def plot_e3_signal_ablation(csv_path: Path = RESULTS_DIR / "e3_signal_ablation.csv") -> Optional[Path]:
    """Generates e3_signal_ablation.png showing user-only vs reviewer-only vs combined feedback."""
    if not csv_path.exists():
        logger.warning("Cannot generate E3 figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    cond_styles = {
        "combined": {"color": "#2ca02c", "marker": "o", "label": "Combined Feedback (Effective Rule)"},
        "user_only": {"color": "#1f77b4", "marker": "s", "label": "User-Only Signals (Delivered HIGH)"},
        "reviewer_only": {"color": "#ff7f0e", "marker": "^", "label": "Reviewer-Only Signals (Borderline)"},
    }

    for cond, style in cond_styles.items():
        sub = df[df["condition"] == cond].sort_values("n_feedback")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["n_feedback"],
            sub["mean_pr_auc"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
        )

    ax.set_title("E3: Signal Ablation under Protocol B [SIMULATED]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feedback Sample Size (n)", fontsize=11)
    ax.set_ylabel("Mean PR-AUC (Weighted)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)

    out_png = RESULTS_DIR / "e3_signal_ablation.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def plot_e4_centroid_update(csv_path: Path = RESULTS_DIR / "e4_centroid_update.csv") -> Optional[Path]:
    """Generates e4_centroid_update.png comparing B2, B3, and B5."""
    if not csv_path.exists():
        logger.warning("Cannot generate E4 figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    models = df["model"].unique()
    colors = {"B2: Fixed Seed Vector": "#7f7f7f", "B3: Updated Centroid (Rocchio)": "#1f77b4", "B5: Classifier (Embedding LogReg)": "#2ca02c"}

    for m in models:
        sub = df[df["model"] == m].sort_values("feedback_size")
        color = colors.get(m, "#333333")
        if m == "B2: Fixed Seed Vector":
            ax.axhline(sub["mean_pr_auc"].values[0], color=color, linestyle=":", linewidth=2, label=m)
        else:
            ax.plot(sub["feedback_size"], sub["mean_pr_auc"], label=m, color=color, marker="o", linewidth=2)

    ax.set_title("E4: Centroid Adaptation vs. Fixed Seed & Classifier [SIMULATED]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feedback Sample Size (n)", fontsize=11)
    ax.set_ylabel("Mean PR-AUC", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)

    out_png = RESULTS_DIR / "e4_centroid_update.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def plot_e5_efficiency(csv_path: Path = RESULTS_DIR / "e5_efficiency.csv") -> Optional[Path]:
    """Generates e5_efficiency.png showing LLM call reduction and cost savings."""
    if not csv_path.exists():
        logger.warning("Cannot generate E5 figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    labels = [f"P_high={r['p_high']:.0f} / P_low={r['p_low']:.0f}" for _, r in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    # Bar chart 1: LLM Call Reduction & Cost Savings %
    ax1.bar(x - width/2, df["llm_call_reduction_pct"], width, label="LLM Call Reduction %", color="#1f77b4")
    ax1.bar(x + width/2, df["cost_savings_pct"], width, label="Estimated Cost Savings %", color="#2ca02c")
    ax1.set_title("E5: LLM Call & Cost Savings vs. Summarize-All", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Percentage (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="lower right")

    # Bar chart 2: Volume breakdown (Summarized vs Discarded vs Review)
    ax2.bar(x, df["low_discarded"], width=0.5, label="LOW (Discarded)", color="#d62728", alpha=0.8)
    ax2.bar(x, df["review_queue_workload"], width=0.5, bottom=df["low_discarded"], label="BORDERLINE (Review Queue)", color="#ff7f0e", alpha=0.8)
    ax2.bar(x, df["papers_summarized"], width=0.5, bottom=df["low_discarded"] + df["review_queue_workload"], label="HIGH (Summarized)", color="#2ca02c", alpha=0.8)
    ax2.set_title("E5: Routing Volume Breakdown", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Paper Count", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper right")

    plt.suptitle("E5 — Relevance Gating Efficiency & Workload Reduction", fontsize=13, y=1.02)
    plt.tight_layout()

    out_png = RESULTS_DIR / "e5_efficiency.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def plot_e6_human_review_value(csv_path: Path = RESULTS_DIR / "e6_human_review_value.csv") -> Optional[Path]:
    """Generates e6_human_review_value.png comparing decision strategies."""
    if not csv_path.exists():
        logger.warning("Cannot generate E6 figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    strategies = [
        "Auto-Reject (HIGH only)",
        "Auto-Accept (HIGH + BORDER)",
        "Gate + Human Review",
    ]
    x = np.arange(len(strategies))
    width = 0.25

    ax.bar(x - width, df["precision"], width, label="Delivered Precision", color="#1f77b4")
    ax.bar(x, df["recall"], width, label="Delivered Recall", color="#2ca02c")
    ax.bar(x + width, df["f1"], width, label="Delivered F1", color="#ff7f0e")

    ax.set_title("E6: Human Review Escalation Value on BORDERLINE Papers [SIMULATED]", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper left")

    out_png = RESULTS_DIR / "e6_human_review_value.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def plot_e7_pilot_metrics(csv_path: Path = RESULTS_DIR / "final_e7_pilot.csv") -> Optional[Path]:
    """
    Generates final_e7_pilot_metrics.png showing H1 (like rate before vs after)
    and H2 (review burden before vs after) across pilot participants.
    """
    if not csv_path.exists():
        logger.warning("Cannot generate E7 figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    if df.empty or "user_id" not in df.columns:
        return None

    # Filter out users with neither test evaluated
    sub_df = df.dropna(subset=["like_rate_pre", "like_rate_post"], how="all")
    if sub_df.empty:
        # If no like rates, plot survey ratings if available
        if "usefulness_score" in df.columns and not df["usefulness_score"].dropna().empty:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            users = df["user_id"].astype(str)
            x = np.arange(len(users))
            width = 0.35
            ax.bar(x - width / 2, df["usefulness_score"].fillna(0), width, label="Usefulness (1-5)", color="#1f77b4")
            ax.bar(x + width / 2, df["summary_helpfulness_score"].fillna(0), width, label="Helpfulness (1-5)", color="#2ca02c")
            ax.axhline(4.0, color="#d62728", linestyle="--", label="H3 Threshold (>=4.0)")
            ax.set_xticks(x)
            ax.set_xticklabels(users, fontsize=9, rotation=15)
            ax.set_ylim(0, 5.5)
            ax.set_ylabel("Rating (1-5 Likert)", fontsize=10)
            ax.set_title("E7 Pilot: End-of-Pilot Survey Scores (H3)", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.5, axis="y")
            out_png = RESULTS_DIR / "final_e7_pilot_metrics.png"
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
            logger.info("Saved %s", out_png)
            return out_png
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    users = sub_df["user_id"].astype(str).tolist()
    x = np.arange(len(users))
    width = 0.35

    # Panel 1: Like rate before vs after (H1)
    lr_pre = sub_df["like_rate_pre"].fillna(0).values
    lr_post = sub_df["like_rate_post"].fillna(0).values
    ax1.bar(x - width / 2, lr_pre, width, label="Before Adaptation", color="#aec7e8")
    ax1.bar(x + width / 2, lr_post, width, label="After Adaptation", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(users, fontsize=9, rotation=15)
    ax1.set_ylabel("Digest Like Rate", fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("H1: Digest Like Rate Before vs. After Adaptation", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel 2: Review burden before vs after (H2)
    rb_pre = sub_df["review_burden_pre"].fillna(0).values
    rb_post = sub_df["review_burden_post"].fillna(0).values
    ax2.bar(x - width / 2, rb_pre, width, label="Before Adaptation", color="#ffbb78")
    ax2.bar(x + width / 2, rb_post, width, label="After Adaptation", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(users, fontsize=9, rotation=15)
    ax2.set_ylabel("Review Tasks / Cycle", fontsize=10)
    ax2.set_title("H2: Review Queue Workload (Tasks/Cycle)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.suptitle("Phase 13 / E7: Pilot User Adaptation and Experience", fontsize=13, fontweight="bold")
    out_png = RESULTS_DIR / "final_e7_pilot_metrics.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def plot_pilot_before_after(csv_path: Path = RESULTS_DIR / "final_pilot_before_after.csv") -> Optional[Path]:
    """Generates final_pilot_before_after.png showing routing distribution shifts."""
    if not csv_path.exists():
        logger.warning("Cannot generate Before/After figure: %s not found.", csv_path)
        return None

    df = pd.read_csv(csv_path)
    if df.empty or "user_id" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    users = df["user_id"].unique()
    indices = np.arange(len(users))
    width = 0.2

    # High share pre vs post
    high_pre = df["pre_high_share"].fillna(0).values
    high_post = df["post_high_share"].fillna(0).values
    border_pre = df["pre_borderline_share"].fillna(0).values
    border_post = df["post_borderline_share"].fillna(0).values

    ax.bar(indices - width * 1.5, high_pre, width, label="HIGH (Pre)", color="#1f77b4", alpha=0.7)
    ax.bar(indices - width * 0.5, high_post, width, label="HIGH (Post)", color="#1f77b4")
    ax.bar(indices + width * 0.5, border_pre, width, label="BORDERLINE (Pre)", color="#ff7f0e", alpha=0.7)
    ax.bar(indices + width * 1.5, border_post, width, label="BORDERLINE (Post)", color="#ff7f0e")

    ax.set_xticks(indices)
    ax.set_xticklabels(users, fontsize=9)
    ax.set_ylabel("Routing Share of Scored Papers", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Routing Distribution Shift Before vs. After Vector Activation", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    out_png = RESULTS_DIR / "final_pilot_before_after.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_png)
    return out_png


def regenerate_all_figures() -> list[Path]:
    """Regenerates all publication figures from existing CSV results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for fn in [
        plot_e2_learning_curves,
        plot_e3_signal_ablation,
        plot_e4_centroid_update,
        plot_e5_efficiency,
        plot_e6_human_review_value,
        plot_e7_pilot_metrics,
        plot_pilot_before_after,
    ]:
        p = fn()
        if p is not None:
            generated.append(p)
    return generated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("Regenerating all publication figures...")
    gen = regenerate_all_figures()
    print(f"Generated {len(gen)} figure(s) under {RESULTS_DIR}")

