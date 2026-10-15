"""
run_phase12_experiments.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Master CLI Runner for Phase 12 Offline Experiment Suite (Roadmap Sections 8.4 & 8.5).

Runs:
  - E1: Gate Quality (Test split, weighted, vs B0-B2, B4 at matched volume, bootstrap CIs).
  - E2: Learning Curves (Protocols A and B, PR-AUC and F1 vs n, B2 & B4 baselines, switch point).
  - E3: Signal Ablation (Protocol B, user-only vs reviewer-only vs combined).
  - E4: Centroid Update (B2 fixed vs B3 updated centroid vs B5 classifier at equal budgets).
  - E5: Efficiency (LLM-call reduction, tokens & costs, review workload, percentile sensitivity).
  - E6: Human-Review Value (Escalation vs auto-accept vs auto-reject on BORDERLINE papers).

Usage:
  python scripts/run_phase12_experiments.py --all
  python scripts/run_phase12_experiments.py --all --simulated-oracle
  python scripts/run_phase12_experiments.py --e1 --e5
"""

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

# Add src and root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from research_monitor.config import config
from research_monitor.evaluation.experiments import (
    BlockedExperimentError,
    run_experiment_e1_gate_quality,
    run_experiment_e2_learning_curves,
    run_experiment_e3_signal_ablation,
    run_experiment_e4_centroid_update,
    run_experiment_e5_efficiency,
    run_experiment_e6_human_review_value,
)
from research_monitor.evaluation.labeling import load_latest_labels
from research_monitor.evaluation.simulation import SimulationOracle
from scripts.generate_figures import regenerate_all_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 12 Offline Experiment Suite (E1 through E6)."
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments (E1-E6).")
    parser.add_argument("--e1", action="store_true", help="Run E1: Gate Quality.")
    parser.add_argument("--e2", action="store_true", help="Run E2: Learning Curves.")
    parser.add_argument("--e3", action="store_true", help="Run E3: Signal Ablation.")
    parser.add_argument("--e4", action="store_true", help="Run E4: Centroid Update.")
    parser.add_argument("--e5", action="store_true", help="Run E5: Efficiency.")
    parser.add_argument("--e6", action="store_true", help="Run E6: Human Review Value.")
    parser.add_argument(
        "--simulated-oracle",
        action="store_true",
        help="Use simulation oracle if human ground-truth labels are not recorded yet.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure PNG generation.",
    )
    args = parser.parse_args()

    # Default to running all if none specified
    if not (args.all or args.e1 or args.e2 or args.e3 or args.e4 or args.e5 or args.e6):
        args.all = True

    run_all = args.all
    do_e1 = run_all or args.e1
    do_e2 = run_all or args.e2
    do_e3 = run_all or args.e3
    do_e4 = run_all or args.e4
    do_e5 = run_all or args.e5
    do_e6 = run_all or args.e6

    print("\n" + "=" * 70)
    print("  PHASE 12 — OFFLINE EXPERIMENT SUITE")
    print(f"  Execution Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # 1. Inspect required inputs
    sample_path = Path("data/evaluation_sample.parquet")
    if sample_path.exists():
        sample_df = pd.read_parquet(sample_path)
    else:
        sample_path = Path("data/evaluation_sample.csv")
        if not sample_path.exists():
            logger.error("Missing evaluation sample. Run scripts/prepare_evaluation_sample.py first.")
            sys.exit(1)
        sample_df = pd.read_csv(sample_path, dtype={"paper_id": str})

    corpus_path = Path("data/corpus.parquet")
    emb_path = Path("data/embeddings.npy")
    if not corpus_path.exists() or not emb_path.exists():
        logger.error("Missing %s or %s.", corpus_path, emb_path)
        sys.exit(1)

    logger.info("Loading corpus and embeddings...")
    corpus_df = pd.read_parquet(corpus_path)
    embeddings = np.load(emb_path)

    # Align evaluation sample with embeddings
    corpus_pid_to_idx = {str(pid): i for i, pid in enumerate(corpus_df["paper_id"].values)}
    sample_indices = [corpus_pid_to_idx[str(pid)] for pid in sample_df["paper_id"].values]
    sample_embeddings = embeddings[sample_indices]

    # Split into dev and test
    dev_mask = (sample_df["split"] == "dev").values
    test_mask = (sample_df["split"] == "test").values

    dev_df = sample_df[dev_mask].reset_index(drop=True)
    dev_embeddings = sample_embeddings[dev_mask]

    test_df = sample_df[test_mask].reset_index(drop=True)
    test_embeddings = sample_embeddings[test_mask]

    logger.info("Evaluation sample partitioned: %d dev, %d test papers.", len(dev_df), len(test_df))

    # Reference sample for percentiles
    ref_mask = corpus_df["is_reference"].values
    ref_embeddings = embeddings[ref_mask]

    # Compute seed vector
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    seed_vec = model.encode(config.SEED_INTEREST, convert_to_numpy=True)
    seed_vec = seed_vec / np.linalg.norm(seed_vec)

    # Compute reference percentiles
    ref_scores = ref_embeddings @ seed_vec
    t_high = float(np.percentile(ref_scores, config.PROVISIONAL_P_HIGH))
    t_low = float(np.percentile(ref_scores, config.PROVISIONAL_P_LOW))

    logger.info("Reference score cut points: t_high=%.4f (P%.1f), t_low=%.4f (P%.1f)", t_high, config.PROVISIONAL_P_HIGH, t_low, config.PROVISIONAL_P_LOW)

    # Check for human labels
    labels_df = load_latest_labels()
    has_human_labels = labels_df is not None and len(labels_df) > 0

    gt_map = {}
    if has_human_labels:
        gt_map = dict(zip(labels_df["paper_id"].astype(str), labels_df["label"].astype(str)))
        logger.info("Found %d human ground-truth labels.", len(gt_map))
    elif args.simulated_oracle:
        tmp_oracle = SimulationOracle(ground_truth_map={}, noise_rate=0.0)
        gt_map = {
            str(r["paper_id"]): ("relevant" if tmp_oracle.evaluate_paper(str(r["paper_id"]), {"title": r["title"], "abstract": r["abstract"]}) == 1 else "not_relevant")
            for _, r in sample_df.iterrows()
        }
        logger.info("Generated %d simulated oracle labels for simulation mode.", len(gt_map))
    else:
        logger.warning("No human ground-truth labels found in data/labels/latest_labels.csv.")

    # Oracle setup
    oracle = SimulationOracle(ground_truth_map=gt_map if (has_human_labels or args.simulated_oracle) else None, noise_rate=0.0)

    corpus_texts = (corpus_df["title"].fillna("") + " " + corpus_df["abstract"].fillna("")).tolist()[:2000]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed_experiments = []
    blocked_experiments = []

    # -------------------------------------------------------------------------
    # Experiment E1: Gate Quality
    # -------------------------------------------------------------------------
    if do_e1:
        print("\n--- Running E1: Gate Quality ---")
        if not has_human_labels and not args.simulated_oracle:
            msg = "BLOCKED: Requires human ground-truth labels on the test split (run scripts/label_papers.py or use --simulated-oracle for simulation)."
            print(f"  [E1 STATUS] {msg}")
            blocked_experiments.append(("E1", msg))
        else:
            sim_flag = not has_human_labels
            e1_labels = gt_map if has_human_labels else {
                str(r["paper_id"]): ("relevant" if oracle.evaluate_paper(r["paper_id"], {"title": r["title"], "abstract": r["abstract"]}) == 1 else "not_relevant")
                for _, r in sample_df.iterrows()
            }
            df_e1 = run_experiment_e1_gate_quality(
                test_df=test_df,
                test_embeddings=test_embeddings,
                seed_vector=seed_vec,
                ground_truth_labels=e1_labels,
                dev_df=dev_df,
                t_high=t_high,
                t_low=t_low,
                corpus_texts=corpus_texts,
                simulated=sim_flag,
            )
            print(df_e1[["model_or_baseline", "matched_volume", "precision", "recall", "f1", "pr_auc", "roc_auc"]].to_string(index=False))
            completed_experiments.append("E1 (Gate Quality)")

    # -------------------------------------------------------------------------
    # Experiment E2: Learning Curves (Protocols A & B)
    # -------------------------------------------------------------------------
    if do_e2:
        print("\n--- Running E2: Learning Curves (Protocols A & B) ---")
        df_e2 = run_experiment_e2_learning_curves(
            dev_df=dev_df,
            dev_embeddings=dev_embeddings,
            test_df=test_df,
            test_embeddings=test_embeddings,
            seed_vector=seed_vec,
            oracle=oracle,
            t_high=t_high,
            t_low=t_low,
            corpus_texts=corpus_texts,
        )
        print(df_e2[["series", "n_feedback", "model_type", "mean_pr_auc", "mean_f1", "is_switch_point"]].head(12).to_string(index=False))
        completed_experiments.append("E2 (Learning Curves)")

    # -------------------------------------------------------------------------
    # Experiment E3: Signal Ablation
    # -------------------------------------------------------------------------
    if do_e3:
        print("\n--- Running E3: Signal Ablation (Protocol B) ---")
        df_e3 = run_experiment_e3_signal_ablation(
            dev_df=dev_df,
            dev_embeddings=dev_embeddings,
            test_df=test_df,
            test_embeddings=test_embeddings,
            seed_vector=seed_vec,
            oracle=oracle,
            t_high=t_high,
            t_low=t_low,
        )
        print(df_e3.to_string(index=False))
        completed_experiments.append("E3 (Signal Ablation)")

    # -------------------------------------------------------------------------
    # Experiment E4: Centroid Update
    # -------------------------------------------------------------------------
    if do_e4:
        print("\n--- Running E4: Centroid Adaptation vs. Fixed Seed & Classifier ---")
        df_e4 = run_experiment_e4_centroid_update(
            dev_df=dev_df,
            dev_embeddings=dev_embeddings,
            test_df=test_df,
            test_embeddings=test_embeddings,
            seed_vector=seed_vec,
            oracle=oracle,
        )
        print(df_e4.to_string(index=False))
        completed_experiments.append("E4 (Centroid Update)")

    # -------------------------------------------------------------------------
    # Experiment E5: Efficiency
    # -------------------------------------------------------------------------
    if do_e5:
        print("\n--- Running E5: Relevance Gating Efficiency ---")
        test_scores = test_embeddings @ seed_vec
        df_e5 = run_experiment_e5_efficiency(scores=test_scores)
        print(df_e5[["p_high", "p_low", "total_papers_scored", "papers_summarized", "llm_call_reduction_pct", "cost_savings_pct"]].to_string(index=False))
        completed_experiments.append("E5 (Efficiency)")

    # -------------------------------------------------------------------------
    # Experiment E6: Human Review Value
    # -------------------------------------------------------------------------
    if do_e6:
        print("\n--- Running E6: Human Review Escalation Value on BORDERLINE Papers ---")
        test_scores = test_embeddings @ seed_vec
        # y_true from oracle
        y_test_oracle = np.array([
            oracle.evaluate_paper(pid, {"title": r["title"], "abstract": r["abstract"]})
            for pid, (_, r) in zip(test_df["paper_id"].values, test_df.iterrows())
        ], dtype=np.int32)

        df_e6 = run_experiment_e6_human_review_value(
            scores=test_scores,
            y_true=y_test_oracle,
            t_high=t_high,
            t_low=t_low,
            weights=test_df["sampling_weight"].values if "sampling_weight" in test_df.columns else None,
            oracle=oracle,
            paper_ids=test_df["paper_id"].values,
        )
        print(df_e6[["strategy", "papers_delivered", "papers_reviewed", "precision", "recall", "f1"]].to_string(index=False))
        completed_experiments.append("E6 (Human Review Value)")

    # -------------------------------------------------------------------------
    # Regenerate Figures
    # -------------------------------------------------------------------------
    if not args.no_figures:
        print("\n--- Generating Publication PNG Figures ---")
        gen_figs = regenerate_all_figures()
        for f in gen_figs:
            print(f"  [FIGURE] Generated: {f}")

    print("\n" + "=" * 70)
    print("  PHASE 12 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Completed Experiments: {len(completed_experiments)}")
    for c in completed_experiments:
        print(f"    - {c}")
    if blocked_experiments:
        print(f"\n  Blocked Experiments:   {len(blocked_experiments)}")
        for exp_id, reason in blocked_experiments:
            print(f"    - [{exp_id}] {reason}")
    print(f"\n  Results saved in:      {RESULTS_DIR.resolve()}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
