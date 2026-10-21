"""
Final E1–E7 Evaluation Suite Orchestrator (Phase 13, Section 8 & Section 2.C).

Orchestrates:
  - Step 1: Pre-flight readiness check (readiness.py)
  - Step 2: Reproducible pilot dataset export (pilot_export.py)
  - Step 3: Final E1-E6 offline evaluation (using exact Phase 12 baselines B0-B5 and frozen test split)
  - Step 4: Final E7 pilot evaluation (Hypotheses H1-H3, survey stats, Section 8.6 exclusions)
  - Step 5: Before-and-after adaptation analysis (adaptation.py)
  - Step 6: Operational efficiency & workload analysis (efficiency.py)
  - Step 7: Reproducible deliverables and readiness reporting (honesty safeguards, no premature freezing)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from research_monitor.config import config
from research_monitor.database.session import get_db_session
from research_monitor.evaluation.access import log_test_access
from research_monitor.evaluation.adaptation import (
    CAUSATION_DISCLAIMER,
    analyze_all_pilot_adaptation,
    export_adaptation_table,
)
from research_monitor.evaluation.efficiency import (
    calculate_operational_efficiency,
    export_efficiency_report,
)
from research_monitor.evaluation.experiments import (
    BlockedExperimentError,
    RESULTS_DIR,
    run_experiment_e1_gate_quality,
)
from research_monitor.evaluation.labeling import load_latest_labels
from research_monitor.evaluation.pilot_export import export_pilot_dataset
from research_monitor.evaluation.readiness import (
    ReadinessTier,
    check_pilot_readiness,
)
from research_monitor.evaluation.survey import (
    compute_survey_statistics,
    load_all_surveys,
)

logger = logging.getLogger(__name__)


def _compute_seed_vector() -> np.ndarray:
    """Computes normalized seed interest embedding using the configured model."""
    from research_monitor.relevance.user_manager import _generate_seed_vector
    return _generate_seed_vector(config.SEED_INTEREST)


def run_experiment_e7_pilot(
    pilot_export_dir: Path | str = Path("data/pilot_export"),
    survey_dir: Path | str = Path("data/surveys"),
    output_dir: Path | str = RESULTS_DIR,
) -> Dict[str, Any]:
    """
    E7 — Pilot Evaluation (Roadmap Section 8.5 & 8.6):
    Tests Hypotheses:
      - H1: within each user, digest like rate after vector/model activation is >= before.
      - H2: review burden (borderline tasks/cycle) does not increase after activation.
      - H3: users rate digest as useful for triage and summaries as helpful (median >= 4 of 5).
    Enforces Section 8.6 exclusion rules:
      - Users with < 5 active days or < 30 effective labels excluded from H1/H2 (kept for H3).
      - Outage days logged and excluded.
      - Pre-onboarding decisions excluded.
      - Developer data reported separately.
    """
    p_dir = Path(pilot_export_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pilot_records_path = p_dir / "pilot_records_clean.csv"
    if not pilot_records_path.exists():
        logger.warning("Pilot records not found at %s. Returning empty E7 results.", pilot_records_path)
        return {
            "status": "NO_PILOT_DATA",
            "message": f"Clean pilot records file {pilot_records_path} does not exist.",
            "hypotheses": {"h1": None, "h2": None, "h3": None},
            "participants": [],
        }

    pilot_df = pd.read_csv(pilot_records_path)
    audit_log_path = p_dir / "audit_log.json"
    audit_log: Dict[str, Any] = {}
    if audit_log_path.exists():
        try:
            with open(audit_log_path, "r", encoding="utf-8") as f:
                audit_log = json.load(f)
        except Exception as e:
            logger.warning("Could not read audit log: %s", e)

    # 1. Survey evaluation (H3)
    surveys = [survey for survey in load_all_surveys(survey_dir=survey_dir)
        if survey["user_id"] in audit_log.get("user_eligibility", {})
        and not audit_log["user_eligibility"][survey["user_id"]].get("is_developer")
        and not audit_log["user_eligibility"][survey["user_id"]].get("is_simulated")]
    h3_stats = compute_survey_statistics(surveys)

    # 2. Section 8.6 eligibility and adaptation analysis
    eligibility_info = audit_log.get("user_eligibility", {})
    adaptation_res = analyze_all_pilot_adaptation(pilot_df, user_eligibility=eligibility_info)
    adaptation_users_raw = adaptation_res.get("users", {})
    adaptation_users = list(adaptation_users_raw.values()) if isinstance(adaptation_users_raw, dict) else adaptation_users_raw

    # 3. Process each participant against Section 8.6 exclusions
    participant_rows = []
    developer_ids = {"developer", "dev_phase5", "dev"}

    h1_supported_count = 0
    h1_tested_count = 0
    h2_supported_count = 0
    h2_tested_count = 0

    for u_res in adaptation_users:
        uid = u_res.get("user_id")
        is_dev = uid in developer_ids or u_res.get("is_developer", False)

        u_elig = eligibility_info.get(uid, {})
        active_days = u_elig.get("active_days", 0)
        eff_labels = u_elig.get("effective_labels", 0)
        h1_h2_eligible = u_res.get("is_h1_h2_eligible", u_elig.get("h1_h2_eligible", active_days >= 5 and eff_labels >= 30))
        h1_h2_reason = u_res.get("exclusion_reason", u_elig.get("h1_h2_exclusion_reason"))

        # H1 Like rate test
        h1_test = u_res.get("h1_like_rate_test") or {}
        h1_status = u_res.get("status", "NOT_EVALUATED")
        h1_supported = h1_test.get("supported", h1_test.get("h1_supported"))
        p_pre = h1_test.get("pre_like_rate", h1_test.get("like_rate_pre"))
        p_post = h1_test.get("post_like_rate", h1_test.get("like_rate_post"))

        # H2 Review burden test
        h2_test = u_res.get("h2_review_burden_test") or {}
        h2_status = u_res.get("status", "NOT_EVALUATED")
        h2_supported = h2_test.get("supported", h2_test.get("h2_supported"))
        b_pre = h2_test.get("pre_burden_per_cycle", h2_test.get("review_burden_pre"))
        b_post = h2_test.get("post_burden_per_cycle", h2_test.get("review_burden_post"))

        # Individual survey scores if available
        u_survey = next((s for s in surveys if str(s.get("user_id")) == uid), None)
        u_useful = u_survey.get("usefulness_score") if u_survey else None
        u_helpful = u_survey.get("summary_helpfulness_score") if u_survey else None
        u_triage = u_survey.get("triage_time_minutes") if u_survey else None

        if h1_h2_eligible and not is_dev:
            if h1_supported is not None:
                h1_tested_count += 1
                if h1_supported:
                    h1_supported_count += 1
            if h2_supported is not None:
                h2_tested_count += 1
                if h2_supported:
                    h2_supported_count += 1

        participant_rows.append({
            "user_id": uid,
            "is_developer": is_dev,
            "is_h1_h2_eligible": h1_h2_eligible,
            "exclusion_reason": h1_h2_reason,
            "active_days": active_days,
            "effective_labels": eff_labels,
            "h1_status": h1_status,
            "h1_supported": h1_supported,
            "like_rate_pre": p_pre,
            "like_rate_post": p_post,
            "h2_status": h2_status,
            "h2_supported": h2_supported,
            "review_burden_pre": b_pre,
            "review_burden_post": b_post,
            "usefulness_score": u_useful,
            "summary_helpfulness_score": u_helpful,
            "triage_time_minutes": u_triage,
        })

    # Save final E7 CSV
    df_e7 = pd.DataFrame(participant_rows)
    if df_e7.empty:
        df_e7 = pd.DataFrame(columns=["user_id", "like_rate_pre", "like_rate_post", "review_burden_pre", "review_burden_post", "usefulness_score", "summary_helpfulness_score"])
    e7_csv_path = out_dir / "final_e7_pilot.csv"
    df_e7.to_csv(e7_csv_path, index=False)
    logger.info("Saved Final E7 Pilot table to %s", e7_csv_path)

    # Aggregate summaries
    h1_summary = {
        "hypothesis": "H1: digest like rate after vector/model activation is >= before",
        "eligible_pilot_users_tested": h1_tested_count,
        "users_supporting_h1": h1_supported_count,
        "support_rate": (h1_supported_count / h1_tested_count) if h1_tested_count > 0 else None,
        "status": "SUPPORTED" if h1_tested_count > 0 and h1_supported_count == h1_tested_count else "PENDING_OR_MIXED",
    }

    h2_summary = {
        "hypothesis": "H2: review burden (borderline tasks/cycle) does not increase after activation",
        "eligible_pilot_users_tested": h2_tested_count,
        "users_supporting_h2": h2_supported_count,
        "support_rate": (h2_supported_count / h2_tested_count) if h2_tested_count > 0 else None,
        "status": "SUPPORTED" if h2_tested_count > 0 and h2_supported_count == h2_tested_count else "PENDING_OR_MIXED",
    }

    return {
        "status": "EVALUATED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": CAUSATION_DISCLAIMER,
        "hypotheses": {
            "h1": h1_summary,
            "h2": h2_summary,
            "h3": h3_stats,
        },
        "participants_count": len(participant_rows),
        "participants": participant_rows,
        "output_csv": str(e7_csv_path),
    }


def run_final_evaluation_suite(
    db_session: Optional[Session] = None,
    pilot_export_dir: Path | str = Path("data/pilot_export"),
    survey_dir: Path | str = Path("data/surveys"),
    results_dir: Path | str = RESULTS_DIR,
    run_offline_e1: bool = True,
    run_offline_simulations: bool = False,
    allow_partial_labels: bool = False,
) -> Dict[str, Any]:
    """
    Master Phase 13 Evaluation Suite Runner.

    Executes all Phase 13 deliverables and checks readiness safeguards.
    """
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    p_dir = Path(pilot_export_dir)

    # -------------------------------------------------------------
    # Step 1: Pre-flight Readiness Check
    # -------------------------------------------------------------
    logger.info("Running Step 1: Pre-flight readiness check...")
    if db_session is not None:
        readiness = check_pilot_readiness(
            session=db_session,
            survey_dir=survey_dir,
            results_dir=res_dir,
        )
    else:
        with get_db_session() as session:
            readiness = check_pilot_readiness(
                session=session,
                survey_dir=survey_dir,
                results_dir=res_dir,
            )
    readiness_report_path = res_dir / "readiness_report.json"
    with open(readiness_report_path, "w", encoding="utf-8") as f:
        json.dump(readiness.to_dict(), f, indent=2)

    logger.info(
        "Readiness tier: %s (Can freeze: %s)",
        readiness.tier.value,
        readiness.can_freeze_final_numbers,
    )

    # -------------------------------------------------------------
    # Step 2: Pilot Data Export (with Section 8.6 exclusions)
    # -------------------------------------------------------------
    logger.info("Running Step 2: Reproducible pilot data export...")
    if db_session is not None:
        export_summary = export_pilot_dataset(
            session=db_session,
            export_dir=p_dir,
        )
    else:
        with get_db_session() as session:
            export_summary = export_pilot_dataset(
                session=session,
                export_dir=p_dir,
            )

    # -------------------------------------------------------------
    # Step 3: E1 Gate Quality on Frozen Test Split
    # -------------------------------------------------------------
    e1_results = None
    simulation_results = {"status": "NOT_REQUESTED"}
    label_coverage = {"total": 0, "labeled": 0, "missing": 0, "scope": "not_requested"}
    if run_offline_e1 or run_offline_simulations:
        try:
            sample_path = Path("data/evaluation_sample.parquet")
            sample_df = pd.read_parquet(sample_path)
            labels_df = load_latest_labels()
            if labels_df is not None and "user_id" in labels_df:
                labels_df = labels_df[labels_df["user_id"] == "developer"]
            gt_map = dict(zip(labels_df["paper_id"].astype(str), labels_df["label"])) if labels_df is not None else {}
            missing = set(sample_df["paper_id"].astype(str)) - set(gt_map)
            label_coverage = {"total": len(sample_df), "labeled": len(sample_df)-len(missing),
                "missing": len(missing), "scope": "labeled_subset" if missing else "full_sample"}
            if missing and not allow_partial_labels:
                raise BlockedExperimentError(f"{len(missing)} of {len(sample_df)} frozen sample papers lack human labels. Final evaluation is held; missing labels are not negatives.")
            if missing:
                sample_df = sample_df[sample_df["paper_id"].astype(str).isin(gt_map)].reset_index(drop=True)
            if sample_df.empty:
                raise BlockedExperimentError("No human-labeled papers are available yet; operational and feedback reports are still produced.")
            corpus_df = pd.read_parquet("data/corpus.parquet")
            all_embs = np.load("data/embeddings.npy", allow_pickle=False)
            if len(all_embs) != len(corpus_df):
                raise ValueError("Corpus and embeddings are not aligned.")
            lookup = {str(pid): i for i, pid in enumerate(corpus_df["paper_id"])}
            sample_embeddings = all_embs[[lookup[str(pid)] for pid in sample_df["paper_id"]]]
            dev_mask = (sample_df["split"] == "dev").values
            test_mask = (sample_df["split"] == "test").values
            dev_df = sample_df[dev_mask].reset_index(drop=True)
            test_df = sample_df[test_mask].reset_index(drop=True)
            dev_embeddings, test_embeddings = sample_embeddings[dev_mask], sample_embeddings[test_mask]
            seed_vector = _compute_seed_vector()
            ref_mask = corpus_df["is_reference"].values
            ref_scores = all_embs[ref_mask] @ seed_vector
            t_high, t_low = np.percentile(ref_scores, [config.active_p_high, config.active_p_low])
            corpus_texts = (corpus_df.loc[ref_mask, "title"] + " " + corpus_df.loc[ref_mask, "abstract"]).tolist()
            if not allow_partial_labels and sum(gt_map[str(pid)] == "relevant" for pid in test_df["paper_id"]) < 30:
                raise BlockedExperimentError("Frozen test split requires at least 30 relevant human labels.")
            if run_offline_e1:
                table = run_experiment_e1_gate_quality(test_df, test_embeddings, seed_vector,
                    ground_truth_labels=gt_map, dev_df=dev_df, t_high=t_high, t_low=t_low,
                    corpus_texts=corpus_texts, output_dir=res_dir)
                path = res_dir / "final_e1_gate_quality.csv"
                table["evaluation_scope"] = label_coverage["scope"]
                table["human_labeled_sample"] = label_coverage["labeled"]
                table["planned_sample"] = label_coverage["total"]
                table.to_csv(path, index=False)
                e1_results = {"status": "COMPLETED_PARTIAL" if missing else "COMPLETED", "test_sample_count": len(test_df),
                    "labeled_overlap": len(test_df), "output_csv": str(path),
                    "message": "Descriptive results on the available human-labeled subset; missing labels may introduce selection bias. These are not final full-sample claims." if missing else "Complete human-labeled sample evaluated."}
            if run_offline_simulations:
                from research_monitor.evaluation import experiments as exp
                from research_monitor.evaluation.simulation import SimulationOracle
                oracle = SimulationOracle(ground_truth_map=gt_map)
                common = dict(dev_df=dev_df, dev_embeddings=dev_embeddings, test_df=test_df,
                    test_embeddings=test_embeddings, seed_vector=seed_vector, oracle=oracle, output_dir=res_dir)
                exp.run_experiment_e2_learning_curves(**common, t_high=t_high, t_low=t_low, corpus_texts=corpus_texts)
                exp.run_experiment_e3_signal_ablation(**common, t_high=t_high, t_low=t_low)
                exp.run_experiment_e4_centroid_update(**common)
                exp.run_experiment_e5_efficiency(test_embeddings @ seed_vector, output_dir=res_dir)
                exp.run_experiment_e6_human_review_value(test_embeddings @ seed_vector,
                    np.array([int(gt_map[str(pid)] == "relevant") for pid in test_df["paper_id"]]),
                    t_high, t_low, weights=test_df["sampling_weight"].values,
                    oracle=oracle, paper_ids=test_df["paper_id"].astype(str).tolist(), output_dir=res_dir)
                simulation_results = {"status": "COMPLETED", "is_simulated": True}
        except (BlockedExperimentError, FileNotFoundError) as exc:
            result = {"status": "INSUFFICIENT_LABELED_DATA" if allow_partial_labels else "BLOCKED_INCOMPLETE_DATA", "message": str(exc)}
            e1_results = result if run_offline_e1 else None
            simulation_results = result if run_offline_simulations else simulation_results
        except Exception as exc:
            result = {"status": "ERROR", "error": str(exc)}
            e1_results = result if run_offline_e1 else None
            simulation_results = result if run_offline_simulations else simulation_results

    # -------------------------------------------------------------
    # Step 4: E7 Pilot Evaluation (Hypotheses H1-H3)
    # -------------------------------------------------------------
    logger.info("Running Step 4: E7 Pilot Evaluation...")
    e7_results = run_experiment_e7_pilot(
        pilot_export_dir=p_dir,
        survey_dir=survey_dir,
        output_dir=res_dir,
    )

    # -------------------------------------------------------------
    # Step 5: Before-and-After Adaptation Analysis
    # -------------------------------------------------------------
    logger.info("Running Step 5: Before-and-after adaptation analysis...")
    clean_records_path = p_dir / "pilot_records_clean.csv"
    if clean_records_path.exists():
        clean_df = pd.read_csv(clean_records_path)
        eligibility_info = export_summary.to_dict().get("user_eligibility", {})
        adaptation_dict = analyze_all_pilot_adaptation(clean_df, user_eligibility=eligibility_info)
        adaptation_csv_path = export_adaptation_table(adaptation_dict, res_dir)
        adaptation_results = {
            "status": "COMPLETED",
            "users_analyzed": len(adaptation_dict.get("users", [])),
            "output_csv": str(adaptation_csv_path),
            "disclaimer": CAUSATION_DISCLAIMER,
        }
    else:
        adaptation_results = {"status": "NO_CLEAN_RECORDS", "users_analyzed": 0}

    # -------------------------------------------------------------
    # Step 6: Operational Efficiency & LLM-Call Reduction
    # -------------------------------------------------------------
    logger.info("Running Step 6: Operational efficiency & LLM-call reduction...")
    if db_session is not None:
        eff_summary = calculate_operational_efficiency(session=db_session)
    else:
        with get_db_session() as session:
            eff_summary = calculate_operational_efficiency(session=session)

    eff_csv_path = export_efficiency_report(eff_summary, res_dir)
    efficiency_results = {
        "status": "COMPLETED",
        "measured_savings_status": eff_summary.measured_savings_status,
        "measured_llm_call_reduction": eff_summary.measured_llm_call_reduction,
        "theoretical_call_reduction": eff_summary.theoretical_call_reduction,
        "llm_call_reduction": eff_summary.overall_llm_call_reduction,
        "has_token_telemetry": eff_summary.has_token_telemetry,
        "actual_cost_usd": eff_summary.actual_cost_usd,
        "estimated_cost_savings_usd": eff_summary.estimated_cost_savings_usd,
        "total_review_overflow": eff_summary.total_review_overflow,
        "output_csv": str(eff_csv_path),
        "notes": eff_summary.missing_telemetry_reasons,
    }

    # -------------------------------------------------------------
    # Step 7: Master Deliverable Report
    # -------------------------------------------------------------
    numbers_frozen = readiness.can_freeze_final_numbers and bool(e1_results and e1_results.get("status") == "COMPLETED") and (not run_offline_simulations or simulation_results.get("status") == "COMPLETED")
    report = {
        "phase": 13,
        "title": "Phase 13 Pilot Completion and Final Evaluation Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness.to_dict(),
        "numbers_frozen": numbers_frozen,
        "completion_status": (
            "COMPLETE" if numbers_frozen else ("COMPLETED_WITH_AVAILABLE_DATA" if allow_partial_labels else "IN_PROGRESS_PARTIAL_DATA")
        ),
        "label_coverage": label_coverage,
        "e1_gate_quality": e1_results,
        "offline_simulations": simulation_results,
        "e7_pilot_evaluation": e7_results,
        "adaptation_analysis": adaptation_results,
        "efficiency_analysis": efficiency_results,
        "export_summary": export_summary.to_dict(),
        "notes": [
            "Final numbers are held until the pilot completes >= 7 consecutive daily cycles and validation labeling reaches target quotas.",
            CAUSATION_DISCLAIMER,
        ],
    }

    final_report_path = res_dir / "final_evaluation_report.json"
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Master Phase 13 Evaluation Report saved to %s", final_report_path)
    return report
