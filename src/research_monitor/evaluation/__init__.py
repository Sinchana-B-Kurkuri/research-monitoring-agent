"""
Evaluation and Threshold Calibration Package (Phase 4).
"""

from .access import get_test_access_history, log_test_access
from .baselines import score_embedding_b2, score_random_b0, score_tfidf_b1
from .calibration import (
    CalibrationResult,
    calibrate_thresholds,
    load_calibration_result,
    save_calibration_result,
    sensitivity_analysis,
)
from .labeling import (
    VALID_CANONICAL_LABELS,
    export_labels_to_csv,
    get_label_statistics,
    load_latest_labels,
    normalize_label,
    record_validation_label,
)
from .metrics import (
    binary_map_labels,
    borderline_composition,
    low_miss_rate,
    stratified_bootstrap_ci,
    weighted_f1,
    weighted_pr_auc,
    weighted_precision,
    weighted_recall,
)
from .sampling import (
    DEFAULT_STRATA_TARGETS,
    ReferencePercentiles,
    assign_strata,
    compute_reference_percentiles,
    draw_stratified_evaluation_sample,
    split_strata_dev_test,
    verify_sampling_weights,
)

__all__ = [
    "ReferencePercentiles",
    "compute_reference_percentiles",
    "assign_strata",
    "draw_stratified_evaluation_sample",
    "split_strata_dev_test",
    "verify_sampling_weights",
    "DEFAULT_STRATA_TARGETS",
    "VALID_CANONICAL_LABELS",
    "normalize_label",
    "record_validation_label",
    "export_labels_to_csv",
    "load_latest_labels",
    "get_label_statistics",
    "log_test_access",
    "get_test_access_history",
    "score_random_b0",
    "score_tfidf_b1",
    "score_embedding_b2",
    "binary_map_labels",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "weighted_pr_auc",
    "low_miss_rate",
    "borderline_composition",
    "stratified_bootstrap_ci",
    "CalibrationResult",
    "calibrate_thresholds",
    "sensitivity_analysis",
    "save_calibration_result",
    "load_calibration_result",
]
