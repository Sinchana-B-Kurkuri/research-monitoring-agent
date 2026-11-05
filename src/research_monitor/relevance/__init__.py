"""
relevance
~~~~~~~~~
Phase 5, 10, and 11 Relevance Agent, Preference Memory, and Classifier.
"""

from importlib import import_module

from .classifier import (
    ClassifierEligibility,
    ClassifierTrainingResult,
    DecisionExplanation,
    calibrate_classifier_thresholds,
    check_classifier_eligibility,
    find_nearest_feedback_papers,
    train_and_evaluate_classifier,
)
from .preference import (
    VectorUpdateEligibility,
    VectorUpdateResult,
    check_vector_update_eligibility,
    compute_rocchio_centroid,
    get_user_seed_embedding,
    update_user_interest_vector,
)
from .router import RoutingResult, route_score
from .state import DecisionStatus, PaperCandidate, PipelineState, Route
from .thresholds import Thresholds, compute_vector_thresholds, load_reference_embeddings
from .user_manager import create_user, get_active_vector_version

_PIPELINE_EXPORTS = {
    "discover_stage", "embed_stage", "preprocess_stage",
    "route_stage", "run_monitoring_cycle", "score_stage",
}


def __getattr__(name):
    """Load ingestion dependencies only in callers using the pipeline."""
    if name not in _PIPELINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".pipeline", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Route",
    "DecisionStatus",
    "PaperCandidate",
    "PipelineState",
    "RoutingResult",
    "route_score",
    "Thresholds",
    "compute_vector_thresholds",
    "load_reference_embeddings",
    "create_user",
    "get_active_vector_version",
    "run_monitoring_cycle",
    "discover_stage",
    "preprocess_stage",
    "embed_stage",
    "score_stage",
    "route_stage",
    "compute_rocchio_centroid",
    "check_vector_update_eligibility",
    "update_user_interest_vector",
    "VectorUpdateEligibility",
    "VectorUpdateResult",
    "get_user_seed_embedding",
    "check_classifier_eligibility",
    "train_and_evaluate_classifier",
    "calibrate_classifier_thresholds",
    "find_nearest_feedback_papers",
    "ClassifierEligibility",
    "ClassifierTrainingResult",
    "DecisionExplanation",
]
