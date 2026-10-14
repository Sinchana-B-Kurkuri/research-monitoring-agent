"""
relevance
~~~~~~~~~
Phase 5 Relevance Agent and Explicit State Machine.
"""

from .pipeline import (
    discover_stage,
    embed_stage,
    preprocess_stage,
    route_stage,
    run_monitoring_cycle,
    score_stage,
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
]
