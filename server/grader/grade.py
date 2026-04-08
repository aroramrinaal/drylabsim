"""Main grading entry point.

grade_episode() is a pure, deterministic function that takes the final
episode observation and hidden latent state, then returns a GradeResult
with score guaranteed to be in [0.0, 1.0].

This is the function judges call to evaluate an agent run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .biology import score_biology
from .pipeline import score_pipeline
from .types import GradeResult

if TYPE_CHECKING:
    from ...models import ClonalClaim, ConclusionClaim, ExperimentObservation, PipelineStepRecord
    from ..simulator.latent_state import FullLatentState

_W_PIPELINE = 0.30
_W_BIOLOGY = 0.55
_W_EFFICIENCY = 0.15
_MULTICLONE_EXPERT_SCENARIO = "venetoclax_resistance_multiclone"
_EXPERT_MIN_VALIDATIONS = 2


def _successful_action_counts(
    pipeline_history: List["PipelineStepRecord"],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for step in pipeline_history:
        if not step.success:
            continue
        action_name = step.action_type.value
        counts[action_name] = counts.get(action_name, 0) + 1
    return counts


def _extract_clonal_claims(
    conclusions: List["ConclusionClaim"],
) -> List["ClonalClaim"]:
    clonal_claims: List["ClonalClaim"] = []
    for conclusion in conclusions:
        clonal_claims.extend(conclusion.clonal_claims)
    return clonal_claims


def _normalized_markers(markers: List[str]) -> set[str]:
    return {marker.strip().upper() for marker in markers if str(marker).strip()}


def _normalized_pathways(pathways: List[str]) -> set[str]:
    return {pathway.strip().lower() for pathway in pathways if str(pathway).strip()}


def _matched_resistant_truths(
    latent: "FullLatentState",
    clonal_claims: List["ClonalClaim"],
) -> Dict[str, "ClonalClaim"]:
    truth_clones = {
        name: truth
        for name, truth in latent.biology.clone_truth.items()
        if truth.get("is_resistant", True)
    }
    matched: Dict[str, "ClonalClaim"] = {}
    remaining_truth = set(truth_clones)

    for claim in sorted(clonal_claims, key=lambda item: len(item.markers), reverse=True):
        claim_markers = _normalized_markers(claim.markers)
        claim_pathways = _normalized_pathways(claim.supporting_pathways)
        best_truth = None
        best_marker_overlap = 0
        best_pathway_overlap = 0
        for truth_name in remaining_truth:
            truth = truth_clones[truth_name]
            truth_markers = _normalized_markers(truth.get("markers", []))
            truth_pathways = _normalized_pathways(list(truth.get("pathways", {}).keys()))
            marker_overlap = len(claim_markers & truth_markers)
            pathway_overlap = len(claim_pathways & truth_pathways)
            combined_score = 2 * marker_overlap + pathway_overlap
            best_combined = 2 * best_marker_overlap + best_pathway_overlap
            if combined_score > best_combined:
                best_truth = truth_name
                best_marker_overlap = marker_overlap
                best_pathway_overlap = pathway_overlap
        if best_truth is not None and (
            best_marker_overlap >= 2
            or (best_marker_overlap >= 1 and best_pathway_overlap >= 1)
        ):
            matched[best_truth] = claim
            remaining_truth.remove(best_truth)

    return matched


def _distinct_clonal_mechanisms(clonal_claims: List["ClonalClaim"]) -> int:
    mechanisms = {
        claim.mechanism.strip().lower()
        for claim in clonal_claims
        if claim.mechanism.strip()
    }
    return len(mechanisms)


def _promotes_distractor_as_resistance(
    latent: "FullLatentState",
    clonal_claims: List["ClonalClaim"],
    matched_truths: Dict[str, "ClonalClaim"],
) -> bool:
    non_resistant_truths = [
        truth
        for truth in latent.biology.clone_truth.values()
        if not truth.get("is_resistant", True)
    ]
    if not non_resistant_truths:
        return False

    distractor_markers = set()
    distractor_pathways = set()
    distractor_mechanisms = set()
    for truth in non_resistant_truths:
        distractor_markers.update(_normalized_markers(truth.get("markers", [])))
        distractor_pathways.update(
            _normalized_pathways(list(truth.get("pathways", {}).keys()))
        )
        mechanism = str(truth.get("mechanism", "")).strip().lower()
        if mechanism:
            distractor_mechanisms.add(mechanism)

    for truth_name, claim in matched_truths.items():
        claim_markers = _normalized_markers(claim.markers)
        claim_pathways = _normalized_pathways(claim.supporting_pathways)
        mechanism = claim.mechanism.strip().lower()
        if len(claim_markers & distractor_markers) >= 2:
            return True
        if len(claim_pathways & distractor_pathways) >= 1:
            return True
        if mechanism and mechanism in distractor_mechanisms:
            return True

        truth_markers = _normalized_markers(
            latent.biology.clone_truth.get(truth_name, {}).get("markers", [])
        )
        if not (claim_markers & truth_markers):
            return True

    return False


def _apply_expert_terminal_caps(
    score: float,
    obs: "ExperimentObservation",
    latent: "FullLatentState",
) -> Tuple[float, Dict[str, Any]]:
    if latent.scenario_name != _MULTICLONE_EXPERT_SCENARIO:
        return score, {}

    clonal_claims = _extract_clonal_claims(obs.conclusions)
    matched_truths = _matched_resistant_truths(latent, clonal_claims)
    action_counts = _successful_action_counts(obs.pipeline_history)
    validation_count = action_counts.get("validate_marker", 0)
    cap = 1.0
    penalty = 0.0
    applied_caps: Dict[str, float] = {}

    if not clonal_claims:
        cap = min(cap, 0.15)
        applied_caps["flat_conclusion_cap"] = 0.15

    if len(matched_truths) < 2:
        cap = min(cap, 0.25)
        applied_caps["matched_clone_cap"] = 0.25

    if _distinct_clonal_mechanisms(clonal_claims) < 2:
        cap = min(cap, 0.25)
        applied_caps["mechanism_diversity_cap"] = 0.25

    if validation_count < _EXPERT_MIN_VALIDATIONS or not latent.progress.markers_validated:
        cap = min(cap, 0.20)
        applied_caps["validation_cap"] = 0.20

    if latent.technical.batch_effects and not latent.progress.batches_integrated:
        cap = min(cap, 0.20)
        applied_caps["batch_integration_cap"] = 0.20

    if not latent.progress.trajectories_inferred or not latent.progress.networks_inferred:
        cap = min(cap, 0.15)
        applied_caps["branch_resolution_cap"] = 0.15

    if _promotes_distractor_as_resistance(latent, clonal_claims, matched_truths):
        penalty += 0.25

    adjusted = max(0.0, min(1.0, min(score, cap) - penalty))
    return adjusted, {
        "expert_cap": cap,
        "expert_penalty": penalty,
        "expert_validation_count": validation_count,
        "expert_matched_clone_claims": len(matched_truths),
        "expert_distinct_mechanisms": _distinct_clonal_mechanisms(clonal_claims),
        "expert_applied_caps": applied_caps,
    }


def grade_episode(
    obs: "ExperimentObservation",
    latent: "FullLatentState",
) -> GradeResult:
    """Grade a completed episode. Pure function — deterministic and reproducible.

    Args:
        obs: The final ExperimentObservation from the environment.
        latent: The FullLatentState (hidden ground truth) for this episode.

    Returns:
        GradeResult with score in [0.0, 1.0].
    """
    completeness = score_pipeline(latent)

    biology = score_biology(
        state=latent,
        discovered_markers=obs.discovered_markers,
        candidate_mechanisms=obs.candidate_mechanisms,
        conclusions=obs.conclusions,
    )

    res = latent.resources
    budget_eff = res.budget_remaining / max(res.budget_total, 1.0)
    time_eff = res.time_remaining_days / max(res.time_limit_days, 1.0)
    efficiency = 0.5 * budget_eff + 0.5 * time_eff

    score = (
        _W_PIPELINE * completeness + _W_BIOLOGY * biology + _W_EFFICIENCY * efficiency
    )
    score, expert_breakdown = _apply_expert_terminal_caps(score, obs, latent)

    return GradeResult(
        score=score,
        completeness=completeness,
        biology_score=biology,
        efficiency_score=efficiency,
        breakdown={
            "completeness": completeness,
            "biology_score": biology,
            "efficiency_score": efficiency,
            "weight_pipeline": _W_PIPELINE,
            "weight_biology": _W_BIOLOGY,
            "weight_efficiency": _W_EFFICIENCY,
            **expert_breakdown,
        },
    )
