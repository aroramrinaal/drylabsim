"""Biological scoring: markers, mechanisms, pathways, and expert clone claims."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

try:
    from ..biology.gene_index import (
        marker_set_score,
        mechanism_set_score,
        score_pathways,
    )
except ImportError:
    from server.biology.gene_index import (
        marker_set_score,
        mechanism_set_score,
        score_pathways,
    )

try:
    from ...models import ClonalClaim, ConclusionClaim
    from ..simulator.latent_state import FullLatentState
except ImportError:
    from models import ClonalClaim, ConclusionClaim
    from server.simulator.latent_state import FullLatentState


_MULTICLONE_EXPERT_SCENARIO = "venetoclax_resistance_multiclone"


def _dedupe_nonempty(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        norm = str(item).strip().upper()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(str(item).strip())
    return deduped


def _marker_precision_factor(predicted: List[str], truth: List[str]) -> float:
    deduped_pred = _dedupe_nonempty(predicted)
    if not deduped_pred:
        return 0.0
    return min(1.0, (len(truth) * 2.0) / max(len(deduped_pred), 1))


def _confounder_penalty(
    predicted_pathways: Dict[str, float],
    confounders: Dict[str, float],
) -> float:
    if not predicted_pathways or not confounders:
        return 0.0

    confounder_names = {name.strip().lower() for name in confounders}
    overlap = sum(
        1 for pathway in predicted_pathways if pathway.strip().lower() in confounder_names
    )
    return min(0.5, 0.15 * overlap)


def _pathway_dict(pathways: Iterable[str]) -> Dict[str, float]:
    return {pathway: 1.0 for pathway in _dedupe_nonempty(pathways)}


def _size_estimate_score(estimate: Any, truth_size: float, sigma: float = 0.10) -> float:
    try:
        if estimate is None:
            return 0.0
        pred = float(estimate)
    except (TypeError, ValueError):
        return 0.0
    if pred < 0.0:
        return 0.0
    return float(math.exp(-((pred - truth_size) ** 2) / (2.0 * sigma**2)))


def _collect_predictions(
    discovered_markers: List[str],
    candidate_mechanisms: List[str],
    conclusions: List["ConclusionClaim"],
) -> Tuple[List[str], List[str], Dict[str, float]]:
    pred_markers: List[str] = []
    pred_mechs: List[str] = []
    pred_pathways: Dict[str, float] = {}

    if conclusions:
        for conclusion in conclusions:
            pred_markers.extend(conclusion.top_markers)
            pred_mechs.extend(conclusion.causal_mechanisms)
            pred_pathways.update(conclusion.predicted_pathways)
            for clonal_claim in conclusion.clonal_claims:
                pred_markers.extend(clonal_claim.markers)
                if clonal_claim.mechanism:
                    pred_mechs.append(clonal_claim.mechanism)
                pred_pathways.update(_pathway_dict(clonal_claim.supporting_pathways))

    if not pred_markers and discovered_markers:
        pred_markers = list(discovered_markers)
    if not pred_mechs and candidate_mechanisms:
        pred_mechs = list(candidate_mechanisms)

    return (
        _dedupe_nonempty(pred_markers),
        _dedupe_nonempty(pred_mechs),
        pred_pathways,
    )


def _base_biology_score(
    state: FullLatentState,
    pred_markers: List[str],
    pred_mechs: List[str],
    pred_pathways: Dict[str, float],
) -> float:
    true_markers = state.biology.true_markers
    true_mechanisms = state.biology.causal_mechanisms
    true_pathways = state.biology.true_pathways

    m_score = 0.0
    if true_markers:
        marker_recall = marker_set_score(pred_markers, true_markers)
        m_score = marker_recall * _marker_precision_factor(pred_markers, true_markers)
    mech_score = (
        mechanism_set_score(pred_mechs, true_mechanisms) if true_mechanisms else 0.0
    )
    pw_score = score_pathways(pred_pathways, true_pathways) if true_pathways else 0.0

    has_markers = bool(true_markers)
    has_mechs = bool(true_mechanisms)
    has_pathways = bool(true_pathways)

    total_weight = 0.0
    weighted = 0.0

    if has_markers:
        weighted += 0.40 * m_score
        total_weight += 0.40
    if has_mechs:
        weighted += 0.35 * mech_score
        total_weight += 0.35
    if has_pathways:
        weighted += 0.25 * pw_score
        total_weight += 0.25

    if total_weight == 0.0:
        return 0.0

    score = weighted / total_weight
    score -= _confounder_penalty(pred_pathways, state.biology.confounders)
    return max(0.0, min(1.0, score))


def _resistant_truth_clones(state: FullLatentState) -> Dict[str, Dict[str, Any]]:
    return {
        name: truth
        for name, truth in state.biology.clone_truth.items()
        if truth.get("is_resistant", True)
    }


def _all_truth_markers(state: FullLatentState) -> Dict[str, set[str]]:
    return {
        name: {marker.upper() for marker in truth.get("markers", [])}
        for name, truth in state.biology.clone_truth.items()
    }


def _extract_clonal_claims(
    conclusions: List["ConclusionClaim"],
) -> Tuple[List["ClonalClaim"], Dict[str, float]]:
    clonal_claims: List[ClonalClaim] = []
    clone_size_estimates: Dict[str, float] = {}
    for conclusion in conclusions:
        clonal_claims.extend(conclusion.clonal_claims)
        clone_size_estimates.update(conclusion.clone_size_estimates)
    return clonal_claims, clone_size_estimates


def _match_clonal_claims(
    clonal_claims: List["ClonalClaim"],
    truth_clones: Dict[str, Dict[str, Any]],
) -> Dict[str, ClonalClaim]:
    assignments: Dict[str, ClonalClaim] = {}
    remaining_truth = set(truth_clones)
    ordered_claims = sorted(
        clonal_claims,
        key=lambda claim: len(_dedupe_nonempty(claim.markers)),
        reverse=True,
    )

    for claim in ordered_claims:
        best_name = None
        best_score = 0.0
        claim_markers = _dedupe_nonempty(claim.markers)
        claim_pathways = _pathway_dict(claim.supporting_pathways)
        for truth_name in remaining_truth:
            truth = truth_clones[truth_name]
            marker_score = marker_set_score(claim_markers, truth.get("markers", []))
            pathway_score = score_pathways(claim_pathways, truth.get("pathways", {}))
            combined = 0.7 * marker_score + 0.3 * pathway_score
            if combined > best_score:
                best_name = truth_name
                best_score = combined
        if best_name is not None and best_score > 0.15:
            assignments[best_name] = claim
            remaining_truth.remove(best_name)

    return assignments


def _cross_contamination_penalty(
    state: FullLatentState,
    matched_truth_name: str,
    claim: "ClonalClaim",
) -> float:
    predicted = {marker.upper() for marker in claim.markers}
    if not predicted:
        return 0.0

    all_truth_markers = _all_truth_markers(state)
    matched_markers = all_truth_markers.get(matched_truth_name, set())
    off_target = set()
    non_resistant_hit = False

    for truth_name, truth_markers in all_truth_markers.items():
        if truth_name == matched_truth_name:
            continue
        overlap = predicted & truth_markers
        if not overlap:
            continue
        off_target.update(overlap)
        if not state.biology.clone_truth.get(truth_name, {}).get("is_resistant", True):
            non_resistant_hit = True

    penalty = 0.0
    if len(off_target) >= 2:
        penalty += 0.15
    if non_resistant_hit:
        penalty += 0.15
    return min(0.30, penalty)


def _score_multiclone_expert(
    state: FullLatentState,
    clonal_claims: List["ClonalClaim"],
    clone_size_estimates: Dict[str, float],
) -> float:
    truth_clones = _resistant_truth_clones(state)
    if not truth_clones:
        return 0.0

    assignments = _match_clonal_claims(clonal_claims, truth_clones)
    if not assignments:
        return 0.0

    marker_scores: List[float] = []
    mechanism_scores: List[float] = []
    pathway_scores: List[float] = []
    size_scores: List[float] = []
    penalties = 0.0

    for truth_name, truth in truth_clones.items():
        claim = assignments.get(truth_name)
        if claim is None:
            marker_scores.append(0.0)
            mechanism_scores.append(0.0)
            pathway_scores.append(0.0)
            size_scores.append(0.0)
            continue

        truth_markers = truth.get("markers", [])
        claim_markers = _dedupe_nonempty(claim.markers)
        marker_score = marker_set_score(claim_markers, truth_markers)
        marker_score *= _marker_precision_factor(claim_markers, truth_markers)
        marker_scores.append(marker_score)

        mechanism_scores.append(
            mechanism_set_score(
                [claim.mechanism] if claim.mechanism else [],
                [truth.get("mechanism", "")],
            )
        )
        pathway_scores.append(
            score_pathways(
                _pathway_dict(claim.supporting_pathways),
                truth.get("pathways", {}),
            )
        )
        size_scores.append(
            _size_estimate_score(
                clone_size_estimates.get(claim.subpopulation_id),
                float(truth.get("size", 0.0)),
            )
        )
        penalties += _cross_contamination_penalty(state, truth_name, claim)

    structured = (
        0.30 * (sum(marker_scores) / len(marker_scores))
        + 0.35 * (sum(mechanism_scores) / len(mechanism_scores))
        + 0.20 * (sum(pathway_scores) / len(pathway_scores))
        + 0.15 * (sum(size_scores) / len(size_scores))
    )
    structured -= penalties
    return max(0.0, min(1.0, structured))


def score_biology(
    state: FullLatentState,
    discovered_markers: List[str],
    candidate_mechanisms: List[str],
    conclusions: List["ConclusionClaim"],
) -> float:
    """Composite biological accuracy score in [0.0, 1.0]."""
    pred_markers, pred_mechs, pred_pathways = _collect_predictions(
        discovered_markers,
        candidate_mechanisms,
        conclusions,
    )
    score = _base_biology_score(state, pred_markers, pred_mechs, pred_pathways)

    if (
        state.scenario_name == _MULTICLONE_EXPERT_SCENARIO
        and state.biology.clone_truth
    ):
        clonal_claims, clone_size_estimates = _extract_clonal_claims(conclusions)
        if not clonal_claims:
            score = min(score, 0.30)
        else:
            structured_score = _score_multiclone_expert(
                state,
                clonal_claims,
                clone_size_estimates,
            )
            score = 0.25 * score + 0.75 * structured_score

    if conclusions and any(
        conclusion.top_markers or any(claim.markers for claim in conclusion.clonal_claims)
        for conclusion in conclusions
    ):
        if not state.progress.markers_validated:
            score *= 0.7

    return max(0.0, min(1.0, score))
