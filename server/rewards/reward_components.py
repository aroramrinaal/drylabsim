"""Helper functions for reward computation."""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from ...models import (
        ActionType,
        ConclusionClaim,
        ExperimentAction,
        IntermediateOutput,
        META_ACTIONS,
        TOOL_REGISTRY,
        WET_LAB_ACTIONS,
    )
    from ..biology.gene_index import (
        marker_set_score,
        mechanism_set_score,
        score_pathways,
    )
    from ..simulator.latent_state import FullLatentState
except ImportError:  # pragma: no cover - direct module import path
    from models import (
        ActionType,
        ConclusionClaim,
        ExperimentAction,
        IntermediateOutput,
        META_ACTIONS,
        TOOL_REGISTRY,
        WET_LAB_ACTIONS,
    )
    from server.biology.gene_index import (
        marker_set_score,
        mechanism_set_score,
        score_pathways,
    )
    from server.simulator.latent_state import FullLatentState


# Mapping from method strings to tool names
_METHOD_TO_TOOL: Dict[str, str] = {
    "scanpy.pp.calculate_qc_metrics": "Scanpy",
    "scanpy.pp.filter_cells": "Scanpy",
    "scanpy.pp.filter_genes": "Scanpy",
    "scanpy.pp.normalize_total": "Scanpy",
    "scanpy.pp.log1p": "Scanpy",
    "scanpy.pp.highly_variable_genes": "Scanpy",
    "scanpy.pp.neighbors": "Scanpy",
    "scanpy.tl.leiden": "Leiden",
    "scanpy.tl.louvain": "Louvain",
    "scanpy.tl.rank_genes_groups": "Scanpy",
    "scanpy.tl.paga": "PAGA",
    "scanpy.tl.umap": "UMAP",
    "gseapy.prerank": "Scanpy",
    "gseapy.gsea": "Scanpy",
    "10x_chromium": "CellRanger",
    "NovaSeq": "CellRanger",
}


def _strict_mechanism_credit(score: float) -> float:
    if score <= 0.4:
        return 0.0
    if score >= 0.7:
        return 1.0
    return (score - 0.4) / 0.3


def ordering_score(action: ExperimentAction, s: FullLatentState) -> float:
    """Heuristic: 1.0 if natural next, 0.3 if acceptable, -1.0 if premature."""
    at = action.action_type
    p = s.progress
    NATURAL_NEXT = {
        ActionType.COLLECT_SAMPLE: not p.samples_collected,
        ActionType.PREPARE_LIBRARY: p.samples_collected and not p.library_prepared,
        ActionType.SEQUENCE_CELLS: p.library_prepared and not p.cells_sequenced,
        ActionType.RUN_QC: p.cells_sequenced and not p.qc_performed,
        ActionType.FILTER_DATA: p.qc_performed and not p.data_filtered,
        ActionType.NORMALIZE_DATA: p.data_filtered and not p.data_normalized,
        ActionType.CLUSTER_CELLS: p.data_normalized and not p.cells_clustered,
        ActionType.DIFFERENTIAL_EXPRESSION: p.data_normalized and not p.de_performed,
        ActionType.PATHWAY_ENRICHMENT: p.de_performed and not p.pathways_analyzed,
        ActionType.MARKER_SELECTION: p.de_performed and not p.markers_discovered,
        ActionType.VALIDATE_MARKER: p.markers_discovered and not p.markers_validated,
        ActionType.SYNTHESIZE_CONCLUSION: (p.de_performed or p.cells_clustered)
        and not p.conclusion_reached,
    }
    if NATURAL_NEXT.get(at, False):
        return 1.0

    has_evidence = any(
        [
            p.cells_clustered,
            p.de_performed,
            p.trajectories_inferred,
            p.pathways_analyzed,
            p.networks_inferred,
            p.markers_discovered,
        ]
    )
    if at in META_ACTIONS and not has_evidence:
        return -1.0

    return 0.3


def potential(s: FullLatentState) -> float:
    """Progress potential φ(s) — counts completed milestones.

    Returns 0.0 at terminal states so that the shaping signal
    telescopes correctly over the episode.
    """
    if s.progress.conclusion_reached:
        return 0.0
    p = s.progress
    milestones = [
        p.samples_collected,
        p.library_prepared,
        p.cells_sequenced,
        p.qc_performed,
        p.data_filtered,
        p.data_normalized,
        p.cells_clustered,
        p.de_performed,
        p.pathways_analyzed,
        p.markers_discovered,
        p.markers_validated,
        p.conclusion_reached,
    ]
    return sum(milestones) / len(milestones)


def completeness(s: FullLatentState) -> float:
    p = s.progress
    core = [
        p.samples_collected,
        p.cells_sequenced,
        p.qc_performed,
        p.data_filtered,
        p.data_normalized,
        p.de_performed or p.cells_clustered,
        p.conclusion_reached,
    ]
    return sum(core) / len(core)


def calibration(s: FullLatentState, conclusions: List[ConclusionClaim]) -> float:
    """Structured set-similarity calibration against hidden ground truth.

    Uses pathway-weighted Gaussian similarity for markers, semantic
    similarity for mechanisms, and activity-weighted matching for pathways.
    Falls back to legacy substring matching when structured fields are empty.
    """
    if not conclusions:
        return 0.0

    pred_markers = [g for c in conclusions for g in c.top_markers]
    pred_mechs = [m for c in conclusions for m in c.causal_mechanisms]
    pred_pathways = {p: v for c in conclusions for p, v in c.predicted_pathways.items()}
    mech_conf = {
        mech: float(conf)
        for c in conclusions
        for mech, conf in c.mechanism_confidence.items()
    }

    has_structured = bool(pred_markers or pred_mechs or pred_pathways)

    if has_structured:
        m_score = marker_set_score(pred_markers, s.biology.true_markers)
        mech_score = mechanism_set_score(pred_mechs, s.biology.causal_mechanisms)
        pw_score = score_pathways(pred_pathways, s.biology.true_pathways)
        calibrated_mech_score = _strict_mechanism_credit(mech_score)
        if mech_conf and s.biology.causal_mechanisms:
            confidence_penalties: List[float] = []
            truth_lower = [m.lower() for m in s.biology.causal_mechanisms]
            for mech, conf in mech_conf.items():
                mech_lower = mech.lower()
                is_truth_like = any(
                    mech_lower in t or t in mech_lower for t in truth_lower
                )
                confidence_penalties.append(
                    1.0 - abs(float(conf) - (1.0 if is_truth_like else 0.0))
                )
            if confidence_penalties:
                calibrated_mech_score = 0.7 * calibrated_mech_score + 0.3 * (
                    sum(confidence_penalties) / len(confidence_penalties)
                )
        return 0.50 * m_score + 0.35 * calibrated_mech_score + 0.15 * pw_score

    return legacy_calibration(s, conclusions)


def legacy_calibration(s: FullLatentState, conclusions: List[ConclusionClaim]) -> float:
    """Substring-based calibration kept for backward compatibility."""
    true_mechanisms = set(s.biology.causal_mechanisms)
    true_markers = set(s.biology.true_markers)
    score = 0.0
    n = len(conclusions)

    for c in conclusions:
        claim_lower = c.claim.lower()
        match = any(m.lower() in claim_lower for m in true_mechanisms)
        marker_match = any(m.lower() in claim_lower for m in true_markers)
        if match or marker_match:
            score += 1.0
        else:
            score -= 0.3
    return max(0.0, min(1.0, score / max(n, 1)))


def tool_fit_score(action: ExperimentAction, s: FullLatentState) -> float:
    """Score how well the chosen tool matches the task modality.

    Returns +1.0 for a perfect match, 0.0 if no tool specified,
    -1.0 for a known tool used on an incompatible modality.
    """
    method = action.method
    if not method:
        return 0.0
    resolved = _METHOD_TO_TOOL.get(method, method)
    tool_spec = TOOL_REGISTRY.get(resolved)
    if tool_spec is None:
        return -0.5
    modality = getattr(s, "task_modality", None)
    if not modality or not tool_spec.modalities:
        return 0.0
    if modality in tool_spec.modalities:
        return 1.0
    return -1.0


def overconfidence_penalty(
    s: FullLatentState, conclusions: List[ConclusionClaim]
) -> float:
    """Penalise high-confidence claims that disagree with ground truth.

    Checks structured fields (top_markers, causal_mechanisms) first;
    falls back to claim substring matching for backward compatibility.
    """
    penalty = 0.0
    true_markers_lower = {m.lower() for m in s.biology.true_markers}
    true_mechs_lower = {m.lower() for m in s.biology.causal_mechanisms}
    true_set = true_markers_lower | true_mechs_lower

    for c in conclusions:
        if c.mechanism_confidence:
            for mech, conf in c.mechanism_confidence.items():
                if conf <= 0.8:
                    continue
                mech_lower = mech.lower()
                is_correct = any(
                    mech_lower in t.lower() or t.lower() in mech_lower
                    for t in s.biology.causal_mechanisms
                )
                if not is_correct:
                    penalty -= 0.5 * float(conf)

        if c.confidence <= 0.8:
            continue

        has_structured = bool(c.top_markers or c.causal_mechanisms)
        if has_structured:
            marker_hit = any(
                g.upper().strip() in {m.upper() for m in s.biology.true_markers}
                for g in c.top_markers
            )
            mech_hit = any(
                any(kw in m.lower() for kw in t.lower().split())
                for m in c.causal_mechanisms
                for t in s.biology.causal_mechanisms
            )
            is_correct = marker_hit or mech_hit
        else:
            is_correct = any(t in c.claim.lower() for t in true_set)

        if not is_correct:
            penalty -= 0.5 * c.confidence

    return penalty


def discovery_alignment(
    s: FullLatentState,
    discovered_markers: List[str],
    candidate_mechanisms: List[str],
) -> float:
    """Symmetric end-of-episode similarity for discovered biology.

    Forward scoring measures recall against hidden truth. Reverse scoring
    measures how well the agent's discoveries map back onto real biology,
    which penalizes extra hallucinated markers or mechanisms.
    """
    components: List[float] = []

    if s.biology.true_markers or discovered_markers:
        marker_recall = marker_set_score(
            discovered_markers,
            s.biology.true_markers,
        )
        marker_precision = marker_set_score(
            s.biology.true_markers,
            discovered_markers,
        )
        components.append((marker_recall + marker_precision) / 2.0)

    if s.biology.causal_mechanisms or candidate_mechanisms:
        mechanism_recall = mechanism_set_score(
            candidate_mechanisms,
            s.biology.causal_mechanisms,
        )
        mechanism_precision = mechanism_set_score(
            s.biology.causal_mechanisms,
            candidate_mechanisms,
        )
        components.append((mechanism_recall + mechanism_precision) / 2.0)

    if not components:
        return 1.0
    return sum(components) / len(components)


def conclusion_alignment(
    s: FullLatentState,
    conclusions: List[ConclusionClaim],
) -> float:
    if not conclusions:
        return 0.0

    pred_markers = [
        marker for conclusion in conclusions for marker in conclusion.top_markers
    ]
    pred_mechanisms = [
        mechanism
        for conclusion in conclusions
        for mechanism in conclusion.causal_mechanisms
    ]

    if not pred_markers and not pred_mechanisms:
        return legacy_calibration(s, conclusions)

    components: List[float] = []
    if s.biology.true_markers or pred_markers:
        marker_recall = marker_set_score(pred_markers, s.biology.true_markers)
        marker_precision = marker_set_score(s.biology.true_markers, pred_markers)
        components.append((marker_recall + marker_precision) / 2.0)

    if s.biology.causal_mechanisms or pred_mechanisms:
        mechanism_recall = mechanism_set_score(
            pred_mechanisms,
            s.biology.causal_mechanisms,
        )
        mechanism_precision = mechanism_set_score(
            s.biology.causal_mechanisms,
            pred_mechanisms,
        )
        components.append((mechanism_recall + mechanism_precision) / 2.0)

    if not components:
        return 1.0
    return sum(components) / len(components)
