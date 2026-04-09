"""Redundancy checks — prevent repeating completed steps."""

from __future__ import annotations

from typing import List

try:
    from ...models import ActionType, ExperimentAction
    from ..simulator.latent_state import FullLatentState
except ImportError:  # pragma: no cover
    from models import ActionType, ExperimentAction
    from server.simulator.latent_state import FullLatentState

from .types import RuleViolation, Severity

_REDUNDANT = {
    ActionType.COLLECT_SAMPLE: "samples_collected",
    ActionType.PREPARE_LIBRARY: "library_prepared",
    ActionType.SEQUENCE_CELLS: "cells_sequenced",
    ActionType.RUN_QC: "qc_performed",
    ActionType.FILTER_DATA: "data_filtered",
    ActionType.NORMALIZE_DATA: "data_normalized",
    ActionType.CLUSTER_CELLS: "cells_clustered",
    ActionType.DIFFERENTIAL_EXPRESSION: "de_performed",
    ActionType.TRAJECTORY_ANALYSIS: "trajectories_inferred",
    ActionType.PATHWAY_ENRICHMENT: "pathways_analyzed",
    ActionType.REGULATORY_NETWORK_INFERENCE: "networks_inferred",
    ActionType.MARKER_SELECTION: "markers_discovered",
    ActionType.DESIGN_FOLLOWUP: "followup_designed",
    ActionType.REQUEST_SUBAGENT_REVIEW: "subagent_review_requested",
    ActionType.SYNTHESIZE_CONCLUSION: "conclusion_reached",
}


def _marker_pair_key(action: ExperimentAction) -> str:
    marker = str(action.parameters.get("marker", "")).strip()
    subpop = str(action.parameters.get("subpopulation_id", "")).strip()
    return f"{marker}::{subpop}" if subpop else marker


def check_redundancy(
    action: ExperimentAction, s: FullLatentState
) -> List[RuleViolation]:
    vs: List[RuleViolation] = []
    at = action.action_type
    p = s.progress

    flag = _REDUNDANT.get(at)
    if flag and getattr(p, flag, False):
        vs.append(
            RuleViolation(
                rule_id=f"redundant_{at.value}",
                severity=Severity.HARD,
                message=f"Step '{at.value}' already completed — redundant action blocked",
            )
        )

    if at == ActionType.VALIDATE_MARKER and p.markers_validated:
        pair_key = _marker_pair_key(action)
        if pair_key in s.validated_marker_pairs:
            vs.append(
                RuleViolation(
                    rule_id="redundant_validate_marker_pair",
                    severity=Severity.HARD,
                    message=(
                        f"Marker '{action.parameters.get('marker', '')}' in "
                        f"subpopulation '{action.parameters.get('subpopulation_id', '')}' "
                        f"already validated — redundant action blocked"
                    ),
                )
            )

    return vs
