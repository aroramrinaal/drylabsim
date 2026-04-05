"""Prerequisite checks for biological experiment actions."""

from __future__ import annotations

from typing import List

try:
    from ...models import ActionType, ExperimentAction
    from ..simulator.latent_state import FullLatentState
except ImportError:  # pragma: no cover
    from models import ActionType, ExperimentAction
    from server.simulator.latent_state import FullLatentState

from .types import RuleViolation, Severity

_REQUIRES = {
    ActionType.PREPARE_LIBRARY: [
        ("samples_collected", "Cannot prepare library without collected samples"),
    ],
    ActionType.SEQUENCE_CELLS: [
        ("library_prepared", "Cannot sequence without library preparation"),
    ],
    ActionType.RUN_QC: [
        ("cells_sequenced", "Cannot run QC before sequencing"),
    ],
    ActionType.FILTER_DATA: [
        ("qc_performed", "Cannot filter data before QC"),
    ],
    ActionType.NORMALIZE_DATA: [
        ("data_filtered", "Cannot normalise before filtering"),
    ],
    ActionType.INTEGRATE_BATCHES: [
        ("data_normalized", "Cannot integrate batches before normalisation"),
    ],
    ActionType.CLUSTER_CELLS: [
        ("data_normalized", "Cannot cluster before normalisation"),
    ],
    ActionType.DIFFERENTIAL_EXPRESSION: [
        ("data_normalized", "Cannot run DE before normalisation"),
    ],
    ActionType.TRAJECTORY_ANALYSIS: [
        ("data_normalized", "Cannot infer trajectories before normalisation"),
    ],
    ActionType.PATHWAY_ENRICHMENT: [
        ("de_performed", "Cannot run pathway enrichment without DE results"),
    ],
    ActionType.REGULATORY_NETWORK_INFERENCE: [
        ("data_normalized", "Cannot infer networks before normalisation"),
    ],
    ActionType.MARKER_SELECTION: [
        ("de_performed", "Cannot select markers without DE results"),
    ],
    ActionType.VALIDATE_MARKER: [
        ("markers_discovered", "Cannot validate markers before discovery"),
    ],
    ActionType.PERTURB_GENE: [
        ("samples_collected", "Cannot perturb without samples"),
    ],
    ActionType.PERTURB_COMPOUND: [
        ("samples_collected", "Cannot perturb without samples"),
    ],
    ActionType.CULTURE_CELLS: [
        ("samples_collected", "Cannot culture without samples"),
    ],
    ActionType.SYNTHESIZE_CONCLUSION: [
        ("data_normalized", "Cannot synthesize conclusions before data normalization"),
    ],
}


def check_prerequisites(
    action: ExperimentAction, s: FullLatentState
) -> List[RuleViolation]:
    vs: List[RuleViolation] = []
    at = action.action_type
    p = s.progress

    for flag, msg in _REQUIRES.get(at, []):
        if not getattr(p, flag, False):
            vs.append(
                RuleViolation(
                    rule_id=f"prereq_{at.value}_{flag}",
                    severity=Severity.HARD,
                    message=msg,
                )
            )
    return vs
