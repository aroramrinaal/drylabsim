from __future__ import annotations

from typing import Any, Dict, List

try:
    from ....models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )
except ImportError:  # pragma: no cover
    from models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )

from ..latent_state import FullLatentState
from ..noise import NoiseModel


def design_followup(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    evidence_signals = sum(
        [
            int(s.progress.cells_clustered),
            int(s.progress.de_performed),
            int(s.progress.trajectories_inferred),
            int(s.progress.pathways_analyzed),
            int(s.progress.networks_inferred),
            int(s.progress.markers_discovered),
            int(s.progress.markers_validated),
        ]
    )
    return IntermediateOutput(
        output_type=OutputType.FOLLOWUP_DESIGN,
        step_index=idx,
        quality_score=min(0.75, 0.2 + 0.08 * evidence_signals),
        summary=(
            f"Follow-up experiment design proposed "
            f"(evidence_signals={evidence_signals})"
        ),
        data={
            "proposal": action.parameters,
            "evidence_signals": evidence_signals,
        },
        uncertainty=max(0.25, 0.8 - 0.08 * evidence_signals),
        artifacts_available=["followup_proposal"],
    )


def subagent_review(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    evidence_signals = sum(
        [
            int(s.progress.cells_clustered),
            int(s.progress.de_performed),
            int(s.progress.trajectories_inferred),
            int(s.progress.pathways_analyzed),
            int(s.progress.networks_inferred),
            int(s.progress.markers_discovered),
            int(s.progress.markers_validated),
        ]
    )
    return IntermediateOutput(
        output_type=OutputType.SUBAGENT_REPORT,
        step_index=idx,
        quality_score=min(0.7, 0.15 + 0.07 * evidence_signals),
        summary=f"Subagent review ({action.invoked_subagent or 'general'})",
        data={
            "subagent": action.invoked_subagent,
            "notes": "Review complete.",
            "evidence_signals": evidence_signals,
        },
        uncertainty=max(0.3, 0.85 - 0.08 * evidence_signals),
        artifacts_available=["subagent_report"],
    )


def synthesize_conclusion(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    return IntermediateOutput(
        output_type=OutputType.CONCLUSION,
        step_index=idx,
        summary="Conclusion synthesised from pipeline evidence",
        data={"claims": action.parameters.get("claims", [])},
        artifacts_available=["conclusion_report"],
    )


def default_handler(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    return IntermediateOutput(
        output_type=OutputType.FAILURE_REPORT,
        step_index=idx,
        success=False,
        summary=f"Unhandled action type: {action.action_type}",
        data={},
    )
