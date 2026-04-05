"""Causal validity checks — ensure actions are backed by sufficient evidence."""

from __future__ import annotations

from typing import List

try:
    from ...models import ActionType, ExperimentAction
    from ..simulator.latent_state import FullLatentState
except ImportError:  # pragma: no cover
    from models import ActionType, ExperimentAction
    from server.simulator.latent_state import FullLatentState

from .types import RuleViolation, Severity


def _has_analysis_evidence(s: FullLatentState) -> bool:
    p = s.progress
    return any(
        [
            p.cells_clustered,
            p.de_performed,
            p.trajectories_inferred,
            p.pathways_analyzed,
            p.networks_inferred,
            p.markers_discovered,
            p.markers_validated,
        ]
    )


def _has_marker_evidence(s: FullLatentState) -> bool:
    p = s.progress
    return p.markers_discovered or p.markers_validated


def _has_mechanism_evidence(s: FullLatentState) -> bool:
    p = s.progress
    return p.pathways_analyzed or p.networks_inferred


def check_causal_validity(
    action: ExperimentAction, s: FullLatentState
) -> List[RuleViolation]:
    vs: List[RuleViolation] = []
    has_analysis_evidence = _has_analysis_evidence(s)

    if action.action_type == ActionType.DESIGN_FOLLOWUP:
        if not has_analysis_evidence:
            vs.append(
                RuleViolation(
                    rule_id="premature_followup_design",
                    severity=Severity.HARD,
                    message=(
                        "Follow-up design without prior analysis is blocked; "
                        "complete wet-lab and computational steps first"
                    ),
                )
            )

    if action.action_type == ActionType.REQUEST_SUBAGENT_REVIEW:
        if not has_analysis_evidence:
            vs.append(
                RuleViolation(
                    rule_id="premature_subagent_review",
                    severity=Severity.HARD,
                    message=(
                        "Subagent review without prior analysis is blocked; "
                        "generate evidence first"
                    ),
                )
            )

    if action.action_type == ActionType.SYNTHESIZE_CONCLUSION:
        if not s.progress.de_performed and not s.progress.cells_clustered:
            vs.append(
                RuleViolation(
                    rule_id="premature_conclusion",
                    severity=Severity.HARD,
                    message="Cannot synthesise conclusion without substantive analysis",
                )
            )

        if not _has_marker_evidence(s):
            vs.append(
                RuleViolation(
                    rule_id="conclusion_without_marker_evidence",
                    severity=Severity.HARD,
                    message="Cannot synthesise conclusion before discovering or validating markers",
                )
            )

        if not _has_mechanism_evidence(s):
            vs.append(
                RuleViolation(
                    rule_id="conclusion_without_mechanism_evidence",
                    severity=Severity.HARD,
                    message="Cannot synthesise conclusion before inferring pathways or mechanisms",
                )
            )

        claims = action.parameters.get("claims", [])
        for claim in claims:
            if isinstance(claim, dict) and claim.get("claim_type") == "causal":
                if (
                    not s.progress.markers_validated
                    and not s.progress.networks_inferred
                ):
                    vs.append(
                        RuleViolation(
                            rule_id="unsupported_causal_claim",
                            severity=Severity.SOFT,
                            message="Causal claim without validation or network evidence",
                        )
                    )
                    break

    if action.action_type == ActionType.PATHWAY_ENRICHMENT:
        if not s.progress.de_performed:
            vs.append(
                RuleViolation(
                    rule_id="pathway_without_de",
                    severity=Severity.SOFT,
                    message="Pathway enrichment without DE may yield unreliable results",
                )
            )
    return vs
