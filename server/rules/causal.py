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

_MULTICLONE_EXPERT_SCENARIO = "venetoclax_resistance_multiclone"


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
    is_multiclone_expert = s.scenario_name == _MULTICLONE_EXPERT_SCENARIO

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

        if is_multiclone_expert:
            if not s.progress.trajectories_inferred:
                vs.append(
                    RuleViolation(
                        rule_id="expert_conclusion_without_trajectory",
                        severity=Severity.HARD,
                        message=(
                            "Cannot synthesise the multiclone resistance conclusion "
                            "before trajectory evidence resolves parallel branches"
                        ),
                    )
                )
            if not s.progress.networks_inferred:
                vs.append(
                    RuleViolation(
                        rule_id="expert_conclusion_without_network",
                        severity=Severity.HARD,
                        message=(
                            "Cannot synthesise the multiclone resistance conclusion "
                            "before regulatory evidence resolves clone-specific wiring"
                        ),
                    )
                )
            if len(s.discovered_clusters) < 2:
                vs.append(
                    RuleViolation(
                        rule_id="expert_conclusion_without_multiple_clusters",
                        severity=Severity.HARD,
                        message=(
                            "Cannot synthesise the multiclone resistance conclusion "
                            "before at least two relapse clusters are resolved"
                        ),
                    )
                )
            if len(s.mechanism_confidence) < 2:
                vs.append(
                    RuleViolation(
                        rule_id="expert_conclusion_without_both_mechanisms",
                        severity=Severity.HARD,
                        message=(
                            "Cannot synthesise the multiclone resistance conclusion "
                            "before both resistant mechanisms have supporting evidence"
                        ),
                    )
                )

        claims = action.parameters.get("claims", [])
        unique_claim_mechs = set()
        for claim in claims:
            if isinstance(claim, dict):
                unique_claim_mechs.update(claim.get("causal_mechanisms", []))
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
        if is_multiclone_expert and claims and len(unique_claim_mechs) < 2:
            vs.append(
                RuleViolation(
                    rule_id="expert_single_mechanism_conclusion",
                    severity=Severity.SOFT,
                    message=(
                        "Conclusion names fewer than two resistant mechanisms in a "
                        "scenario whose ground truth is explicitly multiclonal"
                    ),
                )
            )

    if action.action_type == ActionType.PATHWAY_ENRICHMENT:
        if not s.progress.de_performed:
            vs.append(
                RuleViolation(
                    rule_id="pathway_without_de",
                    severity=Severity.SOFT,
                    message="Pathway enrichment without DE may yield unreliable results",
                )
            )
    if is_multiclone_expert and action.action_type in {
        ActionType.CLUSTER_CELLS,
        ActionType.DIFFERENTIAL_EXPRESSION,
    }:
        if len(s.technical.batch_effects) > 1 and not s.progress.batches_integrated:
            vs.append(
                RuleViolation(
                    rule_id="expert_multiclone_without_batch_integration",
                    severity=Severity.SOFT,
                    message=(
                        "Skipping batch integration may merge or obscure the minor "
                        "resistant clone in this multiclone scenario"
                    ),
                )
            )
    return vs
