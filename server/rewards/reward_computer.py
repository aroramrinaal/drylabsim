"""Computes step-wise and terminal rewards."""

from __future__ import annotations

from typing import List, Optional

from .reward_breakdown import RewardBreakdown
from .reward_components import (
    ordering_score,
    potential,
    completeness,
    calibration,
    legacy_calibration,
    tool_fit_score,
    overconfidence_penalty,
    discovery_alignment,
    conclusion_alignment,
)

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
    from server.simulator.latent_state import FullLatentState


class RewardComputer:
    """Computes step-wise and terminal rewards.

    Parameters
    ----------
    efficiency_weight : float
        Relative importance of resource efficiency.
    """

    def __init__(
        self,
        efficiency_weight: float = 0.3,
        info_gain_weight: float = 0.4,
        validity_weight: float = 0.3,
    ):
        self.w_eff = efficiency_weight
        self.w_ig = info_gain_weight
        self.w_val = validity_weight

    # ── step reward ─────────────────────────────────────────────────────

    def step_reward(
        self,
        action: ExperimentAction,
        prev_state: FullLatentState,
        next_state: FullLatentState,
        output: IntermediateOutput,
        hard_violations: List[str],
        soft_violations: List[str],
    ) -> RewardBreakdown:
        rb = RewardBreakdown()

        # validity
        if hard_violations:
            rb.validity = -1.0
            rb.penalty = -0.5 * len(hard_violations)
            rb.components["hard_violations"] = len(hard_violations)
            return rb

        rb.validity = self.w_val * (1.0 if output.success else 0.0)

        ordering = ordering_score(action, prev_state)
        rb.ordering = 0.2 * ordering
        if ordering < 0:
            rb.penalty += ordering * 0.3

        # information gain proxy: quality × (1 - uncertainty)
        rb.info_gain = self.w_ig * output.quality_score * (1.0 - output.uncertainty)
        if action.action_type in META_ACTIONS and not (
            prev_state.progress.de_performed or prev_state.progress.cells_clustered
        ):
            # Meta actions before substantive analysis should not dominate reward.
            rb.info_gain *= 0.2

        # efficiency: normalised cost relative to budget
        budget_frac = (
            next_state.resources.budget_used - prev_state.resources.budget_used
        ) / max(next_state.resources.budget_total, 1)
        rb.efficiency = self.w_eff * max(0.0, 1.0 - 5.0 * budget_frac)

        # novelty: small bonus for non-redundant steps
        if not soft_violations:
            rb.novelty = 0.1

        # tool-modality fit bonus/penalty
        tool_fit = tool_fit_score(action, prev_state)
        rb.components["tool_fit"] = tool_fit
        rb.validity += 0.15 * tool_fit

        # penalties
        rb.penalty = -0.15 * len(soft_violations)
        if action.action_type in META_ACTIONS and not (
            prev_state.progress.de_performed or prev_state.progress.cells_clustered
        ):
            rb.penalty -= 0.25
            rb.components["premature_meta_action_penalty"] = -0.25

        # potential-based shaping (γ=1 so it doesn't depend on the
        # training algorithm's discount factor)
        phi_prev = potential(prev_state)
        phi_next = potential(next_state)
        rb.shaping = phi_next - phi_prev

        return rb

    # ── terminal reward ─────────────────────────────────────────────────

    def terminal_reward(
        self,
        state: FullLatentState,
        conclusions: List[ConclusionClaim],
        task_success_criteria: List[str],
        discovered_markers: Optional[List[str]] = None,
        candidate_mechanisms: Optional[List[str]] = None,
    ) -> RewardBreakdown:
        rb = RewardBreakdown()
        discovered_markers = discovered_markers or []
        candidate_mechanisms = candidate_mechanisms or []

        # pipeline completeness (0-1)
        comp = completeness(state)
        rb.components["completeness"] = comp

        # calibration: how well conclusions align with hidden ground truth
        cal = calibration(state, conclusions)
        rb.components["calibration"] = cal

        # efficiency bonus at terminal
        budget_eff = state.resources.budget_remaining / max(
            state.resources.budget_total, 1
        )
        time_eff = state.resources.time_remaining_days / max(
            state.resources.time_limit_days, 1
        )
        rb.components["budget_efficiency"] = budget_eff
        rb.components["time_efficiency"] = time_eff

        # over-confidence penalty
        overconf = overconfidence_penalty(state, conclusions)
        rb.components["overconfidence_penalty"] = overconf

        discovery_align = discovery_alignment(
            state,
            discovered_markers,
            candidate_mechanisms,
        )
        discovery_error_penalty = -6.0 * (1.0 - discovery_align)
        if discovery_align < 0.25:
            discovery_error_penalty -= 2.0
        rb.components["discovery_alignment"] = discovery_align
        rb.components["discovery_error_penalty"] = discovery_error_penalty

        conclusion_align = conclusion_alignment(state, conclusions)
        conclusion_error_penalty = -4.0 * (1.0 - conclusion_align)
        if conclusions and conclusion_align < 0.25:
            conclusion_error_penalty -= 1.5
        rb.components["conclusion_alignment"] = conclusion_align
        rb.components["conclusion_error_penalty"] = conclusion_error_penalty

        eff_bonus = (budget_eff + time_eff) / 2.0 if comp >= 0.3 else 0.0
        rb.terminal = (
            3.0 * comp
            + 4.0 * cal
            + 1.0 * eff_bonus
            + overconf
            + discovery_error_penalty
            + conclusion_error_penalty
        )
        return rb
