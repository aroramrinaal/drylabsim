"""Resource constraint checks (budget, time)."""

from __future__ import annotations

from typing import List

try:
    from ...models import ActionType, ExperimentAction
    from ..simulator.latent_state import FullLatentState
    from ..simulator.transition import compute_action_cost
except ImportError:  # pragma: no cover
    from models import ActionType, ExperimentAction
    from server.simulator.latent_state import FullLatentState
    from server.simulator.transition import compute_action_cost

from .types import RuleViolation, Severity


def check_resource_constraints(
    action: ExperimentAction, s: FullLatentState
) -> List[RuleViolation]:
    vs: List[RuleViolation] = []
    if s.resources.budget_exhausted:
        vs.append(
            RuleViolation(
                rule_id="budget_exhausted",
                severity=Severity.HARD,
                message="Budget exhausted - no further actions possible",
            )
        )
    if s.resources.time_exhausted:
        vs.append(
            RuleViolation(
                rule_id="time_exhausted",
                severity=Severity.HARD,
                message="Time limit reached - no further actions possible",
            )
        )

    remaining = s.resources.budget_remaining
    cost, _ = compute_action_cost(action)
    if cost > remaining and remaining > 0:
        vs.append(
            RuleViolation(
                rule_id="budget_insufficient",
                severity=Severity.HARD,
                message=f"Action costs ${cost:,.0f} but only ${remaining:,.0f} remains",
            )
        )
    return vs
