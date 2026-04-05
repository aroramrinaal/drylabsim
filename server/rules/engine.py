"""Biological rule engine — hard and soft constraint checking.

Hard constraints block action execution entirely.
Soft constraints allow execution but degrade output quality and incur penalties.
"""

from __future__ import annotations

from typing import List

try:
    from ...models import ActionType, ExperimentAction
    from ..simulator.latent_state import FullLatentState
except ImportError:  # pragma: no cover
    from models import ActionType, ExperimentAction
    from server.simulator.latent_state import FullLatentState

from .causal import check_causal_validity
from .prerequisites import check_prerequisites
from .redundancy import check_redundancy
from .resources import check_resource_constraints
from .tool_compatibility import check_tool_compatibility
from .types import RuleViolation, Severity


class RuleEngine:
    """Evaluates biological and resource constraints against the current
    latent state before each action is applied.
    """

    def check(
        self, action: ExperimentAction, state: FullLatentState
    ) -> List[RuleViolation]:
        violations: List[RuleViolation] = []
        violations.extend(check_prerequisites(action, state))
        violations.extend(check_resource_constraints(action, state))
        violations.extend(check_redundancy(action, state))
        violations.extend(check_causal_validity(action, state))
        violations.extend(check_tool_compatibility(action, state))
        return violations

    def hard_violations(self, violations: List[RuleViolation]) -> List[str]:
        return [v.message for v in violations if v.severity == Severity.HARD]

    def soft_violations(self, violations: List[RuleViolation]) -> List[str]:
        return [v.message for v in violations if v.severity == Severity.SOFT]
