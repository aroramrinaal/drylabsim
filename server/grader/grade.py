"""Main grading entry point.

grade_episode() is a pure, deterministic function that takes the final
episode observation and hidden latent state, then returns a GradeResult
with score guaranteed to be in [0.0, 1.0].

This is the function judges call to evaluate an agent run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .biology import score_biology
from .pipeline import score_pipeline
from .types import GradeResult

if TYPE_CHECKING:
    from ...models import ExperimentObservation
    from ...simulator.latent_state import FullLatentState

_W_PIPELINE = 0.30
_W_BIOLOGY = 0.55
_W_EFFICIENCY = 0.15


def grade_episode(
    obs: "ExperimentObservation",
    latent: "FullLatentState",
) -> GradeResult:
    """Grade a completed episode. Pure function — deterministic and reproducible.

    Args:
        obs: The final ExperimentObservation from the environment.
        latent: The FullLatentState (hidden ground truth) for this episode.

    Returns:
        GradeResult with score in [0.0, 1.0].
    """
    completeness = score_pipeline(latent)

    biology = score_biology(
        state=latent,
        discovered_markers=obs.discovered_markers,
        candidate_mechanisms=obs.candidate_mechanisms,
        conclusions=obs.conclusions,
    )

    res = latent.resources
    budget_eff = res.budget_remaining / max(res.budget_total, 1.0)
    time_eff = res.time_remaining_days / max(res.time_limit_days, 1.0)
    efficiency = 0.5 * budget_eff + 0.5 * time_eff

    score = (
        _W_PIPELINE * completeness + _W_BIOLOGY * biology + _W_EFFICIENCY * efficiency
    )

    return GradeResult(
        score=score,
        completeness=completeness,
        biology_score=biology,
        efficiency_score=efficiency,
        breakdown={
            "completeness": completeness,
            "biology_score": biology,
            "efficiency_score": efficiency,
            "weight_pipeline": _W_PIPELINE,
            "weight_biology": _W_BIOLOGY,
            "weight_efficiency": _W_EFFICIENCY,
        },
    )
