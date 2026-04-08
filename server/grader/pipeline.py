"""Pipeline completeness and validity scoring.

Pure function of FullLatentState.progress — no randomness, no side effects.
Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

try:
    from ..simulator.latent_state import FullLatentState
except ImportError:
    from server.simulator.latent_state import FullLatentState

_CORE_MILESTONES = [
    "samples_collected",
    "cells_sequenced",
    "qc_performed",
    "data_filtered",
    "data_normalized",
    "de_performed",
    "conclusion_reached",
]

_OPTIONAL_MILESTONES = [
    "cells_clustered",
    "pathways_analyzed",
    "markers_discovered",
    "markers_validated",
    "trajectories_inferred",
    "networks_inferred",
]


def score_pipeline(state: FullLatentState) -> float:
    """Score pipeline completeness from progress flags.

    Core milestones (weight 1.0 each): samples → sequencing → QC → filter
    → normalize → DE/cluster → conclusion.

    Optional milestones (weight 0.5 each): clustering, pathways, markers,
    validation, trajectory, network inference.

    Returns 0.0 if no milestones completed, 1.0 if all completed.
    """
    p = state.progress

    core_score = sum(getattr(p, m) for m in _CORE_MILESTONES) / len(_CORE_MILESTONES)

    optional_score = 0.0
    if _OPTIONAL_MILESTONES:
        optional_score = sum(getattr(p, m) for m in _OPTIONAL_MILESTONES) / len(
            _OPTIONAL_MILESTONES
        )

    return 0.7 * core_score + 0.3 * optional_score
