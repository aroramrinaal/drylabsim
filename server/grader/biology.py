"""Biological scoring: markers, mechanisms, pathways.

Wraps calls to gene_index.py scoring functions. All inputs come from
the agent's discoveries/conclusions and the hidden ground truth in
FullLatentState. Returns a float in [0.0, 1.0].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

try:
    from ...biology.gene_index import (
        marker_set_score,
        mechanism_set_score,
        score_pathways,
    )
except ImportError:
    from server.biology.gene_index import (
        marker_set_score,
        mechanism_set_score,
        score_pathways,
    )

try:
    from ...models import ConclusionClaim
    from ...simulator.latent_state import FullLatentState
except ImportError:
    from models import ConclusionClaim
    from server.simulator.latent_state import FullLatentState


def score_biology(
    state: FullLatentState,
    discovered_markers: List[str],
    candidate_mechanisms: List[str],
    conclusions: List["ConclusionClaim"],
) -> float:
    """Composite biological accuracy score in [0.0, 1.0].

    Combines three sub-scores with weights that sum to 1.0:
      - marker recall/precision:    0.40
      - mechanism semantic match:   0.35
      - pathway activity match:     0.25

    If the agent submitted structured conclusions (top_markers,
    causal_mechanisms, predicted_pathways), those are scored directly.
    Otherwise the grader falls back to discovered_markers and
    candidate_mechanisms accumulated during the episode.
    """
    true_markers = state.biology.true_markers
    true_mechanisms = state.biology.causal_mechanisms
    true_pathways = state.biology.true_pathways

    pred_markers: List[str] = []
    pred_mechs: List[str] = []
    pred_pathways: Dict[str, float] = {}

    if conclusions:
        for c in conclusions:
            pred_markers.extend(c.top_markers)
            pred_mechs.extend(c.causal_mechanisms)
            pred_pathways.update(c.predicted_pathways)

    if not pred_markers and discovered_markers:
        pred_markers = list(discovered_markers)
    if not pred_mechs and candidate_mechanisms:
        pred_mechs = list(candidate_mechanisms)

    m_score = marker_set_score(pred_markers, true_markers) if true_markers else 0.0
    mech_score = (
        mechanism_set_score(pred_mechs, true_mechanisms) if true_mechanisms else 0.0
    )
    pw_score = score_pathways(pred_pathways, true_pathways) if true_pathways else 0.0

    has_markers = bool(true_markers)
    has_mechs = bool(true_mechanisms)
    has_pathways = bool(true_pathways)

    total_weight = 0.0
    weighted = 0.0

    if has_markers:
        weighted += 0.40 * m_score
        total_weight += 0.40
    if has_mechs:
        weighted += 0.35 * mech_score
        total_weight += 0.35
    if has_pathways:
        weighted += 0.25 * pw_score
        total_weight += 0.25

    if total_weight == 0.0:
        return 0.0

    return weighted / total_weight
