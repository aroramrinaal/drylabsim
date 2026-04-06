# Generate simulated intermediate outputs conditioned on latent state.

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from ...models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )
except ImportError:  # pragma: no cover - direct module import path
    from models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )

from .latent_state import FullLatentState
from .noise import NoiseModel

from .handlers.wet_lab import (
    collect_sample,
    select_cohort,
    prepare_library,
    culture_cells,
    perturb_gene,
    perturb_compound,
    sequence_cells,
)

from .handlers.computational import (
    run_qc,
    filter_data,
    normalize_data,
    integrate_batches,
    cluster_cells,
    differential_expression,
)

from .handlers.analysis import (
    trajectory_analysis,
    pathway_enrichment,
    regulatory_network,
    marker_selection,
    validate_marker,
)

from .handlers.meta import (
    design_followup,
    subagent_review,
    synthesize_conclusion,
    default_handler,
)


class OutputGenerator:
    """Creates structured ``IntermediateOutput`` objects conditioned on the
    hidden latent state, the action taken, and a stochastic noise model.
    """

    def __init__(self, noise: NoiseModel):
        self.noise = noise

    def generate(
        self,
        action: ExperimentAction,
        state: FullLatentState,
        step_index: int,
    ) -> IntermediateOutput:
        handler = _HANDLERS.get(action.action_type, default_handler)
        return handler(self, action, state, step_index)


_HANDLERS = {
    ActionType.COLLECT_SAMPLE: collect_sample,
    ActionType.SELECT_COHORT: select_cohort,
    ActionType.PREPARE_LIBRARY: prepare_library,
    ActionType.CULTURE_CELLS: culture_cells,
    ActionType.PERTURB_GENE: perturb_gene,
    ActionType.PERTURB_COMPOUND: perturb_compound,
    ActionType.SEQUENCE_CELLS: sequence_cells,
    ActionType.RUN_QC: run_qc,
    ActionType.FILTER_DATA: filter_data,
    ActionType.NORMALIZE_DATA: normalize_data,
    ActionType.INTEGRATE_BATCHES: integrate_batches,
    ActionType.CLUSTER_CELLS: cluster_cells,
    ActionType.DIFFERENTIAL_EXPRESSION: differential_expression,
    ActionType.TRAJECTORY_ANALYSIS: trajectory_analysis,
    ActionType.PATHWAY_ENRICHMENT: pathway_enrichment,
    ActionType.REGULATORY_NETWORK_INFERENCE: regulatory_network,
    ActionType.MARKER_SELECTION: marker_selection,
    ActionType.VALIDATE_MARKER: validate_marker,
    ActionType.DESIGN_FOLLOWUP: design_followup,
    ActionType.REQUEST_SUBAGENT_REVIEW: subagent_review,
    ActionType.SYNTHESIZE_CONCLUSION: synthesize_conclusion,
}
