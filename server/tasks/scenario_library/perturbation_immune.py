"""Perturbation immune scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..scenarios import Scenario
else:
    # Import the same way as scenarios.py
    try:
        from ....models import ExpectedFinding, PaperReference, TaskSpec
        from ...simulator.latent_state import (
            CellPopulation,
            LatentBiologicalState,
            TechnicalState,
        )
        from ..scenarios import Scenario
    except ImportError:  # pragma: no cover - direct module import path
        from models import ExpectedFinding, PaperReference, TaskSpec
        from server.simulator.latent_state import (
            CellPopulation,
            LatentBiologicalState,
            TechnicalState,
        )
        from server.tasks.scenarios import Scenario


def perturbation_immune() -> Scenario:
    return Scenario(
        name="perturbation_immune",
        difficulty="hard",
        tags=["perturbation", "scRNA-seq", "immune"],
        task=TaskSpec(
            problem_statement=(
                "Determine the effect of JAK inhibitor treatment on "
                "T-cell activation states in rheumatoid arthritis."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="synovial_fluid",
            conditions=["untreated_RA", "JAK_inhibitor_treated"],
            budget_limit=80_000.0,
            time_limit_days=120.0,
            prior_observations=[
                "Elevated JAK-STAT signalling observed in prior bulk RNA-seq",
            ],
            success_criteria=[
                "Quantify shift in T-cell activation states",
                "Identify pathways modulated by JAK inhibitor",
                "Propose validation strategy",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="CD4_Th1",
                    proportion=0.20,
                    marker_genes=["IFNG", "TBX21", "IL2"],
                    state="activated",
                    condition_response={"JAK_inhibitor_treated": 0.5},
                ),
                CellPopulation(
                    name="CD4_Th17",
                    proportion=0.15,
                    marker_genes=["IL17A", "RORC", "CCR6"],
                    state="activated",
                    condition_response={"JAK_inhibitor_treated": 0.6},
                ),
                CellPopulation(
                    name="CD4_Treg",
                    proportion=0.08,
                    marker_genes=["FOXP3", "IL2RA", "CTLA4"],
                    state="regulatory",
                    condition_response={"JAK_inhibitor_treated": 1.2},
                ),
                CellPopulation(
                    name="CD8_cytotoxic",
                    proportion=0.18,
                    marker_genes=["GZMB", "PRF1", "CD8A"],
                    state="activated",
                    condition_response={"JAK_inhibitor_treated": 0.7},
                ),
                CellPopulation(
                    name="macrophage",
                    proportion=0.15,
                    marker_genes=["CD68", "CD163", "MARCO"],
                    state="inflammatory",
                ),
                CellPopulation(
                    name="fibroblast",
                    proportion=0.14,
                    marker_genes=["COL1A1", "FAP", "THY1"],
                    state="activated",
                ),
                CellPopulation(
                    name="B_cell",
                    proportion=0.10,
                    marker_genes=["CD19", "MS4A1", "CD79A"],
                    state="quiescent",
                ),
            ],
            true_de_genes={
                "treated_vs_untreated": {
                    "IFNG": -1.8,
                    "TBX21": -1.2,
                    "IL17A": -1.5,
                    "RORC": -0.9,
                    "JAK1": -0.3,
                    "STAT1": -1.0,
                    "STAT3": -0.8,
                    "SOCS1": 1.5,
                    "SOCS3": 1.3,
                    "FOXP3": 0.6,
                    "IL10": 0.7,
                },
            },
            true_pathways={
                "JAK_STAT_signalling": 0.3,
                "Th1_differentiation": 0.35,
                "Th17_differentiation": 0.4,
                "cytokine_signalling": 0.45,
                "regulatory_T_cell_function": 0.7,
                "negative_feedback_of_cytokine_signalling": 0.76,
                "immune_state_rebalancing": 0.72,
            },
            perturbation_effects={
                "JAK_inhibitor": {
                    "STAT1": -0.8,
                    "STAT3": -0.7,
                    "IFNG": -1.5,
                    "IL17A": -1.3,
                    "SOCS1": 1.2,
                },
            },
            confounders={
                "TNF_NFkB_signalling": 0.83,
                "interferon_response": 0.78,
                "antigen_presentation": 0.74,
            },
            true_markers=[
                "STAT1",
                "STAT3",
                "SOCS1",
                "SOCS3",
                "IFNG",
                "TBX21",
                "IL17A",
                "RORC",
                "FOXP3",
                "IL10",
            ],
            causal_mechanisms=[
                "JAK-STAT pathway inhibition reduces Th1/Th17 activation",
                "Compensatory Treg expansion under JAK inhibition",
            ],
            n_true_cells=18_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_ctrl": 0.12, "batch_treated": 0.18},
            ambient_rna_fraction=0.07,
            dropout_rate=0.10,
        ),
        hidden_failure_conditions=[
            "High ambient RNA may confound DE in low-abundance transcripts",
        ],
    )
