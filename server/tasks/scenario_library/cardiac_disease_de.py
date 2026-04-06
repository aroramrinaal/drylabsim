"""Cardiac disease DE scenario."""

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


def cardiac_disease_de() -> Scenario:
    return Scenario(
        name="cardiac_disease_de",
        difficulty="easy",
        tags=["de", "scRNA-seq", "cardiac"],
        task=TaskSpec(
            problem_statement=(
                "Identify differentially expressed genes between diseased "
                "and healthy cardiomyocytes using single-cell RNA sequencing."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="heart",
            conditions=["healthy", "dilated_cardiomyopathy"],
            budget_limit=80_000.0,
            time_limit_days=120.0,
            success_criteria=[
                "Identify DE genes between conditions",
                "Validate at least one candidate marker",
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="cardiomyocyte",
                    proportion=0.35,
                    marker_genes=["TNNT2", "MYH7", "ACTC1"],
                    state="contractile",
                    condition_response={"dilated_cardiomyopathy": 0.8},
                ),
                CellPopulation(
                    name="fibroblast",
                    proportion=0.25,
                    marker_genes=["COL1A1", "DCN", "LUM"],
                    state="quiescent",
                    condition_response={"dilated_cardiomyopathy": 1.3},
                ),
                CellPopulation(
                    name="endothelial",
                    proportion=0.15,
                    marker_genes=["PECAM1", "VWF", "CDH5"],
                    state="quiescent",
                ),
                CellPopulation(
                    name="macrophage",
                    proportion=0.10,
                    marker_genes=["CD68", "CD163", "CSF1R"],
                    state="activated",
                    condition_response={"dilated_cardiomyopathy": 1.5},
                ),
                CellPopulation(
                    name="smooth_muscle",
                    proportion=0.15,
                    marker_genes=["ACTA2", "MYH11", "TAGLN"],
                    state="quiescent",
                ),
            ],
            true_de_genes={
                "disease_vs_healthy": {
                    "NPPA": 2.5,
                    "NPPB": 3.1,
                    "MYH7": 1.8,
                    "COL1A1": 1.6,
                    "COL3A1": 1.4,
                    "POSTN": 2.0,
                    "CCL2": 1.2,
                    "IL6": 0.9,
                    "TGFB1": 1.1,
                    "ANKRD1": 2.2,
                    "XIRP2": -1.3,
                    "MYL2": -0.8,
                },
            },
            true_pathways={
                "cardiac_muscle_contraction": 0.4,
                "extracellular_matrix_organisation": 0.85,
                "inflammatory_response": 0.7,
                "TGF_beta_signalling": 0.75,
                "apoptosis": 0.55,
            },
            true_markers=["NPPA", "NPPB", "POSTN", "COL1A1"],
            causal_mechanisms=[
                "TGF-beta-driven fibrosis",
                "inflammatory macrophage infiltration",
            ],
            n_true_cells=12_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_1": 0.15, "batch_2": 0.10},
            doublet_rate=0.05,
            dropout_rate=0.08,
        ),
    )
