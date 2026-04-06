"""Hematopoiesis trajectory scenario."""

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


def hematopoiesis_trajectory() -> Scenario:
    return Scenario(
        name="hematopoiesis_trajectory",
        difficulty="medium",
        tags=["trajectory", "scRNA-seq", "hematopoiesis"],
        task=TaskSpec(
            problem_statement=(
                "Infer the developmental trajectory of hematopoietic "
                "stem cells differentiating into mature blood lineages."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="bone_marrow",
            conditions=["steady_state"],
            budget_limit=100_000.0,
            time_limit_days=150.0,
            success_criteria=[
                "Reconstruct branching lineage structure",
                "Identify key transcription factors driving fate decisions",
            ],
            paper_references=[
                PaperReference(
                    title=(
                        "Single-cell RNA-sequencing uncovers transcriptional "
                        "states and fate decisions in haematopoiesis"
                    ),
                    citation="Nature Communications (2018)",
                    doi="10.1038/s41467-017-02305-6",
                    url=("https://www.nature.com/articles/s41467-017-02305-6"),
                ),
            ],
            expected_findings=[
                ExpectedFinding(
                    finding=(
                        "Trajectory analysis should recover branching blood "
                        "lineages rooted in HSCs."
                    ),
                    category="trajectory",
                    keywords=["HSC", "branching", "lineage", "trajectory"],
                ),
                ExpectedFinding(
                    finding=(
                        "GATA1 should appear as a driver of erythroid fate commitment."
                    ),
                    category="regulatory_network",
                    keywords=["GATA1", "erythroid", "commitment"],
                ),
                ExpectedFinding(
                    finding=("CEBPA and SPI1 should support myeloid branch decisions."),
                    category="regulatory_network",
                    keywords=["CEBPA", "SPI1", "myeloid", "branch"],
                ),
            ],
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="HSC",
                    proportion=0.05,
                    marker_genes=["CD34", "KIT", "THY1"],
                    state="stem",
                ),
                CellPopulation(
                    name="CMP",
                    proportion=0.10,
                    marker_genes=["CD34", "FLT3"],
                    state="progenitor",
                ),
                CellPopulation(
                    name="GMP",
                    proportion=0.12,
                    marker_genes=["CSF3R", "CEBPA"],
                    state="progenitor",
                ),
                CellPopulation(
                    name="MEP",
                    proportion=0.10,
                    marker_genes=["GATA1", "KLF1"],
                    state="progenitor",
                ),
                CellPopulation(
                    name="erythrocyte",
                    proportion=0.20,
                    marker_genes=["HBA1", "HBB", "GYPA"],
                    state="mature",
                ),
                CellPopulation(
                    name="neutrophil",
                    proportion=0.18,
                    marker_genes=["ELANE", "MPO", "CTSG"],
                    state="mature",
                ),
                CellPopulation(
                    name="monocyte",
                    proportion=0.15,
                    marker_genes=["CD14", "CSF1R", "FCGR3A"],
                    state="mature",
                ),
                CellPopulation(
                    name="megakaryocyte",
                    proportion=0.10,
                    marker_genes=["ITGA2B", "GP1BA"],
                    state="mature",
                ),
            ],
            true_de_genes={},
            true_pathways={
                "hematopoietic_cell_lineage": 0.9,
                "MAPK_signalling": 0.6,
                "JAK_STAT_signalling": 0.7,
            },
            true_trajectory={
                "root": "HSC",
                "n_lineages": 3,
                "branching": True,
                "branches": [
                    ["HSC", "CMP", "GMP", "neutrophil"],
                    ["HSC", "CMP", "GMP", "monocyte"],
                    ["HSC", "MEP", "erythrocyte"],
                    ["HSC", "MEP", "megakaryocyte"],
                ],
            },
            true_regulatory_network={
                "GATA1": ["KLF1", "HBB", "HBA1", "GYPA"],
                "CEBPA": ["CSF3R", "ELANE", "MPO"],
                "SPI1": ["CSF1R", "CD14", "FCGR3A"],
                "RUNX1": ["CD34", "KIT"],
            },
            true_markers=["GATA1", "CEBPA", "SPI1"],
            causal_mechanisms=[
                "GATA1-driven erythroid commitment",
                "PU.1/CEBPA antagonism at myeloid branch point",
            ],
            n_true_cells=15_000,
        ),
        technical=TechnicalState(dropout_rate=0.12, doublet_rate=0.06),
    )
