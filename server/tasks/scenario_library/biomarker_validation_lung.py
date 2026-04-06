"""Biomarker validation lung scenario."""

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


def biomarker_validation_lung() -> Scenario:
    return Scenario(
        name="biomarker_validation_lung",
        difficulty="medium",
        tags=["biomarker", "validation", "scRNA-seq", "lung"],
        task=TaskSpec(
            problem_statement=(
                "Design a follow-up validation experiment for candidate "
                "biomarker SPP1 in idiopathic pulmonary fibrosis (IPF)."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="lung",
            conditions=["healthy", "IPF"],
            budget_limit=90_000.0,
            time_limit_days=150.0,
            prior_observations=[
                "A macrophage subpopulation shows elevated expression in IPF tissue relative to controls",
                "Pro-fibrotic macrophage enrichment has been observed in fibrotic regions by spatial profiling",
            ],
            success_criteria=[
                "Validate SPP1 as a marker for pro-fibrotic macrophages",
                "Confirm spatial localisation in fibrotic tissue",
            ],
            paper_references=[
                PaperReference(
                    title=(
                        "Proliferating SPP1/MERTK-expressing macrophages in "
                        "idiopathic pulmonary fibrosis"
                    ),
                    citation="European Respiratory Journal (2019)",
                    doi="10.1183/13993003.02441-2018",
                    pmid="31221805",
                    url="https://pubmed.ncbi.nlm.nih.gov/31221805/",
                ),
            ],
            expected_findings=[
                ExpectedFinding(
                    finding=(
                        "SPP1-positive macrophages should be enriched in IPF "
                        "fibrotic regions."
                    ),
                    category="marker",
                    keywords=["SPP1", "macrophage", "IPF", "fibrotic"],
                ),
                ExpectedFinding(
                    finding=(
                        "MERTK should co-occur with the profibrotic macrophage state."
                    ),
                    category="marker",
                    keywords=["MERTK", "macrophage", "SPP1"],
                ),
                ExpectedFinding(
                    finding=(
                        "Extracellular matrix organization should emerge as a "
                        "top fibrotic program."
                    ),
                    category="pathway",
                    keywords=["extracellular_matrix", "fibrosis", "pathway"],
                ),
            ],
            dataset_metadata={
                "literature_grounding": "single_cell_ipf_macrophages",
            },
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="alveolar_macrophage",
                    proportion=0.18,
                    marker_genes=["MARCO", "FABP4", "MCEMP1"],
                    state="resident",
                ),
                CellPopulation(
                    name="SPP1_macrophage",
                    proportion=0.12,
                    marker_genes=["SPP1", "MERTK", "MMP9", "TREM2"],
                    state="pro-fibrotic",
                    condition_response={"IPF": 2.0},
                ),
                CellPopulation(
                    name="AT2",
                    proportion=0.20,
                    marker_genes=["SFTPC", "SFTPB", "ABCA3"],
                    state="normal",
                ),
                CellPopulation(
                    name="fibroblast",
                    proportion=0.22,
                    marker_genes=["COL1A1", "COL3A1", "POSTN"],
                    state="activated",
                    condition_response={"IPF": 1.5},
                ),
                CellPopulation(
                    name="endothelial",
                    proportion=0.13,
                    marker_genes=["PECAM1", "CLDN5"],
                    state="quiescent",
                ),
                CellPopulation(
                    name="T_cell",
                    proportion=0.15,
                    marker_genes=["CD3D", "CD3E", "IL7R"],
                    state="quiescent",
                ),
            ],
            true_de_genes={
                "IPF_vs_healthy": {
                    "SPP1": 3.2,
                    "MERTK": 1.4,
                    "MMP9": 1.8,
                    "TREM2": 1.5,
                    "COL1A1": 2.1,
                    "COL3A1": 1.9,
                    "POSTN": 2.4,
                    "SFTPC": -1.2,
                    "AGER": -1.6,
                },
            },
            true_pathways={
                "extracellular_matrix_organisation": 0.9,
                "integrin_signalling": 0.75,
                "macrophage_activation": 0.8,
                "Wnt_signalling": 0.6,
            },
            true_markers=["SPP1", "MERTK", "POSTN", "MMP9"],
            causal_mechanisms=[
                "SPP1+ macrophage-driven fibroblast activation",
                "Integrin-mediated SPP1 signalling in fibrosis",
            ],
            n_true_cells=14_000,
        ),
        technical=TechnicalState(
            batch_effects={"batch_1": 0.10},
            dropout_rate=0.09,
            sample_quality=0.85,
        ),
    )
