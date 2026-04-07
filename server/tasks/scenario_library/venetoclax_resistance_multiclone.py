"""Expert AML venetoclax resistance scenario with parallel resistant clones."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..scenarios import Scenario
else:
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


def venetoclax_resistance_multiclone() -> Scenario:
    return Scenario(
        name="venetoclax_resistance_multiclone",
        difficulty="expert",
        tags=["resistance", "scRNA-seq", "AML", "bone_marrow", "expert"],
        task=TaskSpec(
            problem_statement=(
                "Disentangle parallel mechanisms of acquired venetoclax resistance "
                "in acute myeloid leukemia and determine whether the resistant "
                "state is explained by one clone or multiple resistant subclones."
            ),
            modality="scRNA-seq",
            organism="human",
            tissue="bone_marrow",
            conditions=["pre_treatment", "post_venetoclax_resistant"],
            budget_limit=150_000.0,
            time_limit_days=210.0,
            prior_observations=[
                "The patient initially responded to venetoclax-based therapy before relapsing",
                "Bulk expression suggested both apoptotic rewiring and cytokine signalling, but no single mechanism was definitive",
                "Relapse marrow contains a rare blast population that is near the clustering detection limit",
            ],
            success_criteria=[
                "Resolve whether relapse is driven by one resistant state or multiple resistant subclones",
                "Identify the anti-apoptotic MCL1 program in one resistant branch",
                "Identify the JAK2-STAT5 survival program in the second resistant branch",
                "Synthesize a calibrated conclusion that names both mechanisms without overclaiming certainty for the smaller clone",
            ],
            paper_references=[
                PaperReference(
                    title="Recent single-cell studies of venetoclax resistance in acute myeloid leukemia",
                    citation="Literature-grounded composite scenario",
                ),
            ],
            expected_findings=[
                ExpectedFinding(
                    finding=(
                        "Bulk post-treatment DE should look mixed and fail to cleanly "
                        "explain relapse with one dominant mechanism."
                    ),
                    category="confounded_de",
                    keywords=["bulk", "mixed", "inconclusive", "relapse"],
                ),
                ExpectedFinding(
                    finding=(
                        "Clustering should separate two relapse-enriched blast "
                        "subclones rather than one unified resistant population."
                    ),
                    category="clonal_structure",
                    keywords=["two_clones", "relapse", "branch", "cluster"],
                ),
                ExpectedFinding(
                    finding=(
                        "One resistant branch should show anti-apoptotic escape "
                        "through MCL1 and BCL2A1."
                    ),
                    category="mechanism",
                    keywords=["MCL1", "BCL2A1", "apoptosis_escape"],
                ),
                ExpectedFinding(
                    finding=(
                        "The second resistant branch should show JAK2-STAT5-PIM1 "
                        "survival signalling."
                    ),
                    category="mechanism",
                    keywords=["JAK2", "STAT5", "PIM1", "survival"],
                ),
                ExpectedFinding(
                    finding=(
                        "Trajectory analysis should support divergence from a shared "
                        "founder blast state into two distinct resistant branches."
                    ),
                    category="trajectory",
                    keywords=["founder", "divergence", "two_branches"],
                ),
            ],
            dataset_metadata={
                "literature_grounding": "aml_venetoclax_multiclone_resistance",
                "designed_failure_mode": "bulk_de_confounding",
            },
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="AML_founder_blast",
                    proportion=0.30,
                    marker_genes=["FLT3", "KIT", "WT1", "MKI67"],
                    state="proliferative",
                    condition_response={"post_venetoclax_resistant": 0.65},
                ),
                CellPopulation(
                    name="MCL1_resistant_clone",
                    proportion=0.18,
                    marker_genes=["MCL1", "BCL2A1", "SOX4", "IL1RAP"],
                    state="resistant_apoptosis_escape",
                    condition_response={"post_venetoclax_resistant": 1.9},
                ),
                CellPopulation(
                    name="JAK2_STAT5_resistant_clone",
                    proportion=0.12,
                    marker_genes=["JAK2", "STAT5A", "PIM1", "SOCS2"],
                    state="resistant_cytokine_survival",
                    condition_response={"post_venetoclax_resistant": 2.1},
                ),
                CellPopulation(
                    name="GMP_like",
                    proportion=0.10,
                    marker_genes=["CEBPA", "CSF3R", "MPO"],
                    state="progenitor",
                ),
                CellPopulation(
                    name="monocyte_like",
                    proportion=0.10,
                    marker_genes=["LST1", "FCN1", "CTSS"],
                    state="differentiated",
                ),
                CellPopulation(
                    name="T_cell",
                    proportion=0.08,
                    marker_genes=["CD3D", "IL7R", "LTB"],
                    state="quiescent",
                ),
                CellPopulation(
                    name="stromal",
                    proportion=0.07,
                    marker_genes=["COL1A1", "CXCL12", "DCN"],
                    state="supportive",
                ),
                CellPopulation(
                    name="erythroid",
                    proportion=0.05,
                    marker_genes=["HBB", "HBA1", "ALAS2"],
                    state="mature",
                ),
            ],
            true_de_genes={
                "post_vs_pre_bulk": {
                    "MCL1": 0.9,
                    "BCL2A1": 1.1,
                    "SOX4": 0.6,
                    "JAK2": 0.7,
                    "STAT5A": 0.8,
                    "PIM1": 1.2,
                    "SOCS2": 1.0,
                    "BCL2": -0.9,
                    "BCL2L11": -0.5,
                    "MKI67": -0.4,
                },
            },
            true_pathways={
                "intrinsic_apoptosis_regulation": 0.88,
                "JAK_STAT_signalling": 0.84,
                "oxidative_phosphorylation": 0.64,
                "stress_response": 0.58,
            },
            true_trajectory={
                "root": "AML_founder_blast",
                "n_lineages": 2,
                "branching": True,
                "branches": [
                    ["AML_founder_blast", "MCL1_resistant_clone"],
                    ["AML_founder_blast", "JAK2_STAT5_resistant_clone"],
                ],
            },
            true_regulatory_network={
                "CREB1": ["MCL1", "BCL2A1", "SOX4"],
                "ATF4": ["MCL1", "DDIT3", "EIF2AK3"],
                "STAT5A": ["PIM1", "SOCS2", "BCL2L1"],
                "MYC": ["MKI67", "MCL1", "IL1RAP"],
            },
            perturbation_effects={
                "venetoclax": {
                    "BCL2": -1.4,
                    "MCL1": 0.4,
                    "JAK2": 0.5,
                    "STAT5A": 0.5,
                    "PIM1": 0.6,
                },
            },
            true_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1", "SOCS2"],
            causal_mechanisms=[
                "An MCL1/BCL2A1 anti-apoptotic escape program sustains one resistant AML subclone under venetoclax pressure",
                "A JAK2-STAT5-PIM1 survival program sustains a second resistant AML subclone in parallel",
            ],
            n_true_cells=18_500,
        ),
        technical=TechnicalState(
            batch_effects={"pre_treatment_batch": 0.10, "relapse_batch": 0.18},
            ambient_rna_fraction=0.08,
            doublet_rate=0.08,
            dropout_rate=0.13,
            sample_quality=0.79,
            library_complexity=0.76,
            sequencing_depth_factor=1.05,
            capture_efficiency=0.58,
        ),
        hidden_failure_conditions=[
            "Bulk DE partially averages away the clone-specific resistance programs",
            "The smaller JAK2-STAT5 clone is close to the detection threshold and can be lost with aggressive filtering",
            "Pre/post treatment batch structure can masquerade as a single dominant relapse program if batches are not integrated",
        ],
    )
