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
            budget_limit=85_000.0,
            time_limit_days=130.0,
            prior_observations=[
                "The patient initially responded to venetoclax-based therapy before relapsing",
                "Bulk expression suggested both apoptotic rewiring and cytokine signalling, but no single mechanism was definitive",
                "Relapse marrow shows heterogeneous blast-state remodeling after treatment",
                "A subset of post-treatment blasts remains highly proliferative, raising concern for a non-mechanistic distractor state",
            ],
            success_criteria=[
                "Resolve whether relapse is driven by one resistant state or multiple resistant subclones",
                "Identify a dominant anti-apoptotic survival program in one resistant branch",
                "Identify a distinct cytokine-responsive survival program in the second resistant branch",
                "Avoid misclassifying a fast-cycling post-treatment blast cluster as a bona fide resistance mechanism",
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
                        "through an anti-apoptotic survival program."
                    ),
                    category="mechanism",
                    keywords=["anti_apoptotic", "survival_program", "escape"],
                ),
                ExpectedFinding(
                    finding=(
                        "The second resistant branch should show a cytokine-driven "
                        "survival signalling program."
                    ),
                    category="mechanism",
                    keywords=["cytokine_signalling", "parallel_branch", "survival"],
                ),
                ExpectedFinding(
                    finding=(
                        "Trajectory analysis should support divergence from a shared "
                        "founder blast state into parallel post-treatment branches, "
                        "only two of which are resistant."
                    ),
                    category="trajectory",
                    keywords=["founder", "divergence", "parallel_branches"],
                ),
                ExpectedFinding(
                    finding=(
                        "A cycling post-treatment blast branch should show a strong "
                        "proliferation signal but should not be treated as a true "
                        "venetoclax resistance mechanism."
                    ),
                    category="distractor",
                    keywords=["cycling", "cell_cycle", "distractor", "non_resistant"],
                ),
            ],
            dataset_metadata={
                "literature_grounding": "aml_venetoclax_multiclone_resistance",
                "designed_failure_mode": "bulk_de_confounding",
                "adversarial_distractor": "cycling_post_treatment_blast",
            },
        ),
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(
                    name="AML_founder_blast",
                    proportion=0.24,
                    marker_genes=["FLT3", "KIT", "WT1", "MKI67"],
                    state="proliferative",
                    condition_response={"post_venetoclax_resistant": 0.65},
                ),
                CellPopulation(
                    name="MCL1_resistant_clone",
                    proportion=0.17,
                    marker_genes=["MCL1", "BCL2A1", "SOX4", "IL1RAP"],
                    state="resistant_apoptosis_escape",
                    condition_response={"post_venetoclax_resistant": 1.9},
                ),
                CellPopulation(
                    name="JAK2_STAT5_resistant_clone",
                    proportion=0.11,
                    marker_genes=["JAK2", "STAT5A", "PIM1", "SOCS2"],
                    state="resistant_cytokine_survival",
                    condition_response={"post_venetoclax_resistant": 2.1},
                ),
                CellPopulation(
                    name="cycling_distractor_clone",
                    proportion=0.08,
                    marker_genes=["MKI67", "TOP2A", "PCNA", "UBE2C"],
                    state="cycling_rebound",
                    condition_response={"post_venetoclax_resistant": 1.55},
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
                    "TOP2A": 0.9,
                    "PCNA": 0.7,
                    "BCL2": -0.9,
                    "BCL2L11": -0.5,
                    "MKI67": 0.8,
                },
            },
            true_pathways={
                "intrinsic_apoptosis_regulation": 0.88,
                "JAK_STAT_signalling": 0.84,
                "cytokine_receptor_signalling": 0.73,
                "integrated_stress_response": 0.69,
                "oxidative_phosphorylation": 0.64,
                "stress_response": 0.58,
            },
            true_trajectory={
                "root": "AML_founder_blast",
                "n_lineages": 3,
                "branching": True,
                "branches": [
                    ["AML_founder_blast", "MCL1_resistant_clone"],
                    ["AML_founder_blast", "JAK2_STAT5_resistant_clone"],
                    ["AML_founder_blast", "cycling_distractor_clone"],
                ],
            },
            true_regulatory_network={
                "CREB1": ["MCL1", "BCL2A1", "SOX4"],
                "ATF4": ["MCL1", "DDIT3", "EIF2AK3"],
                "STAT5A": ["PIM1", "SOCS2", "BCL2L1"],
                "FOXM1": ["MKI67", "TOP2A", "PCNA"],
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
            clone_truth={
                "MCL1_resistant_clone": {
                    "is_resistant": True,
                    "size": 0.18,
                    "markers": [
                        "MCL1",
                        "BCL2A1",
                        "SOX4",
                        "IL1RAP",
                        "BCL2L1",
                        "DDIT3",
                    ],
                    "decoy_markers": ["BAX", "BAK1", "BID", "CASP3"],
                    "de_genes": {
                        "MCL1": 1.8,
                        "BCL2A1": 1.6,
                        "SOX4": 1.1,
                        "IL1RAP": 1.0,
                        "BCL2L1": 0.9,
                        "DDIT3": 0.8,
                        "EIF2AK3": 0.7,
                        "BCL2": -1.2,
                    },
                    "pathways": {
                        "intrinsic_apoptosis_regulation": 0.95,
                        "integrated_stress_response": 0.81,
                        "oxidative_phosphorylation": 0.78,
                    },
                    "regulators": ["CREB1", "ATF4", "MYC", "XBP1"],
                    "mechanism": (
                        "An MCL1/BCL2A1 anti-apoptotic escape program sustains "
                        "one resistant AML subclone under venetoclax pressure"
                    ),
                },
                "JAK2_STAT5_resistant_clone": {
                    "is_resistant": True,
                    "size": 0.12,
                    "markers": [
                        "JAK2",
                        "STAT5A",
                        "PIM1",
                        "SOCS2",
                        "CISH",
                        "BCL2L1",
                    ],
                    "decoy_markers": ["JAK1", "STAT3", "IL6ST", "TYK2"],
                    "de_genes": {
                        "JAK2": 1.5,
                        "STAT5A": 1.4,
                        "PIM1": 1.7,
                        "SOCS2": 1.3,
                        "CISH": 1.1,
                        "BCL2L1": 0.9,
                        "IL7R": 0.8,
                        "BCL2": -0.8,
                    },
                    "pathways": {
                        "JAK_STAT_signalling": 0.97,
                        "cytokine_receptor_signalling": 0.86,
                        "stress_response": 0.62,
                    },
                    "regulators": ["STAT5A", "JAK2", "MYC", "IRF8"],
                    "mechanism": (
                        "A JAK2-STAT5-PIM1 survival program sustains a second "
                        "resistant AML subclone in parallel"
                    ),
                },
                "cycling_distractor_clone": {
                    "is_resistant": False,
                    "size": 0.08,
                    "markers": ["MKI67", "TOP2A", "PCNA", "UBE2C", "CDK1"],
                    "decoy_markers": ["CDK4", "CDK6", "CCND1", "RB1"],
                    "de_genes": {
                        "MKI67": 1.6,
                        "TOP2A": 1.8,
                        "PCNA": 1.3,
                        "UBE2C": 1.2,
                        "CDK1": 1.1,
                    },
                    "pathways": {
                        "cell_cycle": 0.96,
                        "DNA_replication": 0.89,
                    },
                    "regulators": ["FOXM1", "E2F1", "MYBL2"],
                    "mechanism": (
                        "A transient proliferative rebound state expands after "
                        "treatment but does not itself explain venetoclax resistance"
                    ),
                },
            },
            confounders={
                "IL6_STAT3_feedback": 0.89,
                "cell_cycle": 0.91,
                "DNA_replication": 0.87,
                "oxidative_stress_adaptation": 0.84,
                "inflammatory_cytokine_response": 0.81,
            },
            true_markers=[
                "MCL1",
                "BCL2A1",
                "SOX4",
                "IL1RAP",
                "DDIT3",
                "JAK2",
                "STAT5A",
                "PIM1",
                "SOCS2",
                "CISH",
            ],
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
