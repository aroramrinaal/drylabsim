"""Tests for the latent-state simulator modules."""

import pytest

from models import ActionType, ExperimentAction, OutputType
from server.simulator.latent_state import (
    CellPopulation,
    ExperimentProgress,
    FullLatentState,
    LatentBiologicalState,
    ResourceState,
    TechnicalState,
)
from server.simulator.noise import NoiseModel
from server.simulator.output_generator import OutputGenerator
from server.simulator.transition import TransitionEngine


def _make_state() -> FullLatentState:
    return FullLatentState(
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(name="A", proportion=0.6, marker_genes=["G1"]),
                CellPopulation(name="B", proportion=0.4, marker_genes=["G2"]),
            ],
            true_de_genes={"disease_vs_healthy": {"G1": 2.0, "G2": -1.5}},
            true_pathways={"apoptosis": 0.7},
            true_markers=["G1"],
            causal_mechanisms=["G1-driven apoptosis"],
            n_true_cells=5000,
        ),
        technical=TechnicalState(dropout_rate=0.1, doublet_rate=0.04),
        progress=ExperimentProgress(),
        resources=ResourceState(budget_total=50_000, time_limit_days=90),
    )


def _make_multiclone_state() -> FullLatentState:
    return FullLatentState(
        biology=LatentBiologicalState(
            cell_populations=[
                CellPopulation(name="AML_founder_blast", proportion=0.35, marker_genes=["FLT3"]),
                CellPopulation(name="MCL1_resistant_clone", proportion=0.18, marker_genes=["MCL1", "BCL2A1"]),
                CellPopulation(name="JAK2_STAT5_resistant_clone", proportion=0.12, marker_genes=["JAK2", "STAT5A"]),
                CellPopulation(name="GMP_like", proportion=0.20, marker_genes=["CEBPA"]),
                CellPopulation(name="T_cell", proportion=0.15, marker_genes=["CD3D"]),
            ],
            true_de_genes={"post_vs_pre_bulk": {"MCL1": 0.9, "JAK2": 0.7, "PIM1": 1.0}},
            true_pathways={"intrinsic_apoptosis_regulation": 0.88, "JAK_STAT_signalling": 0.84},
            true_trajectory={"root": "AML_founder_blast", "n_lineages": 2, "branching": True},
            true_regulatory_network={"CREB1": ["MCL1"], "STAT5A": ["PIM1"]},
            clone_truth={
                "MCL1_resistant_clone": {
                    "size": 0.18,
                    "markers": ["MCL1", "BCL2A1"],
                    "de_genes": {"MCL1": 1.8, "BCL2A1": 1.4},
                    "pathways": {"intrinsic_apoptosis_regulation": 0.95},
                    "regulators": ["CREB1", "ATF4"],
                    "mechanism": "MCL1 escape",
                },
                "JAK2_STAT5_resistant_clone": {
                    "size": 0.12,
                    "markers": ["JAK2", "STAT5A", "PIM1"],
                    "de_genes": {"JAK2": 1.5, "STAT5A": 1.4, "PIM1": 1.7},
                    "pathways": {"JAK_STAT_signalling": 0.97},
                    "regulators": ["STAT5A", "JAK2"],
                    "mechanism": "STAT5 survival",
                },
            },
            true_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            causal_mechanisms=["MCL1 escape", "STAT5 survival"],
            n_true_cells=9000,
        ),
        technical=TechnicalState(
            batch_effects={"pre": 0.10, "post": 0.18},
            dropout_rate=0.12,
            doublet_rate=0.05,
        ),
        progress=ExperimentProgress(data_normalized=True, batches_integrated=True),
        resources=ResourceState(budget_total=120_000, time_limit_days=180),
        scenario_name="venetoclax_resistance_multiclone",
    )


class TestNoiseModel:
    def test_deterministic_with_seed(self):
        n1 = NoiseModel(seed=42)
        n2 = NoiseModel(seed=42)
        assert n1.sample_qc_metric(0.5, 0.1) == n2.sample_qc_metric(0.5, 0.1)

    def test_false_positives(self):
        n = NoiseModel(seed=0)
        fps = n.generate_false_positives(1000, 0.01)
        assert all(g.startswith("FP_GENE_") for g in fps)

    def test_quality_degradation_bounded(self):
        n = NoiseModel(seed=0)
        for _ in range(100):
            q = n.quality_degradation(0.9, [0.8, 0.7])
            assert 0.0 <= q <= 1.0


class TestOutputGenerator:
    def test_collect_sample(self):
        noise = NoiseModel(seed=1)
        gen = OutputGenerator(noise)
        s = _make_state()
        action = ExperimentAction(
            action_type=ActionType.COLLECT_SAMPLE,
            parameters={"n_samples": 4},
        )
        out = gen.generate(action, s, 1)
        assert out.output_type == OutputType.SAMPLE_COLLECTION_RESULT
        assert out.data["n_samples"] == 4

    def test_de_includes_true_genes(self):
        noise = NoiseModel(seed=42)
        gen = OutputGenerator(noise)
        s = _make_state()
        s.progress.data_normalized = True
        action = ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
            parameters={"comparison": "disease_vs_healthy"},
        )
        out = gen.generate(action, s, 5)
        assert out.output_type == OutputType.DE_RESULT
        gene_names = [g["gene"] for g in out.data["top_genes"]]
        assert "G1" in gene_names or "G2" in gene_names

    def test_expert_multiclone_de_emits_clone_specific_results(self):
        noise = NoiseModel(seed=42)
        gen = OutputGenerator(noise)
        s = _make_multiclone_state()
        s.progress.cells_clustered = True
        action = ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
            parameters={"comparison": "post_vs_pre_bulk"},
        )
        out = gen.generate(action, s, 5)
        assert out.output_type == OutputType.DE_RESULT
        assert out.data["bulk_signal_is_mixed"] is True
        assert "clone_de" in out.data
        assert "subpopulation_0" in out.data["clone_de"]
        assert "subpopulation_1" in out.data["clone_de"]
        assert "mechanism" not in out.data["clone_de"]["subpopulation_0"]
        assert "relapse_enriched_clusters" not in out.data

    def test_expert_pathway_enrichment_emits_clone_pathways(self):
        noise = NoiseModel(seed=7)
        gen = OutputGenerator(noise)
        s = _make_multiclone_state()
        s.progress.cells_clustered = True
        s.progress.de_performed = True
        action = ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT)
        out = gen.generate(action, s, 6)
        assert out.output_type == OutputType.PATHWAY_RESULT
        assert "clone_pathways" in out.data
        assert "subpopulation_0" in out.data["clone_pathways"]
        assert "inferred_mechanisms" not in out.data

    def test_expert_clustering_omits_relapse_hint_and_noises_sizes(self):
        noise = NoiseModel(seed=11)
        gen = OutputGenerator(noise)
        s = _make_multiclone_state()
        out = gen.generate(ExperimentAction(action_type=ActionType.CLUSTER_CELLS), s, 4)
        assert out.output_type == OutputType.CLUSTER_RESULT
        assert "relapse_enriched_clusters" not in out.data
        assert out.data["cluster_markers_available"] is True
        assert sum(out.data["cluster_sizes"]) == s.biology.n_true_cells


class TestTransitionEngine:
    def test_progress_flags_set(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        action = ExperimentAction(action_type=ActionType.COLLECT_SAMPLE)
        result = engine.step(s, action)
        assert result.next_state.progress.samples_collected is True

    def test_hard_violation_blocks(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        result = engine.step(
            s,
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            hard_violations=["test_block"],
        )
        assert result.output.success is False
        assert result.output.output_type == OutputType.FAILURE_REPORT

    def test_resource_deduction(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        action = ExperimentAction(action_type=ActionType.SEQUENCE_CELLS)
        s.progress.library_prepared = True
        result = engine.step(s, action)
        assert result.next_state.resources.budget_used == 15_000

    def test_conclusion_ends_episode(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_state()
        s.progress.de_performed = True
        action = ExperimentAction(action_type=ActionType.SYNTHESIZE_CONCLUSION)
        result = engine.step(s, action)
        assert result.done is True

    def test_expert_analysis_outputs_no_longer_autopropagate_mechanisms(self):
        noise = NoiseModel(seed=0)
        engine = TransitionEngine(noise)
        s = _make_multiclone_state()
        s.progress.cells_clustered = True
        s.progress.de_performed = True
        result = engine.step(s, ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT))
        assert "mechanism_confidence" not in result.next_state.model_dump()
