"""Tests for the biological rule engine."""

from models import ActionType, ExperimentAction
from server.rules.engine import RuleEngine, Severity
from server.simulator.latent_state import (
    ExperimentProgress,
    FullLatentState,
    ResourceState,
)


def _state(**progress_flags) -> FullLatentState:
    return FullLatentState(
        progress=ExperimentProgress(**progress_flags),
        resources=ResourceState(budget_total=100_000, time_limit_days=180),
    )


class TestPrerequisites:
    def test_sequence_without_library_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            _state(samples_collected=True),
        )
        hard = engine.hard_violations(violations)
        assert any("library" in m.lower() for m in hard)

    def test_sequence_with_library_allowed(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            _state(samples_collected=True, library_prepared=True),
        )
        hard = engine.hard_violations(violations)
        assert not hard

    def test_de_without_normalization_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION),
            _state(cells_sequenced=True, qc_performed=True, data_filtered=True),
        )
        hard = engine.hard_violations(violations)
        assert any("normalis" in m.lower() or "normaliz" in m.lower() for m in hard)

    def test_validate_marker_without_discovery_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.VALIDATE_MARKER),
            _state(de_performed=True),
        )
        hard = engine.hard_violations(violations)
        assert any("marker" in m.lower() for m in hard)

    def test_expert_de_without_clustering_blocked(self):
        engine = RuleEngine()
        s = _state(
            samples_collected=True,
            library_prepared=True,
            cells_sequenced=True,
            qc_performed=True,
            data_filtered=True,
            data_normalized=True,
        )
        s.scenario_name = "venetoclax_resistance_multiclone"
        violations = engine.check(
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("clustering" in m.lower() for m in hard)


class TestRedundancy:
    def test_double_qc_is_hard_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.RUN_QC),
            _state(cells_sequenced=True, qc_performed=True),
        )
        hard = engine.hard_violations(violations)
        assert any("redundant" in m.lower() for m in hard)

    def test_repeated_followup_design_is_hard_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.DESIGN_FOLLOWUP),
            _state(followup_designed=True, de_performed=True),
        )
        hard = engine.hard_violations(violations)
        assert any("redundant" in m.lower() for m in hard)


class TestMetaActionTiming:
    def test_followup_design_without_analysis_is_hard_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.DESIGN_FOLLOWUP),
            _state(),
        )
        hard = engine.hard_violations(violations)
        assert any("follow-up design" in m.lower() for m in hard)

    def test_subagent_review_without_analysis_is_hard_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.REQUEST_SUBAGENT_REVIEW),
            _state(),
        )
        hard = engine.hard_violations(violations)
        assert any("subagent review" in m.lower() for m in hard)

    def test_conclusion_without_marker_or_mechanism_evidence_is_hard_blocked(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SYNTHESIZE_CONCLUSION),
            _state(data_normalized=True, cells_clustered=True),
        )
        hard = engine.hard_violations(violations)
        assert any("markers" in m.lower() for m in hard)
        assert any("pathways or mechanisms" in m.lower() for m in hard)

    def test_conclusion_with_marker_and_mechanism_evidence_is_allowed(self):
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SYNTHESIZE_CONCLUSION),
            _state(
                data_normalized=True,
                cells_clustered=True,
                markers_discovered=True,
                pathways_analyzed=True,
            ),
        )
        hard = engine.hard_violations(violations)
        assert not hard

    def test_expert_conclusion_requires_branch_resolving_evidence(self):
        engine = RuleEngine()
        s = _state(
            data_normalized=True,
            cells_clustered=True,
            de_performed=True,
            markers_discovered=True,
            pathways_analyzed=True,
        )
        s.scenario_name = "venetoclax_resistance_multiclone"
        s.discovered_clusters = ["cluster_1", "cluster_2"]
        s.discovered_clone_markers = {"cluster_1": ["MCL1"]}
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SYNTHESIZE_CONCLUSION),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("trajectory" in m.lower() for m in hard)
        assert any("regulatory" in m.lower() or "wiring" in m.lower() for m in hard)
        assert any("marker-supported evidence" in m.lower() for m in hard)

    def test_strict_scenario_requires_mechanism_claims(self):
        engine = RuleEngine()
        s = _state(
            data_normalized=True,
            cells_clustered=True,
            de_performed=True,
            markers_discovered=True,
            pathways_analyzed=True,
        )
        s.scenario_name = "perturbation_immune"
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [{"top_markers": ["STAT1"], "evidence_steps": [3, 4]}]
                },
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("causal mechanism" in m.lower() for m in hard)

    def test_mechanism_claims_require_evidence_steps(self):
        engine = RuleEngine()
        s = _state(
            data_normalized=True,
            cells_clustered=True,
            de_performed=True,
            markers_discovered=True,
            pathways_analyzed=True,
        )
        s.scenario_name = "perturbation_immune"
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "top_markers": ["STAT1"],
                            "causal_mechanisms": ["JAK-STAT pathway inhibition"],
                            "evidence_steps": [],
                        }
                    ]
                },
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("evidence_steps" in m.lower() for m in hard)

    def test_marker_claim_without_validation_is_soft_violation(self):
        engine = RuleEngine()
        s = _state(
            data_normalized=True,
            cells_clustered=True,
            de_performed=True,
            markers_discovered=True,
            pathways_analyzed=True,
        )
        s.scenario_name = "perturbation_immune"
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "top_markers": ["STAT1"],
                            "causal_mechanisms": ["JAK-STAT pathway inhibition"],
                            "evidence_steps": [4, 5],
                        }
                    ]
                },
            ),
            s,
        )
        soft = engine.soft_violations(violations)
        assert any("validating at least one marker" in m.lower() for m in soft)

    def test_malformed_mechanism_payload_does_not_crash_rule_engine(self):
        engine = RuleEngine()
        s = _state(
            data_normalized=True,
            cells_clustered=True,
            de_performed=True,
            markers_discovered=True,
            pathways_analyzed=True,
        )
        s.scenario_name = "perturbation_immune"
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "top_markers": ["STAT1"],
                            "causal_mechanisms": [{"unexpected": "shape"}],
                            "evidence_steps": [4, 5],
                        }
                    ]
                },
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("causal mechanism" in m.lower() for m in hard)

    def test_expert_single_clone_conclusion_is_hard_blocked(self):
        engine = RuleEngine()
        s = _state(
            data_normalized=True,
            cells_clustered=True,
            de_performed=True,
            pathways_analyzed=True,
            networks_inferred=True,
            trajectories_inferred=True,
            markers_discovered=True,
            markers_validated=True,
        )
        s.scenario_name = "venetoclax_resistance_multiclone"
        s.discovered_clusters = ["cluster_1", "cluster_2"]
        s.discovered_clone_markers = {
            "cluster_1": ["MCL1", "BCL2A1"],
            "cluster_2": ["JAK2", "STAT5A"],
        }
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "clonal_claims": [
                                {
                                    "subpopulation_id": "cluster_1",
                                    "markers": ["MCL1", "BCL2A1"],
                                    "mechanism": "An MCL1/BCL2A1 anti-apoptotic escape program",
                                }
                            ],
                            "causal_mechanisms": [
                                "An MCL1/BCL2A1 anti-apoptotic escape program"
                            ],
                            "evidence_steps": [8, 9, 10],
                        }
                    ]
                },
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("two clone-resolved claims" in m.lower() for m in hard)
        assert any("fewer than two resistant mechanisms" in m.lower() for m in hard)


class TestValidateMarkerRedundancy:
    def test_same_marker_same_subpop_is_blocked(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = ["MCL1::cluster_1"]
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "MCL1", "subpopulation_id": "cluster_1"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("already validated" in m.lower() for m in hard)

    def test_same_marker_different_subpop_is_allowed(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = ["MCL1::cluster_1"]
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "MCL1", "subpopulation_id": "cluster_2"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert not any("redundant" in m.lower() for m in hard)

    def test_different_marker_same_subpop_is_allowed(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = ["MCL1::cluster_1"]
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "BCL2A1", "subpopulation_id": "cluster_1"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert not any("redundant" in m.lower() for m in hard)

    def test_different_marker_different_subpop_is_allowed(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = ["MCL1::cluster_1"]
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "JAK2", "subpopulation_id": "cluster_2"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert not any("redundant" in m.lower() for m in hard)

    def test_no_subpop_same_marker_is_blocked(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = ["NPPA"]
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "NPPA"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("already validated" in m.lower() for m in hard)

    def test_no_subpop_different_marker_is_allowed(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = ["NPPA"]
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "BCL2A1"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert not any("redundant" in m.lower() for m in hard)

    def test_first_validate_marker_never_blocked_by_pair_check(self):
        engine = RuleEngine()
        s = _state(markers_discovered=True, markers_validated=True)
        s.validated_marker_pairs = []
        violations = engine.check(
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "MCL1", "subpopulation_id": "cluster_1"},
            ),
            s,
        )
        hard = engine.hard_violations(violations)
        assert not any("redundant" in m.lower() for m in hard)


class TestResourceConstraints:
    def test_exhausted_budget_blocked(self):
        s = _state()
        s.resources.budget_used = 100_000
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("budget" in m.lower() for m in hard)
