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
        s.discovered_clusters = ["MCL1_resistant_clone", "JAK2_STAT5_resistant_clone"]
        s.mechanism_confidence = {
            "An MCL1/BCL2A1 anti-apoptotic escape program sustains one resistant AML subclone under venetoclax pressure": 0.8,
        }
        violations = engine.check(
            ExperimentAction(action_type=ActionType.SYNTHESIZE_CONCLUSION),
            s,
        )
        hard = engine.hard_violations(violations)
        assert any("trajectory" in m.lower() for m in hard)
        assert any("regulatory" in m.lower() or "wiring" in m.lower() for m in hard)
        assert any("both resistant mechanisms" in m.lower() for m in hard)


class TestResourceConstraints:
    def test_exhausted_budget_blocked(self):
        s = _state()
        s.resources.budget_used = 100_000
        engine = RuleEngine()
        violations = engine.check(
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE), s,
        )
        hard = engine.hard_violations(violations)
        assert any("budget" in m.lower() for m in hard)
