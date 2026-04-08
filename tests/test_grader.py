"""Tests for the DryLabSim grader.

Covers:
  - Perfect episode → score close to 1.0
  - Zero conclusion / empty episode → score close to 0.0
  - Partial episode → score between 0.2 and 0.6
  - Wrong markers but correct mechanisms → higher than all wrong
  - Reproducibility: same inputs, same score twice
  - Boundary: score always in [0.0, 1.0] across random scenarios
  - Difficulty distinction: easy > hard for same agent quality
"""

from models import (
    ActionType,
    ClonalClaim,
    ConclusionClaim,
    ExperimentObservation,
    IntermediateOutput,
    OutputType,
    PipelineStepRecord,
    ResourceUsage,
    TaskSpec,
)
from server.grader import GradeResult, grade_episode
from server.grader.biology import score_biology
from server.grader.pipeline import score_pipeline
from server.simulator.latent_state import (
    ExperimentProgress,
    FullLatentState,
    LatentBiologicalState,
    ResourceState,
)


def _make_obs(
    discovered_markers=None,
    candidate_mechanisms=None,
    conclusions=None,
    step_index=0,
    done=True,
    reward=0.0,
):
    return ExperimentObservation(
        task=TaskSpec(),
        step_index=step_index,
        pipeline_history=[],
        available_assays=[],
        available_tools=[],
        resource_usage=ResourceUsage(),
        latest_output=None,
        all_outputs=[],
        discovered_markers=discovered_markers or [],
        candidate_mechanisms=candidate_mechanisms or [],
        uncertainty_summary={},
        subagent_outputs=[],
        conclusions=conclusions or [],
        rule_violations=[],
        step_reward_breakdown={},
        done=done,
        reward=reward,
    )


def _make_latent(
    progress=None,
    true_markers=None,
    causal_mechanisms=None,
    true_pathways=None,
    clone_truth=None,
    confounders=None,
    budget_total=100_000,
    budget_used=0,
    time_limit=180,
    time_used=0,
    scenario_name="",
):
    return FullLatentState(
        biology=LatentBiologicalState(
            true_markers=true_markers or [],
            causal_mechanisms=causal_mechanisms or [],
            true_pathways=true_pathways or {},
            clone_truth=clone_truth or {},
            confounders=confounders or {},
        ),
        progress=progress or ExperimentProgress(),
        resources=ResourceState(
            budget_total=budget_total,
            budget_used=budget_used,
            time_limit_days=time_limit,
            time_used_days=time_used,
        ),
        scenario_name=scenario_name,
    )


class TestGradeResultBounds:
    def test_score_always_in_01(self):
        result = GradeResult(
            score=1.5,
            completeness=2.0,
            biology_score=-0.5,
            efficiency_score=0.8,
        )
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.completeness <= 1.0
        assert 0.0 <= result.biology_score <= 1.0
        assert 0.0 <= result.efficiency_score <= 1.0


class TestPerfectEpisode:
    def test_perfect_episode_scores_near_1(self):
        progress = ExperimentProgress(
            samples_collected=True,
            cells_sequenced=True,
            qc_performed=True,
            data_filtered=True,
            data_normalized=True,
            de_performed=True,
            cells_clustered=True,
            pathways_analyzed=True,
            markers_discovered=True,
            markers_validated=True,
            conclusion_reached=True,
        )
        latent = _make_latent(
            progress=progress,
            true_markers=["NPPA", "NPPB", "POSTN"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
            true_pathways={"cardiac_muscle_contraction": 0.8},
            budget_total=100_000,
            budget_used=30_000,
            time_used=60,
        )
        obs = _make_obs(
            discovered_markers=["NPPA", "NPPB", "POSTN"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["NPPA", "NPPB", "POSTN"],
                    causal_mechanisms=["TGF-beta-driven fibrosis"],
                    predicted_pathways={"cardiac_muscle_contraction": 0.8},
                    confidence=0.9,
                ),
            ],
        )
        result = grade_episode(obs, latent)
        assert result.score > 0.84
        assert result.completeness >= 0.89
        assert result.biology_score > 0.86
        assert 0.0 <= result.score <= 1.0


class TestEmptyEpisode:
    def test_no_conclusion_no_discoveries_scores_near_0(self):
        latent = _make_latent(
            progress=ExperimentProgress(),
            true_markers=["NPPA", "NPPB"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
            true_pathways={"cardiac_muscle_contraction": 0.8},
        )
        obs = _make_obs()
        result = grade_episode(obs, latent)
        assert result.score < 0.20
        assert result.completeness == 0.0
        assert result.biology_score == 0.0
        assert 0.0 <= result.score <= 1.0


class TestPartialEpisode:
    def test_partial_pipeline_partial_biology(self):
        progress = ExperimentProgress(
            samples_collected=True,
            cells_sequenced=True,
            qc_performed=True,
            data_filtered=True,
            data_normalized=True,
            de_performed=True,
        )
        latent = _make_latent(
            progress=progress,
            true_markers=["NPPA", "NPPB", "POSTN"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
            true_pathways={"cardiac_muscle_contraction": 0.8},
            budget_total=100_000,
            budget_used=60_000,
        )
        obs = _make_obs(
            discovered_markers=["NPPA"],
            candidate_mechanisms=[],
            conclusions=[
                ConclusionClaim(
                    top_markers=["NPPA"],
                    causal_mechanisms=[],
                    confidence=0.5,
                ),
            ],
        )
        result = grade_episode(obs, latent)
        assert 0.2 <= result.score <= 0.6
        assert result.completeness > 0.0
        assert result.biology_score > 0.0
        assert result.biology_score < 1.0
        assert 0.0 <= result.score <= 1.0


class TestWrongVsPartial:
    def test_wrong_markers_correct_mechanisms_beats_all_wrong(self):
        true_markers = ["NPPA", "NPPB"]
        true_mechs = ["TGF-beta-driven fibrosis"]

        latent = _make_latent(
            progress=ExperimentProgress(conclusion_reached=True),
            true_markers=true_markers,
            causal_mechanisms=true_mechs,
        )

        obs_wrong_markers = _make_obs(
            discovered_markers=["WRONG1", "WRONG2"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["WRONG1"],
                    causal_mechanisms=["TGF-beta-driven fibrosis"],
                ),
            ],
        )
        obs_all_wrong = _make_obs(
            discovered_markers=["WRONG1", "WRONG2"],
            candidate_mechanisms=["unrelated process"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["WRONG1"],
                    causal_mechanisms=["unrelated process"],
                ),
            ],
        )

        r_partial = grade_episode(obs_wrong_markers, latent)
        r_all_wrong = grade_episode(obs_all_wrong, latent)

        assert r_partial.biology_score > r_all_wrong.biology_score
        assert r_partial.score > r_all_wrong.score


class TestReproducibility:
    def test_same_inputs_same_score(self):
        progress = ExperimentProgress(
            samples_collected=True,
            cells_sequenced=True,
            qc_performed=True,
            data_normalized=True,
            de_performed=True,
            conclusion_reached=True,
        )
        latent = _make_latent(
            progress=progress,
            true_markers=["NPPA", "NPPB"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
            true_pathways={"cardiac_muscle_contraction": 0.8},
            budget_total=100_000,
            budget_used=50_000,
        )
        obs = _make_obs(
            discovered_markers=["NPPA"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["NPPA"],
                    causal_mechanisms=["TGF-beta-driven fibrosis"],
                    confidence=0.8,
                ),
            ],
        )

        result1 = grade_episode(obs, latent)
        result2 = grade_episode(obs, latent)

        assert result1.score == result2.score
        assert result1.completeness == result2.completeness
        assert result1.biology_score == result2.biology_score
        assert result1.efficiency_score == result2.efficiency_score

    def test_same_inputs_same_score_with_mechanism_confidence(self):
        progress = ExperimentProgress(
            samples_collected=True,
            cells_sequenced=True,
            qc_performed=True,
            data_normalized=True,
            de_performed=True,
            conclusion_reached=True,
        )
        latent = _make_latent(
            progress=progress,
            true_markers=["MCL1", "JAK2"],
            causal_mechanisms=["MCL1 escape", "STAT5 survival"],
            true_pathways={"JAK_STAT_signalling": 0.8},
            budget_total=120_000,
            budget_used=60_000,
        )
        obs = _make_obs(
            discovered_markers=["MCL1", "JAK2"],
            candidate_mechanisms=["MCL1 escape", "STAT5 survival"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["MCL1", "JAK2"],
                    causal_mechanisms=["MCL1 escape", "STAT5 survival"],
                    mechanism_confidence={
                        "MCL1 escape": 0.82,
                        "STAT5 survival": 0.67,
                    },
                    confidence=0.78,
                ),
            ],
        )

        result1 = grade_episode(obs, latent)
        result2 = grade_episode(obs, latent)

        assert result1.score == result2.score
        assert 0.0 <= result1.score <= 1.0


class TestMechanismConfidence:
    def test_wrong_high_confidence_mechanism_is_penalized(self):
        latent = _make_latent(
            progress=ExperimentProgress(conclusion_reached=True),
            true_markers=["MCL1", "JAK2"],
            causal_mechanisms=["MCL1 escape", "STAT5 survival"],
            true_pathways={"JAK_STAT_signalling": 0.8},
        )
        obs_good = _make_obs(
            conclusions=[
                ConclusionClaim(
                    causal_mechanisms=["MCL1 escape", "STAT5 survival"],
                    mechanism_confidence={
                        "MCL1 escape": 0.8,
                        "STAT5 survival": 0.65,
                    },
                    confidence=0.75,
                ),
            ],
        )
        obs_bad = _make_obs(
            conclusions=[
                ConclusionClaim(
                    causal_mechanisms=["MCL1 escape", "totally wrong mechanism"],
                    mechanism_confidence={
                        "MCL1 escape": 0.8,
                        "totally wrong mechanism": 0.95,
                    },
                    confidence=0.75,
                ),
            ],
        )

        good = grade_episode(obs_good, latent)
        bad = grade_episode(obs_bad, latent)

        assert good.score > bad.score


class TestBoundaryAcrossScenarios:
    def test_score_bounded_across_20_random_configs(self):
        import random

        random.seed(42)

        all_markers = [
            "NPPA",
            "NPPB",
            "POSTN",
            "COL1A1",
            "MCL1",
            "BCL2A1",
            "GATA1",
            "CEBPA",
            "SPI1",
            "STAT5A",
            "SOCS2",
            "IFNG",
        ]
        all_mechs = [
            "TGF-beta-driven fibrosis",
            "inflammatory macrophage infiltration",
            "GATA1-driven erythroid commitment",
            "PU.1/CEBPA antagonism at myeloid branch point",
            "JAK-STAT pathway inhibition reduces Th1/Th17 activation",
            "Compensatory Treg expansion under JAK inhibition",
            "An MCL1/BCL2A1 anti-apoptotic escape program sustains one resistant AML subclone under venetoclax pressure",
        ]

        for i in range(20):
            n_markers = random.randint(0, 4)
            n_mechs = random.randint(0, 2)
            n_true_markers = random.randint(1, 4)
            n_true_mechs = random.randint(1, 2)

            true_markers = random.sample(
                all_markers, min(n_true_markers, len(all_markers))
            )
            true_mechs = random.sample(all_mechs, min(n_true_mechs, len(all_mechs)))

            discovered = random.sample(all_markers, min(n_markers, len(all_markers)))
            candidate = random.sample(all_mechs, min(n_mechs, len(all_mechs)))

            milestone_flags = {}
            for flag in [
                "samples_collected",
                "cells_sequenced",
                "qc_performed",
                "data_filtered",
                "data_normalized",
                "de_performed",
                "conclusion_reached",
            ]:
                milestone_flags[flag] = random.choice([True, False])

            progress = ExperimentProgress(**milestone_flags)
            budget_used = random.uniform(0, 100_000)
            time_used = random.uniform(0, 180)

            latent = _make_latent(
                progress=progress,
                true_markers=true_markers,
                causal_mechanisms=true_mechs,
                true_pathways={"some_pathway": random.uniform(0.3, 0.9)},
                budget_total=100_000,
                budget_used=budget_used,
                time_used=time_used,
            )
            obs = _make_obs(
                discovered_markers=discovered,
                candidate_mechanisms=candidate,
            )

            result = grade_episode(obs, latent)
            assert 0.0 <= result.score <= 1.0, f"Scenario {i}: score={result.score}"
            assert 0.0 <= result.completeness <= 1.0
            assert 0.0 <= result.biology_score <= 1.0
            assert 0.0 <= result.efficiency_score <= 1.0


class TestDifficultyDistinction:
    def test_easy_task_higher_score_than_hard_for_same_agent_quality(self):
        """For the same agent performance, easy tasks should score higher
        because they have fewer ground-truth items to discover."""
        agent_markers = ["NPPA"]
        agent_mechs = ["TGF-beta-driven fibrosis"]

        easy_latent = _make_latent(
            progress=ExperimentProgress(
                samples_collected=True,
                cells_sequenced=True,
                qc_performed=True,
                data_normalized=True,
                de_performed=True,
                markers_discovered=True,
                conclusion_reached=True,
            ),
            true_markers=["NPPA", "NPPB"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
            true_pathways={"cardiac_muscle_contraction": 0.8},
            budget_total=80_000,
            budget_used=40_000,
        )

        hard_latent = _make_latent(
            progress=ExperimentProgress(
                samples_collected=True,
                cells_sequenced=True,
                qc_performed=True,
                data_normalized=True,
                de_performed=True,
                markers_discovered=True,
                conclusion_reached=True,
            ),
            true_markers=["NPPA", "NPPB", "POSTN", "COL1A1", "STAT1"],
            causal_mechanisms=[
                "TGF-beta-driven fibrosis",
                "inflammatory macrophage infiltration",
                "JAK-STAT pathway inhibition",
            ],
            true_pathways={
                "cardiac_muscle_contraction": 0.8,
                "inflammatory_response": 0.7,
                "JAK_STAT_signalling": 0.3,
            },
            budget_total=120_000,
            budget_used=60_000,
        )

        obs = _make_obs(
            discovered_markers=agent_markers,
            candidate_mechanisms=agent_mechs,
            conclusions=[
                ConclusionClaim(
                    top_markers=agent_markers,
                    causal_mechanisms=agent_mechs,
                    confidence=0.8,
                ),
            ],
        )

        easy_result = grade_episode(obs, easy_latent)
        hard_result = grade_episode(obs, hard_latent)

        assert easy_result.biology_score > hard_result.biology_score, (
            f"Easy biology={easy_result.biology_score:.3f} should beat "
            f"hard biology={hard_result.biology_score:.3f} for same agent output"
        )


class TestPipelineScoring:
    def test_no_milestones_zero(self):
        state = _make_latent(progress=ExperimentProgress())
        assert score_pipeline(state) == 0.0

    def test_all_milestones_one(self):
        progress = ExperimentProgress(
            samples_collected=True,
            cells_sequenced=True,
            qc_performed=True,
            data_filtered=True,
            data_normalized=True,
            de_performed=True,
            cells_clustered=True,
            pathways_analyzed=True,
            markers_discovered=True,
            markers_validated=True,
            conclusion_reached=True,
            trajectories_inferred=True,
            networks_inferred=True,
        )
        state = _make_latent(progress=progress)
        assert score_pipeline(state) == 1.0

    def test_partial_milestones_between_0_and_1(self):
        progress = ExperimentProgress(
            samples_collected=True,
            cells_sequenced=True,
            qc_performed=True,
        )
        state = _make_latent(progress=progress)
        score = score_pipeline(state)
        assert 0.0 < score < 1.0


class TestBiologyScoring:
    def test_empty_truth_returns_zero(self):
        state = _make_latent()
        score = score_biology(state, ["NPPA"], ["fibrosis"], [])
        assert score == 0.0

    def test_empty_predictions_returns_zero(self):
        state = _make_latent(
            true_markers=["NPPA"],
            causal_mechanisms=["fibrosis"],
        )
        score = score_biology(state, [], [], [])
        assert score == 0.0

    def test_perfect_match_high_score(self):
        state = _make_latent(
            true_markers=["NPPA", "NPPB"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
        )
        score = score_biology(
            state,
            discovered_markers=["NPPA", "NPPB"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[],
        )
        assert score > 0.8

    def test_marker_dumping_gets_precision_penalty(self):
        state = _make_latent(
            true_markers=["NPPA", "NPPB"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
        )
        focused = score_biology(
            state,
            discovered_markers=["NPPA", "NPPB"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[],
        )
        dumped = score_biology(
            state,
            discovered_markers=["NPPA", "NPPB", "WRONG1", "WRONG2", "WRONG3", "WRONG4"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[],
        )
        assert focused > dumped

    def test_confounder_pathways_reduce_biology_score(self):
        state = _make_latent(
            true_markers=["NPPA"],
            causal_mechanisms=["TGF-beta-driven fibrosis"],
            true_pathways={"cardiac_muscle_contraction": 0.8},
        )
        state.biology.confounders = {"TNF_NFkB_signalling": 0.9}

        clean = score_biology(
            state,
            discovered_markers=["NPPA"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["NPPA"],
                    causal_mechanisms=["TGF-beta-driven fibrosis"],
                    predicted_pathways={"cardiac_muscle_contraction": 0.8},
                )
            ],
        )
        confounded = score_biology(
            state,
            discovered_markers=["NPPA"],
            candidate_mechanisms=["TGF-beta-driven fibrosis"],
            conclusions=[
                ConclusionClaim(
                    top_markers=["NPPA"],
                    causal_mechanisms=["TGF-beta-driven fibrosis"],
                    predicted_pathways={
                        "cardiac_muscle_contraction": 0.8,
                        "TNF_NFkB_signalling": 0.9,
                    },
                )
            ],
        )
        assert clean > confounded

    def test_expert_multiclone_caps_flat_claims_and_rewards_clonal_resolution(self):
        clone_truth = {
            "MCL1_resistant_clone": {
                "is_resistant": True,
                "size": 0.18,
                "markers": ["MCL1", "BCL2A1", "SOX4"],
                "pathways": {"intrinsic_apoptosis_regulation": 0.95},
                "mechanism": "An MCL1/BCL2A1 anti-apoptotic escape program",
            },
            "JAK2_STAT5_resistant_clone": {
                "is_resistant": True,
                "size": 0.12,
                "markers": ["JAK2", "STAT5A", "PIM1"],
                "pathways": {"JAK_STAT_signalling": 0.97},
                "mechanism": "A JAK2-STAT5-PIM1 survival program",
            },
            "cycling_distractor_clone": {
                "is_resistant": False,
                "size": 0.08,
                "markers": ["MKI67", "TOP2A", "PCNA"],
                "pathways": {"cell_cycle": 0.96},
                "mechanism": "Cycling distractor",
            },
        }
        state = _make_latent(
            progress=ExperimentProgress(markers_validated=True),
            true_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            causal_mechanisms=[
                "An MCL1/BCL2A1 anti-apoptotic escape program",
                "A JAK2-STAT5-PIM1 survival program",
            ],
            true_pathways={
                "intrinsic_apoptosis_regulation": 0.9,
                "JAK_STAT_signalling": 0.85,
            },
            clone_truth=clone_truth,
            confounders={"cell_cycle": 0.9},
            scenario_name="venetoclax_resistance_multiclone",
        )

        flat = score_biology(
            state,
            discovered_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            candidate_mechanisms=[
                "An MCL1/BCL2A1 anti-apoptotic escape program",
                "A JAK2-STAT5-PIM1 survival program",
            ],
            conclusions=[
                ConclusionClaim(
                    top_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
                    causal_mechanisms=[
                        "An MCL1/BCL2A1 anti-apoptotic escape program",
                        "A JAK2-STAT5-PIM1 survival program",
                    ],
                    predicted_pathways={
                        "intrinsic_apoptosis_regulation": 0.9,
                        "JAK_STAT_signalling": 0.85,
                    },
                )
            ],
        )
        structured = score_biology(
            state,
            discovered_markers=[],
            candidate_mechanisms=[],
            conclusions=[
                ConclusionClaim(
                    clonal_claims=[
                        ClonalClaim(
                            subpopulation_id="cluster_1",
                            markers=["MCL1", "BCL2A1", "SOX4"],
                            mechanism="An MCL1/BCL2A1 anti-apoptotic escape program",
                            supporting_pathways=["intrinsic_apoptosis_regulation"],
                        ),
                        ClonalClaim(
                            subpopulation_id="cluster_2",
                            markers=["JAK2", "STAT5A", "PIM1"],
                            mechanism="A JAK2-STAT5-PIM1 survival program",
                            supporting_pathways=["JAK_STAT_signalling"],
                        ),
                    ],
                    clone_size_estimates={"cluster_1": 0.18, "cluster_2": 0.12},
                    predicted_pathways={
                        "intrinsic_apoptosis_regulation": 0.9,
                        "JAK_STAT_signalling": 0.85,
                    },
                )
            ],
        )
        contaminated = score_biology(
            state,
            discovered_markers=[],
            candidate_mechanisms=[],
            conclusions=[
                ConclusionClaim(
                    clonal_claims=[
                        ClonalClaim(
                            subpopulation_id="cluster_1",
                            markers=["MCL1", "BCL2A1", "MKI67", "TOP2A"],
                            mechanism="An MCL1/BCL2A1 anti-apoptotic escape program",
                            supporting_pathways=["intrinsic_apoptosis_regulation", "cell_cycle"],
                        ),
                        ClonalClaim(
                            subpopulation_id="cluster_2",
                            markers=["JAK2", "STAT5A", "PIM1"],
                            mechanism="A JAK2-STAT5-PIM1 survival program",
                            supporting_pathways=["JAK_STAT_signalling"],
                        ),
                    ],
                    clone_size_estimates={"cluster_1": 0.18, "cluster_2": 0.12},
                )
            ],
        )

        assert flat <= 0.30
        assert structured > flat
        assert structured > contaminated


class TestExpertTerminalCaps:
    def test_expert_grade_caps_without_two_validations(self):
        clone_truth = {
            "MCL1_resistant_clone": {
                "is_resistant": True,
                "size": 0.18,
                "markers": ["MCL1", "BCL2A1", "SOX4"],
                "pathways": {"intrinsic_apoptosis_regulation": 0.95},
                "mechanism": "An MCL1/BCL2A1 anti-apoptotic escape program",
            },
            "JAK2_STAT5_resistant_clone": {
                "is_resistant": True,
                "size": 0.12,
                "markers": ["JAK2", "STAT5A", "PIM1"],
                "pathways": {"JAK_STAT_signalling": 0.97},
                "mechanism": "A JAK2-STAT5-PIM1 survival program",
            },
        }
        latent = _make_latent(
            progress=ExperimentProgress(
                samples_collected=True,
                cells_sequenced=True,
                qc_performed=True,
                data_filtered=True,
                data_normalized=True,
                batches_integrated=True,
                cells_clustered=True,
                de_performed=True,
                pathways_analyzed=True,
                networks_inferred=True,
                trajectories_inferred=True,
                markers_discovered=True,
                markers_validated=True,
                conclusion_reached=True,
            ),
            true_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            causal_mechanisms=[
                "An MCL1/BCL2A1 anti-apoptotic escape program",
                "A JAK2-STAT5-PIM1 survival program",
            ],
            true_pathways={
                "intrinsic_apoptosis_regulation": 0.9,
                "JAK_STAT_signalling": 0.85,
            },
            clone_truth=clone_truth,
            scenario_name="venetoclax_resistance_multiclone",
        )
        obs = _make_obs(
            discovered_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            conclusions=[
                ConclusionClaim(
                    clonal_claims=[
                        ClonalClaim(
                            subpopulation_id="cluster_1",
                            markers=["MCL1", "BCL2A1", "SOX4"],
                            mechanism="An MCL1/BCL2A1 anti-apoptotic escape program",
                            supporting_pathways=["intrinsic_apoptosis_regulation"],
                        ),
                        ClonalClaim(
                            subpopulation_id="cluster_2",
                            markers=["JAK2", "STAT5A", "PIM1"],
                            mechanism="A JAK2-STAT5-PIM1 survival program",
                            supporting_pathways=["JAK_STAT_signalling"],
                        ),
                    ],
                    clone_size_estimates={"cluster_1": 0.18, "cluster_2": 0.12},
                )
            ],
        )
        obs.pipeline_history = [
            PipelineStepRecord(
                step_index=1,
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "MCL1", "subpopulation_id": "cluster_1"},
                output_summary="Validated MCL1 in resistant cluster",
                output_type=OutputType.VALIDATION_RESULT,
                success=True,
            )
        ]

        result = grade_episode(obs, latent)

        assert result.score <= 0.20
        assert result.breakdown["expert_validation_count"] == 1

    def test_expert_grade_penalizes_distractor_promotion(self):
        clone_truth = {
            "MCL1_resistant_clone": {
                "is_resistant": True,
                "size": 0.18,
                "markers": ["MCL1", "BCL2A1", "SOX4"],
                "pathways": {"intrinsic_apoptosis_regulation": 0.95},
                "mechanism": "An MCL1/BCL2A1 anti-apoptotic escape program",
            },
            "JAK2_STAT5_resistant_clone": {
                "is_resistant": True,
                "size": 0.12,
                "markers": ["JAK2", "STAT5A", "PIM1"],
                "pathways": {"JAK_STAT_signalling": 0.97},
                "mechanism": "A JAK2-STAT5-PIM1 survival program",
            },
            "cycling_distractor_clone": {
                "is_resistant": False,
                "size": 0.08,
                "markers": ["MKI67", "TOP2A", "PCNA"],
                "pathways": {"cell_cycle": 0.96},
                "mechanism": "Cycling distractor",
            },
        }
        latent = _make_latent(
            progress=ExperimentProgress(
                samples_collected=True,
                cells_sequenced=True,
                qc_performed=True,
                data_filtered=True,
                data_normalized=True,
                batches_integrated=True,
                cells_clustered=True,
                de_performed=True,
                pathways_analyzed=True,
                networks_inferred=True,
                trajectories_inferred=True,
                markers_discovered=True,
                markers_validated=True,
                conclusion_reached=True,
            ),
            true_markers=["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            causal_mechanisms=[
                "An MCL1/BCL2A1 anti-apoptotic escape program",
                "A JAK2-STAT5-PIM1 survival program",
            ],
            true_pathways={
                "intrinsic_apoptosis_regulation": 0.9,
                "JAK_STAT_signalling": 0.85,
            },
            clone_truth=clone_truth,
            scenario_name="venetoclax_resistance_multiclone",
        )
        clean = _make_obs(
            conclusions=[
                ConclusionClaim(
                    clonal_claims=[
                        ClonalClaim(
                            subpopulation_id="cluster_1",
                            markers=["MCL1", "BCL2A1", "SOX4"],
                            mechanism="An MCL1/BCL2A1 anti-apoptotic escape program",
                            supporting_pathways=["intrinsic_apoptosis_regulation"],
                        ),
                        ClonalClaim(
                            subpopulation_id="cluster_2",
                            markers=["JAK2", "STAT5A", "PIM1"],
                            mechanism="A JAK2-STAT5-PIM1 survival program",
                            supporting_pathways=["JAK_STAT_signalling"],
                        ),
                    ],
                )
            ],
        )
        contaminated = _make_obs(
            conclusions=[
                ConclusionClaim(
                    clonal_claims=[
                        ClonalClaim(
                            subpopulation_id="cluster_1",
                            markers=["MCL1", "BCL2A1", "MKI67", "TOP2A"],
                            mechanism="An MCL1/BCL2A1 anti-apoptotic escape program",
                            supporting_pathways=[
                                "intrinsic_apoptosis_regulation",
                                "cell_cycle",
                            ],
                        ),
                        ClonalClaim(
                            subpopulation_id="cluster_2",
                            markers=["JAK2", "STAT5A", "PIM1"],
                            mechanism="A JAK2-STAT5-PIM1 survival program",
                            supporting_pathways=["JAK_STAT_signalling"],
                        ),
                    ],
                )
            ],
        )
        validation_steps = [
            PipelineStepRecord(
                step_index=1,
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "MCL1", "subpopulation_id": "cluster_1"},
                output_summary="Validated MCL1",
                output_type=OutputType.VALIDATION_RESULT,
                success=True,
            ),
            PipelineStepRecord(
                step_index=2,
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "JAK2", "subpopulation_id": "cluster_2"},
                output_summary="Validated JAK2",
                output_type=OutputType.VALIDATION_RESULT,
                success=True,
            ),
        ]
        clean.pipeline_history = validation_steps
        contaminated.pipeline_history = validation_steps

        clean_result = grade_episode(clean, latent)
        contaminated_result = grade_episode(contaminated, latent)

        assert clean_result.score > contaminated_result.score
        assert contaminated_result.breakdown["expert_penalty"] == 0.25
