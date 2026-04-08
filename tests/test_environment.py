"""Integration tests for the full BioExperimentEnvironment."""

from fastapi.testclient import TestClient

from models import ActionType, ExperimentAction
from server.app import app
from server.drylabsim_environment import BioExperimentEnvironment


class TestEnvironmentLifecycle:
    def test_reset_returns_valid_observation(self):
        env = BioExperimentEnvironment()
        obs = env.reset()
        assert obs.step_index == 0
        assert obs.done is False
        assert obs.task.problem_statement != ""

    def test_reset_accepts_task_alias(self):
        env = BioExperimentEnvironment(domain_randomise=False)
        obs = env.reset(task_name="easy")
        assert obs.metadata["task_name"] == "easy"
        assert obs.metadata["scenario_name"] == "cardiac_disease_de"

    def test_step_increments_step_count(self):
        env = BioExperimentEnvironment()
        env.reset()
        obs = env.step(ExperimentAction(action_type=ActionType.COLLECT_SAMPLE))
        assert obs.step_index == 1
        assert env.state.step_count == 1

    def test_valid_pipeline_trajectory(self):
        env = BioExperimentEnvironment()
        env.reset()

        actions = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE,
                             parameters={"n_samples": 6}),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY,
                             method="10x_chromium"),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                             parameters={"comparison": "disease_vs_healthy"}),
        ]

        for a in actions:
            obs = env.step(a)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True, (
                f"Step {a.action_type} failed: {obs.rule_violations}"
            )

        assert obs.step_index == len(actions)
        assert obs.resource_usage.budget_used > 0

    def test_premature_de_blocked(self):
        env = BioExperimentEnvironment()
        env.reset()
        obs = env.step(ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
        ))
        assert obs.latest_output is not None
        assert obs.latest_output.success is False

    def test_premature_followup_design_is_flagged(self):
        env = BioExperimentEnvironment()
        env.reset()
        obs = env.step(ExperimentAction(
            action_type=ActionType.DESIGN_FOLLOWUP,
            parameters={"assay": "qPCR"},
        ))
        assert obs.latest_output is not None
        assert obs.latest_output.success is False
        assert any("follow-up design" in msg.lower() for msg in obs.rule_violations)

    def test_conclusion_ends_episode(self):
        env = BioExperimentEnvironment(
            scenario_name="cardiac_disease_de",
            domain_randomise=False,
        )
        env.reset()

        quick_pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                             parameters={"comparison": "disease_vs_healthy"}),
            ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT),
            ExperimentAction(action_type=ActionType.MARKER_SELECTION),
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={"claims": [
                    {"claim": "Test conclusion", "confidence": 0.7,
                     "claim_type": "correlational"},
                ]},
            ),
        ]
        for a in quick_pipeline:
            obs = env.step(a)

        assert obs.done is True
        assert obs.reward != 0.0
        assert 0.0 <= obs.metadata["score"] <= 1.0
        assert 0.0 <= obs.metadata["completeness"] <= 1.0
        assert 0.0 <= obs.metadata["biology_score"] <= 1.0
        assert 0.0 <= obs.metadata["efficiency_score"] <= 1.0
        assert obs.metadata["grade_source"] == "grade_episode"
        assert "grade_breakdown" in obs.metadata

    def test_blocked_conclusion_does_not_persist_claims(self):
        env = BioExperimentEnvironment(
            scenario_name="cardiac_disease_de",
            domain_randomise=False,
        )
        env.reset()

        pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
        ]
        for action in pipeline:
            obs = env.step(action)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True

        obs = env.step(ExperimentAction(
            action_type=ActionType.SYNTHESIZE_CONCLUSION,
            parameters={"claims": [
                {"claim": "Premature conclusion", "confidence": 0.9},
            ]},
        ))

        assert obs.latest_output is not None
        assert obs.latest_output.success is False
        assert obs.conclusions == []
        assert any("markers" in msg.lower() for msg in obs.rule_violations)

    def test_expert_scenario_blocks_de_until_clustering(self):
        env = BioExperimentEnvironment(
            scenario_name="venetoclax_resistance_multiclone",
            domain_randomise=False,
        )
        env.reset()

        pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
        ]
        for action in pipeline:
            obs = env.step(action)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True

        obs = env.step(ExperimentAction(action_type=ActionType.DIFFERENTIAL_EXPRESSION))
        assert obs.latest_output is not None
        assert obs.latest_output.success is False
        assert any("clustering" in msg.lower() for msg in obs.rule_violations)

    def test_expert_scenario_blocks_conclusion_without_second_wave_evidence(self):
        env = BioExperimentEnvironment(
            scenario_name="venetoclax_resistance_multiclone",
            domain_randomise=False,
        )
        env.reset()

        pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.INTEGRATE_BATCHES),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(
                action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                parameters={"comparison": "post_vs_pre_bulk"},
            ),
            ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT),
            ExperimentAction(action_type=ActionType.MARKER_SELECTION),
        ]
        for action in pipeline:
            obs = env.step(action)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True

        obs = env.step(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "claim": "One resistant mechanism dominates relapse",
                            "causal_mechanisms": [
                                "An MCL1/BCL2A1 anti-apoptotic escape program sustains one resistant AML subclone under venetoclax pressure"
                            ],
                            "confidence": 0.9,
                            "claim_type": "causal",
                        }
                    ]
                },
            )
        )
        assert obs.latest_output is not None
        assert obs.latest_output.success is False
        assert any("trajectory" in msg.lower() for msg in obs.rule_violations)

    def test_hard_scenario_blocks_conclusion_without_mechanism_claims(self):
        env = BioExperimentEnvironment(
            scenario_name="perturbation_immune",
            domain_randomise=False,
        )
        env.reset()

        pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(
                action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                parameters={"comparison": "treated_vs_untreated"},
            ),
            ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT),
            ExperimentAction(action_type=ActionType.MARKER_SELECTION),
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "STAT1", "assay": "qPCR"},
            ),
        ]
        for action in pipeline:
            obs = env.step(action)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True

        obs = env.step(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "claim": "Treatment changes immune state composition",
                            "top_markers": ["STAT1", "SOCS1"],
                            "confidence": 0.8,
                            "claim_type": "correlational",
                            "evidence_steps": [8, 9],
                        }
                    ]
                },
            )
        )
        assert obs.latest_output is not None
        assert obs.latest_output.success is False
        assert any("causal mechanism" in msg.lower() for msg in obs.rule_violations)

    def test_hard_scenario_blocks_mechanism_claims_without_evidence_steps(self):
        env = BioExperimentEnvironment(
            scenario_name="perturbation_immune",
            domain_randomise=False,
        )
        env.reset()

        pipeline = [
            ExperimentAction(action_type=ActionType.COLLECT_SAMPLE),
            ExperimentAction(action_type=ActionType.PREPARE_LIBRARY),
            ExperimentAction(action_type=ActionType.SEQUENCE_CELLS),
            ExperimentAction(action_type=ActionType.RUN_QC),
            ExperimentAction(action_type=ActionType.FILTER_DATA),
            ExperimentAction(action_type=ActionType.NORMALIZE_DATA),
            ExperimentAction(action_type=ActionType.CLUSTER_CELLS),
            ExperimentAction(
                action_type=ActionType.DIFFERENTIAL_EXPRESSION,
                parameters={"comparison": "treated_vs_untreated"},
            ),
            ExperimentAction(action_type=ActionType.PATHWAY_ENRICHMENT),
            ExperimentAction(action_type=ActionType.MARKER_SELECTION),
            ExperimentAction(
                action_type=ActionType.VALIDATE_MARKER,
                parameters={"marker": "STAT1", "assay": "qPCR"},
            ),
        ]
        for action in pipeline:
            obs = env.step(action)
            assert obs.latest_output is not None
            assert obs.latest_output.success is True

        obs = env.step(
            ExperimentAction(
                action_type=ActionType.SYNTHESIZE_CONCLUSION,
                parameters={
                    "claims": [
                        {
                            "claim": "JAK inhibition suppresses inflammatory signaling",
                            "top_markers": ["STAT1", "SOCS1"],
                            "causal_mechanisms": [
                                "JAK-STAT pathway inhibition reduces Th1/Th17 activation"
                            ],
                            "confidence": 0.85,
                            "claim_type": "causal",
                            "evidence_steps": [],
                        }
                    ]
                },
            )
        )
        assert obs.latest_output is not None
        assert obs.latest_output.success is False
        assert any("evidence_steps" in msg.lower() for msg in obs.rule_violations)


class TestSessionBackedHttpAPI:
    def test_reset_returns_session_id_and_step_uses_it(self):
        with TestClient(app) as client:
            reset_resp = client.post("/reset", json={"task_name": "easy"})
            assert reset_resp.status_code == 200
            reset_payload = reset_resp.json()

            session_id = reset_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            assert reset_payload["observation"]["metadata"]["task_name"] == "easy"

            step_resp = client.post(
                "/step",
                json={
                    "session_id": session_id,
                    "action": {"action_type": "collect_sample", "parameters": {"n_samples": 6}},
                },
            )
            assert step_resp.status_code == 200
            step_payload = step_resp.json()
            assert step_payload["session_id"] == session_id
            assert step_payload["observation"]["step_index"] == 1

    def test_state_is_session_scoped_and_invalid_session_fails_cleanly(self):
        with TestClient(app) as client:
            reset_a = client.post("/reset", json={"task_name": "easy"}).json()
            reset_b = client.post("/reset", json={"task_name": "medium"}).json()

            session_a = reset_a["session_id"]
            session_b = reset_b["session_id"]
            assert session_a != session_b

            step_a = client.post(
                "/step",
                json={
                    "session_id": session_a,
                    "action": {"action_type": "collect_sample", "parameters": {"n_samples": 6}},
                },
            )
            assert step_a.status_code == 200

            state_a = client.get("/state", params={"session_id": session_a})
            state_b = client.get("/state", params={"session_id": session_b})
            assert state_a.status_code == 200
            assert state_b.status_code == 200
            assert state_a.json()["step_count"] == 1
            assert state_b.json()["step_count"] == 0

            bad_step = client.post(
                "/step",
                json={
                    "session_id": "missing-session",
                    "action": {"action_type": "collect_sample"},
                },
            )
            assert bad_step.status_code == 404
