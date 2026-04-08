from inference import _mechanism_hypotheses, build_fallback_action, recommend_next_action


class TestInferenceMechanismRecovery:
    def test_hard_task_derives_mechanisms_from_outputs(self):
        obs = {
            "discovered_markers": ["STAT1", "IFNG", "FOXP3", "IL10"],
            "all_outputs": [
                {
                    "output_type": "pathway_result",
                    "data": {
                        "top_pathways": [
                            {"pathway": "JAK_STAT_signalling", "score": 0.8},
                            {"pathway": "regulatory_T_cell_function", "score": 0.7},
                        ]
                    },
                }
            ],
        }

        mechanisms = _mechanism_hypotheses("hard", obs)
        assert mechanisms == [
            "Evidence supports pathway-level remodeling involving JAK_STAT_signalling, regulatory_T_cell_function"
        ]

    def test_expert_fallback_conclusion_uses_cluster_resolved_claims(self):
        obs = {
            "discovered_markers": ["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            "all_outputs": [
                {
                    "output_type": "cluster_result",
                    "data": {
                        "cluster_names": ["cluster_0", "cluster_1", "cluster_2"],
                        "cluster_sizes": [5000, 1800, 1200],
                    },
                },
                {
                    "output_type": "pathway_result",
                    "data": {
                        "top_pathways": [{"pathway": "intrinsic_apoptosis_regulation", "score": 0.9}],
                        "cluster_pathways": {
                            "cluster_1": {
                                "top_pathways": [
                                    {"pathway": "intrinsic_apoptosis_regulation", "score": 0.9}
                                ]
                            },
                            "cluster_2": {
                                "top_pathways": [
                                    {"pathway": "JAK_STAT_signalling", "score": 0.88}
                                ]
                            },
                        },
                    },
                },
                {
                    "output_type": "marker_result",
                    "data": {
                        "cluster_markers": {
                            "cluster_1": ["MCL1", "BCL2A1", "SOX4"],
                            "cluster_2": ["JAK2", "STAT5A", "PIM1"],
                        }
                    },
                },
            ],
            "pipeline_history": [
                {"step_index": 9, "success": True, "action_type": "pathway_enrichment"},
                {
                    "step_index": 10,
                    "success": True,
                    "action_type": "regulatory_network_inference",
                },
                {"step_index": 11, "success": True, "action_type": "marker_selection"},
            ],
        }

        action = build_fallback_action("expert", obs, "synthesize_conclusion")
        claim = action["parameters"]["claims"][0]
        assert len(claim["causal_mechanisms"]) == 2
        assert claim["claim_type"] == "causal"
        assert claim["evidence_steps"] == [9, 10, 11]
        assert len(claim["clonal_claims"]) == 2
        assert claim["clone_size_estimates"]["cluster_1"] == 0.225

    def test_expert_recommends_second_validation_before_conclusion(self):
        pipeline_history = [
            {"step_index": 1, "success": True, "action_type": "collect_sample"},
            {"step_index": 2, "success": True, "action_type": "prepare_library"},
            {"step_index": 3, "success": True, "action_type": "sequence_cells"},
            {"step_index": 4, "success": True, "action_type": "run_qc"},
            {"step_index": 5, "success": True, "action_type": "filter_data"},
            {"step_index": 6, "success": True, "action_type": "normalize_data"},
            {"step_index": 7, "success": True, "action_type": "integrate_batches"},
            {"step_index": 8, "success": True, "action_type": "cluster_cells"},
            {"step_index": 9, "success": True, "action_type": "differential_expression"},
            {"step_index": 10, "success": True, "action_type": "pathway_enrichment"},
            {
                "step_index": 11,
                "success": True,
                "action_type": "regulatory_network_inference",
            },
            {"step_index": 12, "success": True, "action_type": "trajectory_analysis"},
            {"step_index": 13, "success": True, "action_type": "marker_selection"},
            {
                "step_index": 14,
                "success": True,
                "action_type": "validate_marker",
                "parameters": {"marker": "MCL1", "subpopulation_id": "cluster_1"},
            },
        ]

        assert recommend_next_action("expert", pipeline_history) == "validate_marker"

    def test_expert_validation_fallback_targets_unvalidated_branch(self):
        obs = {
            "discovered_markers": ["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            "all_outputs": [
                {
                    "output_type": "cluster_result",
                    "data": {
                        "cluster_names": ["cluster_0", "cluster_1", "cluster_2"],
                        "cluster_sizes": [5000, 1800, 1200],
                    },
                },
                {
                    "output_type": "marker_result",
                    "data": {
                        "cluster_markers": {
                            "cluster_1": ["MCL1", "BCL2A1", "SOX4"],
                            "cluster_2": ["JAK2", "STAT5A", "PIM1"],
                        }
                    },
                },
            ],
            "pipeline_history": [
                {
                    "step_index": 14,
                    "success": True,
                    "action_type": "validate_marker",
                    "parameters": {"marker": "MCL1", "subpopulation_id": "cluster_1"},
                }
            ],
        }

        action = build_fallback_action("expert", obs, "validate_marker")

        assert action["parameters"]["marker"] == "JAK2"
        assert action["parameters"]["subpopulation_id"] == "cluster_2"
