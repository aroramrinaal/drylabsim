from inference import _mechanism_hypotheses, build_fallback_action


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
        assert "JAK-STAT pathway inhibition reduces Th1/Th17 activation" in mechanisms
        assert "Compensatory Treg expansion under JAK inhibition" in mechanisms

    def test_expert_fallback_conclusion_uses_inferred_mechanisms(self):
        obs = {
            "discovered_markers": ["MCL1", "BCL2A1", "JAK2", "STAT5A", "PIM1"],
            "all_outputs": [
                {
                    "output_type": "pathway_result",
                    "data": {
                        "top_pathways": [
                            {"pathway": "intrinsic_apoptosis_regulation", "score": 0.9},
                            {"pathway": "JAK_STAT_signalling", "score": 0.88},
                        ]
                    },
                },
                {
                    "output_type": "network_result",
                    "data": {
                        "top_regulators": ["CREB1", "STAT5A"],
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
