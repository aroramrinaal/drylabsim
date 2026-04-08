from __future__ import annotations

from typing import Any, Dict, List

try:
    from ....models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )
except ImportError:  # pragma: no cover
    from models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )

from ..latent_state import FullLatentState
from ..noise import NoiseModel

from .helpers import NOISE_TFS

_MULTICLONE_EXPERT_SCENARIO = "venetoclax_resistance_multiclone"


def _is_multiclone_expert(s: FullLatentState) -> bool:
    return (
        s.scenario_name == _MULTICLONE_EXPERT_SCENARIO
        and bool(s.biology.clone_truth)
    )


def _clone_alias_map(s: FullLatentState) -> Dict[str, str]:
    return {
        clone_name: f"subpopulation_{idx}"
        for idx, clone_name in enumerate(s.biology.clone_truth.keys())
    }


def _minor_clone_name(s: FullLatentState) -> str | None:
    if not s.biology.clone_truth:
        return None
    return min(
        s.biology.clone_truth.items(),
        key=lambda item: item[1].get("size", 1.0),
    )[0]


def _dominant_clone_name(s: FullLatentState) -> str | None:
    if not s.biology.clone_truth:
        return None
    return max(
        s.biology.clone_truth.items(),
        key=lambda item: item[1].get("size", 0.0),
    )[0]


def trajectory_analysis(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Trajectory analysis handler."""

    if _is_multiclone_expert(s):
        integrated = s.progress.batches_integrated
        clone_truth = s.biology.clone_truth
        clone_aliases = _clone_alias_map(s)
        minor_clone = _minor_clone_name(s)
        dominant_clone = _dominant_clone_name(s)
        clone_lineages = {}
        branch_confidence: Dict[str, float] = {}
        for clone_name, truth in clone_truth.items():
            conf = 0.80 if clone_name == dominant_clone else (
                0.68 if integrated else 0.45
            )
            clone_alias = clone_aliases[clone_name]
            clone_lineages[clone_alias] = {
                "path": ["founder_like_state", clone_alias],
                "detected": conf > 0.5,
            }
            if clone_name == minor_clone and not integrated:
                clone_lineages[clone_alias]["detected"] = False
            branch_confidence[clone_alias] = conf

        return IntermediateOutput(
            output_type=OutputType.TRAJECTORY_RESULT,
            step_index=idx,
            quality_score=gen.noise.quality_degradation(
                0.85 if integrated else 0.67,
                [0.95 if integrated else 0.8],
            ),
            summary="Trajectory analysis supports divergence from a shared founder-like state into multiple resistant branches",
            data={
                "method": action.method or "monocle3",
                "n_lineages": len(clone_truth),
                "pseudotime_range": [0.0, 1.0],
                "branching_detected": True,
                "clone_lineages": clone_lineages,
                "branch_confidence": branch_confidence,
                "minor_branch_detected": (
                    branch_confidence.get(clone_aliases.get(minor_clone or ""), 0.0) > 0.5
                ),
            },
            uncertainty=0.18 if integrated else 0.36,
            artifacts_available=["pseudotime_values", "lineage_graph"],
        )

    has_trajectory = s.biology.true_trajectory is not None
    quality = gen.noise.quality_degradation(0.7 if has_trajectory else 0.3, [0.9])
    summary_data: Dict[str, Any] = {"method": action.method or "monocle3"}
    if has_trajectory:
        true_n_lineages = s.biology.true_trajectory.get("n_lineages", 1)
        true_branching = s.biology.true_trajectory.get("branching", False)
        noisy_n_lineages = max(
            1, true_n_lineages + int(gen.noise.rng.choice([-1, 0, 0, 1]))
        )
        noisy_branching = (
            true_branching if not gen.noise.coin_flip(0.20) else not true_branching
        )
        summary_data.update(
            {
                "n_lineages": noisy_n_lineages,
                "pseudotime_range": [0.0, 1.0],
                "branching_detected": noisy_branching,
            }
        )
    else:
        summary_data["n_lineages"] = gen.noise.sample_count(1) + 1
        summary_data["pseudotime_range"] = [0.0, 1.0]
        summary_data["branching_detected"] = gen.noise.coin_flip(0.3)

    return IntermediateOutput(
        output_type=OutputType.TRAJECTORY_RESULT,
        step_index=idx,
        quality_score=quality,
        summary="Trajectory / pseudotime analysis complete",
        data=summary_data,
        uncertainty=0.2 if has_trajectory else 0.6,
        artifacts_available=["pseudotime_values", "lineage_graph"],
    )


def pathway_enrichment(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Pathway enrichment handler."""

    if _is_multiclone_expert(s):
        clone_truth = s.biology.clone_truth
        integrated = s.progress.batches_integrated
        clone_aliases = _clone_alias_map(s)
        minor_clone = _minor_clone_name(s)
        clone_pathways: Dict[str, Any] = {}
        flattened: List[Dict[str, Any]] = []
        for clone_name, truth in clone_truth.items():
            pathway_scores = truth.get("pathways", {})
            clone_detected = not (
                clone_name == minor_clone and not integrated
            )
            clone_top = [
                {
                    "pathway": pw,
                    "score": round(
                        activity + float(gen.noise.rng.normal(0, 0.06 if integrated else 0.12)),
                        3,
                    ),
                }
                for pw, activity in sorted(
                    pathway_scores.items(), key=lambda kv: kv[1], reverse=True
                )
            ]
            clone_pathways[clone_aliases[clone_name]] = {
                "detected": clone_detected,
                "top_pathways": clone_top,
            }
            if clone_detected:
                for item in clone_top[:2]:
                    flattened.append(
                        {**item, "subpopulation": clone_aliases[clone_name]}
                    )

        for pathway, activity in sorted(
            s.biology.confounders.items(),
            key=lambda kv: kv[1],
            reverse=True,
        ):
            flattened.append(
                {
                    "pathway": pathway,
                    "score": round(
                        activity
                        + float(gen.noise.rng.normal(0, 0.05 if integrated else 0.10)),
                        3,
                    ),
                    "subpopulation": "mixed_signal",
                }
            )

        flattened.sort(key=lambda item: item.get("score", 0.0), reverse=True)

        return IntermediateOutput(
            output_type=OutputType.PATHWAY_RESULT,
            step_index=idx,
            quality_score=gen.noise.quality_degradation(
                0.88 if integrated else 0.69,
                [0.95 if integrated else 0.8],
            ),
            summary="Pathway enrichment highlights multiple relapse-associated programs with residual mixed-signal confounders",
            data={
                "method": action.method or "GSEA",
                "top_pathways": flattened[:10],
                "clone_pathways": clone_pathways,
            },
            uncertainty=0.19 if integrated else 0.35,
            artifacts_available=["enrichment_table", "clone_enrichment_table"],
        )

    true_pathways = s.biology.true_pathways
    de_genes_found = s.progress.n_de_genes_found or 0
    de_was_run = s.progress.de_performed
    if de_was_run and de_genes_found > 0:
        noise_level = max(0.05, 0.25 - 0.001 * min(de_genes_found, 200))
        n_fp_mean = max(1, int(5 - de_genes_found / 50))
    else:
        noise_level = 0.40
        n_fp_mean = 8

    observed: Dict[str, float] = {}
    for pw, activity in true_pathways.items():
        observed[pw] = activity + float(gen.noise.rng.normal(0, noise_level))

    for pw, activity in s.biology.confounders.items():
        observed[pw] = activity + float(gen.noise.rng.normal(0, noise_level * 0.8))

    for i in range(gen.noise.sample_count(n_fp_mean)):
        observed[f"FP_PATHWAY_{i}"] = float(gen.noise.rng.uniform(0.3, 0.6))

    top = sorted(observed.items(), key=lambda kv: kv[1], reverse=True)[:15]
    base_quality = 0.80 if de_was_run else 0.45
    return IntermediateOutput(
        output_type=OutputType.PATHWAY_RESULT,
        step_index=idx,
        quality_score=gen.noise.quality_degradation(base_quality, [0.95]),
        summary=f"Pathway enrichment: {len(top)} significant pathways",
        data={
            "method": action.method or "GSEA",
            "top_pathways": [{"pathway": p, "score": round(sc, 3)} for p, sc in top],
        },
        uncertainty=noise_level,
        artifacts_available=["enrichment_table"],
    )


def regulatory_network(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Regulatory network handler."""

    if _is_multiclone_expert(s):
        clone_truth = s.biology.clone_truth
        integrated = s.progress.batches_integrated
        clone_aliases = _clone_alias_map(s)
        minor_clone = _minor_clone_name(s)
        clone_regulators: Dict[str, Any] = {}
        top_regulators: List[str] = []
        for clone_name, truth in clone_truth.items():
            regulators = list(truth.get("regulators", []))
            clone_detected = not (
                clone_name == minor_clone and not integrated
            )
            shuffled = gen.noise.shuffle_ranking(regulators, 0.3)
            clone_regulators[clone_aliases[clone_name]] = {
                "detected": clone_detected,
                "top_regulators": shuffled[:5],
            }
            if clone_detected:
                top_regulators.extend(shuffled[:3])

        return IntermediateOutput(
            output_type=OutputType.NETWORK_RESULT,
            step_index=idx,
            quality_score=gen.noise.quality_degradation(
                0.84 if integrated else 0.68,
                [0.92 if integrated else 0.78],
            ),
            summary="Regulatory network inference resolves distinct relapse-associated regulon modules",
            data={
                "method": action.method or "SCENIC",
                "n_regulons": len(top_regulators) + gen.noise.sample_count(2),
                "n_edges": 40 + gen.noise.sample_count(10),
                "top_regulators": top_regulators[:10],
                "clone_regulators": clone_regulators,
            },
            uncertainty=0.22 if integrated else 0.39,
            artifacts_available=["regulon_table", "grn_adjacency", "clone_grn_summary"],
        )

    true_net = s.biology.true_regulatory_network
    n_edges_true = sum(len(v) for v in true_net.values())
    noise_edges = gen.noise.sample_count(max(5, int(n_edges_true * 0.3)))

    true_tfs = list(true_net.keys())
    fn_set = set(gen.noise.generate_false_negatives(true_tfs, 0.25))
    observed_tfs = [tf for tf in true_tfs if tf not in fn_set]
    fp_candidates = [t for t in NOISE_TFS if t not in set(true_tfs)]
    n_fp = gen.noise.sample_count(max(2, int(len(true_tfs) * 0.5) + 2))
    if fp_candidates and n_fp > 0:
        chosen = gen.noise.rng.choice(
            fp_candidates,
            size=min(n_fp, len(fp_candidates)),
            replace=False,
        )
        observed_tfs.extend(chosen.tolist())
    observed_tfs = gen.noise.shuffle_ranking(observed_tfs, 0.5)

    return IntermediateOutput(
        output_type=OutputType.NETWORK_RESULT,
        step_index=idx,
        quality_score=gen.noise.quality_degradation(0.6, [0.9]),
        summary=f"Regulatory network inferred: {n_edges_true + noise_edges} edges",
        data={
            "method": action.method or "SCENIC",
            "n_regulons": len(true_net) + gen.noise.sample_count(3),
            "n_edges": n_edges_true + noise_edges,
            "top_regulators": observed_tfs[:10],
        },
        uncertainty=0.35,
        artifacts_available=["regulon_table", "grn_adjacency"],
    )


def marker_selection(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Marker selection handler."""

    if _is_multiclone_expert(s):
        clone_truth = s.biology.clone_truth
        integrated = s.progress.batches_integrated
        clone_aliases = _clone_alias_map(s)
        minor_clone = _minor_clone_name(s)
        clone_markers: Dict[str, List[str]] = {}
        observed_markers: List[str] = []
        for clone_name, truth in clone_truth.items():
            markers = list(truth.get("markers", []))
            if clone_name == minor_clone and not integrated:
                markers = markers[:2]
            clone_markers[clone_aliases[clone_name]] = markers
            observed_markers.extend(markers)
        fp = gen.noise.generate_false_positives(200, 0.01)
        observed_markers.extend(fp)
        deduped = list(dict.fromkeys(observed_markers))
        return IntermediateOutput(
            output_type=OutputType.MARKER_RESULT,
            step_index=idx,
            quality_score=gen.noise.quality_degradation(
                0.86 if integrated else 0.70,
                [0.9 if integrated else 0.78],
            ),
            summary="Marker selection recovered subgroup-resolved resistant markers",
            data={
                "markers": deduped[:20],
                "n_candidates": len(deduped),
                "clone_markers": clone_markers,
            },
            uncertainty=0.16 if integrated else 0.31,
            artifacts_available=["marker_list", "clone_marker_list"],
        )

    true_markers = list(s.biology.true_markers)
    noise_level = 0.2
    observed_markers = [m for m in true_markers if not gen.noise.coin_flip(noise_level)]
    fp = gen.noise.generate_false_positives(200, 0.01)
    observed_markers.extend(fp)
    return IntermediateOutput(
        output_type=OutputType.MARKER_RESULT,
        step_index=idx,
        quality_score=gen.noise.quality_degradation(0.75, [0.9]),
        summary=f"Selected {len(observed_markers)} candidate markers",
        data={
            "markers": observed_markers[:20],
            "n_candidates": len(observed_markers),
        },
        uncertainty=noise_level,
        artifacts_available=["marker_list"],
    )


def validate_marker(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Validate marker handler."""

    marker = action.parameters.get("marker", "unknown")
    is_true = marker in s.biology.true_markers
    validation_correct = not gen.noise.coin_flip(0.1)
    validated = is_true == validation_correct
    return IntermediateOutput(
        output_type=OutputType.VALIDATION_RESULT,
        step_index=idx,
        quality_score=0.9 if validation_correct else 0.4,
        summary=f"Marker {marker}: {'validated' if validated else 'not validated'}",
        data={
            "marker": marker,
            "validated": validated,
            "assay": action.method or "qPCR",
            "effect_size": gen.noise.sample_qc_metric(
                0.85 if is_true else 0.45, 0.4, -0.5, 5.0
            ),
        },
        artifacts_available=["validation_data"],
    )
