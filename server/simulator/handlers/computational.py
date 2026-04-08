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

from .helpers import partition_by_population

import math

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


def _expert_cluster_labels(s: FullLatentState) -> List[str]:
    background = [
        p.name
        for p in s.biology.cell_populations
        if p.name not in set(s.biology.clone_truth.keys())
    ]
    internal_cluster_labels = ["AML_founder_blast", *list(s.biology.clone_truth.keys())]
    for pop_name in background:
        if pop_name not in internal_cluster_labels:
            internal_cluster_labels.append(pop_name)

    integrated = s.progress.batches_integrated
    minor_clone = _minor_clone_name(s)
    if not integrated and minor_clone in internal_cluster_labels:
        internal_cluster_labels.remove(minor_clone)
        internal_cluster_labels.append("merged_relapse_state")
    return internal_cluster_labels


def _expert_cluster_alias_map(s: FullLatentState) -> Dict[str, str]:
    labels = _expert_cluster_labels(s)
    alias_map = {label: f"cluster_{idx}" for idx, label in enumerate(labels)}
    minor_clone = _minor_clone_name(s)
    if (
        not s.progress.batches_integrated
        and minor_clone
        and minor_clone not in alias_map
        and "merged_relapse_state" in alias_map
    ):
        alias_map[minor_clone] = alias_map["merged_relapse_state"]
    return alias_map


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


def _noisy_cluster_sizes(
    sizes: List[int],
    rng,
    total: int,
) -> List[int]:
    if not sizes:
        return []

    noisy = []
    for size in sizes:
        scaled = size * float(rng.uniform(0.88, 1.12))
        noisy.append(max(1, int(round(scaled))))

    diff = total - sum(noisy)
    noisy[0] += diff
    if noisy[0] <= 0:
        noisy[0] = 1
        overflow = sum(noisy) - total
        if overflow > 0:
            for idx in range(1, len(noisy)):
                removable = min(noisy[idx] - 1, overflow)
                noisy[idx] -= removable
                overflow -= removable
                if overflow == 0:
                    break
    return noisy


def run_qc(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Run QC handler."""

    doublet_frac = gen.noise.sample_qc_metric(s.technical.doublet_rate, 0.01, 0.0, 0.2)
    _stressed_states = {"activated", "stressed", "pro-fibrotic", "inflammatory"}
    has_stressed_cells = any(
        p.state in _stressed_states for p in s.biology.cell_populations
    )
    mito_mean = 0.09 if has_stressed_cells else 0.06
    mito_frac = gen.noise.sample_qc_metric(mito_mean, 0.03, 0.0, 0.3)
    ambient_frac = gen.noise.sample_qc_metric(
        s.technical.ambient_rna_fraction, 0.01, 0.0, 0.2
    )
    warnings: List[str] = []
    if doublet_frac > 0.08:
        warnings.append(f"High doublet rate ({doublet_frac:.1%})")
    if mito_frac > 0.1:
        warnings.append(f"High mitochondrial fraction ({mito_frac:.1%})")
    if _is_multiclone_expert(s) and ambient_frac > 0.06:
        warnings.append(
            "Post-treatment ambient RNA may blur bulk resistance signals"
        )
    quality = 1.0 - (doublet_frac + mito_frac + ambient_frac)
    return IntermediateOutput(
        output_type=OutputType.QC_METRICS,
        step_index=idx,
        quality_score=max(0.0, quality),
        summary="QC metrics computed",
        data={
            "doublet_fraction": doublet_frac,
            "mitochondrial_fraction": mito_frac,
            "ambient_rna_fraction": ambient_frac,
            "median_genes_per_cell": gen.noise.sample_count(2500),
            "median_umi_per_cell": gen.noise.sample_count(8000),
        },
        warnings=warnings,
        artifacts_available=["qc_report"],
    )


def filter_data(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Filter data handler."""

    retain_frac = (
        s.last_retain_frac
        if s.last_retain_frac is not None
        else gen.noise.sample_qc_metric(0.85, 0.05, 0.5, 1.0)
    )
    n_before = s.progress.n_cells_sequenced or s.biology.n_true_cells
    n_after = s.progress.n_cells_after_filter or max(100, int(n_before * retain_frac))
    return IntermediateOutput(
        output_type=OutputType.COUNT_MATRIX_SUMMARY,
        step_index=idx,
        quality_score=retain_frac,
        summary=f"Filtered {n_before} → {n_after} cells ({retain_frac:.0%} retained)",
        data={
            "n_cells_before": n_before,
            "n_cells_after": n_after,
            "n_genes_retained": gen.noise.sample_count(15_000),
            "retain_fraction": retain_frac,
        },
        artifacts_available=["filtered_count_matrix"],
    )


def normalize_data(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Normalize data handler."""

    method = action.method or "log_normalize"
    return IntermediateOutput(
        output_type=OutputType.COUNT_MATRIX_SUMMARY,
        step_index=idx,
        summary=f"Normalized with {method}",
        data={"method": method, "n_hvg": gen.noise.sample_count(2000)},
        artifacts_available=["normalized_matrix", "hvg_list"],
    )


def integrate_batches(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Integrate batches handler."""

    method = action.method or "harmony"
    residual = gen.noise.sample_qc_metric(0.05, 0.03, 0.0, 0.3)
    return IntermediateOutput(
        output_type=OutputType.EMBEDDING_SUMMARY,
        step_index=idx,
        quality_score=1.0 - residual,
        summary=f"Batch integration ({method}), residual batch effect={residual:.2f}",
        data={
            "method": method,
            "residual_batch_effect": residual,
            "n_batches": len(s.technical.batch_effects) or 1,
        },
        artifacts_available=["integrated_embedding"],
    )


def cluster_cells(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Cluster cells handler."""

    if _is_multiclone_expert(s):
        internal_cluster_labels = _expert_cluster_labels(s)
        cluster_aliases = _expert_cluster_alias_map(s)
        cluster_names = [cluster_aliases[label] for label in internal_cluster_labels]
        n_clusters = len(cluster_names)
        n_cells = s.progress.n_cells_after_filter or s.biology.n_true_cells
        integrated = s.progress.batches_integrated
        quality = gen.noise.quality_degradation(
            0.88 if integrated else 0.64,
            [0.95 if integrated else 0.75],
        )
        sizes = partition_by_population(
            n_cells, n_clusters, s.biology.cell_populations, gen.noise.rng
        )
        sizes = _noisy_cluster_sizes(sizes, gen.noise.rng, n_cells)
        relapse_clusters = [
            cluster_aliases[label]
            for label in internal_cluster_labels
            if label in s.biology.clone_truth or label == "merged_relapse_state"
        ]
        return IntermediateOutput(
            output_type=OutputType.CLUSTER_RESULT,
            step_index=idx,
            quality_score=quality,
            summary=(
                "Clustering resolved multiple relapse-enriched subpopulations"
                if integrated
                else "Clustering found relapse structure, but one small resistant branch remains partially merged"
            ),
            data={
                "n_clusters": n_clusters,
                "cluster_names": cluster_names,
                "cluster_sizes": sizes,
                "silhouette_score": gen.noise.sample_qc_metric(
                    0.44 if integrated else 0.28, 0.08, -1.0, 1.0
                ),
                "cluster_markers_available": bool(relapse_clusters),
                "batch_integration_recommended": not integrated,
            },
            uncertainty=0.18 if integrated else 0.42,
            artifacts_available=["cluster_assignments", "umap_embedding"],
        )

    n_true = len(s.biology.cell_populations) or 5
    quality = gen.noise.quality_degradation(0.8, [0.95])
    n_clusters = (
        s.last_n_clusters
        if s.last_n_clusters is not None
        else gen.noise.sample_cluster_count(n_true, quality)
    )
    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
    n_cells = s.progress.n_cells_after_filter or s.biology.n_true_cells
    sizes = partition_by_population(
        n_cells, n_clusters, s.biology.cell_populations, gen.noise.rng
    )
    return IntermediateOutput(
        output_type=OutputType.CLUSTER_RESULT,
        step_index=idx,
        quality_score=quality,
        summary=f"Found {n_clusters} clusters",
        data={
            "n_clusters": n_clusters,
            "cluster_names": cluster_names,
            "cluster_sizes": sizes,
            "silhouette_score": gen.noise.sample_qc_metric(0.35, 0.1, -1.0, 1.0),
        },
        uncertainty=abs(n_clusters - n_true) / max(n_true, 1),
        artifacts_available=["cluster_assignments", "umap_embedding"],
    )


def differential_expression(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    """Differential expression handler."""

    if _is_multiclone_expert(s):
        comparison = action.parameters.get("comparison", "post_vs_pre_bulk")
        true_effects = s.biology.true_de_genes.get(comparison, {})
        integrated = s.progress.batches_integrated
        clone_truth = s.biology.clone_truth
        cluster_aliases = _expert_cluster_alias_map(s)
        minor_clone = _minor_clone_name(s)
        dominant_clone = _dominant_clone_name(s)
        batch_noise = sum(s.technical.batch_effects.values()) / max(
            len(s.technical.batch_effects), 1
        )
        base_noise = (
            s.technical.dropout_rate
            + 0.1 * (1.0 - s.technical.sample_quality)
            + 0.5 * batch_noise
        )
        noise_level = base_noise * (0.85 if integrated else 1.15)
        observed = gen.noise.sample_effect_sizes(
            true_effects,
            s.progress.n_cells_after_filter or s.biology.n_true_cells,
            noise_level,
        )

        cluster_de: Dict[str, Any] = {}
        pooled_top: List[tuple[str, float]] = []
        for clone_name, truth in clone_truth.items():
            clone_effects = truth.get("de_genes", {})
            clone_noise = noise_level + (0.10 if not integrated else 0.0)
            if clone_name == minor_clone and not integrated:
                clone_noise += 0.15
            observed_clone = gen.noise.sample_effect_sizes(
                clone_effects,
                max(
                    2000,
                    int(
                        (s.progress.n_cells_after_filter or s.biology.n_true_cells)
                        * truth.get("size", 0.1)
                    ),
                ),
                clone_noise,
            )
            clone_top = sorted(
                observed_clone.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:8]
            clone_detected = not (
                clone_name == minor_clone and not integrated
            )
            cluster_id = cluster_aliases[clone_name]
            if integrated:
                cluster_de[cluster_id] = {
                    "detected": clone_detected,
                    "top_genes": [
                        {"gene": g, "log2FC": round(fc, 3)} for g, fc in clone_top
                    ],
                    "confidence": 0.81 if clone_name == dominant_clone else 0.66,
                }
            if clone_detected:
                pooled_top.extend(clone_top[:5])

        if not pooled_top:
            pooled_top = list(observed.items())
        fp_genes = gen.noise.generate_false_positives(5000, 0.002 + noise_level * 0.01)
        for g in fp_genes[:5]:
            observed[g] = float(gen.noise.rng.normal(0, 0.25))
        merged = dict(pooled_top)
        merged.update(observed)
        top_genes = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:50]
        warnings: List[str] = []
        if not integrated:
            warnings.append(
                "Residual batch structure may understate the smaller resistant branch"
            )
            warnings.append(
                "Bulk contrast shows mixed signal; consider clustering-resolved analysis before assigning mechanisms"
            )
        return IntermediateOutput(
            output_type=OutputType.DE_RESULT,
            step_index=idx,
            quality_score=gen.noise.quality_degradation(
                0.86 if integrated else 0.66,
                [1.0 - min(noise_level, 0.9)],
            ),
            summary=(
                "Bulk DE remains mixed and cannot cleanly separate parallel post-treatment programs"
                if not integrated
                else "Cluster-resolved DE distinguishes parallel post-treatment programs with competing resistance hypotheses"
            ),
            data={
                "comparison": comparison,
                "n_tested": len(merged),
                "top_genes": [
                    {"gene": g, "log2FC": round(fc, 3)} for g, fc in top_genes
                ],
                "n_significant": sum(1 for _, fc in merged.items() if abs(fc) > 0.5),
                "bulk_signal_is_mixed": True,
                **({"cluster_de": cluster_de} if integrated else {}),
            },
            uncertainty=0.24 if integrated else 0.46,
            warnings=warnings,
            artifacts_available=(
                ["de_table", "cluster_specific_de"]
                if integrated
                else ["de_table"]
            ),
        )

    comparison = action.parameters.get("comparison", "disease_vs_healthy")
    if comparison not in s.biology.true_de_genes and s.biology.true_de_genes:
        comparison = next(iter(s.biology.true_de_genes))
    true_effects = s.biology.true_de_genes.get(comparison, {})

    n_cells = s.progress.n_cells_after_filter or s.biology.n_true_cells
    batch_noise = sum(s.technical.batch_effects.values()) / max(
        len(s.technical.batch_effects), 1
    )
    noise_level = (
        s.technical.dropout_rate
        + 0.1 * (1.0 - s.technical.sample_quality)
        + 0.5 * batch_noise
    )
    observed = gen.noise.sample_effect_sizes(true_effects, n_cells, noise_level)

    fp_genes = gen.noise.generate_false_positives(5000, 0.002 + noise_level * 0.01)
    for g in fp_genes:
        observed[g] = float(gen.noise.rng.normal(0, 0.3))

    fn_genes = gen.noise.generate_false_negatives(list(true_effects.keys()), 0.15)
    for g in fn_genes:
        observed.pop(g, None)

    top_genes = sorted(observed.items(), key=lambda kv: abs(kv[1]), reverse=True)[:50]
    return IntermediateOutput(
        output_type=OutputType.DE_RESULT,
        step_index=idx,
        quality_score=gen.noise.quality_degradation(0.8, [1.0 - noise_level]),
        summary=f"DE analysis ({comparison}): {len(observed)} genes tested, {len(top_genes)} top hits",
        data={
            "comparison": comparison,
            "n_tested": len(observed),
            "top_genes": [{"gene": g, "log2FC": round(fc, 3)} for g, fc in top_genes],
            "n_significant": sum(1 for _, fc in observed.items() if abs(fc) > 0.5),
        },
        uncertainty=noise_level,
        artifacts_available=["de_table"],
    )
