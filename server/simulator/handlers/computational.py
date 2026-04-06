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
