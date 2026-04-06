from __future__ import annotations

from typing import Any, Dict, List

try:
    from ....models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )
except ImportError:  # pragma: no cover - direct module import path
    from models import (
        ActionType,
        ExperimentAction,
        IntermediateOutput,
        OutputType,
    )

from ..latent_state import FullLatentState
from ..noise import NoiseModel


def collect_sample(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    n_samples = action.parameters.get("n_samples", 6)
    quality = gen.noise.quality_degradation(
        s.technical.sample_quality, [s.technical.capture_efficiency]
    )
    return IntermediateOutput(
        output_type=OutputType.SAMPLE_COLLECTION_RESULT,
        step_index=idx,
        quality_score=quality,
        summary=f"Collected {n_samples} samples (quality={quality:.2f})",
        data={
            "n_samples": n_samples,
            "quality": quality,
            "organism": "human",
            "tissue": "blood",
        },
        artifacts_available=["raw_samples"],
    )


def select_cohort(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    criteria = action.parameters.get("criteria", {})
    n_selected = action.parameters.get("n_selected", 4)
    return IntermediateOutput(
        output_type=OutputType.COHORT_RESULT,
        step_index=idx,
        summary=f"Selected cohort of {n_selected} samples with criteria {criteria}",
        data={"n_selected": n_selected, "criteria": criteria},
        artifacts_available=["cohort_manifest"],
    )


def prepare_library(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    complexity = gen.noise.quality_degradation(
        s.technical.library_complexity,
        [s.technical.sample_quality],
    )
    return IntermediateOutput(
        output_type=OutputType.LIBRARY_PREP_RESULT,
        step_index=idx,
        quality_score=complexity,
        summary=f"Library prepared (complexity={complexity:.2f})",
        data={
            "library_complexity": complexity,
            "method": action.method or "10x_chromium",
        },
        artifacts_available=["prepared_library"],
    )


def culture_cells(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    days = action.parameters.get("days", 7)
    decay = 0.005 * days
    viability = gen.noise.sample_qc_metric(max(0.50, 0.95 - decay), 0.05, 0.30, 1.0)
    return IntermediateOutput(
        output_type=OutputType.CULTURE_RESULT,
        step_index=idx,
        quality_score=viability,
        summary=f"Cultured for {days}d, viability={viability:.2f}",
        data={"days": days, "viability": viability},
        artifacts_available=["cultured_cells"],
    )


def perturb_gene(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    target = action.parameters.get("target", "unknown")
    efficiency = (
        s.last_perturbation_efficiency
        if s.last_perturbation_efficiency is not None
        else gen.noise.sample_qc_metric(0.80, 0.12, 0.0, 1.0)
    )
    off_target_risk = gen.noise.sample_qc_metric(0.10, 0.05, 0.0, 0.5)
    return IntermediateOutput(
        output_type=OutputType.PERTURBATION_RESULT,
        step_index=idx,
        quality_score=efficiency,
        summary=(
            f"Genetic perturbation of {target} "
            f"(efficiency={efficiency:.2f}, off-target risk={off_target_risk:.2f})"
        ),
        data={
            "target": target,
            "efficiency": efficiency,
            "type": action.action_type.value,
            "off_target_risk": off_target_risk,
        },
        artifacts_available=["perturbed_cells"],
    )


def perturb_compound(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    target = action.parameters.get("target", "unknown")
    dose_um = action.parameters.get("dose_uM", 1.0)
    efficiency = (
        s.last_perturbation_efficiency
        if s.last_perturbation_efficiency is not None
        else gen.noise.sample_qc_metric(0.70, 0.15, 0.0, 1.0)
    )
    on_target_frac = gen.noise.sample_qc_metric(0.75, 0.10, 0.0, 1.0)
    return IntermediateOutput(
        output_type=OutputType.PERTURBATION_RESULT,
        step_index=idx,
        quality_score=efficiency * on_target_frac,
        summary=(
            f"Compound perturbation targeting {target} at {dose_um} µM "
            f"(efficiency={efficiency:.2f}, on-target={on_target_frac:.2f})"
        ),
        data={
            "target": target,
            "efficiency": efficiency,
            "type": action.action_type.value,
            "dose_uM": dose_um,
            "on_target_fraction": on_target_frac,
        },
        artifacts_available=["perturbed_cells"],
    )


def sequence_cells(
    gen, action: ExperimentAction, s: FullLatentState, idx: int
) -> IntermediateOutput:
    import math

    depth = s.technical.sequencing_depth_factor
    n_cells = s.progress.n_cells_sequenced or gen.noise.sample_count(
        s.biology.n_true_cells * s.technical.capture_efficiency
    )
    max_genes = 20_000
    saturation_arg = depth * s.technical.library_complexity * 0.8
    n_genes = gen.noise.sample_count(int(max_genes * (1.0 - math.exp(-saturation_arg))))
    median_umi = gen.noise.sample_count(int(3000 * depth))
    quality = gen.noise.quality_degradation(
        s.technical.sample_quality,
        [s.technical.library_complexity, s.technical.capture_efficiency],
    )
    return IntermediateOutput(
        output_type=OutputType.SEQUENCING_RESULT,
        step_index=idx,
        quality_score=quality,
        summary=(
            f"Sequenced {n_cells} cells, {n_genes} genes detected, "
            f"median UMI={median_umi}"
        ),
        data={
            "n_cells": n_cells,
            "n_genes": n_genes,
            "median_umi": median_umi,
            "sequencing_saturation": gen.noise.sample_qc_metric(0.7, 0.1),
        },
        artifacts_available=["raw_count_matrix"],
    )
