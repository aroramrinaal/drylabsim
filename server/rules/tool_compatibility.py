"""Tool / modality compatibility checks."""

from __future__ import annotations

from typing import List

try:
    from ...models import ActionType, ExperimentAction, TOOL_REGISTRY
    from ..simulator.latent_state import FullLatentState
except ImportError:  # pragma: no cover
    from models import ActionType, ExperimentAction, TOOL_REGISTRY
    from server.simulator.latent_state import FullLatentState

from .types import RuleViolation, Severity

_KNOWN_METHODS = {
    "scanpy.pp.calculate_qc_metrics",
    "scanpy.pp.filter_cells",
    "scanpy.pp.filter_genes",
    "scanpy.pp.normalize_total",
    "scanpy.pp.log1p",
    "scanpy.pp.highly_variable_genes",
    "scanpy.pp.neighbors",
    "scanpy.tl.leiden",
    "scanpy.tl.louvain",
    "scanpy.tl.rank_genes_groups",
    "scanpy.tl.paga",
    "scanpy.tl.umap",
    "gseapy.prerank",
    "gseapy.gsea",
    "10x_chromium",
    "NovaSeq",
}
_METHOD_TO_TOOL = {
    "scanpy.pp.calculate_qc_metrics": "Scanpy",
    "scanpy.pp.filter_cells": "Scanpy",
    "scanpy.pp.filter_genes": "Scanpy",
    "scanpy.pp.normalize_total": "Scanpy",
    "scanpy.pp.log1p": "Scanpy",
    "scanpy.pp.highly_variable_genes": "Scanpy",
    "scanpy.pp.neighbors": "Scanpy",
    "scanpy.tl.leiden": "Leiden",
    "scanpy.tl.louvain": "Louvain",
    "scanpy.tl.rank_genes_groups": "Scanpy",
    "scanpy.tl.paga": "PAGA",
    "scanpy.tl.umap": "UMAP",
    "gseapy.prerank": "Scanpy",
    "gseapy.gsea": "Scanpy",
    "10x_chromium": "CellRanger",
    "NovaSeq": "CellRanger",
}


def check_tool_compatibility(
    action: ExperimentAction, s: FullLatentState
) -> List[RuleViolation]:
    vs: List[RuleViolation] = []
    method = action.method
    if not method:
        return vs

    resolved = _METHOD_TO_TOOL.get(method, method)
    tool_spec = TOOL_REGISTRY.get(resolved)
    if tool_spec is None and method not in _KNOWN_METHODS:
        vs.append(
            RuleViolation(
                rule_id="unknown_tool",
                severity=Severity.SOFT,
                message=f"Tool '{method}' is not in the registry — results may be unreliable",
            )
        )
        return vs
    if tool_spec is None:
        return vs

    task_modality = getattr(s, "task_modality", None)
    if task_modality and tool_spec.modalities:
        if task_modality not in tool_spec.modalities:
            vs.append(
                RuleViolation(
                    rule_id="tool_modality_mismatch",
                    severity=Severity.SOFT,
                    message=(
                        f"Tool '{method}' is designed for "
                        f"{', '.join(tool_spec.modalities)} but task modality "
                        f"is '{task_modality}'"
                    ),
                )
            )

    return vs
