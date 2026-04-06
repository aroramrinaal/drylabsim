"""Perturbation templates for biological building blocks.

Provides therapeutic perturbation effect profiles with gene-level
effects for simulating drug treatments, CRISPR knockouts, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PerturbationTemplate:
    name: str
    target_pathway: str
    gene_effects: Dict[str, float]
    description: str


PERTURBATION_TEMPLATES: Dict[str, PerturbationTemplate] = {
    "JAK_inhibitor": PerturbationTemplate(
        name="JAK_inhibitor",
        target_pathway="JAK_STAT_signalling",
        gene_effects={
            "STAT1": -0.8,
            "STAT3": -0.7,
            "IFNG": -1.5,
            "IL17A": -1.3,
            "SOCS1": 1.2,
        },
        description="JAK inhibitor treatment",
    ),
    "anti_TNF": PerturbationTemplate(
        name="anti_TNF",
        target_pathway="TNF_signalling",
        gene_effects={
            "TNF": -1.5,
            "IL6": -1.0,
            "IL1B": -0.8,
            "MMP3": -1.2,
            "SOCS3": 0.8,
        },
        description="anti-TNF biologic therapy",
    ),
    "PD1_blockade": PerturbationTemplate(
        name="PD1_blockade",
        target_pathway="immune_checkpoint",
        gene_effects={"PDCD1": -1.0, "GZMB": 1.5, "IFNG": 1.2, "PRF1": 1.0, "TNF": 0.8},
        description="anti-PD-1 immune checkpoint blockade",
    ),
    "BRAF_inhibitor": PerturbationTemplate(
        name="BRAF_inhibitor",
        target_pathway="MAPK_signalling",
        gene_effects={
            "BRAF": -0.5,
            "MAPK1": -1.0,
            "MKI67": -1.5,
            "CCND1": -1.2,
            "FOS": -0.8,
        },
        description="BRAF inhibitor treatment",
    ),
    "TGFb_inhibitor": PerturbationTemplate(
        name="TGFb_inhibitor",
        target_pathway="TGF_beta_signalling",
        gene_effects={
            "TGFB1": -0.8,
            "COL1A1": -1.2,
            "COL3A1": -1.0,
            "POSTN": -1.5,
            "ACTA2": -0.8,
        },
        description="TGF-beta pathway inhibitor",
    ),
    "mTOR_inhibitor": PerturbationTemplate(
        name="mTOR_inhibitor",
        target_pathway="PI3K_AKT_signalling",
        gene_effects={
            "MTOR": -0.8,
            "AKT1": -0.6,
            "MKI67": -1.2,
            "CCND1": -1.0,
            "HIF1A": -0.7,
        },
        description="mTOR inhibitor treatment",
    ),
    "CRISPR_TP53_KO": PerturbationTemplate(
        name="CRISPR_TP53_KO",
        target_pathway="p53_signalling",
        gene_effects={
            "TP53": -2.0,
            "BAX": -1.0,
            "CDKN1A": -1.5,
            "MDM2": -0.8,
            "MKI67": 1.0,
        },
        description="CRISPR knockout of TP53",
    ),
}
