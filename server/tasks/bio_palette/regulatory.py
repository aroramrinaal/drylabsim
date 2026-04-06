"""Regulatory network templates for biological building blocks.

Provides transcription factor -> target gene relationships
for constructing biologically plausible regulatory networks.
"""

from typing import Dict, List


REGULATORY_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "erythroid": {
        "GATA1": ["KLF1", "HBB", "HBA1", "GYPA", "ALAS2"],
        "KLF1": ["HBB", "HBA1", "SLC4A1"],
    },
    "myeloid": {
        "CEBPA": ["CSF3R", "ELANE", "MPO", "CTSG"],
        "SPI1": ["CSF1R", "CD14", "FCGR3A", "CD68"],
    },
    "lymphoid": {
        "TCF7": ["CD3D", "CD3E", "IL7R", "LEF1"],
        "PAX5": ["CD19", "MS4A1", "CD79A"],
    },
    "fibrotic": {
        "SMAD3": ["COL1A1", "COL3A1", "FN1", "POSTN"],
        "TGFB1": ["ACTA2", "COL1A1", "CTGF"],
    },
    "inflammatory": {
        "NFKB1": ["TNF", "IL6", "IL1B", "CCL2", "CXCL8"],
        "STAT1": ["IRF1", "CXCL10", "MX1", "OAS1"],
    },
    "stem_cell": {
        "RUNX1": ["CD34", "KIT", "FLT3"],
        "MYC": ["CDK4", "CCND1", "E2F1"],
    },
    "neuronal": {
        "NEUROD1": ["SLC17A7", "NRGN", "SNAP25"],
        "DLX1": ["GAD1", "GAD2", "SLC32A1"],
    },
}
