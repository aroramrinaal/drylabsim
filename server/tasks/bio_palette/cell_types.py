"""Cell type templates for biological building blocks.

Provides tissue-specific cell type definitions with marker genes,
proportion ranges, states, and disease responsiveness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CellTypeTemplate:
    name: str
    marker_genes: List[str]
    proportion_range: Tuple[float, float] = (0.05, 0.30)
    states: List[str] = field(default_factory=lambda: ["quiescent"])
    disease_responsive: bool = False
    response_range: Tuple[float, float] = (0.5, 1.5)


TISSUE_CELL_TYPES: Dict[str, List[CellTypeTemplate]] = {
    "heart": [
        CellTypeTemplate(
            "cardiomyocyte",
            ["TNNT2", "MYH7", "ACTC1"],
            (0.25, 0.40),
            ["contractile", "stressed"],
        ),
        CellTypeTemplate(
            "cardiac_fibroblast",
            ["COL1A1", "DCN", "LUM"],
            (0.15, 0.30),
            ["quiescent", "activated"],
            True,
            (1.1, 1.8),
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "VWF", "CDH5"], (0.10, 0.20), ["quiescent"]
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "CSF1R"],
            (0.05, 0.15),
            ["quiescent", "activated", "inflammatory"],
            True,
            (1.2, 2.0),
        ),
        CellTypeTemplate(
            "smooth_muscle", ["ACTA2", "MYH11", "TAGLN"], (0.08, 0.18), ["quiescent"]
        ),
        CellTypeTemplate(
            "pericyte", ["PDGFRB", "RGS5", "NOTCH3"], (0.03, 0.10), ["quiescent"]
        ),
    ],
    "lung": [
        CellTypeTemplate(
            "AT2", ["SFTPC", "SFTPB", "ABCA3"], (0.15, 0.25), ["normal", "stressed"]
        ),
        CellTypeTemplate("AT1", ["AGER", "PDPN", "CAV1"], (0.10, 0.18), ["normal"]),
        CellTypeTemplate(
            "alveolar_macrophage",
            ["MARCO", "FABP4", "MCEMP1"],
            (0.10, 0.20),
            ["resident", "activated"],
        ),
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "COL3A1", "POSTN"],
            (0.12, 0.25),
            ["quiescent", "activated"],
            True,
            (1.2, 2.0),
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "CLDN5", "VWF"], (0.08, 0.15), ["quiescent"]
        ),
        CellTypeTemplate(
            "T_cell", ["CD3D", "CD3E", "IL7R"], (0.08, 0.18), ["quiescent", "activated"]
        ),
        CellTypeTemplate(
            "ciliated", ["FOXJ1", "DNAH5", "TPPP3"], (0.05, 0.12), ["normal"]
        ),
    ],
    "brain": [
        CellTypeTemplate(
            "excitatory_neuron",
            ["SLC17A7", "CAMK2A", "NRGN"],
            (0.25, 0.40),
            ["normal", "stressed"],
        ),
        CellTypeTemplate(
            "inhibitory_neuron", ["GAD1", "GAD2", "SLC32A1"], (0.12, 0.22), ["normal"]
        ),
        CellTypeTemplate(
            "astrocyte",
            ["GFAP", "AQP4", "SLC1A3"],
            (0.10, 0.20),
            ["quiescent", "activated"],
            True,
            (1.2, 1.8),
        ),
        CellTypeTemplate(
            "microglia",
            ["CX3CR1", "P2RY12", "TMEM119"],
            (0.05, 0.12),
            ["homeostatic", "activated", "inflammatory"],
            True,
            (1.3, 2.5),
        ),
        CellTypeTemplate(
            "oligodendrocyte", ["MBP", "PLP1", "MOG"], (0.10, 0.18), ["myelinating"]
        ),
        CellTypeTemplate(
            "OPC", ["PDGFRA", "CSPG4", "OLIG2"], (0.03, 0.08), ["progenitor"]
        ),
        CellTypeTemplate(
            "endothelial", ["CLDN5", "FLT1", "PECAM1"], (0.03, 0.08), ["quiescent"]
        ),
    ],
    "liver": [
        CellTypeTemplate(
            "hepatocyte",
            ["ALB", "APOB", "CYP3A4"],
            (0.55, 0.70),
            ["normal", "stressed"],
        ),
        CellTypeTemplate(
            "cholangiocyte", ["KRT19", "KRT7", "EPCAM"], (0.05, 0.10), ["normal"]
        ),
        CellTypeTemplate(
            "kupffer_cell",
            ["CD68", "CLEC4F", "MARCO"],
            (0.08, 0.15),
            ["quiescent", "activated", "inflammatory"],
            True,
            (1.2, 2.0),
        ),
        CellTypeTemplate(
            "stellate_cell",
            ["ACTA2", "LRAT", "PDGFRB"],
            (0.05, 0.12),
            ["quiescent", "activated"],
            True,
            (1.3, 2.0),
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "LYVE1", "STAB2"], (0.05, 0.10), ["quiescent"]
        ),
        CellTypeTemplate(
            "NK_cell",
            ["NKG7", "GNLY", "KLRD1"],
            (0.03, 0.08),
            ["quiescent", "activated"],
        ),
    ],
    "bone_marrow": [
        CellTypeTemplate("HSC", ["CD34", "KIT", "THY1"], (0.03, 0.08), ["stem"]),
        CellTypeTemplate("CMP", ["CD34", "FLT3"], (0.08, 0.15), ["progenitor"]),
        CellTypeTemplate("GMP", ["CSF3R", "CEBPA"], (0.08, 0.15), ["progenitor"]),
        CellTypeTemplate("MEP", ["GATA1", "KLF1"], (0.06, 0.12), ["progenitor"]),
        CellTypeTemplate(
            "erythrocyte", ["HBA1", "HBB", "GYPA"], (0.15, 0.25), ["mature"]
        ),
        CellTypeTemplate(
            "neutrophil", ["ELANE", "MPO", "CTSG"], (0.12, 0.22), ["mature"]
        ),
        CellTypeTemplate(
            "monocyte", ["CD14", "CSF1R", "FCGR3A"], (0.10, 0.18), ["mature"]
        ),
        CellTypeTemplate(
            "megakaryocyte", ["ITGA2B", "GP1BA", "PF4"], (0.05, 0.12), ["mature"]
        ),
    ],
    "kidney": [
        CellTypeTemplate(
            "proximal_tubule",
            ["SLC34A1", "LRP2", "CUBN"],
            (0.30, 0.45),
            ["normal", "stressed"],
        ),
        CellTypeTemplate(
            "distal_tubule", ["SLC12A3", "CALB1"], (0.10, 0.18), ["normal"]
        ),
        CellTypeTemplate(
            "collecting_duct", ["AQP2", "FXYD4"], (0.08, 0.15), ["normal"]
        ),
        CellTypeTemplate(
            "podocyte", ["NPHS1", "NPHS2", "WT1"], (0.05, 0.10), ["normal", "stressed"]
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "EMCN", "FLT1"], (0.05, 0.12), ["quiescent"]
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "CSF1R"],
            (0.05, 0.10),
            ["quiescent", "inflammatory"],
            True,
            (1.3, 2.0),
        ),
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "PDGFRA", "DCN"],
            (0.05, 0.12),
            ["quiescent", "activated"],
            True,
            (1.2, 1.8),
        ),
    ],
    "colon": [
        CellTypeTemplate(
            "colonocyte",
            ["CA2", "AQP8", "SLC26A3"],
            (0.25, 0.40),
            ["normal", "stressed"],
        ),
        CellTypeTemplate(
            "goblet_cell", ["MUC2", "TFF3", "FCGBP"], (0.10, 0.18), ["secretory"]
        ),
        CellTypeTemplate(
            "stem_cell", ["LGR5", "ASCL2", "OLFM4"], (0.05, 0.10), ["stem"]
        ),
        CellTypeTemplate(
            "T_cell", ["CD3D", "CD3E", "IL7R"], (0.10, 0.18), ["quiescent", "activated"]
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "CSF1R"],
            (0.05, 0.12),
            ["quiescent", "inflammatory"],
            True,
            (1.3, 2.0),
        ),
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "COL3A1", "VIM"],
            (0.08, 0.15),
            ["quiescent", "activated"],
            True,
            (1.2, 1.8),
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "VWF", "CDH5"], (0.05, 0.10), ["quiescent"]
        ),
    ],
    "pancreas": [
        CellTypeTemplate(
            "beta_cell",
            ["INS", "MAFA", "NKX6-1"],
            (0.25, 0.40),
            ["normal", "stressed"],
            True,
            (0.4, 0.8),
        ),
        CellTypeTemplate(
            "alpha_cell", ["GCG", "ARX", "IRX2"], (0.15, 0.25), ["normal"]
        ),
        CellTypeTemplate("delta_cell", ["SST", "HHEX"], (0.05, 0.10), ["normal"]),
        CellTypeTemplate("ductal", ["KRT19", "SOX9", "CFTR"], (0.10, 0.18), ["normal"]),
        CellTypeTemplate(
            "acinar", ["PRSS1", "CPA1", "CELA3A"], (0.10, 0.20), ["normal"]
        ),
        CellTypeTemplate(
            "stellate",
            ["ACTA2", "PDGFRA", "COL1A1"],
            (0.05, 0.10),
            ["quiescent", "activated"],
            True,
            (1.2, 1.8),
        ),
        CellTypeTemplate(
            "macrophage", ["CD68", "CD163"], (0.03, 0.08), ["quiescent", "inflammatory"]
        ),
    ],
    "skin": [
        CellTypeTemplate(
            "keratinocyte",
            ["KRT14", "KRT5", "KRT1"],
            (0.40, 0.55),
            ["basal", "differentiated"],
        ),
        CellTypeTemplate(
            "melanocyte",
            ["MLANA", "PMEL", "TYR"],
            (0.05, 0.10),
            ["normal", "activated"],
        ),
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "COL3A1", "DCN"],
            (0.10, 0.20),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "T_cell", ["CD3D", "CD3E", "IL7R"], (0.08, 0.15), ["quiescent", "activated"]
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "CSF1R"],
            (0.05, 0.10),
            ["quiescent", "inflammatory"],
        ),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF"], (0.05, 0.10), ["quiescent"]),
    ],
    "breast": [
        CellTypeTemplate(
            "luminal_epithelial",
            ["KRT8", "KRT18", "EPCAM"],
            (0.25, 0.40),
            ["normal", "stressed"],
        ),
        CellTypeTemplate(
            "basal_epithelial", ["KRT14", "KRT5", "TP63"], (0.10, 0.20), ["normal"]
        ),
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "COL3A1", "FAP"],
            (0.10, 0.20),
            ["quiescent", "activated"],
            True,
            (1.2, 1.8),
        ),
        CellTypeTemplate(
            "T_cell",
            ["CD3D", "CD3E", "CD8A"],
            (0.08, 0.15),
            ["quiescent", "activated", "exhausted"],
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "CSF1R"],
            (0.05, 0.12),
            ["quiescent", "inflammatory"],
            True,
            (1.3, 2.0),
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "VWF", "CDH5"], (0.05, 0.10), ["quiescent"]
        ),
    ],
    "synovium": [
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "FAP", "THY1"],
            (0.20, 0.30),
            ["quiescent", "activated"],
            True,
            (1.2, 1.8),
        ),
        CellTypeTemplate(
            "CD4_T_cell",
            ["CD3D", "CD4", "IL7R"],
            (0.12, 0.22),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "CD8_T_cell",
            ["CD3D", "CD8A", "GZMB"],
            (0.08, 0.15),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "MARCO"],
            (0.10, 0.18),
            ["quiescent", "inflammatory"],
            True,
            (1.3, 2.0),
        ),
        CellTypeTemplate(
            "B_cell", ["CD19", "MS4A1", "CD79A"], (0.05, 0.12), ["quiescent"]
        ),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF"], (0.05, 0.10), ["quiescent"]),
        CellTypeTemplate(
            "mast_cell", ["KIT", "TPSAB1", "CPA3"], (0.03, 0.08), ["quiescent"]
        ),
    ],
    "aorta": [
        CellTypeTemplate(
            "smooth_muscle",
            ["ACTA2", "MYH11", "TAGLN"],
            (0.30, 0.45),
            ["contractile", "synthetic"],
            True,
            (0.6, 0.9),
        ),
        CellTypeTemplate(
            "endothelial",
            ["PECAM1", "VWF", "CDH5"],
            (0.15, 0.25),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "TREM2"],
            (0.08, 0.15),
            ["quiescent", "inflammatory"],
            True,
            (1.5, 2.5),
        ),
        CellTypeTemplate(
            "fibroblast",
            ["COL1A1", "LUM", "DCN"],
            (0.08, 0.15),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "T_cell", ["CD3D", "CD3E", "IL7R"], (0.05, 0.12), ["quiescent", "activated"]
        ),
        CellTypeTemplate(
            "dendritic_cell", ["FCER1A", "CD1C", "CLEC10A"], (0.03, 0.08), ["quiescent"]
        ),
    ],
    "blood": [
        CellTypeTemplate(
            "CD4_T_cell",
            ["CD3D", "CD4", "IL7R"],
            (0.15, 0.25),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "CD8_T_cell",
            ["CD3D", "CD8A", "GZMB"],
            (0.10, 0.18),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "B_cell", ["CD19", "MS4A1", "CD79A"], (0.08, 0.15), ["quiescent"]
        ),
        CellTypeTemplate(
            "NK_cell",
            ["NKG7", "GNLY", "KLRD1"],
            (0.05, 0.12),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "monocyte",
            ["CD14", "CSF1R", "FCGR3A"],
            (0.15, 0.25),
            ["classical", "non_classical"],
        ),
        CellTypeTemplate(
            "neutrophil", ["ELANE", "MPO", "CTSG"], (0.10, 0.20), ["mature"]
        ),
        CellTypeTemplate(
            "platelet", ["ITGA2B", "GP1BA", "PF4"], (0.03, 0.08), ["normal"]
        ),
    ],
    "spleen": [
        CellTypeTemplate(
            "B_cell",
            ["CD19", "MS4A1", "CD79A"],
            (0.20, 0.35),
            ["quiescent", "activated"],
        ),
        CellTypeTemplate(
            "T_cell", ["CD3D", "CD3E", "IL7R"], (0.15, 0.25), ["quiescent", "activated"]
        ),
        CellTypeTemplate(
            "macrophage",
            ["CD68", "CD163", "CSF1R"],
            (0.10, 0.18),
            ["quiescent", "inflammatory"],
        ),
        CellTypeTemplate(
            "dendritic_cell", ["FCER1A", "CD1C", "CLEC10A"], (0.05, 0.10), ["quiescent"]
        ),
        CellTypeTemplate(
            "NK_cell", ["NKG7", "GNLY", "KLRD1"], (0.05, 0.12), ["quiescent"]
        ),
        CellTypeTemplate(
            "endothelial", ["PECAM1", "STAB2"], (0.05, 0.10), ["quiescent"]
        ),
    ],
    "thymus": [
        CellTypeTemplate(
            "double_negative_T", ["CD3D", "PTCRA"], (0.10, 0.18), ["progenitor"]
        ),
        CellTypeTemplate(
            "double_positive_T", ["CD3D", "CD4", "CD8A"], (0.30, 0.45), ["progenitor"]
        ),
        CellTypeTemplate("CD4_SP", ["CD3D", "CD4", "IL7R"], (0.10, 0.18), ["mature"]),
        CellTypeTemplate("CD8_SP", ["CD3D", "CD8A", "CD8B"], (0.08, 0.15), ["mature"]),
        CellTypeTemplate(
            "thymic_epithelial",
            ["FOXN1", "KRT5", "KRT8"],
            (0.05, 0.12),
            ["cortical", "medullary"],
        ),
        CellTypeTemplate(
            "dendritic_cell", ["FCER1A", "CD1C"], (0.03, 0.08), ["quiescent"]
        ),
        CellTypeTemplate("macrophage", ["CD68", "CD163"], (0.03, 0.08), ["quiescent"]),
    ],
}
