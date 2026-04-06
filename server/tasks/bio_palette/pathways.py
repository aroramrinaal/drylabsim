"""Pathway gene libraries for biological building blocks.

Provides pathway-specific gene sets for enrichment analysis,
regulatory network construction, and mechanism annotation.
"""

from typing import Dict, List


PATHWAY_LIBRARY: Dict[str, List[str]] = {
    "TGF_beta_signalling": ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "SMAD4", "ACVR1"],
    "Wnt_signalling": ["WNT3A", "CTNNB1", "APC", "AXIN2", "LGR5", "TCF7L2"],
    "MAPK_signalling": ["KRAS", "BRAF", "MAP2K1", "MAPK1", "MAPK3", "FOS", "JUN"],
    "JAK_STAT_signalling": [
        "JAK1",
        "JAK2",
        "STAT1",
        "STAT3",
        "STAT5A",
        "SOCS1",
        "SOCS3",
    ],
    "PI3K_AKT_signalling": ["PIK3CA", "AKT1", "MTOR", "PTEN", "TSC2"],
    "NF_kB_signalling": ["NFKB1", "RELA", "IKBKB", "TNF", "IL1B"],
    "cell_cycle": ["CDK4", "CDK6", "CCND1", "CCNE1", "RB1", "E2F1", "MKI67"],
    "apoptosis": ["BCL2", "BAX", "BAK1", "CASP3", "CASP9", "TP53", "BID"],
    "inflammatory_response": ["TNF", "IL6", "IL1B", "CCL2", "CXCL8", "NFKB1"],
    "extracellular_matrix_organisation": [
        "COL1A1",
        "COL3A1",
        "FN1",
        "POSTN",
        "MMP2",
        "MMP9",
        "TIMP1",
    ],
    "complement_cascade": ["C1QA", "C1QB", "C3", "C4A", "C5", "CFB"],
    "neuroinflammation": ["TREM2", "CX3CR1", "P2RY12", "IL1B", "TNF", "C1QA"],
    "synaptic_signalling": ["SLC17A7", "GRIA1", "GRIN1", "DLG4", "SNAP25", "SYP"],
    "hematopoietic_cell_lineage": ["CD34", "KIT", "FLT3", "GATA1", "CEBPA", "SPI1"],
    "insulin_signalling": ["INS", "INSR", "IRS1", "PIK3CA", "AKT1", "SLC2A4"],
    "ER_stress_response": ["DDIT3", "ATF4", "XBP1", "HSPA5", "EIF2AK3"],
    "oxidative_stress": ["SOD1", "SOD2", "CAT", "GPX1", "NFE2L2", "HMOX1"],
    "angiogenesis": ["VEGFA", "VEGFB", "KDR", "FLT1", "ANGPT1", "ANGPT2"],
    "EMT": ["CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1"],
    "immune_checkpoint": ["CD274", "PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT"],
    "T_cell_activation": ["CD3D", "CD28", "LCK", "ZAP70", "IL2", "IFNG"],
    "T_cell_exhaustion": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1"],
    "TNF_signalling": ["TNF", "TNFRSF1A", "TRADD", "RIPK1", "NFKB1", "CASP8"],
    "Th17_differentiation": ["IL17A", "IL17F", "RORC", "IL23R", "CCR6", "STAT3"],
    "interferon_signalling": ["IFNG", "IFNB1", "STAT1", "IRF1", "IRF7", "MX1", "OAS1"],
    "lipid_metabolism": ["APOE", "APOB", "LDLR", "HMGCR", "ABCA1", "PPARG"],
    "myelination": ["MBP", "PLP1", "MOG", "MAG", "OLIG2", "SOX10"],
    "foam_cell_formation": ["CD36", "MSR1", "ABCA1", "APOE", "LGALS3", "TREM2"],
    "smooth_muscle_contraction": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK"],
    "glucagon_signalling": ["GCG", "GCGR", "CREB1", "PCK1", "G6PC"],
    "matrix_metalloproteinase_activity": [
        "MMP1",
        "MMP2",
        "MMP3",
        "MMP9",
        "TIMTIMP1",
        "TIMP2",
    ],
    "estrogen_signalling": ["ESR1", "ESR2", "PGR", "GREB1", "TFF1"],
    "melanogenesis": ["MITF", "TYR", "TYRP1", "DCT", "MLANA", "PMEL"],
    "VEGF_signalling": ["VEGFA", "VEGFB", "KDR", "FLT1", "NRP1"],
}
