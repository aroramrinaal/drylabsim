"""Hidden failure condition templates for biological building blocks.

Provides realistic technical failure scenarios that can confound
differential expression analysis and other bioinformatics tasks.
"""

from typing import List


HIDDEN_FAILURE_TEMPLATES: List[str] = [
    "High ambient RNA may confound DE in low-abundance transcripts",
    "Strong batch effects between conditions may inflate false positives",
    "Low cell viability in disease samples reduces statistical power",
    "Doublet contamination in dense populations obscures rare cell types",
    "Sample degradation during processing introduces 3' bias artefacts",
    "Dissociation-induced gene expression changes confound stress signatures",
    "Unbalanced sample sizes between conditions reduce DE sensitivity",
]
