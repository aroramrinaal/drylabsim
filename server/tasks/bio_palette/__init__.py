"""Curated biological building blocks for procedural scenario generation.

Provides tissue-specific cell types, disease profiles, pathway libraries,
regulatory network templates, and perturbation effect profiles.  The
procedural generator composes these into complete ``Scenario`` objects
with fully populated ``LatentBiologicalState``.
"""

from __future__ import annotations

# Re-export all symbols from the refactored modules to maintain
# backward compatibility with existing imports
from .cell_types import CellTypeTemplate, TISSUE_CELL_TYPES
from .diseases import DiseaseProfile, DISEASE_PROFILES
from .pathways import PATHWAY_LIBRARY
from .regulatory import REGULATORY_TEMPLATES
from .perturbations import PerturbationTemplate, PERTURBATION_TEMPLATES
from .trajectories import TrajectoryTemplate, TRAJECTORY_TEMPLATES
from .failures import HIDDEN_FAILURE_TEMPLATES


__all__ = [
    # Classes
    "CellTypeTemplate",
    "DiseaseProfile",
    "PerturbationTemplate",
    "TrajectoryTemplate",
    # Dictionaries/Lists
    "TISSUE_CELL_TYPES",
    "DISEASE_PROFILES",
    "PATHWAY_LIBRARY",
    "REGULATORY_TEMPLATES",
    "PERTURBATION_TEMPLATES",
    "TRAJECTORY_TEMPLATES",
    "HIDDEN_FAILURE_TEMPLATES",
]
