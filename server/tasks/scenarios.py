"""Pre-defined biological scenarios for task generation.

Each ``Scenario`` bundles a task specification together with the matching
hidden ground-truth biology so the simulator can instantiate consistent
episodes.  The library is intentionally diverse: it covers differential
expression, trajectory inference, perturbation response, and biomarker
validation across tissues and modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from ...models import ExpectedFinding, PaperReference, TaskSpec
    from ..simulator.latent_state import (
        CellPopulation,
        GeneProgram,
        LatentBiologicalState,
        TechnicalState,
    )
except ImportError:  # pragma: no cover - direct module import path
    from models import ExpectedFinding, PaperReference, TaskSpec
    from server.simulator.latent_state import (
        CellPopulation,
        GeneProgram,
        LatentBiologicalState,
        TechnicalState,
    )


@dataclass
class Scenario:
    """A reproducible (task, ground-truth) pair."""

    name: str
    task: TaskSpec
    biology: LatentBiologicalState
    technical: TechnicalState = field(default_factory=TechnicalState)
    hidden_failure_conditions: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)


# ── Scenario library ────────────────────────────────────────────────────────

from .scenario_library.cardiac_disease_de import cardiac_disease_de
from .scenario_library.hematopoiesis_trajectory import hematopoiesis_trajectory
from .scenario_library.perturbation_immune import perturbation_immune
from .scenario_library.venetoclax_resistance_multiclone import (
    venetoclax_resistance_multiclone,
)

SCENARIO_LIBRARY: List[Scenario] = [
    cardiac_disease_de(),
    hematopoiesis_trajectory(),
    perturbation_immune(),
    venetoclax_resistance_multiclone(),
]
