"""Grade result dataclass for the DryLabSim grader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class GradeResult:
    """Deterministic grading output guaranteed to be in [0.0, 1.0]."""

    score: float
    completeness: float
    biology_score: float
    efficiency_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, self.score))
        self.completeness = max(0.0, min(1.0, self.completeness))
        self.biology_score = max(0.0, min(1.0, self.biology_score))
        self.efficiency_score = max(0.0, min(1.0, self.efficiency_score))
