"""Reward breakdown dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardBreakdown:
    validity: float = 0.0
    ordering: float = 0.0
    info_gain: float = 0.0
    efficiency: float = 0.0
    novelty: float = 0.0
    penalty: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (
            self.validity
            + self.ordering
            + self.info_gain
            + self.efficiency
            + self.novelty
            + self.penalty
            + self.shaping
            + self.terminal
        )

    def to_dict(self) -> Dict[str, float]:
        d = {
            "validity": self.validity,
            "ordering": self.ordering,
            "info_gain": self.info_gain,
            "efficiency": self.efficiency,
            "novelty": self.novelty,
            "penalty": self.penalty,
            "shaping": self.shaping,
            "terminal": self.terminal,
            "total": self.total,
        }
        d.update(self.components)
        return d
