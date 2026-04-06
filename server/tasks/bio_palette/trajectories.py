"""Trajectory templates for biological building blocks.

Provides developmental trajectory templates through cell populations
for simulating lineage relationships and branching structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TrajectoryTemplate:
    """Template for a developmental trajectory through cell populations."""

    root_population: str
    branches: List[List[str]]
    n_lineages: int
    tissue: str


TRAJECTORY_TEMPLATES: List[TrajectoryTemplate] = [
    TrajectoryTemplate(
        root_population="HSC",
        branches=[
            ["HSC", "CMP", "GMP", "neutrophil"],
            ["HSC", "CMP", "GMP", "monocyte"],
            ["HSC", "MEP", "erythrocyte"],
            ["HSC", "MEP", "megakaryocyte"],
        ],
        n_lineages=3,
        tissue="bone_marrow",
    ),
    TrajectoryTemplate(
        root_population="double_negative_T",
        branches=[
            ["double_negative_T", "double_positive_T", "CD4_SP"],
            ["double_negative_T", "double_positive_T", "CD8_SP"],
        ],
        n_lineages=2,
        tissue="thymus",
    ),
    TrajectoryTemplate(
        root_population="stem_cell",
        branches=[
            ["stem_cell", "colonocyte"],
            ["stem_cell", "goblet_cell"],
        ],
        n_lineages=2,
        tissue="colon",
    ),
    TrajectoryTemplate(
        root_population="OPC",
        branches=[
            ["OPC", "oligodendrocyte"],
        ],
        n_lineages=1,
        tissue="brain",
    ),
]
