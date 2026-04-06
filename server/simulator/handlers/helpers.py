from __future__ import annotations

from typing import List
import numpy as np

# Pool of common transcription factors used to generate realistic false-positive
# regulators, so the agent cannot trivially distinguish true vs. false hits by
# gene-name format alone.
NOISE_TFS: List[str] = [
    "NR3C1",
    "KLF4",
    "EGR1",
    "IRF1",
    "FOSL2",
    "JUN",
    "FOS",
    "ATF3",
    "NFKB1",
    "RELA",
    "SP1",
    "MYC",
    "MAX",
    "E2F1",
    "CTCF",
    "YY1",
    "TP53",
    "STAT5A",
    "SMAD3",
    "TCF7L2",
    "NFE2L2",
    "HIF1A",
    "CREB1",
]


def random_partition(total: int, k: int, rng: np.random.Generator) -> List[int]:
    """Random partition helper."""

    if k <= 0:
        return []
    fracs = rng.dirichlet(alpha=[1.0] * k)
    sizes = [max(1, int(total * f)) for f in fracs]
    diff = total - sum(sizes)
    sizes[0] += diff
    return sizes


def partition_by_population(
    total: int,
    k: int,
    populations: List,
    rng: np.random.Generator,
) -> List[int]:
    """Partition by population helper."""

    if k <= 0:
        return []
    if populations:
        raw = [max(p.proportion, 1e-3) for p in populations]
        if len(raw) >= k:
            alpha = raw[:k]
        else:
            alpha = raw + [sum(raw) / len(raw)] * (k - len(raw))
        scale = k / max(sum(alpha), 1e-6)
        alpha = [a * scale for a in alpha]
    else:
        alpha = [1.0] * k
    fracs = rng.dirichlet(alpha=alpha)
    sizes = [max(1, int(total * f)) for f in fracs]
    diff = total - sum(sizes)
    sizes[0] += diff
    return sizes
