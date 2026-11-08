from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

from qnn_edge_agent.types import TierScore


@dataclass(frozen=True)
class McNemarResult:
    b: int
    c: int
    n_discordant: int
    p_value: float


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> TierScore:
    if n <= 0:
        return TierScore(rate=0.0, ci_low=0.0, ci_high=0.0, n=0)

    p = k / n
    z = float(NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0))
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(
        (p * (1.0 - p)) / n + (z**2) / (4.0 * (n**2))
    )

    ci_low = 0.0 if k == 0 else max(0.0, float(center - margin))
    ci_high = 1.0 if k == n else min(1.0, float(center + margin))
    return TierScore(rate=p, ci_low=ci_low, ci_high=ci_high, n=n)


def mcnemar_exact(
    outcomes_a: Sequence[bool],
    outcomes_b: Sequence[bool],
) -> McNemarResult:
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(
            f"Mismatched sequence lengths: {len(outcomes_a)} vs {len(outcomes_b)}"
        )

    b = 0
    c = 0
    for a, b_val in zip(outcomes_a, outcomes_b):
        if a and not b_val:
            b += 1
        elif not a and b_val:
            c += 1

    n_discordant = b + c
    if n_discordant == 0:
        return McNemarResult(b=b, c=c, n_discordant=0, p_value=1.0)

    k_tail = min(b, c)
    cdf_tail = sum(math.comb(n_discordant, i) for i in range(k_tail + 1)) / (2.0**n_discordant)
    p_val = min(1.0, 2.0 * cdf_tail)
    return McNemarResult(
        b=b,
        c=c,
        n_discordant=n_discordant,
        p_value=float(p_val),
    )
