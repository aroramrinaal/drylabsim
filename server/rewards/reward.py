"""Decomposable reward function for the bio-experiment planning POMDP.

Reward components
─────────────────
  r_validity      — biological validity of the chosen action
  r_ordering      — correct ordering of experiment steps
  r_info_gain     — information gain from the step's output
  r_efficiency    — resource efficiency (budget & time normalised)
  r_novelty       — bonus for non-redundant, non-trivial actions
  r_penalty       — penalties for violations, redundancy, waste
  r_terminal      — terminal quality & calibration against hidden truth

Potential-based shaping
  φ(s)            — progress potential used for dense shaping signal

The final step reward is:
  R_t = r_validity + r_ordering + r_info_gain + r_efficiency
        + r_novelty + r_penalty + [φ(s_{t+1}) − φ(s_t)]

The terminal reward adds:
  R_T += r_terminal
"""

from __future__ import annotations

from .reward_breakdown import RewardBreakdown
from .reward_computer import RewardComputer

__all__ = ["RewardBreakdown", "RewardComputer"]
