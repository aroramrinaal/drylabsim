"""Shared types for the biological rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class RuleViolation:
    rule_id: str
    severity: Severity
    message: str
