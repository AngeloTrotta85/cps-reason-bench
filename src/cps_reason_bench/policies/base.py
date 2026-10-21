"""Minimal policy interface; intentionally not an autonomous-agent framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.protocol import Decision, DecisionConstraints


@dataclass(frozen=True)
class PolicyRequest:
    scenario: ScenarioConfig
    context: dict[str, Any]
    constraints: DecisionConstraints
    invocation_index: int


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision | None
    prompt: tuple[dict[str, str], ...] | None = None
    raw_response: str | None = None
    parsed_response: dict[str, Any] | None = None
    validation_errors: tuple[str, ...] = ()
    attempts: tuple[dict[str, Any], ...] = ()
    token_usage: dict[str, int | float] = field(default_factory=dict)
    latency_seconds: float = 0.0
    rendered_prompt_hash: str | None = None
    response_schema_hash: str | None = None
    response_processing: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.decision is not None and not self.validation_errors


class Policy(ABC):
    policy_name: str

    @abstractmethod
    def decide(self, request: PolicyRequest) -> PolicyResult:
        """Return one protocol decision for one benchmark interaction."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata that contains no secrets."""
