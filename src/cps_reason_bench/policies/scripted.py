"""Deterministic scenario scripts used as transparent reference baselines."""

from __future__ import annotations

import json
from typing import Any

from cps_reason_bench.policies.base import Policy, PolicyRequest, PolicyResult
from cps_reason_bench.protocol import parse_decision


class ScriptedPolicy(Policy):
    policy_name = "scripted"

    def __init__(self, configuration_name: str = "scripted") -> None:
        self.configuration_name = configuration_name

    def decide(self, request: PolicyRequest) -> PolicyResult:
        script = request.scenario.baseline.get("script", [])
        selected: dict[str, Any] | None = None
        for item in script:
            if int(item.get("at_step", -1)) == request.constraints.current_step:
                selected = dict(item.get("decision", {}))
                break
        if selected is None:
            selected = dict(
                request.scenario.baseline.get(
                    "default_decision",
                    {
                        "actions": [],
                        "callback": None,
                        "subscriptions": [],
                        "context_requests": [],
                    },
                )
            )
        parsed = parse_decision(selected, request.constraints)
        raw = json.dumps(selected, sort_keys=True)
        return PolicyResult(
            decision=parsed.decision,
            raw_response=raw,
            parsed_response=parsed.parsed,
            validation_errors=parsed.errors,
            attempts=(
                {
                    "attempt": 1,
                    "raw_response": raw,
                    "parsed_response": parsed.parsed,
                    "validation_errors": list(parsed.errors),
                    "validation_issues": [issue.to_dict() for issue in parsed.issues],
                    "json_valid": parsed.json_valid,
                    "schema_valid": parsed.schema_valid,
                    "decision_valid": parsed.decision_valid,
                },
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "provider": "none",
            "model": "scripted",
            "model_configuration": self.configuration_name,
            "capabilities": {
                "deterministic": True,
                "model_seed": False,
                "token_usage": False,
                "latency": False,
            },
        }
