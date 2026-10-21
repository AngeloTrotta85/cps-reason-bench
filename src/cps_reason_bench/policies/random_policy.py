"""Seeded random-action baseline."""

from __future__ import annotations

import json
import random
from typing import Any

from cps_reason_bench.policies.base import Policy, PolicyRequest, PolicyResult
from cps_reason_bench.protocol import parse_decision


class RandomPolicy(Policy):
    policy_name = "random"

    def __init__(self, seed: int = 0, configuration_name: str = "random") -> None:
        self.seed = seed
        self.configuration_name = configuration_name
        self._rng = random.Random(seed)

    def decide(self, request: PolicyRequest) -> PolicyResult:
        maximum = min(
            request.constraints.max_actions,
            len(request.constraints.allowed_action_ids),
        )
        action_count = self._rng.randint(1, maximum)
        action_ids = self._rng.sample(
            list(request.constraints.allowed_action_ids), action_count
        )
        decision_data: dict[str, Any] = {
            "actions": [{"action_id": action_id} for action_id in action_ids],
            "callback": None,
            "subscriptions": [],
            "context_requests": [],
        }
        remaining_steps = request.constraints.max_steps - request.constraints.current_step
        if (
            request.constraints.allows_callbacks
            and remaining_steps >= request.constraints.min_callback_steps
        ):
            decision_data["callback"] = {
                "after_steps": self._rng.randint(
                    request.constraints.min_callback_steps,
                    min(request.constraints.max_callback_steps, remaining_steps),
                )
            }
        parsed = parse_decision(decision_data, request.constraints)
        raw = json.dumps(decision_data, sort_keys=True)
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
            "model": "random",
            "seed": self.seed,
            "model_configuration": self.configuration_name,
            "capabilities": {
                "deterministic": True,
                "model_seed": False,
                "token_usage": False,
                "latency": False,
            },
        }
