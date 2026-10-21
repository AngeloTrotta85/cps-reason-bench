from __future__ import annotations

import unittest
from unittest.mock import patch

from cps_reason_bench.models import (
    DecodingParameters,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    OllamaBackend,
)
from cps_reason_bench.policies import LLMPolicy, PolicyRequest
from cps_reason_bench.protocol import (
    CallbackRequest,
    DecisionConstraints,
    parse_decision,
)
from cps_reason_bench.scheduling import CallbackScheduler
from cps_reason_bench.subscriptions import SubscriptionManager
from cps_reason_bench.types import SimulatorEvent
from tests.test_config_and_simulator import SCENARIOS
from cps_reason_bench.config import load_scenario


def constraints(**overrides: object) -> DecisionConstraints:
    values: dict[str, object] = {
        "allowed_action_ids": ("wait",),
        "action_parameter_schemas": {"wait": {}},
        "max_actions": 1,
        "current_step": 0,
        "current_time": 0.0,
        "max_steps": 8,
        "time_step": 1.0,
        "allows_callbacks": True,
        "min_callback_steps": 1,
        "max_callback_steps": 5,
        "allows_subscriptions": True,
        "allowed_event_types": ("variable_threshold",),
        "active_subscription_ids": (),
        "allowed_context_requests": ("event_history",),
        "reasoning_enabled": False,
    }
    values.update(overrides)
    return DecisionConstraints(**values)  # type: ignore[arg-type]


class FakeBackend(ModelBackend):
    provider = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return ModelResponse(response, {"total_tokens": 10}, 0.01)


class ProtocolAndInteractionTests(unittest.TestCase):
    def test_valid_decision_parses(self) -> None:
        result = parse_decision(
            {
                "actions": [{"action_id": "wait"}],
                "callback": {"after_steps": 2},
                "subscriptions": [
                    {
                        "operation": "add",
                        "id": "watch",
                        "event_type": "variable_threshold",
                        "filters": {
                            "target": "zone.level",
                            "operator": "ge",
                            "threshold": 5,
                        },
                    }
                ],
                "context_requests": ["event_history"],
                "reasoning": None,
            },
            constraints(),
        )
        self.assertTrue(result.valid, result.errors)

    def test_disabled_callback_is_rejected(self) -> None:
        result = parse_decision(
            {
                "actions": [],
                "callback": {"after_steps": 1},
                "subscriptions": [],
                "context_requests": [],
            },
            constraints(allows_callbacks=False),
        )
        self.assertFalse(result.valid)
        self.assertIn("disabled", result.errors[0])

    def test_unsupported_action_parameters_are_protocol_errors(self) -> None:
        result = parse_decision(
            {
                "actions": [{"action_id": "wait", "parameters": {"invented": "value"}}],
                "callback": None,
                "subscriptions": [],
                "context_requests": [],
            },
            constraints(),
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("unsupported parameters" in item for item in result.errors))

    def test_subscription_trigger_does_not_leak_internal_value(self) -> None:
        parsed = parse_decision(
            {
                "actions": [],
                "callback": None,
                "subscriptions": [
                    {
                        "operation": "add",
                        "id": "watch",
                        "event_type": "variable_threshold",
                        "filters": {
                            "target": "zone.level",
                            "operator": "ge",
                            "threshold": 5,
                        },
                    }
                ],
                "context_requests": [],
            },
            constraints(),
        )
        assert parsed.decision is not None
        manager = SubscriptionManager()
        manager.apply(parsed.decision.subscriptions, step=0)
        event = SimulatorEvent(
            "variable_changed",
            1,
            1.0,
            {"target": "zone.level", "old": 4, "new": 6},
            {"target": "zone.level"},
        )
        trigger = manager.match([event])[0]
        self.assertNotIn("new", trigger.details)
        self.assertNotIn("event_data", trigger.details)

    def test_event_values_use_the_public_allowlist_not_internal_event_data(
        self,
    ) -> None:
        manager = SubscriptionManager()
        parsed = parse_decision(
            {
                "actions": [],
                "callback": None,
                "subscriptions": [
                    {
                        "operation": "add",
                        "id": "watch",
                        "event_type": "variable_threshold",
                        "filters": {
                            "target": "zone.level",
                            "operator": "ge",
                            "threshold": 5,
                        },
                    }
                ],
                "context_requests": [],
            },
            constraints(),
        )
        assert parsed.decision is not None
        manager.apply(parsed.decision.subscriptions, step=0)
        event = SimulatorEvent(
            "variable_changed",
            1,
            1.0,
            {
                "target": "zone.level",
                "old": 4,
                "new": 6,
                "hidden_fault_type": "sensor_bias",
                "internal_id": "secret-fault-17",
            },
            {
                "target": "zone.level",
                "new": 6,
                "internal_id": "must-be-dropped",
            },
        )

        trigger = manager.match([event], include_event_values=True)[0]
        self.assertEqual(
            {"target": "zone.level", "new": 6},
            trigger.details["event_data"],
        )
        self.assertNotIn("hidden_fault_type", trigger.details)
        self.assertNotIn("internal_id", trigger.details)
        self.assertNotIn("old", trigger.details["event_data"])

    def test_scheduler_converts_simulated_time(self) -> None:
        parsed = parse_decision(
            {
                "actions": [],
                "callback": {"after_time": 2.5},
                "subscriptions": [],
                "context_requests": [],
            },
            constraints(time_step=1.0),
        )
        assert parsed.decision is not None and parsed.decision.callback is not None
        scheduler = CallbackScheduler()
        scheduled = scheduler.apply(parsed.decision.callback, 0, 0.0, 1.0)
        assert scheduled is not None
        self.assertEqual(3, scheduled.due_step)

    def test_canceled_callback_never_triggers(self) -> None:
        scheduler = CallbackScheduler()
        scheduler.apply(CallbackRequest(after_steps=2), 0, 0.0, 1.0)
        scheduler.apply(CallbackRequest(cancel=True), 1, 1.0, 1.0)
        self.assertIsNone(scheduler.consume_if_due(2))
        self.assertEqual(1, scheduler.cancel_count)
        self.assertEqual([], scheduler.triggered_history)

    def test_llm_policy_retries_invalid_json_and_aggregates_usage(self) -> None:
        backend = FakeBackend(
            [
                "not json",
                '{"actions": [{"action_id": "wait"}], "callback": null, '
                '"subscriptions": [], "context_requests": [], "reasoning": null}',
            ]
        )
        policy = LLMPolicy(backend, DecodingParameters(), max_output_retries=1)
        scenario = load_scenario(SCENARIOS / "prospective_basic.yaml")
        result = policy.decide(
            PolicyRequest(scenario, {"observation": {}}, constraints(), 0)
        )
        self.assertTrue(result.valid, result.validation_errors)
        self.assertEqual(2, backend.calls)
        self.assertEqual(20, result.token_usage["total_tokens"])
        self.assertEqual(2, len(result.attempts))

    def test_empty_model_output_is_a_protocol_failure_not_a_backend_failure(
        self,
    ) -> None:
        backend = FakeBackend([""])
        policy = LLMPolicy(backend, DecodingParameters(), max_output_retries=0)
        scenario = load_scenario(SCENARIOS / "prospective_basic.yaml")

        result = policy.decide(
            PolicyRequest(scenario, {"observation": {}}, constraints(), 0)
        )

        self.assertFalse(result.valid)
        self.assertEqual("", result.raw_response)
        self.assertEqual(1, backend.calls)
        self.assertNotEqual("backend", result.attempts[0].get("validation_stage"))
        self.assertTrue(result.attempts[0]["validation_errors"])

    @patch("cps_reason_bench.models.ollama._post_json")
    def test_ollama_disables_thinking_by_default(self, post_json: object) -> None:
        mock = post_json
        mock.return_value = {  # type: ignore[attr-defined]
            "message": {"content": "{}"},
            "prompt_eval_count": 2,
            "eval_count": 1,
        }
        backend = OllamaBackend("qwen3:0.6b")
        backend.generate(ModelRequest((), {"type": "object"}, DecodingParameters()))
        payload = mock.call_args.args[1]  # type: ignore[attr-defined]
        self.assertIs(payload["think"], False)


if __name__ == "__main__":
    unittest.main()
