from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from cps_reason_bench.config import ContextConfig, ScenarioConfig, load_scenario
from cps_reason_bench.context import ContextManager
from cps_reason_bench.models.base import ModelBackend, ModelRequest, ModelResponse
from cps_reason_bench.policies import LLMPolicy, ScriptedPolicy
from cps_reason_bench.protocol import ActionCommand, DecisionConstraints, parse_decision
from cps_reason_bench.references import analyze_reference
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.validation import validate_scenario_semantics


ROOT = Path(__file__).resolve().parents[1]
PLANNING = ROOT / "examples" / "scenarios" / "planning_basic.yaml"


class SequenceBackend(ModelBackend):
    provider = "sequence"
    model = "sequence-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        response = self.responses[self.index]
        self.index += 1
        return ModelResponse(response, {"total_tokens": 10}, 0.01)


class CapturingSequenceBackend(SequenceBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return super().generate(request)


def decision(action_id: str) -> str:
    return json.dumps(
        {
            "actions": [{"action_id": action_id}],
            "callback": None,
            "subscriptions": [],
            "context_requests": [],
            "reasoning": None,
        }
    )


class ReferenceFeedbackAndRetryTests(unittest.TestCase):
    def test_reference_solver_establishes_planning_minimum(self) -> None:
        report = analyze_reference(load_scenario(PLANNING))
        self.assertTrue(report.script_valid)
        self.assertTrue(report.task_success)
        self.assertEqual("optimal_established", report.solver.status)
        self.assertEqual(3, report.solver.minimum_action_count)

    def test_semantic_validation_detects_unreachable_goal_and_bad_reference(self) -> None:
        raw = copy.deepcopy(load_scenario(PLANNING).to_mapping())
        raw.pop("difficulty_profile", None)
        raw["graph"]["nodes"]["plant"]["variables"]["locked"] = {"initial": False}
        raw["goals"] = [
            {
                "id": "unreachable",
                "condition": {"target": "plant.locked", "operator": "eq", "value": True},
            }
        ]
        raw["baseline"]["script"][0]["decision"]["actions"] = [
            {"action_id": "repair_output"}
        ]
        report = validate_scenario_semantics(ScenarioConfig.from_mapping(raw))
        self.assertFalse(report.valid)
        self.assertTrue(any("unreachable" in item for item in report.errors))
        self.assertTrue(any("reference script" in item for item in report.errors))

    def test_public_failure_feedback_can_be_clear_generic_or_omitted(self) -> None:
        scenario = load_scenario(PLANNING)
        simulator = GraphCPSSimulator(scenario)
        simulator.apply_actions([ActionCommand("restore_power")])
        outcomes, _ = simulator.apply_actions([ActionCommand("restore_power")])
        clear = outcomes[0].to_public_dict()
        self.assertIn("precondition", clear["message"])
        self.assertIn("unmet_precondition", clear["details"])

        raw = copy.deepcopy(scenario.to_mapping())
        raw.pop("difficulty_profile", None)
        next(action for action in raw["actions"] if action["id"] == "restore_power")[
            "failure_feedback"
        ] = "generic"
        generic_simulator = GraphCPSSimulator(ScenarioConfig.from_mapping(raw))
        generic_simulator.apply_actions([ActionCommand("restore_power")])
        generic_outcomes, _ = generic_simulator.apply_actions(
            [ActionCommand("restore_power")]
        )
        generic = generic_outcomes[0]
        self.assertIn("precondition", generic.message)
        self.assertEqual("action rejected", generic.to_public_dict()["message"])
        self.assertNotIn("unmet_precondition", generic.to_public_dict()["details"])

        context = ContextManager(
            ContextConfig.from_mapping(
                {"mode": "current", "include_action_outcomes": False}
            )
        )
        visible = context.build(
            simulator.observation(), [], [], [], []
        )["observation"]
        self.assertNotIn("last_action_outcomes", visible)
        self.assertNotIn("full_state", visible)

        events_context = ContextManager(
            ContextConfig.from_mapping(
                {"mode": "current", "include_public_events": True}
            )
        ).build(
            simulator.observation(),
            [],
            [],
            [],
            [{"event_type": "safety_violation", "details": {"constraint_id": "c"}}],
        )
        self.assertEqual(
            "safety_violation",
            events_context["public_events_since_last_interaction"][0]["event_type"],
        )

    def test_omitted_outcomes_are_absent_from_observation_history_and_requests(self) -> None:
        raw = copy.deepcopy(load_scenario(PLANNING).to_mapping())
        raw.pop("difficulty_profile", None)
        raw["context"].update(
            {
                "mode": "history",
                "history_limit": 3,
                "include_action_outcomes": False,
                "include_public_events": True,
            }
        )
        first = json.loads(decision("restore_power"))
        first["context_requests"] = ["action_history", "event_history"]
        backend = CapturingSequenceBackend(
            [
                json.dumps(first),
                decision("restore_power"),
                decision("position_maintenance"),
                decision("repair_output"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            BenchmarkRunner(
                ScenarioConfig.from_mapping(raw),
                LLMPolicy(backend),
                directory,
            ).run()

        self.assertGreaterEqual(len(backend.requests), 2)
        second_payload = json.loads(backend.requests[1].messages[1]["content"])
        context = second_payload["context"]
        serialized = json.dumps(context, sort_keys=True)
        self.assertNotIn("last_action_outcomes", serialized)
        self.assertNotIn('"action_outcomes"', serialized)
        action_history = context["requested_context"]["action_history"]
        self.assertTrue(action_history)
        self.assertNotIn("outcome", action_history[0])
        public_events = context.get("public_events_since_last_interaction", [])
        self.assertFalse(
            any(
                item.get("event_type")
                in {
                    "action_applied",
                    "action_failed",
                    "action_scheduled",
                    "actuator_failed",
                    "invalid_action",
                }
                for item in public_events
            )
        )

    def test_repeated_rejection_metrics_identify_feedback_condition(self) -> None:
        source = load_scenario(PLANNING)

        def run_variant(feedback: str, include_outcomes: bool) -> dict[str, int]:
            raw = copy.deepcopy(source.to_mapping())
            raw.pop("difficulty_profile", None)
            raw["context"]["include_action_outcomes"] = include_outcomes
            action = next(
                item for item in raw["actions"] if item["id"] == "restore_power"
            )
            action["failure_feedback"] = feedback
            raw["baseline"]["script"] = [
                {
                    "at_step": step,
                    "decision": {
                        "actions": [{"action_id": "restore_power"}],
                        "callback": None,
                        "subscriptions": [],
                        "context_requests": [],
                    },
                }
                for step in (0, 1, 2)
            ]
            with tempfile.TemporaryDirectory() as directory:
                result = BenchmarkRunner(
                    ScenarioConfig.from_mapping(raw),
                    ScriptedPolicy(),
                    directory,
                ).run()
            return result.metrics["task_specific"]

        clear = run_variant("full", True)
        omitted = run_variant("full", False)
        hidden = run_variant("generic", True)
        self.assertEqual(1, clear["repeated_rejected_actions_with_clear_feedback"])
        self.assertEqual(1, omitted["repeated_rejected_actions_with_omitted_feedback"])
        self.assertEqual(1, hidden["repeated_rejected_actions_with_hidden_cause"])

    def test_retry_metrics_preserve_first_failure_and_added_cost(self) -> None:
        backend = SequenceBackend(
            ["not-json", decision("restore_power"), decision("position_maintenance"), decision("repair_output")]
        )
        policy = LLMPolicy(
            backend,
            max_output_retries=1,
            retry_policy_name="one_correction_assisted",
        )
        with tempfile.TemporaryDirectory() as directory:
            result = BenchmarkRunner(load_scenario(PLANNING), policy, directory).run()
        compliance = result.metrics["protocol_compliance"]
        self.assertEqual(3, compliance["first_attempt_count"])
        self.assertEqual(2, compliance["first_attempt_json_valid"])
        self.assertEqual(1, compliance["correction_requests"])
        self.assertEqual(1, compliance["valid_after_correction"])
        self.assertEqual(1, compliance["additional_model_calls"])
        self.assertEqual(10, compliance["additional_token_usage"]["total_tokens"])
        self.assertAlmostEqual(0.01, compliance["additional_latency_seconds"])
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_json_schema_and_decision_validity_are_distinct(self) -> None:
        constraints = DecisionConstraints(
            allowed_action_ids=("wait",),
            action_parameter_schemas={"wait": {}},
            max_actions=1,
            current_step=0,
            current_time=0.0,
            max_steps=2,
            time_step=1.0,
            allows_callbacks=False,
            min_callback_steps=1,
            max_callback_steps=1,
            allows_subscriptions=False,
            allowed_event_types=(),
            active_subscription_ids=(),
            allowed_context_requests=(),
            reasoning_enabled=False,
        )
        invalid_json = parse_decision("bad", constraints)
        invalid_schema = parse_decision("[]", constraints)
        invalid_decision = parse_decision(
            {
                "actions": [{"action_id": "unknown"}],
                "callback": None,
                "subscriptions": [],
                "context_requests": [],
            },
            constraints,
        )
        self.assertFalse(invalid_json.json_valid)
        self.assertTrue(invalid_schema.json_valid)
        self.assertFalse(invalid_schema.schema_valid)
        self.assertTrue(invalid_decision.schema_valid)
        self.assertFalse(invalid_decision.decision_valid)


if __name__ == "__main__":
    unittest.main()
