from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from cps_reason_bench.config import ScenarioConfig, load_scenario
from cps_reason_bench.generators import generate_scenario
from cps_reason_bench.models import ModelBackend, ModelRequest, ModelResponse
from cps_reason_bench.policies import LLMPolicy
from cps_reason_bench.protocol import SubscriptionCommand
from cps_reason_bench.runner import BenchmarkRunner, RunResult
from cps_reason_bench.subscriptions import SubscriptionManager
from cps_reason_bench.types import SimulatorEvent
from cps_reason_bench.versions import content_hash


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCENARIOS = ROOT / "examples" / "scenarios" / "protocol_sanity"


def _decision(
    actions: list[dict[str, Any]] | None = None,
    *,
    callback: dict[str, Any] | None = None,
    subscriptions: list[dict[str, Any]] | None = None,
    context_requests: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "actions": actions or [],
        "callback": callback,
        "subscriptions": subscriptions or [],
        "context_requests": context_requests or [],
    }


class SequenceBackend(ModelBackend):
    """Return fixed operational decisions while retaining every model request."""

    provider = "audit_mock"
    model = "deterministic-sequence"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self.responses):
            raise AssertionError(f"Unexpected mock model invocation {index}")
        return ModelResponse(
            json.dumps(self.responses[index]),
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            0.0,
            {"mock_response_index": index},
        )


def _run(
    scenario: ScenarioConfig,
    responses: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[RunResult, SequenceBackend, list[dict[str, Any]]]:
    backend = SequenceBackend(responses)
    policy = LLMPolicy(
        backend,
        prompt_template_version="2.0.0",
        max_output_retries=0,
    )
    result = BenchmarkRunner(scenario, policy, output_dir).run()
    records = [
        json.loads(line)
        for line in result.events_path.read_text(encoding="utf-8").splitlines()
    ]
    return result, backend, records


def _records(
    records: list[dict[str, Any]], record_type: str
) -> list[dict[str, Any]]:
    return [record for record in records if record["record_type"] == record_type]


class EndToEndVerificationTests(unittest.TestCase):
    def test_mock_llm_direct_action_changes_state_and_recomputes_goal(self) -> None:
        scenario = load_scenario(PROTOCOL_SCENARIOS / "direct_action.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result, backend, records = _run(
                scenario,
                [_decision([{"action_id": "cool_chamber"}])],
                Path(directory),
            )

        start = _records(records, "run_started")[0]
        interaction = _records(records, "model_interaction")[0]
        terminal = _records(records, "terminal_action_events")[0]
        self.assertEqual(9.0, start["true_simulator_state"]["nodes"]["chamber"]["temperature"])
        self.assertFalse(start["true_simulator_state"]["goals_met"])
        self.assertEqual(5.0, interaction["true_state_after_actions"]["nodes"]["chamber"]["temperature"])
        self.assertTrue(interaction["action_outcomes"][0]["accepted"])
        self.assertTrue(terminal["true_simulator_state"]["goals_met"])
        self.assertTrue(result.metrics["general"]["task_success"])
        self.assertEqual(1, len(backend.requests))

    def test_mock_llm_numeric_parameter_reaches_simulator_operation(self) -> None:
        scenario = load_scenario(PROTOCOL_SCENARIOS / "simple_parameter.yaml")
        response = _decision(
            [{"action_id": "adjust_flow", "parameters": {"amount": 2.0}}]
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(scenario, [response], Path(directory))

        interaction = _records(records, "model_interaction")[0]
        self.assertEqual(2.0, interaction["decision"]["actions"][0]["parameters"]["amount"])
        self.assertEqual(2.0, interaction["action_outcomes"][0]["details"]["applied_amount"])
        self.assertEqual(0.0, interaction["true_state_at_invocation"]["nodes"]["process"]["flow"])
        self.assertEqual(2.0, interaction["true_state_after_actions"]["nodes"]["process"]["flow"])
        self.assertTrue(result.metrics["general"]["task_success"])

        prompt_payload = json.loads(interaction["prompt"][1]["content"])
        parameter_schema = prompt_payload["interaction_protocol"][
            "action_parameter_schemas"
        ]["adjust_flow"]["amount"]
        self.assertEqual(
            {"type": "number", "required": True, "minimum": 2.0, "maximum": 2.0},
            parameter_schema,
        )
        for hidden_answer_key in ("baseline", "expected_action_sequence", "reference_action_count"):
            self.assertNotIn(hidden_answer_key, interaction["prompt"][1]["content"])

    def test_mock_llm_noop_advances_time_without_intervention(self) -> None:
        scenario = load_scenario(PROTOCOL_SCENARIOS / "explicit_noop.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(
                scenario,
                [_decision([{"action_id": "wait"}])],
                Path(directory),
            )

        interaction = _records(records, "model_interaction")[0]
        transition = _records(records, "simulation_transition")[0]
        self.assertEqual("noop", interaction["action_outcomes"][0]["details"]["operation"])
        self.assertTrue(interaction["action_outcomes"][0]["accepted"])
        self.assertEqual(0.0, interaction["time"])
        self.assertEqual((0, 1), (transition["from_step"], transition["to_step"]))
        self.assertEqual(1.0, transition["true_state_after"]["time"])
        self.assertTrue(transition["true_state_before"]["nodes"]["process"]["stable"])
        self.assertTrue(transition["true_state_after"]["nodes"]["process"]["stable"])
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_noop_allows_autonomous_transition_to_recompute_goal(self) -> None:
        source = load_scenario(PROTOCOL_SCENARIOS / "direct_action.yaml")
        raw = copy.deepcopy(source.to_mapping())
        raw["name"] = "audit_autonomous_noop"
        raw["graph"]["nodes"]["chamber"]["variables"]["temperature"]["drift"] = -4.0
        raw["evaluation"]["required_actions"] = ["wait"]
        raw["evaluation"]["expected_action_sequence"] = ["wait"]
        raw["baseline"]["script"][0]["decision"]["actions"] = [
            {"action_id": "wait"}
        ]
        scenario = ScenarioConfig.from_mapping(raw)
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(
                scenario,
                [_decision([{"action_id": "wait"}])],
                Path(directory),
            )

        transition = _records(records, "simulation_transition")[0]
        self.assertEqual(9.0, transition["true_state_before"]["nodes"]["chamber"]["temperature"])
        self.assertEqual(5.0, transition["true_state_after"]["nodes"]["chamber"]["temperature"])
        self.assertFalse(transition["true_state_before"]["goals_met"])
        self.assertTrue(transition["true_state_after"]["goals_met"])
        self.assertEqual(
            5.0,
            transition["visible_observation_after"]["sensors"]["temperature_sensor"]["value"],
        )
        self.assertTrue(
            any(event["event_type"] == "task_condition" for event in transition["events"])
        )
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_rejected_action_preserves_state_and_accepted_actions_persist(self) -> None:
        scenario = generate_scenario("planning", "entry_2", 4202)
        responses = [
            _decision([{"action_id": "activate_output"}]),
            _decision([{"action_id": "enable_power"}]),
            _decision([{"action_id": "activate_output"}]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(scenario, responses, Path(directory))

        interactions = _records(records, "model_interaction")
        rejected, enabled, activated = interactions
        self.assertFalse(rejected["action_outcomes"][0]["accepted"])
        self.assertEqual(
            rejected["true_state_at_invocation"]["nodes"],
            rejected["true_state_after_actions"]["nodes"],
        )
        self.assertEqual(
            rejected["true_state_at_invocation"]["resources"],
            rejected["true_state_after_actions"]["resources"],
        )
        self.assertIn(
            "unmet_precondition", rejected["public_action_outcomes"][0]["details"]
        )

        self.assertTrue(enabled["action_outcomes"][0]["accepted"])
        self.assertTrue(enabled["true_state_after_actions"]["nodes"]["plant"]["power_ready"])
        self.assertTrue(activated["true_state_at_invocation"]["nodes"]["plant"]["power_ready"])
        self.assertTrue(
            activated["visible_observation"]["sensors"]["power_sensor"]["value"]
        )
        self.assertTrue(activated["action_outcomes"][0]["accepted"])
        self.assertTrue(activated["true_state_after_actions"]["nodes"]["plant"]["output_enabled"])
        self.assertTrue(result.metrics["general"]["task_success"])
        self.assertEqual(1, result.metrics["general"]["invalid_actions"])

    def test_mock_llm_callback_becomes_due_and_invokes_policy(self) -> None:
        scenario = generate_scenario("prospective_reasoning", "entry_1", 4101)
        responses = [
            _decision(callback={"after_steps": 1}),
            _decision([{"action_id": "acknowledge_cycle"}]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, backend, records = _run(scenario, responses, Path(directory))

        interactions = _records(records, "model_interaction")
        transitions = _records(records, "simulation_transition")
        self.assertEqual([0, 1], [record["step"] for record in interactions])
        self.assertEqual(1, interactions[0]["scheduled_callback"]["due_step"])
        trigger = transitions[0]["next_invocation_triggers"][0]
        self.assertEqual("scheduled_callback", trigger["event_type"])
        self.assertEqual("scheduled_callback", interactions[1]["invocation_triggers"][0]["event_type"])
        self.assertEqual(2, len(backend.requests))
        self.assertEqual(1, result.metrics["general"]["callback_requests"])
        self.assertEqual(1, result.metrics["general"]["callback_triggers"])
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_mock_llm_subscription_crossing_invokes_policy(self) -> None:
        scenario = generate_scenario("prospective_reasoning", "entry_2", 4102)
        watch = {
            "operation": "add",
            "id": "watch_progress",
            "event_type": "variable_threshold",
            "filters": {
                "target": "process.progress",
                "operator": "ge",
                "threshold": 1.0,
            },
        }
        responses = [
            _decision(subscriptions=[watch], context_requests=["active_subscriptions"]),
            _decision(
                [{"action_id": "acknowledge_cycle"}],
                subscriptions=[{"operation": "remove", "id": "watch_progress"}],
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, backend, records = _run(scenario, responses, Path(directory))

        interactions = _records(records, "model_interaction")
        transition = _records(records, "simulation_transition")[0]
        self.assertEqual("watch_progress", interactions[0]["active_subscriptions"][0]["id"])
        self.assertEqual("variable_threshold", transition["next_invocation_triggers"][0]["event_type"])
        self.assertEqual("watch_progress", interactions[1]["invocation_triggers"][0]["subscription_id"])
        self.assertEqual([], interactions[1]["active_subscriptions"])
        self.assertEqual(2, len(backend.requests))
        self.assertEqual(1, result.metrics["general"]["useful_subscription_triggers"])
        self.assertEqual(0, result.metrics["general"]["active_subscriptions_at_end"])
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_subscription_predicate_ignores_nonmatching_events(self) -> None:
        manager = SubscriptionManager()
        manager.apply(
            [
                SubscriptionCommand(
                    "add",
                    "watch_progress",
                    "variable_threshold",
                    {"target": "process.progress", "operator": "ge", "threshold": 1.0},
                )
            ],
            step=0,
        )
        below = SimulatorEvent(
            "variable_changed",
            1,
            1.0,
            {"target": "process.progress", "old": 0.0, "new": 0.5},
            {"target": "process.progress"},
        )
        crossing = SimulatorEvent(
            "variable_changed",
            2,
            2.0,
            {"target": "process.progress", "old": 0.5, "new": 1.0},
            {"target": "process.progress"},
        )
        self.assertEqual([], manager.match([below]))
        self.assertEqual(1, len(manager.match([crossing])))

    def test_delayed_action_executes_at_declared_step_with_public_history(self) -> None:
        scenario = generate_scenario("temporal_tracking", "entry_1", 4601)
        responses = [
            _decision([{"action_id": "start_channel_a"}]),
            _decision([{"action_id": "route_channel_a"}]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, backend, records = _run(scenario, responses, Path(directory))

        interactions = _records(records, "model_interaction")
        transition = _records(records, "simulation_transition")[0]
        first, second = interactions
        pending = first["true_state_after_actions"]["pending_action_effects"][0]
        self.assertEqual(1, pending["due_step"])
        self.assertFalse(first["true_state_after_actions"]["nodes"]["actuator_bank"]["response_ready"])
        self.assertTrue(transition["true_state_after"]["nodes"]["actuator_bank"]["response_ready"])
        self.assertTrue(any(event["event_type"] == "delayed_action_effect" for event in transition["events"]))
        self.assertEqual(0, second["explicit_context"]["previous_interaction"]["step"])
        self.assertEqual(second["explicit_context"], json.loads(backend.requests[1].messages[1]["content"])["context"])
        prompt_context = backend.requests[1].messages[1]["content"]
        for hidden_state_key in (
            "true_state_at_invocation",
            "pending_action_effects",
            "active_faults",
            "action_runtime",
            "sensor_runtime",
        ):
            self.assertNotIn(hidden_state_key, prompt_context)
        self.assertEqual(1.0, result.metrics["task_specific"]["delayed_effect_attribution_accuracy"])
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_mock_llm_fault_changes_actuator_and_requires_backup(self) -> None:
        scenario = generate_scenario("adaptation", "entry_1", 4501)
        responses = [
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "backup_boost"}]),
            _decision(),
            _decision(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(scenario, responses, Path(directory))

        interactions = _records(records, "model_interaction")
        first_transition = _records(records, "simulation_transition")[0]
        self.assertTrue(interactions[0]["action_outcomes"][0]["accepted"])
        self.assertEqual(2.0, interactions[0]["true_state_after_actions"]["nodes"]["process"]["output"])
        self.assertFalse(first_transition["true_state_after"]["action_runtime"]["primary_boost"]["enabled"])
        self.assertFalse(interactions[1]["action_outcomes"][0]["accepted"])
        self.assertEqual(
            interactions[1]["true_state_at_invocation"]["nodes"],
            interactions[1]["true_state_after_actions"]["nodes"],
        )
        self.assertTrue(interactions[2]["action_outcomes"][0]["accepted"])
        self.assertEqual(5.0, interactions[2]["true_state_after_actions"]["nodes"]["process"]["output"])
        self.assertTrue(result.metrics["task_specific"]["adaptation_recovery_demonstrated"])
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_unannounced_fault_schedule_is_absent_before_activation(self) -> None:
        scenario = generate_scenario("adaptation", "entry_1", 4501)
        responses = [
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "backup_boost"}]),
            _decision(),
            _decision(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, backend, _ = _run(scenario, responses, Path(directory))

        initial_payload = json.loads(backend.requests[0].messages[1]["content"])
        initial_context = json.dumps(initial_payload["context"], sort_keys=True)
        self.assertNotIn("primary_loss", initial_context)
        self.assertNotIn("actuator_disabled", initial_context)
        self.assertNotIn('"at_step"', initial_context)

    def test_announced_fault_activation_remains_public(self) -> None:
        source = generate_scenario("adaptation", "entry_1", 4501)
        raw = copy.deepcopy(source.to_mapping())
        raw["faults"][0]["announced"] = True
        scenario = ScenarioConfig.from_mapping(raw)
        responses = [
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "backup_boost"}]),
            _decision(),
            _decision(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, _, records = _run(scenario, responses, Path(directory))

        interaction_at_fault_step = next(
            record
            for record in _records(records, "model_interaction")
            if record["step"] == 1
        )
        event = next(
            event
            for event in interaction_at_fault_step["explicit_context"][
                "public_events_since_last_interaction"
            ]
            if event["event_type"] == "fault_activated"
        )
        self.assertEqual(
            {
                "fault_id": "primary_loss",
                "kind": "actuator_disabled",
                "target": "primary_boost",
            },
            event["details"],
        )

    def test_protocol_valid_decision_does_not_imply_task_success(self) -> None:
        scenario = load_scenario(PROTOCOL_SCENARIOS / "direct_action.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = _run(
                scenario,
                [_decision([{"action_id": "wait"}])],
                Path(directory),
            )

        self.assertEqual(1, result.metrics["protocol_compliance"]["first_attempt_decision_valid"])
        self.assertEqual(1, result.metrics["general"]["accepted_actions"])
        self.assertFalse(result.metrics["general"]["goals_met"])
        self.assertFalse(result.metrics["general"]["behavior_requirements_met"])
        self.assertFalse(result.metrics["general"]["task_success"])

    def test_jsonl_hashes_and_summary_metrics_are_consistent(self) -> None:
        scenario = load_scenario(PROTOCOL_SCENARIOS / "direct_action.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(
                scenario,
                [_decision([{"action_id": "cool_chamber"}])],
                Path(directory),
            )
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

        self.assertEqual(list(range(len(records))), [record["sequence"] for record in records])
        interaction = _records(records, "model_interaction")[0]
        self.assertEqual(
            content_hash(tuple(interaction["prompt"])),
            interaction["rendered_prompt_hash"],
        )
        prompt_payload = json.loads(interaction["prompt"][1]["content"])
        self.assertEqual(
            content_hash(prompt_payload["response_schema"]),
            interaction["response_schema_hash"],
        )
        completed = _records(records, "run_completed")[0]
        self.assertEqual(result.metrics, completed["metrics"])
        self.assertEqual(result.metrics, summary["metrics"])

    def test_terminal_observation_refreshes_immediate_sensor(self) -> None:
        scenario = load_scenario(PROTOCOL_SCENARIOS / "direct_action.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result, _, records = _run(
                scenario,
                [_decision([{"action_id": "cool_chamber"}])],
                Path(directory),
            )

        terminal = _records(records, "terminal_action_events")[0]
        completed = _records(records, "run_completed")[0]
        self.assertEqual(
            5.0,
            terminal["visible_observation"]["sensors"]["temperature_sensor"]["value"],
        )
        self.assertEqual(
            5.0,
            completed["final_visible_observation"]["sensors"]["temperature_sensor"]["value"],
        )
        self.assertEqual(
            5.0,
            completed["final_true_simulator_state"]["nodes"]["chamber"]["temperature"],
        )
        self.assertTrue(completed["final_true_simulator_state"]["goals_met"])
        self.assertEqual(
            [{"id": "normal_temperature", "met": True}],
            completed["final_visible_observation"]["goal_status"],
        )
        self.assertTrue(result.metrics["general"]["task_success"])

    def test_immediate_refresh_preserves_sensor_semantics(self) -> None:
        source = load_scenario(PROTOCOL_SCENARIOS / "direct_action.yaml")

        delayed_raw = copy.deepcopy(source.to_mapping())
        delayed_raw.pop("difficulty_profile", None)
        delayed_raw["name"] = "audit_delayed_terminal_sensor"
        delayed_raw["sensors"][0]["delay_steps"] = 1
        delayed = ScenarioConfig.from_mapping(delayed_raw)

        stuck_raw = copy.deepcopy(source.to_mapping())
        stuck_raw.pop("difficulty_profile", None)
        stuck_raw["name"] = "audit_stuck_terminal_sensor"
        stuck_raw["faults"] = [
            {
                "id": "stuck_temperature",
                "at_step": 0,
                "kind": "sensor_stuck",
                "target": "temperature_sensor",
                "value": 9.0,
                "announced": False,
            }
        ]
        stuck = ScenarioConfig.from_mapping(stuck_raw)

        unavailable_raw = copy.deepcopy(source.to_mapping())
        unavailable_raw.pop("difficulty_profile", None)
        unavailable_raw["name"] = "audit_unavailable_terminal_sensor"
        unavailable_raw["faults"] = [
            {
                "id": "unavailable_temperature",
                "at_step": 0,
                "kind": "sensor_unavailable",
                "target": "temperature_sensor",
                "announced": False,
            }
        ]
        unavailable = ScenarioConfig.from_mapping(unavailable_raw)

        biased_raw = copy.deepcopy(source.to_mapping())
        biased_raw.pop("difficulty_profile", None)
        biased_raw["name"] = "audit_biased_terminal_sensor"
        biased_raw["sensors"][0]["bias"] = 1.0
        biased = ScenarioConfig.from_mapping(biased_raw)

        noisy_raw = copy.deepcopy(source.to_mapping())
        noisy_raw.pop("difficulty_profile", None)
        noisy_raw["name"] = "audit_noisy_terminal_sensor"
        noisy_raw["sensors"][0]["noise_std"] = 0.1
        noisy = ScenarioConfig.from_mapping(noisy_raw)

        with tempfile.TemporaryDirectory() as directory:
            outputs: dict[str, dict[str, Any]] = {}
            for name, scenario in (
                ("delayed", delayed),
                ("stuck", stuck),
                ("unavailable", unavailable),
                ("biased", biased),
                ("noisy_first", noisy),
                ("noisy_second", noisy),
            ):
                _, _, records = _run(
                    scenario,
                    [_decision([{"action_id": "cool_chamber"}])],
                    Path(directory) / name,
                )
                outputs[name] = _records(records, "run_completed")[0][
                    "final_visible_observation"
                ]["sensors"]["temperature_sensor"]

        self.assertEqual(
            {"target": "chamber.temperature", "status": "delayed", "age_steps": 1, "value": 9.0},
            outputs["delayed"],
        )
        self.assertEqual("stale", outputs["stuck"]["status"])
        self.assertEqual(9.0, outputs["stuck"]["value"])
        self.assertEqual("unavailable", outputs["unavailable"]["status"])
        self.assertIsNone(outputs["unavailable"]["value"])
        self.assertEqual("ok", outputs["biased"]["status"])
        self.assertEqual(6.0, outputs["biased"]["value"])
        self.assertEqual(outputs["noisy_first"], outputs["noisy_second"])
        self.assertLess(abs(outputs["noisy_first"]["value"] - 5.0), 0.5)

    def test_unannounced_fault_event_is_internal_but_consequence_is_public(self) -> None:
        scenario = generate_scenario("adaptation", "entry_1", 4501)
        responses = [
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "primary_boost"}]),
            _decision([{"action_id": "backup_boost"}]),
            _decision(),
            _decision(),
        ]
        with tempfile.TemporaryDirectory() as directory:
            _, _, records = _run(scenario, responses, Path(directory))

        transition_at_fault_step = next(
            record
            for record in _records(records, "simulation_transition")
            if record["to_step"] == 1
        )
        internal_event = next(
            event
            for event in transition_at_fault_step["events"]
            if event["event_type"] == "fault_activated"
        )
        self.assertFalse(internal_event["publicly_observable"])
        self.assertEqual("primary_loss", internal_event["data"]["fault_id"])
        self.assertIn(
            "primary_loss", transition_at_fault_step["true_state_after"]["active_faults"]
        )

        interaction_at_fault_step = next(
            record
            for record in _records(records, "model_interaction")
            if record["step"] == 1
        )
        public_events = interaction_at_fault_step["explicit_context"][
            "public_events_since_last_interaction"
        ]
        self.assertNotIn(
            "fault_activated", [event["event_type"] for event in public_events]
        )
        self.assertFalse(interaction_at_fault_step["action_outcomes"][0]["accepted"])
        self.assertEqual(
            "actuator is unavailable",
            interaction_at_fault_step["public_action_outcomes"][0]["message"],
        )

        next_interaction = next(
            record
            for record in _records(records, "model_interaction")
            if record["step"] == 2
        )
        self.assertIn(
            "actuator_failed",
            [
                event["event_type"]
                for event in next_interaction["explicit_context"][
                    "public_events_since_last_interaction"
                ]
            ],
        )

    def test_hidden_events_do_not_trigger_subscriptions(self) -> None:
        manager = SubscriptionManager()
        manager.apply(
            [SubscriptionCommand("add", "watch_fault", "fault_activated", {})],
            step=0,
        )
        hidden = SimulatorEvent(
            "fault_activated",
            1,
            1.0,
            {"fault_id": "hidden_fault"},
            {},
            publicly_observable=False,
        )
        announced = SimulatorEvent(
            "fault_activated",
            2,
            2.0,
            {"fault_id": "announced_fault"},
            {"fault_id": "announced_fault"},
        )
        self.assertEqual([], manager.match([hidden]))
        self.assertEqual(1, len(manager.match([announced])))


if __name__ == "__main__":
    unittest.main()
