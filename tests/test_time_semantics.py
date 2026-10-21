from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.models import (
    ModelBackend,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleBackend,
    ProviderError,
)
from cps_reason_bench.models.openai_compatible import HTTPJSONResponse
from cps_reason_bench.policies import LLMPolicy
from cps_reason_bench.protocol import (
    CallbackRequest,
    DecisionConstraints,
    parse_decision,
)
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.scheduling import CallbackScheduler
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.subscriptions import SubscriptionManager


DECISIONS = [
    {
        "actions": [{"action_id": "schedule_boost"}],
        "callback": {"after_steps": 3},
        "subscriptions": [
            {
                "operation": "add",
                "id": "level_crossing",
                "event_type": "variable_threshold",
                "filters": {
                    "target": "plant.level",
                    "operator": "ge",
                    "threshold": 2.0,
                },
            }
        ],
        "context_requests": ["event_history"],
    },
    {
        "actions": [],
        "callback": None,
        "subscriptions": [{"operation": "remove", "id": "level_crossing"}],
        "context_requests": [],
    },
    {
        "actions": [{"action_id": "wait"}],
        "callback": None,
        "subscriptions": [],
        "context_requests": [],
    },
]


def _scenario() -> ScenarioConfig:
    return ScenarioConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "time_semantics_probe",
            "description": "In-memory timing probe; not a released benchmark scenario.",
            "task_family": "prospective_reasoning",
            "difficulty": 1,
            "difficulty_level": "timing_probe",
            "seed": 701,
            "max_steps": 4,
            "time_step": 0.5,
            "task": {
                "instructions": "Reach the target without violating the bound.",
                "objectives": ["plant.level >= 8"],
                "safety_rules": ["plant.level <= 20"],
            },
            "graph": {
                "nodes": {
                    "plant": {
                        "variables": {
                            "level": {
                                "initial": 0.0,
                                "drift": 1.0,
                                "hard_min": 0.0,
                                "hard_max": 20.0,
                                "event_value_observable": True,
                            }
                        }
                    }
                },
                "resources": {},
                "edges": [],
            },
            "sensors": [
                {
                    "id": "level_sensor",
                    "target": "plant.level",
                    "observable": True,
                    "precision": 3,
                }
            ],
            "actions": [
                {
                    "id": "schedule_boost",
                    "description": "Schedule a delayed level increase.",
                    "operation": "adjust",
                    "target": "plant.level",
                    "amount": 4.0,
                    "effect_delay_steps": 3,
                },
                {
                    "id": "wait",
                    "description": "Take no physical action.",
                    "operation": "noop",
                },
            ],
            "faults": [
                {
                    "id": "drift_increase",
                    "kind": "drift_change",
                    "target": "plant.level",
                    "value": 2.0,
                    "at_step": 3,
                    "announced": False,
                }
            ],
            "constraints": [
                {
                    "id": "level_bound",
                    "kind": "bounds",
                    "target": "plant.level",
                    "max": 20.0,
                }
            ],
            "goals": [
                {
                    "id": "target_level",
                    "condition": {
                        "target": "plant.level",
                        "operator": "ge",
                        "value": 8.0,
                    },
                    "status_observable": True,
                }
            ],
            "observability": {"full_state": False},
            "interaction": {
                "mode": "scheduled_event",
                "initial_invocation": True,
                "interval_steps": 1,
                "min_callback_steps": 1,
                "max_callback_steps": 4,
                "max_actions_per_interaction": 1,
                "allowed_event_types": ["variable_threshold"],
                "include_event_values": False,
            },
            "context": {
                "mode": "previous",
                "history_limit": 4,
                "max_context_requests": 4,
                "include_action_outcomes": True,
                "include_public_events": True,
            },
            "evaluation": {
                "success_requires_safety": True,
                "allowed_safety_violations": 0,
            },
        }
    )


def _state_signature(simulator: GraphCPSSimulator) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "true_state": simulator.true_state(),
            "advance_calls": getattr(simulator, "advance_calls", 0),
            "sensor_history": simulator.sensor_history,
            "action_history": simulator.action_history,
            "completed_delayed_effects": simulator.completed_delayed_effects,
            "safety_violations": simulator.safety_violations,
        }
    )


class ProbeSimulator(GraphCPSSimulator):
    latest: "ProbeSimulator | None" = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.advance_calls = 0
        super().__init__(*args, **kwargs)
        type(self).latest = self

    def advance(self) -> list[Any]:
        self.advance_calls += 1
        return super().advance()


class ProbeScheduler(CallbackScheduler):
    latest: "ProbeScheduler | None" = None

    def __init__(self) -> None:
        super().__init__()
        self.consume_calls = 0
        type(self).latest = self

    def consume_if_due(self, step: int) -> Any:
        self.consume_calls += 1
        return super().consume_if_due(step)


class ProbeSubscriptionManager(SubscriptionManager):
    latest: "ProbeSubscriptionManager | None" = None

    def __init__(self, maximum: int = 8) -> None:
        super().__init__(maximum)
        self.match_calls = 0
        type(self).latest = self

    def match(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.match_calls += 1
        return super().match(*args, **kwargs)


def _pause_signature() -> dict[str, Any]:
    simulator = ProbeSimulator.latest
    scheduler = ProbeScheduler.latest
    subscriptions = ProbeSubscriptionManager.latest
    if simulator is None or scheduler is None or subscriptions is None:
        raise AssertionError("Timing probe objects were not initialized")
    return {
        "simulator": _state_signature(simulator),
        "pending_callback": copy.deepcopy(
            scheduler.pending.to_dict() if scheduler.pending else None
        ),
        "triggered_callbacks": copy.deepcopy(
            [item.to_dict() for item in scheduler.triggered_history]
        ),
        "scheduler_consume_calls": scheduler.consume_calls,
        "subscriptions": copy.deepcopy(subscriptions.snapshot()),
        "subscription_match_calls": subscriptions.match_calls,
        "useful_subscription_triggers": subscriptions.useful_trigger_count,
    }


class DeterministicTimingBackend(ModelBackend):
    def __init__(
        self,
        responses: list[dict[str, Any] | str],
        *,
        provider: str,
        model: str,
        delays: dict[int, tuple[float, ...]] | None = None,
        guard: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.responses = responses
        self.provider = provider
        self.model = model
        self.delays = dict(delays or {})
        self.guard = guard
        self.calls = 0
        self.pause_checks: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        del request
        index = self.calls
        self.calls += 1
        if index >= len(self.responses):
            raise AssertionError(f"Unexpected model invocation {index}")
        before = self.guard() if self.guard else {}
        delay = 0.0
        for component in self.delays.get(index, ()):
            delay += component
            time.sleep(component)
        after = self.guard() if self.guard else {}
        if self.guard:
            self.pause_checks.append((before, after))
            if before != after:
                raise AssertionError("Physical or scheduler state changed during backend wait")
        response = self.responses[index]
        content = response if isinstance(response, str) else json.dumps(response)
        return ModelResponse(
            content,
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            delay,
            {
                "mock_provider": self.provider,
                "network_delay_seconds": self.delays.get(index, (0.0,))[0],
                "provider_delay_seconds": sum(self.delays.get(index, ())[1:]),
            },
        )


@dataclass(frozen=True)
class Capture:
    records: list[dict[str, Any]]
    metrics: dict[str, Any]
    duration_seconds: float
    summary: dict[str, Any]


def _run_backend(
    backend: ModelBackend,
    *,
    output_retries: int = 0,
    use_probes: bool = False,
) -> Capture:
    ProbeSimulator.latest = None
    ProbeScheduler.latest = None
    ProbeSubscriptionManager.latest = None
    policy = LLMPolicy(
        backend,
        max_output_retries=output_retries,
        prompt_template_version="2.0.0",
    )
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        if use_probes:
            stack.enter_context(
                patch("cps_reason_bench.runner.GraphCPSSimulator", ProbeSimulator)
            )
            stack.enter_context(
                patch("cps_reason_bench.runner.CallbackScheduler", ProbeScheduler)
            )
            stack.enter_context(
                patch(
                    "cps_reason_bench.runner.SubscriptionManager",
                    ProbeSubscriptionManager,
                )
            )
        started = time.perf_counter()
        result = BenchmarkRunner(_scenario(), policy, Path(directory)).run()
        duration = time.perf_counter() - started
        records = [
            json.loads(line)
            for line in result.events_path.read_text(encoding="utf-8").splitlines()
        ]
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        return Capture(records, copy.deepcopy(result.metrics), duration, summary)


def _metric_semantics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    result["general"].pop("token_usage", None)
    result["general"].pop("total_latency_seconds", None)
    result["protocol_compliance"].pop("additional_token_usage", None)
    result["protocol_compliance"].pop("additional_latency_seconds", None)
    return result


def _canonical_trajectory(
    capture: Capture, *, include_metrics: bool = True
) -> dict[str, Any]:
    semantic_records: list[dict[str, Any]] = []
    for record in capture.records:
        record_type = record["record_type"]
        semantic: dict[str, Any] | None = None
        if record_type == "run_started":
            semantic = {
                "true_simulator_state": record["true_simulator_state"],
                "visible_observation": record["visible_observation"],
                "initial_events": record["initial_events"],
            }
        elif record_type == "model_interaction":
            semantic = {
                key: record[key]
                for key in (
                    "interaction_index",
                    "step",
                    "time",
                    "invocation_triggers",
                    "true_state_at_invocation",
                    "true_state_after_actions",
                    "visible_observation",
                    "explicit_context",
                    "parsed_response",
                    "validation_errors",
                    "decision",
                    "action_outcomes",
                    "public_action_outcomes",
                    "callbacks_requested",
                    "scheduled_callback",
                    "subscription_commands",
                    "active_subscriptions",
                )
            }
        elif record_type == "simulation_transition":
            semantic = {
                key: record[key]
                for key in (
                    "from_step",
                    "to_step",
                    "true_state_before",
                    "visible_observation_before",
                    "events",
                    "true_state_after",
                    "visible_observation_after",
                    "next_invocation_triggers",
                )
            }
        elif record_type == "terminal_action_events":
            semantic = {
                key: record[key]
                for key in (
                    "step",
                    "time",
                    "events",
                    "true_simulator_state",
                    "visible_observation",
                )
            }
        elif record_type == "run_completed":
            semantic = {
                key: record[key]
                for key in (
                    "final_true_simulator_state",
                    "final_visible_observation",
                    "active_subscriptions",
                    "pending_callback",
                )
            }
        if semantic is not None:
            semantic_records.append(
                {
                    "sequence": record["sequence"],
                    "record_type": record_type,
                    **semantic,
                }
            )
    result: dict[str, Any] = {"records": semantic_records}
    if include_metrics:
        result["metrics"] = _metric_semantics(capture.metrics)
    return result


def _interaction_records(capture: Capture) -> list[dict[str, Any]]:
    return [
        record
        for record in capture.records
        if record["record_type"] == "model_interaction"
    ]


def _transition_records(capture: Capture) -> list[dict[str, Any]]:
    return [
        record
        for record in capture.records
        if record["record_type"] == "simulation_transition"
    ]


def _cloud_success(decision: dict[str, Any]) -> HTTPJSONResponse:
    return HTTPJSONResponse(
        {
            "id": "mock-generation",
            "model": "mock/model",
            "provider": "mock-provider",
            "choices": [
                {
                    "message": {"content": json.dumps(decision)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        },
        {"x-request-id": "mock-request"},
        200,
    )


def _callback_constraints() -> DecisionConstraints:
    return DecisionConstraints(
        allowed_action_ids=("wait",),
        action_parameter_schemas={"wait": {}},
        max_actions=1,
        current_step=0,
        current_time=0.0,
        max_steps=4,
        time_step=0.5,
        allows_callbacks=True,
        min_callback_steps=1,
        max_callback_steps=8,
        allows_subscriptions=False,
        allowed_event_types=(),
        active_subscription_ids=(),
        allowed_context_requests=(),
        reasoning_enabled=False,
    )


def _openai_backend(sleeper: Callable[[float], None]) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        model="mock/model",
        base_url="https://offline.invalid/v1",
        provider="custom",
        routing={"mode": "direct"},
        structured_output_mode="prompt_only",
        transport_retry={
            "max_attempts": 2,
            "base_backoff_seconds": 0.01,
            "max_backoff_seconds": 1.0,
            "jitter_seconds": 0.0,
        },
        sleeper=sleeper,
    )


class TimeModelTests(unittest.TestCase):
    def test_step_and_simulated_time_are_configurable_and_can_diverge(self) -> None:
        simulator = GraphCPSSimulator(_scenario())
        self.assertEqual((0, 0.0), (simulator.step, simulator.time))
        simulator.advance()
        self.assertEqual((1, 0.5), (simulator.step, simulator.time))
        simulator.advance()
        self.assertEqual((2, 1.0), (simulator.step, simulator.time))

    def test_every_callback_mode_and_cancel_replace_boundaries(self) -> None:
        scheduler = CallbackScheduler()
        by_steps = scheduler.apply(CallbackRequest(after_steps=2), 2, 1.0, 0.5)
        assert by_steps is not None
        self.assertEqual((4, 2.0), (by_steps.due_step, by_steps.due_time))
        self.assertIsNone(scheduler.consume_if_due(3))
        self.assertEqual(by_steps, scheduler.consume_if_due(4))

        by_duration = scheduler.apply(
            CallbackRequest(after_time=1.1), 2, 1.0, 0.5
        )
        assert by_duration is not None
        self.assertEqual((5, 2.5), (by_duration.due_step, by_duration.due_time))
        self.assertLess(by_duration.due_time - 2.1, 0.5)

        absolute = scheduler.apply(CallbackRequest(at_time=2.2), 2, 1.0, 0.5)
        assert absolute is not None
        self.assertEqual((5, 2.5), (absolute.due_step, absolute.due_time))
        self.assertLess(absolute.due_time - 2.2, 0.5)
        self.assertEqual(1, scheduler.replacement_count)

        replacement = scheduler.apply(
            CallbackRequest(after_steps=1), 3, 1.5, 0.5
        )
        assert replacement is not None
        self.assertEqual(4, replacement.due_step)
        self.assertEqual(2, scheduler.replacement_count)
        scheduler.apply(CallbackRequest(cancel=True), 3, 1.5, 0.5)
        self.assertIsNone(scheduler.consume_if_due(4))
        self.assertEqual(1, scheduler.cancel_count)

    def test_callbacks_before_at_and_after_terminal_horizon(self) -> None:
        def parse(after_steps: int) -> Any:
            return parse_decision(
                {
                    "actions": [],
                    "callback": {"after_steps": after_steps},
                    "subscriptions": [],
                    "context_requests": [],
                },
                _callback_constraints(),
            )

        before = parse(3)
        at_horizon = parse(4)
        after = parse(5)

        self.assertTrue(before.valid)
        self.assertFalse(at_horizon.valid)
        self.assertFalse(after.valid)
        self.assertTrue(
            any("terminal episode horizon" in item for item in at_horizon.errors)
        )
        self.assertTrue(
            any("terminal episode horizon" in item for item in after.errors)
        )


class BackendPauseAndEquivalenceTests(unittest.TestCase):
    def test_slow_backend_matches_fast_backend_and_state_is_paused(self) -> None:
        fast = _run_backend(
            DeterministicTimingBackend(
                copy.deepcopy(DECISIONS), provider="mock", model="same-model"
            )
        )
        slow_backend = DeterministicTimingBackend(
            copy.deepcopy(DECISIONS),
            provider="mock",
            model="same-model",
            delays={0: (0.015,), 1: (0.015,), 2: (0.015,)},
            guard=_pause_signature,
        )
        slow = _run_backend(slow_backend, use_probes=True)

        self.assertEqual(_canonical_trajectory(fast), _canonical_trajectory(slow))
        self.assertEqual(3, len(slow_backend.pause_checks))
        self.assertTrue(all(before == after for before, after in slow_backend.pause_checks))
        self.assertGreater(
            slow.metrics["general"]["total_latency_seconds"],
            fast.metrics["general"]["total_latency_seconds"],
        )

    def test_mock_ollama_and_cloud_delays_have_identical_physical_results(self) -> None:
        local = _run_backend(
            DeterministicTimingBackend(
                copy.deepcopy(DECISIONS),
                provider="ollama-mock",
                model="local-equivalent",
            )
        )
        cloud_backend = DeterministicTimingBackend(
            copy.deepcopy(DECISIONS),
            provider="openai-compatible-mock",
            model="cloud-equivalent",
            delays={0: (0.007, 0.009), 1: (0.007, 0.009), 2: (0.007, 0.009)},
            guard=_pause_signature,
        )
        cloud = _run_backend(cloud_backend, use_probes=True)

        self.assertEqual(_canonical_trajectory(local), _canonical_trajectory(cloud))
        self.assertTrue(cloud.metrics["general"]["task_success"])
        self.assertEqual(0, cloud.metrics["general"]["safety_violations"])

    def test_long_inference_delay_freezes_pending_time_driven_mechanisms(self) -> None:
        fast = _run_backend(
            DeterministicTimingBackend(
                copy.deepcopy(DECISIONS), provider="mock", model="long-delay"
            )
        )
        long_delay = max(0.05, fast.duration_seconds * 2.0)
        delayed_backend = DeterministicTimingBackend(
            copy.deepcopy(DECISIONS),
            provider="mock",
            model="long-delay",
            delays={1: (long_delay,)},
            guard=_pause_signature,
        )
        delayed = _run_backend(delayed_backend, use_probes=True)

        self.assertGreater(long_delay, fast.duration_seconds)
        before, after = delayed_backend.pause_checks[1]
        self.assertEqual(before, after)
        self.assertEqual(2, before["simulator"]["true_state"]["step"])
        self.assertEqual(3, before["pending_callback"]["due_step"])
        self.assertEqual(3, before["simulator"]["true_state"]["pending_action_effects"][0]["due_step"])
        self.assertEqual({}, before["simulator"]["true_state"]["active_faults"])
        self.assertEqual(_canonical_trajectory(fast), _canonical_trajectory(delayed))

    def test_output_correction_retry_does_not_evolve_the_simulator(self) -> None:
        baseline = _run_backend(
            DeterministicTimingBackend(
                copy.deepcopy(DECISIONS), provider="mock", model="output-retry"
            )
        )
        retry_backend = DeterministicTimingBackend(
            ["not-json", *copy.deepcopy(DECISIONS)],
            provider="mock",
            model="output-retry",
            delays={1: (0.02,)},
            guard=_pause_signature,
        )
        retried = _run_backend(retry_backend, output_retries=1, use_probes=True)

        self.assertEqual(
            _canonical_trajectory(baseline, include_metrics=False),
            _canonical_trajectory(retried, include_metrics=False),
        )
        self.assertEqual(
            baseline.metrics["general"]["final_system_state"],
            retried.metrics["general"]["final_system_state"],
        )
        self.assertEqual(1, retried.metrics["general"]["output_retries"])
        self.assertTrue(all(before == after for before, after in retry_backend.pause_checks))


class TransportAndTimeoutTests(unittest.TestCase):
    def _retry_capture(
        self, first_error: ProviderError
    ) -> tuple[Capture, list[float]]:
        delays: list[float] = []

        def sleeper(delay: float) -> None:
            before = _pause_signature()
            time.sleep(0.01)
            after = _pause_signature()
            if before != after:
                raise AssertionError("State changed during transport retry backoff")
            delays.append(delay)

        backend = _openai_backend(sleeper)
        responses: list[ProviderError | HTTPJSONResponse] = []
        for decision in DECISIONS:
            responses.extend([first_error.with_attempts([]), _cloud_success(decision)])
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=responses,
        ):
            capture = _run_backend(backend, use_probes=True)
        return capture, delays

    def _no_retry_capture(self) -> Capture:
        backend = _openai_backend(lambda _delay: None)
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[_cloud_success(item) for item in DECISIONS],
        ):
            return _run_backend(backend, use_probes=True)

    def test_rate_limit_retry_after_wait_is_physically_inert(self) -> None:
        baseline = self._no_retry_capture()
        rate_limit = ProviderError(
            "rate limited",
            category="rate_limit",
            retryable=True,
            http_status=429,
            retry_after_seconds=0.25,
            request_state="response_received",
            billing_ambiguity=False,
        )
        retried, delays = self._retry_capture(rate_limit)

        self.assertEqual([0.25, 0.25, 0.25], delays)
        self.assertEqual(_canonical_trajectory(baseline), _canonical_trajectory(retried))

    def test_transient_5xx_backoff_is_physically_inert(self) -> None:
        baseline = self._no_retry_capture()
        transient = ProviderError(
            "temporary server failure",
            category="transient_provider_failure",
            retryable=True,
            http_status=503,
            request_state="response_received",
            billing_ambiguity=False,
        )
        retried, delays = self._retry_capture(transient)

        self.assertEqual([0.01, 0.01, 0.01], delays)
        self.assertEqual(_canonical_trajectory(baseline), _canonical_trajectory(retried))

    def test_timeout_and_connection_failure_are_not_decisions(self) -> None:
        errors = (
            ProviderError(
                "request timed out",
                category="timeout",
                retryable=True,
                request_state="sent_no_response",
                billing_ambiguity=True,
            ),
            ProviderError(
                "connection failed",
                category="connection_error",
                retryable=True,
                request_state="unknown",
                billing_ambiguity=True,
            ),
        )
        for error in errors:
            with self.subTest(category=error.category):
                backend = _openai_backend(lambda _delay: None)

                def fail_after_wait(*_args: Any, **_kwargs: Any) -> Any:
                    before = _pause_signature()
                    time.sleep(0.02)
                    after = _pause_signature()
                    self.assertEqual(before, after)
                    raise error

                with patch(
                    "cps_reason_bench.models.openai_compatible._request_json",
                    side_effect=fail_after_wait,
                ) as request:
                    capture = _run_backend(backend, use_probes=True)

                interaction = _interaction_records(capture)[0]
                transitions = _transition_records(capture)
                self.assertEqual(1, request.call_count)
                self.assertIsNone(interaction["decision"])
                self.assertEqual([], interaction["action_outcomes"])
                self.assertEqual(1, capture.metrics["general"]["backend_errors"])
                self.assertEqual(
                    0,
                    capture.metrics["protocol_compliance"][
                        "final_executable_decisions"
                    ],
                )
                self.assertEqual([], transitions)
                self.assertEqual("infrastructure_failed", capture.summary["completion_status"])
                self.assertIsNone(capture.metrics["general"]["task_success"])
                termination = next(
                    item
                    for item in capture.records
                    if item["record_type"] == "infrastructure_failure_termination"
                )
                self.assertFalse(termination["physical_transition_after_failure"])
                self.assertEqual(0, termination["step"])

    def test_timeout_after_success_stops_at_the_failed_invocation(self) -> None:
        backend = _openai_backend(lambda _delay: None)
        timeout = ProviderError(
            "request timed out",
            category="timeout",
            retryable=True,
            request_state="sent_no_response",
            billing_ambiguity=True,
        )
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[_cloud_success(DECISIONS[0]), timeout],
        ):
            capture = _run_backend(backend, use_probes=True)

        self.assertEqual([0, 2], [item["step"] for item in _interaction_records(capture)])
        self.assertEqual(
            [(0, 1), (1, 2)],
            [
                (item["from_step"], item["to_step"])
                for item in _transition_records(capture)
            ],
        )
        self.assertEqual("infrastructure_failed", capture.summary["completion_status"])
        self.assertEqual("timeout", capture.summary["infrastructure_failure"]["category"])
        self.assertEqual(2, capture.metrics["general"]["simulation_steps"])

    def test_exhausted_429_and_5xx_transport_retries_stop_without_transition(self) -> None:
        errors = (
            ProviderError(
                "rate limited",
                category="rate_limit",
                retryable=True,
                http_status=429,
                request_state="response_received",
                billing_ambiguity=False,
            ),
            ProviderError(
                "server failure",
                category="transient_provider_failure",
                retryable=True,
                http_status=503,
                request_state="response_received",
                billing_ambiguity=False,
            ),
        )
        for error in errors:
            with self.subTest(category=error.category):
                backend = _openai_backend(lambda _delay: None)
                with patch(
                    "cps_reason_bench.models.openai_compatible._request_json",
                    side_effect=[error, error],
                ) as request:
                    capture = _run_backend(backend, use_probes=True)

                self.assertEqual(2, request.call_count)
                self.assertEqual([], _transition_records(capture))
                failure = capture.summary["infrastructure_failure"]
                self.assertEqual(error.category, failure["category"])
                self.assertEqual(2, len(failure["transport_attempts"]))
                self.assertTrue(failure["retryable"])

    def test_permanent_provider_failure_stops_without_launcher_trajectory(self) -> None:
        backend = _openai_backend(lambda _delay: None)
        failure = ProviderError(
            "authentication failed",
            category="authentication_failure",
            retryable=False,
            http_status=401,
            request_state="response_received",
            billing_ambiguity=False,
        )
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=failure,
        ):
            capture = _run_backend(backend, use_probes=True)

        self.assertEqual([], _transition_records(capture))
        self.assertEqual(
            "authentication_failure",
            capture.summary["infrastructure_failure"]["category"],
        )
        self.assertFalse(capture.summary["infrastructure_failure"]["retryable"])


class AutonomousTimingTests(unittest.TestCase):
    def test_terminal_step_callback_is_rejected_and_never_counted(self) -> None:
        backend = DeterministicTimingBackend(
            [
                {
                    "actions": [],
                    "callback": {"after_steps": 4},
                    "subscriptions": [],
                    "context_requests": [],
                }
            ],
            provider="mock",
            model="terminal-callback",
        )
        capture = _run_backend(backend)

        self.assertEqual([0], [item["step"] for item in _interaction_records(capture)])
        self.assertEqual([], _transition_records(capture)[-1]["next_invocation_triggers"])
        self.assertEqual(0, capture.metrics["general"]["callback_requests"])
        self.assertEqual(0, capture.metrics["general"]["callback_triggers"])
        self.assertEqual(1, capture.metrics["general"]["llm_or_policy_invocations"])
        self.assertTrue(
            any(
                "terminal episode horizon" in error
                for error in _interaction_records(capture)[0]["validation_errors"]
            )
        )

    def test_simultaneous_event_and_preexisting_callback_at_horizon_are_nonactionable(self) -> None:
        class HorizonScheduler(CallbackScheduler):
            def __init__(self) -> None:
                super().__init__()
                self.apply(CallbackRequest(after_steps=4), 0, 0.0, 0.5)

        backend = DeterministicTimingBackend(
            [
                {
                    "actions": [],
                    "callback": None,
                    "subscriptions": [
                        {
                            "operation": "add",
                            "id": "horizon_crossing",
                            "event_type": "variable_threshold",
                            "filters": {
                                "target": "plant.level",
                                "operator": "ge",
                                "threshold": 6.0,
                            },
                        }
                    ],
                    "context_requests": [],
                }
            ],
            provider="mock",
            model="terminal-event-callback",
        )
        policy = LLMPolicy(backend, prompt_template_version="2.0.0")
        with tempfile.TemporaryDirectory() as directory, patch(
            "cps_reason_bench.runner.CallbackScheduler", HorizonScheduler
        ):
            result = BenchmarkRunner(_scenario(), policy, directory).run()
            records = [
                json.loads(line)
                for line in result.events_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(1, result.metrics["general"]["callback_requests"])
        self.assertEqual(0, result.metrics["general"]["callback_triggers"])
        self.assertEqual(0, result.metrics["general"]["useful_subscription_triggers"])
        self.assertEqual(1, result.metrics["general"]["llm_or_policy_invocations"])
        terminal = [
            item for item in records if item["record_type"] == "simulation_transition"
        ][-1]
        self.assertEqual(4, terminal["to_step"])
        self.assertEqual([], terminal["next_invocation_triggers"])

    def test_runner_explicitly_advances_callbacks_events_effects_and_faults(self) -> None:
        capture = _run_backend(
            DeterministicTimingBackend(
                copy.deepcopy(DECISIONS), provider="mock", model="timeline"
            )
        )
        interactions = _interaction_records(capture)
        transitions = _transition_records(capture)

        self.assertEqual([0, 2, 3], [record["step"] for record in interactions])
        self.assertEqual(
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [(record["from_step"], record["to_step"]) for record in transitions],
        )
        self.assertEqual(
            "variable_threshold",
            interactions[1]["invocation_triggers"][0]["event_type"],
        )
        self.assertEqual(
            (2, 1.0),
            (
                interactions[1]["invocation_triggers"][0]["step"],
                interactions[1]["invocation_triggers"][0]["time"],
            ),
        )
        self.assertEqual(
            "scheduled_callback",
            interactions[2]["invocation_triggers"][0]["event_type"],
        )
        self.assertEqual(
            (3, 1.5),
            (interactions[2]["step"], interactions[2]["time"]),
        )

        step_three_events = transitions[2]["events"]
        fault = next(
            item for item in step_three_events if item["event_type"] == "fault_activated"
        )
        delayed = next(
            item
            for item in step_three_events
            if item["event_type"] == "delayed_action_effect"
        )
        self.assertEqual((3, 1.5), (fault["step"], fault["time"]))
        self.assertEqual(3, delayed["data"]["due_step"])
        self.assertEqual((3, 1.5), (delayed["data"]["completed_step"], delayed["data"]["completed_time"]))
        self.assertTrue(capture.metrics["general"]["task_success"])
        self.assertEqual(0, capture.metrics["general"]["safety_violations"])


if __name__ == "__main__":
    unittest.main()
