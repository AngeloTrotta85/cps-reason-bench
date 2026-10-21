"""Benchmark harness coordinating policies and simulator episodes."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.context import ContextManager
from cps_reason_bench.metrics import RunStatistics, compute_metrics
from cps_reason_bench.policies import Policy, PolicyRequest
from cps_reason_bench.protocol import DecisionConstraints, decision_response_schema
from cps_reason_bench.public_events import (
    ACTION_OUTCOME_EVENT_TYPES,
    public_event_record,
)
from cps_reason_bench.run_logging import RunLogger, make_run_id
from cps_reason_bench.scheduling import CallbackScheduler
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.subscriptions import SubscriptionManager
from cps_reason_bench.types import EventTrigger, SimulatorEvent
from cps_reason_bench.versions import (
    PACKAGE_VERSION,
    PROMPT_TEMPLATE_ID,
    PROMPT_TEMPLATE_VERSION,
    PROTOCOL_SCHEMA_ID,
    PROTOCOL_SCHEMA_VERSION,
    content_hash,
    prompt_template_content_hash,
)


DEFAULT_EVENT_TYPES = (
    "variable_threshold",
    "resource_threshold",
    "sensor_unavailable",
    "actuator_failed",
    "fault_detected",
    "fault_repaired",
    "edge_changed",
    "safety_violation",
    "task_condition",
    "action_failed",
)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    metrics: dict[str, Any]
    events_path: Path
    summary_path: Path
    run_metadata: dict[str, Any]
    completion_status: str = "completed"


class BenchmarkRunner:
    def __init__(
        self,
        scenario: ScenarioConfig,
        policy: Policy,
        output_dir: str | Path = "runs",
        seed: int | None = None,
        reasoning_enabled: bool = False,
        run_metadata: Mapping[str, Any] | None = None,
        model_call_delay_seconds: float = 0.0,
    ) -> None:
        self.scenario = scenario
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.seed = scenario.seed if seed is None else seed
        self.reasoning_enabled = reasoning_enabled
        self.requested_run_metadata = copy.deepcopy(dict(run_metadata or {}))
        self.model_call_delay_seconds = max(0.0, float(model_call_delay_seconds))

    def run(self) -> RunResult:
        simulator = GraphCPSSimulator(self.scenario, self.seed)
        scheduler = CallbackScheduler()
        subscriptions = SubscriptionManager()
        context_manager = ContextManager(self.scenario.context)
        stats = RunStatistics()
        policy_metadata = self.policy.metadata()
        benchmark_metadata = self._benchmark_metadata(policy_metadata)
        run_configuration = {
            "scenario": self.scenario.to_mapping(),
            "seed": self.seed,
            "policy": policy_metadata,
            "reasoning_enabled": self.reasoning_enabled,
            "run_metadata": benchmark_metadata,
        }
        run_id = make_run_id(
            self.scenario.name,
            str(policy_metadata.get("model", policy_metadata.get("policy", "policy"))),
            self.seed,
            run_configuration,
        )
        logger = RunLogger(self.output_dir, run_id)
        public_action_history: list[dict[str, Any]] = []
        public_event_history: list[dict[str, Any]] = []

        try:
            logger.write(
                "run_started",
                {
                    "run_id": run_id,
                    "scenario_name": self.scenario.name,
                    "scenario_configuration": self.scenario.to_mapping(),
                    "seed": self.seed,
                    "policy": policy_metadata,
                    "run_metadata": benchmark_metadata,
                    "benchmark_versions": benchmark_metadata["benchmark_versions"],
                    "difficulty_level": self.scenario.difficulty_level,
                    "difficulty_profile": self.scenario.difficulty_profile,
                    "interaction_configuration": asdict(self.scenario.interaction),
                    "context_configuration": asdict(self.scenario.context),
                    "true_simulator_state": simulator.true_state(),
                    "visible_observation": simulator.observation(),
                    "initial_events": [
                        event.to_log_dict() for event in simulator.initial_events
                    ],
                },
            )
            self._append_public_events(public_event_history, simulator.initial_events)

            triggers: list[EventTrigger] = []
            if self.scenario.interaction.initial_invocation:
                triggers.append(
                    EventTrigger(
                        "benchmark",
                        "initial_invocation",
                        simulator.step,
                        simulator.time,
                        {},
                    )
                )

            while True:
                if simulator.done:
                    break
                action_events: list[SimulatorEvent] = []
                visible_action_events: list[SimulatorEvent] = []
                if triggers:
                    action_events = self._interact(
                        simulator,
                        scheduler,
                        subscriptions,
                        context_manager,
                        stats,
                        triggers,
                        public_action_history,
                        public_event_history,
                        logger,
                    )
                    visible_action_events = self._model_visible_events(
                        action_events, action_batch=True
                    )
                    self._append_public_events(
                        public_event_history, visible_action_events
                    )

                if stats.infrastructure_failure is not None:
                    logger.write(
                        "infrastructure_failure_termination",
                        {
                            "step": simulator.step,
                            "time": simulator.time,
                            "classification": copy.deepcopy(
                                stats.infrastructure_failure
                            ),
                            "true_simulator_state": simulator.true_state(),
                            "visible_observation": simulator.observation(),
                            "physical_transition_after_failure": False,
                        },
                    )
                    break

                if simulator.done:
                    if action_events:
                        logger.write(
                            "terminal_action_events",
                            {
                                "step": simulator.step,
                                "time": simulator.time,
                                "events": [
                                    event.to_log_dict() for event in action_events
                                ],
                                "true_simulator_state": simulator.true_state(),
                                "visible_observation": simulator.observation(),
                            },
                        )
                    break

                before_state = simulator.true_state()
                before_observation = simulator.observation()
                transition_events = simulator.advance()
                all_events = [*action_events, *transition_events]
                visible_transition_events = self._model_visible_events(
                    transition_events, action_batch=False
                )
                self._append_public_events(
                    public_event_history, visible_transition_events
                )

                runtime_failure_event = next(
                    (
                        event
                        for event in transition_events
                        if event.event_type == "action_runtime_failure"
                    ),
                    None,
                )
                if runtime_failure_event is not None:
                    stats.runtime_error_count += 1
                    stats.infrastructure_failure = {
                        "category": "runtime_execution_failure",
                        "stage": "autonomous_transition",
                        "retryable": False,
                        "billing_ambiguity": False,
                        "request_state": "response_received",
                        "action_id": runtime_failure_event.data.get("action_id"),
                        "message": runtime_failure_event.data.get("message"),
                        "rollback_succeeded": runtime_failure_event.data.get(
                            "rollback_succeeded"
                        ),
                    }

                next_triggers: list[EventTrigger] = []
                if not simulator.done and stats.infrastructure_failure is None:
                    event_triggers = subscriptions.match(
                        [*visible_action_events, *visible_transition_events],
                        include_event_values=self.scenario.interaction.include_event_values,
                    )
                    for trigger in event_triggers:
                        stats.triggered_subscription_ids.add(
                            trigger.subscription_id
                        )
                    next_triggers.extend(event_triggers)

                    callback = scheduler.consume_if_due(simulator.step)
                    if callback is not None:
                        next_triggers.append(
                            EventTrigger(
                                "benchmark_scheduler",
                                "scheduled_callback",
                                simulator.step,
                                simulator.time,
                                callback.to_dict(),
                            )
                        )
                    if (
                        self.scenario.interaction.allows_periodic
                        and simulator.step
                        % self.scenario.interaction.interval_steps
                        == 0
                    ):
                        next_triggers.append(
                            EventTrigger(
                                "benchmark_periodic",
                                "periodic_invocation",
                                simulator.step,
                                simulator.time,
                                {
                                    "interval_steps": self.scenario.interaction.interval_steps
                                },
                            )
                        )

                logger.write(
                    "simulation_transition",
                    {
                        "from_step": before_state["step"],
                        "to_step": simulator.step,
                        "true_state_before": before_state,
                        "visible_observation_before": before_observation,
                        "events": [event.to_log_dict() for event in all_events],
                        "true_state_after": simulator.true_state(),
                        "visible_observation_after": simulator.observation(),
                        "next_invocation_triggers": [
                            trigger.to_dict() for trigger in next_triggers
                        ],
                    },
                )
                triggers = next_triggers
                if stats.infrastructure_failure is not None:
                    logger.write(
                        "infrastructure_failure_termination",
                        {
                            "step": simulator.step,
                            "time": simulator.time,
                            "classification": copy.deepcopy(
                                stats.infrastructure_failure
                            ),
                            "true_simulator_state": simulator.true_state(),
                            "visible_observation": simulator.observation(),
                            "physical_transition_after_failure": False,
                        },
                    )
                    break

            metrics = compute_metrics(
                self.scenario, simulator, stats, scheduler, subscriptions
            )
            completion_status = (
                "infrastructure_failed"
                if stats.infrastructure_failure is not None
                else "completed"
            )
            metrics["general"]["reasoning_metrics_eligible"] = (
                completion_status == "completed"
            )
            if completion_status != "completed":
                metrics["general"]["task_success"] = None
            summary = {
                "run_id": run_id,
                "completion_status": completion_status,
                "infrastructure_failure": copy.deepcopy(
                    stats.infrastructure_failure
                ),
                "scenario_name": self.scenario.name,
                "seed": self.seed,
                "policy": policy_metadata,
                "run_metadata": benchmark_metadata,
                "benchmark_versions": benchmark_metadata["benchmark_versions"],
                "scenario_configuration": self.scenario.to_mapping(),
                "metrics": metrics,
                "events_file": str(logger.events_path),
                "reproduction": {
                    "scenario_configuration": self.scenario.to_mapping(),
                    "simulator_seed": self.seed,
                    "reasoning_enabled": self.reasoning_enabled,
                    "model_configuration": benchmark_metadata.get(
                        "model_configuration"
                    ),
                    "parent_run_id": benchmark_metadata.get("parent_run_id"),
                },
            }
            logger.write(
                "run_completed",
                {
                    "run_id": run_id,
                    "completion_status": completion_status,
                    "infrastructure_failure": copy.deepcopy(
                        stats.infrastructure_failure
                    ),
                    "metrics": metrics,
                    "run_metadata": benchmark_metadata,
                    "final_true_simulator_state": simulator.true_state(),
                    "final_visible_observation": simulator.observation(),
                    "active_subscriptions": subscriptions.snapshot(),
                    "pending_callback": (
                        scheduler.pending.to_dict() if scheduler.pending else None
                    ),
                },
            )
            logger.write_summary(summary)
            return RunResult(
                run_id,
                metrics,
                logger.events_path,
                logger.summary_path,
                benchmark_metadata,
                completion_status,
            )
        except BaseException as exc:
            logger.write(
                "run_failed",
                {
                    "run_id": run_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "step": simulator.step,
                    "time": simulator.time,
                    "true_simulator_state": simulator.true_state(),
                    "visible_observation": simulator.observation(),
                },
            )
            raise
        finally:
            logger.close()

    def _interact(
        self,
        simulator: GraphCPSSimulator,
        scheduler: CallbackScheduler,
        subscriptions: SubscriptionManager,
        context_manager: ContextManager,
        stats: RunStatistics,
        triggers: list[EventTrigger],
        public_action_history: list[dict[str, Any]],
        public_event_history: list[dict[str, Any]],
        logger: RunLogger,
    ) -> list[SimulatorEvent]:
        observation = simulator.observation()
        constraints = self._decision_constraints(simulator, subscriptions)
        context = context_manager.build(
            observation,
            triggers,
            subscriptions.snapshot(),
            public_action_history,
            public_event_history,
        )
        request = PolicyRequest(
            self.scenario,
            context,
            constraints,
            stats.interaction_count,
        )
        if self.model_call_delay_seconds:
            time.sleep(self.model_call_delay_seconds)
        result = self.policy.decide(request)
        true_state_at_invocation = simulator.true_state()
        response_schema_hash = result.response_schema_hash or content_hash(
            decision_response_schema(constraints)
        )

        stats.interaction_count += 1
        stats.invocation_steps.append(simulator.step)
        stats.invocation_causes.append([trigger.event_type for trigger in triggers])
        is_model_policy = self.policy.metadata().get("policy") == "llm"
        if is_model_policy:
            stats.model_call_count += len(result.attempts)
        first_is_backend_error = bool(
            result.attempts and result.attempts[0].get("validation_stage") == "backend"
        )
        if first_is_backend_error:
            stats.backend_error_count += 1
            backend_error = result.attempts[0].get("backend_error", {})
            classification = (
                copy.deepcopy(backend_error)
                if isinstance(backend_error, dict)
                else {"category": "backend_error", "message": str(backend_error)}
            )
            classification.setdefault("category", "backend_error")
            classification.setdefault("retryable", False)
            classification.setdefault("billing_ambiguity", False)
            classification.setdefault("request_state", "unknown")
            classification["stage"] = "model_backend"
            stats.infrastructure_failure = classification
        if result.attempts and not first_is_backend_error:
            first = result.attempts[0]
            stats.first_attempt_count += 1
            stats.first_attempt_json_valid_count += int(bool(first.get("json_valid")))
            stats.first_attempt_schema_valid_count += int(
                bool(first.get("schema_valid"))
            )
            stats.first_attempt_decision_valid_count += int(
                bool(first.get("decision_valid"))
            )
            if result.valid and not first.get("decision_valid"):
                stats.validity_after_correction_count += 1
            retry_attempts = result.attempts[1:]
            if is_model_policy:
                stats.correction_request_count += len(retry_attempts)
                stats.additional_model_call_count += len(retry_attempts)
                for attempt in retry_attempts:
                    stats.add_additional_usage(dict(attempt.get("token_usage", {})))
                    stats.additional_latency_seconds += float(
                        attempt.get("latency_seconds") or 0.0
                    )
        invalid_attempts = sum(
            bool(item.get("validation_errors")) for item in result.attempts
        )
        stats.invalid_output_attempt_count += invalid_attempts
        stats.output_retry_count += max(0, len(result.attempts) - 1)
        stats.total_latency_seconds += result.latency_seconds
        stats.add_usage(result.token_usage)
        if not result.valid and not first_is_backend_error:
            stats.invalid_output_count += 1
        if result.decision is not None:
            stats.final_executable_decision_count += 1

        outcomes: list[Any] = []
        events: list[SimulatorEvent] = []
        scheduled_callback: dict[str, Any] | None = None
        if result.decision is not None:
            subscriptions.apply(result.decision.subscriptions, simulator.step)
            for command in result.decision.subscriptions:
                stats.subscription_records.append(
                    {"step": simulator.step, **command.to_dict()}
                )
            callback = scheduler.apply(
                result.decision.callback,
                simulator.step,
                simulator.time,
                self.scenario.time_step,
            )
            scheduled_callback = callback.to_dict() if callback else None
            context_manager.add_requests(result.decision.context_requests)
            outcomes, events = simulator.apply_actions(result.decision.actions)
            for command, outcome in zip(result.decision.actions, outcomes, strict=True):
                record = {
                    "step": simulator.step,
                    "time": simulator.time,
                    "command": command.to_dict(),
                    "outcome": outcome.to_dict(),
                }
                stats.action_records.append(record)
                public_action_history.append(copy.deepcopy(record))
                public_action_history[-1]["outcome"] = outcome.to_public_dict()
                if outcome.accepted:
                    stats.accepted_action_count += 1
                else:
                    stats.invalid_action_count += 1
            runtime_failure = next(
                (
                    outcome
                    for outcome in outcomes
                    if outcome.reason_code == "runtime_execution_failure"
                ),
                None,
            )
            if runtime_failure is not None:
                stats.runtime_error_count += 1
                stats.infrastructure_failure = {
                    "category": "runtime_execution_failure",
                    "stage": "action_execution",
                    "retryable": False,
                    "billing_ambiguity": False,
                    "request_state": "response_received",
                    "action_id": runtime_failure.action_id,
                    "message": runtime_failure.message,
                    "rollback_succeeded": runtime_failure.details.get(
                        "rollback_succeeded"
                    ),
                }

        context_manager.record(
            simulator.step,
            observation,
            result.decision.to_dict() if result.decision else None,
            [outcome.to_public_dict() for outcome in outcomes],
        )
        logger.write(
            "model_interaction",
            {
                "interaction_index": stats.interaction_count - 1,
                "step": simulator.step,
                "time": simulator.time,
                "invocation_triggers": [trigger.to_dict() for trigger in triggers],
                "true_state_at_invocation": true_state_at_invocation,
                "true_state_after_actions": simulator.true_state(),
                "visible_observation": observation,
                "explicit_context": context,
                "prompt": list(result.prompt) if result.prompt else None,
                "raw_response": result.raw_response,
                **result.response_processing,
                "parsed_response": result.parsed_response,
                "validation_errors": list(result.validation_errors),
                "output_attempts": list(result.attempts),
                "rendered_prompt_hash": result.rendered_prompt_hash,
                "response_schema_hash": response_schema_hash,
                "decision": result.decision.to_dict() if result.decision else None,
                "action_outcomes": [outcome.to_dict() for outcome in outcomes],
                "public_action_outcomes": [
                    outcome.to_public_dict() for outcome in outcomes
                ],
                "callbacks_requested": (
                    result.decision.callback.to_dict()
                    if result.decision and result.decision.callback
                    else None
                ),
                "scheduled_callback": scheduled_callback,
                "subscription_commands": (
                    [item.to_dict() for item in result.decision.subscriptions]
                    if result.decision
                    else []
                ),
                "active_subscriptions": subscriptions.snapshot(),
                "token_usage": result.token_usage,
                "latency_seconds": result.latency_seconds,
            },
        )
        return events

    def _benchmark_metadata(self, policy_metadata: dict[str, Any]) -> dict[str, Any]:
        requested_model_seed = (
            policy_metadata.get("decoding", {}).get("seed")
            if isinstance(policy_metadata.get("decoding"), dict)
            else None
        )
        capabilities = policy_metadata.get("capabilities", {})
        prompt = policy_metadata.get("prompt_template")
        if not isinstance(prompt, dict):
            prompt = {
                "id": self.requested_run_metadata.get(
                    "prompt_template_id", PROMPT_TEMPLATE_ID
                ),
                "version": self.requested_run_metadata.get(
                    "prompt_template_version", PROMPT_TEMPLATE_VERSION
                ),
                "content_hash": None,
            }
        prompt_id = str(prompt.get("id", PROMPT_TEMPLATE_ID))
        prompt_version = str(prompt.get("version", PROMPT_TEMPLATE_VERSION))
        protocol_id = str(
            self.requested_run_metadata.get("protocol_schema_id", PROTOCOL_SCHEMA_ID)
        )
        protocol_version = str(
            self.requested_run_metadata.get(
                "protocol_schema_version", PROTOCOL_SCHEMA_VERSION
            )
        )
        metadata = {
            **self.requested_run_metadata,
            "interface_condition": (
                "markdown_fence_normalized"
                if policy_metadata.get("response_normalization", {}).get("mode") == "markdown_single_fence"
                else "strict_prompt_only"
            ),
            "benchmark_versions": {
                "package_version": PACKAGE_VERSION,
                "scenario_schema_version": self.scenario.schema_version,
                "protocol_schema_id": protocol_id,
                "protocol_schema_version": protocol_version,
                "prompt_template_id": prompt_id,
                "prompt_template_version": prompt_version,
                "prompt_template_hash": prompt.get("content_hash")
                or prompt_template_content_hash(prompt_id, prompt_version),
                "suite_version": self.requested_run_metadata.get("suite_version"),
            },
            "difficulty_level": self.scenario.difficulty_level,
            "difficulty_profile": copy.deepcopy(self.scenario.difficulty_profile),
            "scenario_generation": copy.deepcopy(self.scenario.generation),
            "seeds": {
                "simulator_seed": self.seed,
                "scenario_generation_seed": self.scenario.generation.get(
                    "generation_seed"
                ),
                "random_baseline_seed": policy_metadata.get("seed"),
                "requested_model_seed": requested_model_seed,
                "backend_reports_model_seed_support": capabilities.get("model_seed"),
                "repetition_index": self.requested_run_metadata.get(
                    "repetition_index", 0
                ),
            },
            "scenario_content_hash": content_hash(self.scenario.to_mapping()),
            "model_configuration": self.requested_run_metadata.get(
                "model_configuration"
            ),
        }
        return metadata

    def _decision_constraints(
        self,
        simulator: GraphCPSSimulator,
        subscriptions: SubscriptionManager,
    ) -> DecisionConstraints:
        allowed_events = (
            self.scenario.interaction.allowed_event_types or DEFAULT_EVENT_TYPES
        )
        return DecisionConstraints(
            allowed_action_ids=self.scenario.action_ids,
            action_parameter_schemas={
                str(action["id"]): dict(action.get("parameters", {}))
                for action in self.scenario.actions
            },
            max_actions=self.scenario.interaction.max_actions_per_interaction,
            current_step=simulator.step,
            current_time=simulator.time,
            max_steps=self.scenario.max_steps,
            time_step=self.scenario.time_step,
            allows_callbacks=self.scenario.interaction.allows_callbacks,
            min_callback_steps=self.scenario.interaction.min_callback_steps,
            max_callback_steps=self.scenario.interaction.max_callback_steps,
            allows_subscriptions=self.scenario.interaction.allows_subscriptions,
            allowed_event_types=tuple(allowed_events),
            active_subscription_ids=tuple(subscriptions.active),
            allowed_context_requests=self._allowed_context_requests(
                simulator.observation()
            ),
            reasoning_enabled=self.reasoning_enabled,
        )

    @staticmethod
    def _allowed_context_requests(observation: dict[str, Any]) -> tuple[str, ...]:
        result = [
            "active_subscriptions",
            "action_history",
            "event_history",
            "previous_decision",
        ]
        result.extend(f"sensor:{item}" for item in observation.get("sensors", {}))
        result.extend(f"resource:{item}" for item in observation.get("resources", {}))
        return tuple(result)

    @staticmethod
    def _append_public_events(
        history: list[dict[str, Any]], events: list[SimulatorEvent]
    ) -> None:
        for event in events:
            record = public_event_record(event)
            if record is not None:
                history.append(record)

    def _model_visible_events(
        self,
        events: list[SimulatorEvent],
        *,
        action_batch: bool,
    ) -> list[SimulatorEvent]:
        if self.scenario.context.include_action_outcomes:
            return list(events)
        if action_batch:
            return []
        return [
            event
            for event in events
            if event.event_type not in ACTION_OUTCOME_EVENT_TYPES
        ]
