"""A compact, configuration-driven graph CPS simulator."""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping

from cps_reason_bench.config import ConfigError, ScenarioConfig
from cps_reason_bench.protocol import ActionCommand
from cps_reason_bench.types import ActionOutcome, SimulatorEvent


class GraphCPSSimulator:
    """Evolve generic node variables, resources, sensors, actuators, and faults."""

    def __init__(self, scenario: ScenarioConfig, seed: int | None = None) -> None:
        self.scenario = scenario
        self.seed = scenario.seed if seed is None else seed
        self._rng = random.Random(self.seed)
        self._node_specs = {
            str(node_id): dict(node)
            for node_id, node in scenario.graph["nodes"].items()
        }
        self._edge_specs = {
            str(edge["id"]): dict(edge)
            for edge in scenario.graph.get("edges", [])
        }
        self._sensor_specs = {str(item["id"]): dict(item) for item in scenario.sensors}
        self._action_specs = {str(item["id"]): dict(item) for item in scenario.actions}
        self.reset()

    def reset(self) -> dict[str, Any]:
        self._rng = random.Random(self.seed)
        self.step = 0
        self.time = 0.0
        self.node_values: dict[str, dict[str, Any]] = {}
        for node_id, node in self._node_specs.items():
            variables = dict(node.get("variables", {}))
            self.node_values[node_id] = {
                str(variable_id): copy.deepcopy(variable["initial"])
                for variable_id, variable in variables.items()
            }

        resource_specs = dict(self.scenario.graph.get("resources", {}))
        self.resources = {
            str(resource_id): copy.deepcopy(spec["initial"])
            for resource_id, spec in resource_specs.items()
        }
        self.edge_states = {
            edge_id: bool(spec.get("enabled", True))
            for edge_id, spec in self._edge_specs.items()
        }
        self.sensor_runtime = {
            sensor_id: {
                "available": bool(spec.get("available", True)),
                "bias": float(spec.get("bias", 0.0)),
                "delay_steps": int(spec.get("delay_steps", 0)),
                "stuck_value": None,
            }
            for sensor_id, spec in self._sensor_specs.items()
        }
        self.action_runtime = {
            action_id: {
                "enabled": bool(spec.get("enabled", True)),
                "effect_scale": float(spec.get("effect_scale", 1.0)),
                "remap_target": None,
            }
            for action_id, spec in self._action_specs.items()
        }
        self.drift_overrides: dict[str, float] = {}
        self.active_faults: dict[str, dict[str, Any]] = {}
        self.sensor_history: dict[str, list[Any]] = defaultdict(list)
        self.last_action_outcomes: list[ActionOutcome] = []
        self.last_inspection_results: list[dict[str, Any]] = []
        self.action_history: list[dict[str, Any]] = []
        self.pending_action_effects: list[dict[str, Any]] = []
        self.completed_delayed_effects: list[dict[str, Any]] = []
        self.safety_violations: list[dict[str, Any]] = []
        self._safety_violation_keys: set[tuple[str, int]] = set()
        self.prerequisite_violations = 0
        self.goal_first_met: dict[str, int] = {}
        self.goal_status_history: list[dict[str, Any]] = []
        self._last_goal_status: bool | None = None
        self._activated_fault_ids: set[str] = set()
        self._sensor_reading_cache: dict[str, dict[str, Any]] = {}

        self.initial_events = self._activate_due_faults()
        self._record_sensor_history()
        self._evaluate_goals()
        self.initial_events.extend(self._constraint_events())
        return self.observation()

    @property
    def done(self) -> bool:
        if self.step >= self.scenario.max_steps:
            return True
        return bool(self.scenario.task.get("terminate_on_goals", False) and self.goals_met)

    @property
    def goals_met(self) -> bool:
        return all(self._goal_met(goal) for goal in self.scenario.goals)

    def observation(self) -> dict[str, Any]:
        full_state = bool(self.scenario.observability.get("full_state", False))
        visible_sensors: dict[str, Any] = {}
        for sensor_id, spec in self._sensor_specs.items():
            if spec.get("observable", True):
                visible_sensors[sensor_id] = self._read_sensor(sensor_id)

        visible_resources: dict[str, Any] = {}
        resource_specs = dict(self.scenario.graph.get("resources", {}))
        for resource_id, value in self.resources.items():
            if resource_specs.get(resource_id, {}).get("observable", True):
                visible_resources[resource_id] = value

        topology = {
            "nodes": [
                {
                    "id": node_id,
                    "label": spec.get("label", node_id),
                }
                for node_id, spec in self._node_specs.items()
                if spec.get("observable", True)
            ],
            "edges": [
                {
                    "id": edge_id,
                    "source": spec["source"],
                    "target": spec["target"],
                    **(
                        {"enabled": self.edge_states[edge_id]}
                        if spec.get("state_observable", True)
                        else {}
                    ),
                }
                for edge_id, spec in self._edge_specs.items()
                if spec.get("observable", True)
            ],
        }
        available_actions = []
        for action_id, spec in self._action_specs.items():
            action = {
                "id": action_id,
                "description": spec.get("description", action_id),
            }
            if spec.get("show_preconditions", True) and spec.get("preconditions"):
                action["preconditions"] = copy.deepcopy(spec["preconditions"])
            available_actions.append(action)

        observation: dict[str, Any] = {
            "step": self.step,
            "time": self.time,
            "topology": topology,
            "sensors": visible_sensors,
            "resources": visible_resources,
            "available_actions": available_actions,
            "last_action_outcomes": [
                item.to_public_dict() for item in self.last_action_outcomes
            ],
            "inspection_results": copy.deepcopy(self.last_inspection_results),
        }
        visible_goals = [
            {
                "id": goal["id"],
                "met": self._goal_met(goal),
            }
            for goal in self.scenario.goals
            if goal.get("status_observable", False)
        ]
        if visible_goals:
            observation["goal_status"] = visible_goals
        if full_state:
            observation["full_state"] = self.true_state()
        return observation

    def true_state(self) -> dict[str, Any]:
        state = {
            "step": self.step,
            "time": self.time,
            "nodes": copy.deepcopy(self.node_values),
            "resources": copy.deepcopy(self.resources),
            "edges": copy.deepcopy(self.edge_states),
            "sensor_runtime": copy.deepcopy(self.sensor_runtime),
            "action_runtime": copy.deepcopy(self.action_runtime),
            "active_faults": copy.deepcopy(self.active_faults),
            "goals_met": self.goals_met,
        }
        if self.pending_action_effects:
            state["pending_action_effects"] = copy.deepcopy(
                self.pending_action_effects
            )
        return state

    def apply_actions(
        self, commands: Iterable[ActionCommand]
    ) -> tuple[list[ActionOutcome], list[SimulatorEvent]]:
        commands = list(commands)
        if commands and self.scenario.interaction.action_batch_semantics == "atomic":
            return self._apply_actions_atomic(commands)
        outcomes: list[ActionOutcome] = []
        events: list[SimulatorEvent] = []
        self.last_inspection_results = []
        for command in commands:
            outcome, action_events = self._apply_action(command)
            outcomes.append(outcome)
            events.extend(action_events)
            self.action_history.append(
                {
                    "step": self.step,
                    "time": self.time,
                    "command": command.to_dict(),
                    "outcome": outcome.to_dict(),
                }
            )
            if outcome.reason_code == "runtime_execution_failure":
                break
        self.last_action_outcomes = outcomes
        if commands:
            events.extend(self._constraint_events())
            events.extend(self._goal_events())
        if any(
            event.event_type
            in {"variable_changed", "resource_changed", "edge_changed", "fault_repaired"}
            for event in events
        ):
            self._sensor_reading_cache.clear()
        return outcomes, events

    def _apply_actions_atomic(
        self, commands: list[ActionCommand]
    ) -> tuple[list[ActionOutcome], list[SimulatorEvent]]:
        """Apply an ordered command list as one all-or-nothing transaction."""

        batch_before = self._action_transaction_state()
        attempted_outcomes: list[ActionOutcome] = []
        attempted_events: list[SimulatorEvent] = []
        failure_index: int | None = None
        self.last_inspection_results = []
        for index, command in enumerate(commands):
            outcome, action_events = self._apply_action(command)
            attempted_outcomes.append(outcome)
            attempted_events.extend(action_events)
            if not outcome.accepted:
                failure_index = index
                break

        if failure_index is None:
            outcomes = attempted_outcomes
            events = attempted_events
        else:
            failed_action_id = commands[failure_index].action_id
            attempted_state = self._action_transaction_state()
            self._restore_action_transaction_state(batch_before)
            restored_state = self._action_transaction_state()
            outcomes = []
            for index, command in enumerate(commands):
                if index < failure_index:
                    original = attempted_outcomes[index]
                    outcomes.append(
                        ActionOutcome(
                            command.action_id,
                            False,
                            "atomic action batch rolled back",
                            {
                                "failed_action_id": failed_action_id,
                                "original_outcome": original.to_dict(),
                            },
                            "batch_rolled_back",
                            "action batch rolled back",
                            {"failed_action_id": failed_action_id},
                        )
                    )
                elif index == failure_index:
                    outcomes.append(attempted_outcomes[index])
                else:
                    outcomes.append(
                        ActionOutcome(
                            command.action_id,
                            False,
                            "not executed because the atomic action batch failed",
                            {"failed_action_id": failed_action_id},
                            "batch_not_executed",
                            "action not executed",
                            {"failed_action_id": failed_action_id},
                        )
                    )
            failure_events = [
                event
                for event in attempted_events
                if event.event_type
                in {
                    "invalid_action",
                    "action_failed",
                    "actuator_failed",
                    "action_runtime_failure",
                }
            ]
            events = [
                *failure_events,
                self._event(
                    "action_batch_rolled_back",
                    {
                        "failed_action_id": failed_action_id,
                        "command_count": len(commands),
                        "state_before": batch_before,
                        "state_after_attempt": attempted_state,
                        "state_after_rollback": restored_state,
                        "rollback_succeeded": restored_state == batch_before,
                    },
                    {
                        "failed_action_id": failed_action_id,
                        "command_count": len(commands),
                    },
                ),
            ]

        for command, outcome in zip(commands, outcomes, strict=True):
            self.action_history.append(
                {
                    "step": self.step,
                    "time": self.time,
                    "command": command.to_dict(),
                    "outcome": outcome.to_dict(),
                }
            )
        self.last_action_outcomes = outcomes
        events.extend(self._constraint_events())
        events.extend(self._goal_events())
        if any(
            event.event_type
            in {"variable_changed", "resource_changed", "edge_changed", "fault_repaired"}
            for event in events
        ):
            self._sensor_reading_cache.clear()
        return outcomes, events

    def advance(self) -> list[SimulatorEvent]:
        if self.done:
            return []
        self.step += 1
        self.time = round(self.step * self.scenario.time_step, 10)
        events = self._activate_due_faults()
        events.extend(self._apply_due_action_effects())
        before = copy.deepcopy(self.node_values)
        self._apply_drift()
        self._apply_edge_coupling()
        self._clamp_variables()
        events.extend(self._variable_change_events(before, source="dynamics"))
        self._record_sensor_history()
        events.extend(self._constraint_events())
        events.extend(self._goal_events())
        return events

    def _apply_action(
        self, command: ActionCommand
    ) -> tuple[ActionOutcome, list[SimulatorEvent]]:
        spec = self._action_specs.get(command.action_id)
        if spec is None:
            return self._reject_action(command.action_id, "unknown action", "invalid_action")
        runtime = self.action_runtime[command.action_id]
        if not runtime["enabled"]:
            return self._reject_action(
                command.action_id,
                "actuator is unavailable",
                "actuator_failed",
                public={"action_id": command.action_id},
            )

        unknown_parameters = set(command.parameters) - set(spec.get("parameters", {}))
        if unknown_parameters:
            return self._reject_action(
                command.action_id,
                f"unsupported parameters: {', '.join(sorted(unknown_parameters))}",
                "invalid_action",
            )
        for condition in spec.get("preconditions", []):
            if not self.evaluate_condition(condition):
                self.prerequisite_violations += 1
                return self._reject_action(
                    command.action_id,
                    f"precondition failed: {condition}",
                    "action_failed",
                    public={"action_id": command.action_id, "reason": "precondition"},
                    failure_details={"unmet_precondition": copy.deepcopy(condition)},
                )

        costs = dict(spec.get("costs", {}))
        for resource_id, raw_cost in costs.items():
            cost = float(raw_cost)
            if resource_id not in self.resources:
                return self._reject_action(
                    command.action_id, f"unknown resource {resource_id!r}", "invalid_action"
                )
            minimum = float(
                self.scenario.graph.get("resources", {})
                .get(resource_id, {})
                .get("min", 0.0)
            )
            if float(self.resources[resource_id]) - cost < minimum:
                return self._reject_action(
                    command.action_id,
                    f"insufficient resource {resource_id}",
                    "action_failed",
                    public={"action_id": command.action_id, "reason": "resource"},
                    failure_details={"resource_id": resource_id},
                )

        events: list[SimulatorEvent] = []
        transaction_before = self._action_transaction_state()
        for resource_id, raw_cost in costs.items():
            old = self.resources[resource_id]
            self.resources[resource_id] = old - float(raw_cost)
            events.append(self._resource_event(resource_id, old, self.resources[resource_id]))

        operation = str(spec.get("operation", "noop"))
        effect_delay = int(spec.get("effect_delay_steps", 0))
        if effect_delay:
            due_step = self.step + effect_delay
            pending = {
                "action_id": command.action_id,
                "requested_step": self.step,
                "requested_time": self.time,
                "due_step": due_step,
                "command": command.to_dict(),
            }
            self.pending_action_effects.append(pending)
            details = {
                "operation": operation,
                "effect_delay_steps": effect_delay,
                "due_step": due_step,
            }
            events.append(
                self._event(
                    "action_scheduled",
                    copy.deepcopy(pending),
                    {"action_id": command.action_id, "due_step": due_step},
                )
            )
            return ActionOutcome(
                command.action_id,
                True,
                "scheduled",
                details,
                public_message="scheduled",
                public_details=copy.deepcopy(details),
            ), events
        try:
            details = self._execute_operation(command, spec, runtime, events)
        except (ConfigError, TypeError, ValueError) as exc:
            attempted = self._action_transaction_state()
            self._restore_action_transaction_state(transaction_before)
            restored = self._action_transaction_state()
            return self._runtime_execution_failure(
                command.action_id,
                exc,
                transaction_before,
                attempted,
                restored,
            )

        events.append(
            self._event(
                "action_applied",
                {"action_id": command.action_id, "operation": operation, "details": details},
                {"action_id": command.action_id},
            )
        )
        return ActionOutcome(command.action_id, True, "applied", details), events

    def _apply_due_action_effects(self) -> list[SimulatorEvent]:
        events: list[SimulatorEvent] = []
        due = [
            item for item in self.pending_action_effects if item["due_step"] <= self.step
        ]
        self.pending_action_effects = [
            item for item in self.pending_action_effects if item["due_step"] > self.step
        ]
        for pending in due:
            action_id = str(pending["action_id"])
            spec = self._action_specs[action_id]
            runtime = self.action_runtime[action_id]
            command_data = pending["command"]
            command = ActionCommand(
                action_id,
                dict(command_data.get("parameters", {})),
            )
            completion = {
                **copy.deepcopy(pending),
                "completed_step": self.step,
                "completed_time": self.time,
                "success": False,
            }
            if not runtime["enabled"]:
                completion["reason"] = "actuator is unavailable at effect time"
                events.append(
                    self._event(
                        "actuator_failed",
                        {
                            "action_id": action_id,
                            "message": completion["reason"],
                            "delayed_effect": True,
                        },
                        {"action_id": action_id},
                    )
                )
                self.completed_delayed_effects.append(completion)
                continue
            transaction_before = self._action_transaction_state()
            event_start = len(events)
            try:
                details = self._execute_operation(
                    command, spec, runtime, events
                )
            except (ConfigError, TypeError, ValueError) as exc:
                attempted = self._action_transaction_state()
                self._restore_action_transaction_state(transaction_before)
                restored = self._action_transaction_state()
                del events[event_start:]
                completion["reason"] = str(exc)
                completion["runtime_failure"] = True
                events.append(
                    self._event(
                        "action_runtime_failure",
                        {
                            "action_id": action_id,
                            "message": str(exc),
                            "delayed_effect": True,
                            "exception_type": type(exc).__name__,
                            "state_before": transaction_before,
                            "state_after_failed_execution": attempted,
                            "state_after_rollback": restored,
                            "rollback_succeeded": restored == transaction_before,
                        },
                        {},
                        publicly_observable=False,
                    )
                )
                self.completed_delayed_effects.append(completion)
                continue
            completion.update({"success": True, "details": details})
            self.completed_delayed_effects.append(completion)
            events.append(
                self._event(
                    "delayed_action_effect",
                    copy.deepcopy(completion),
                    {"effect_observed": True},
                )
            )
        return events

    def _execute_operation(
        self,
        command: ActionCommand,
        spec: Mapping[str, Any],
        runtime: Mapping[str, Any],
        events: list[SimulatorEvent],
    ) -> dict[str, Any]:
        operation = str(spec.get("operation", "noop"))
        target = runtime.get("remap_target") or spec.get("target")
        details: dict[str, Any] = {"operation": operation}

        if operation == "noop":
            return details
        if operation == "adjust":
            if not isinstance(target, str):
                raise ConfigError("adjust action requires a target")
            old = self.resolve_target(target)
            amount = command.parameters.get("amount", spec.get("amount"))
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise ConfigError("adjust action requires a numeric amount")
            amount = float(amount) * float(runtime["effect_scale"])
            self.set_target(target, float(old) + amount)
            self._clamp_target(target)
            new = self.resolve_target(target)
            events.extend(self._target_change_event(target, old, new, "action"))
            details.update({"target": target, "applied_amount": amount})
        elif operation == "set":
            if not isinstance(target, str):
                raise ConfigError("set action requires a target")
            old = self.resolve_target(target)
            value = command.parameters.get("value", spec.get("value"))
            self.set_target(target, value)
            self._clamp_target(target)
            new = self.resolve_target(target)
            events.extend(self._target_change_event(target, old, new, "action"))
            details.update({"target": target})
        elif operation == "toggle_edge":
            edge_id = str(target)
            if edge_id not in self.edge_states:
                raise ConfigError(f"unknown edge {edge_id!r}")
            old = self.edge_states[edge_id]
            new = bool(
                command.parameters.get(
                    "enabled", spec.get("edge_enabled", not old)
                )
            )
            self.edge_states[edge_id] = new
            events.append(
                self._event(
                    "edge_changed",
                    {"edge_id": edge_id, "old": old, "new": new},
                    {"edge_id": edge_id, "enabled": new},
                )
            )
            details.update({"edge_id": edge_id, "old": old, "new": new})
        elif operation == "inspect":
            inspection_target = str(target)
            private_result = self._inspect(inspection_target)
            fault_ids = list(private_result.get("faults", []))
            result = {
                key: copy.deepcopy(value)
                for key, value in private_result.items()
                if key != "faults"
            }
            self.last_inspection_results.append(result)
            details["inspection"] = result
            events.append(
                self._event(
                    "inspection_completed",
                    private_result,
                    copy.deepcopy(result),
                )
            )
            if fault_ids:
                events.append(
                    self._event(
                        "fault_detected",
                        {"target": inspection_target, "faults": fault_ids},
                        {"target": inspection_target},
                    )
                )
        elif operation == "repair":
            repair_target = str(target)
            cleared = self._repair(repair_target)
            details.update(
                {"target": repair_target, "repaired_fault_count": len(cleared)}
            )
            events.append(
                self._event(
                    "fault_repaired",
                    {"target": repair_target, "fault_ids": cleared},
                    {"target": repair_target},
                )
            )
        elif operation == "communicate":
            message = str(command.parameters.get("message", spec.get("message", "status update")))
            recipient = str(spec.get("recipient", "peer"))
            details.update({"recipient": recipient, "message": message})
            events.append(
                self._event(
                    "communication_sent",
                    {"recipient": recipient, "message": message},
                    {"recipient": recipient},
                )
            )
        else:
            raise ConfigError(f"unsupported action operation {operation!r}")

        for effect in spec.get("additional_effects", []):
            self._apply_additional_effect(effect, events)
        return details

    def _apply_additional_effect(
        self, effect: Mapping[str, Any], events: list[SimulatorEvent]
    ) -> None:
        operation = effect.get("operation")
        target = str(effect.get("target", ""))
        old = self.resolve_target(target)
        if operation == "set":
            self.set_target(target, effect.get("value"))
        elif operation == "adjust":
            self.set_target(target, float(old) + float(effect.get("amount", 0.0)))
        else:
            raise ConfigError(f"unsupported additional effect {operation!r}")
        self._clamp_target(target)
        events.extend(self._target_change_event(target, old, self.resolve_target(target), "action"))

    def _reject_action(
        self,
        action_id: str,
        message: str,
        event_type: str,
        public: dict[str, Any] | None = None,
        failure_details: dict[str, Any] | None = None,
    ) -> tuple[ActionOutcome, list[SimulatorEvent]]:
        feedback_mode = self._action_specs.get(action_id, {}).get(
            "failure_feedback", "full"
        )
        if feedback_mode == "generic":
            public_message = "action rejected"
            public_details = {"failure_cause_visible": False}
        else:
            public_message = message
            public_details = {
                "failure_cause_visible": True,
                **(failure_details or {}),
            }
        outcome = ActionOutcome(
            action_id,
            False,
            message,
            failure_details or {},
            event_type,
            public_message,
            public_details,
        )
        event = self._event(
            event_type,
            {"action_id": action_id, "message": message},
            public or {"action_id": action_id},
        )
        return outcome, [event]

    def _runtime_execution_failure(
        self,
        action_id: str,
        exc: BaseException,
        state_before: dict[str, Any],
        state_after_failed_execution: dict[str, Any],
        state_after_rollback: dict[str, Any],
    ) -> tuple[ActionOutcome, list[SimulatorEvent]]:
        details = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "state_before": state_before,
            "state_after_failed_execution": state_after_failed_execution,
            "state_after_rollback": state_after_rollback,
            "rollback_succeeded": state_after_rollback == state_before,
        }
        outcome = ActionOutcome(
            action_id,
            False,
            "runtime action execution failed",
            details,
            "runtime_execution_failure",
            "action rejected",
            {
                "failure_cause_visible": False,
                "runtime_failure": True,
            },
        )
        event = self._event(
            "action_runtime_failure",
            {"action_id": action_id, **copy.deepcopy(details)},
            {},
            publicly_observable=False,
        )
        return outcome, [event]

    def _action_transaction_state(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "node_values": self.node_values,
                "resources": self.resources,
                "edge_states": self.edge_states,
                "sensor_runtime": self.sensor_runtime,
                "action_runtime": self.action_runtime,
                "drift_overrides": self.drift_overrides,
                "active_faults": self.active_faults,
                "last_inspection_results": self.last_inspection_results,
                "pending_action_effects": self.pending_action_effects,
                "sensor_reading_cache": self._sensor_reading_cache,
            }
        )

    def _restore_action_transaction_state(self, state: Mapping[str, Any]) -> None:
        self.node_values = copy.deepcopy(state["node_values"])
        self.resources = copy.deepcopy(state["resources"])
        self.edge_states = copy.deepcopy(state["edge_states"])
        self.sensor_runtime = copy.deepcopy(state["sensor_runtime"])
        self.action_runtime = copy.deepcopy(state["action_runtime"])
        self.drift_overrides = copy.deepcopy(state["drift_overrides"])
        self.active_faults = copy.deepcopy(state["active_faults"])
        self.last_inspection_results = copy.deepcopy(
            state["last_inspection_results"]
        )
        self.pending_action_effects = copy.deepcopy(state["pending_action_effects"])
        self._sensor_reading_cache = copy.deepcopy(state["sensor_reading_cache"])

    def _apply_drift(self) -> None:
        for node_id, node in self._node_specs.items():
            for variable_id, spec in node.get("variables", {}).items():
                target = f"{node_id}.{variable_id}"
                drift = self.drift_overrides.get(target, spec.get("drift", 0.0))
                value = self.node_values[node_id][variable_id]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.node_values[node_id][variable_id] = float(value) + float(drift)

    def _apply_edge_coupling(self) -> None:
        before = copy.deepcopy(self.node_values)
        deltas: dict[tuple[str, str], float] = defaultdict(float)
        for edge_id, spec in self._edge_specs.items():
            if not self.edge_states[edge_id]:
                continue
            variable = spec.get("variable")
            coupling = float(spec.get("coupling", 0.0))
            if not variable or coupling == 0:
                continue
            source = str(spec["source"])
            target = str(spec["target"])
            source_value = before[source][variable]
            target_value = before[target][variable]
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (source_value, target_value)
            ):
                continue
            flow = (float(source_value) - float(target_value)) * coupling
            deltas[(target, str(variable))] += flow
            if not spec.get("directed", False):
                deltas[(source, str(variable))] -= flow
        for (node_id, variable_id), delta in deltas.items():
            self.node_values[node_id][variable_id] += delta

    def _activate_due_faults(self) -> list[SimulatorEvent]:
        events: list[SimulatorEvent] = []
        for fault in self.scenario.faults:
            fault_id = str(fault["id"])
            if fault_id in self._activated_fault_ids or int(fault.get("at_step", 0)) != self.step:
                continue
            self._activated_fault_ids.add(fault_id)
            self.active_faults[fault_id] = copy.deepcopy(fault)
            kind = str(fault.get("kind"))
            target = str(fault.get("target"))
            value = fault.get("value")
            public: dict[str, Any] = {}
            if kind == "sensor_bias":
                self.sensor_runtime[target]["bias"] += float(value)
            elif kind == "sensor_stuck":
                sensor = self._sensor_specs[target]
                actual = self.resolve_target(str(sensor["target"]))
                self.sensor_runtime[target]["stuck_value"] = actual if value is None else value
            elif kind == "sensor_unavailable":
                self.sensor_runtime[target]["available"] = False
                public = {"sensor_id": target}
                events.append(self._event("sensor_unavailable", {"sensor_id": target}, public))
            elif kind == "sensor_delay":
                self.sensor_runtime[target]["delay_steps"] = int(value)
            elif kind == "actuator_disabled":
                self.action_runtime[target]["enabled"] = False
            elif kind == "actuator_scale":
                self.action_runtime[target]["effect_scale"] = float(value)
            elif kind == "actuator_remap":
                self.action_runtime[target]["remap_target"] = str(value)
            elif kind == "edge_disabled":
                old = self.edge_states[target]
                self.edge_states[target] = False
                public = {"edge_id": target, "enabled": False}
                events.append(
                    self._event(
                        "edge_changed",
                        {"edge_id": target, "old": old, "new": False},
                        public,
                    )
                )
            elif kind == "drift_change":
                self.drift_overrides[target] = float(value)
            else:
                raise ConfigError(f"Unsupported fault kind {kind!r}")
            announced = bool(fault.get("announced", False))
            if announced:
                public = {"fault_id": fault_id, "kind": kind, "target": target}
            events.append(
                self._event(
                    "fault_activated",
                    {"fault_id": fault_id, "kind": kind, "target": target, "value": value},
                    public,
                    publicly_observable=announced,
                )
            )
        return events

    def _repair(self, target: str) -> list[str]:
        cleared = [
            fault_id
            for fault_id, fault in self.active_faults.items()
            if str(fault.get("target")) == target
        ]
        for fault_id in cleared:
            del self.active_faults[fault_id]
        if target in self.sensor_runtime:
            spec = self._sensor_specs[target]
            self.sensor_runtime[target] = {
                "available": bool(spec.get("available", True)),
                "bias": float(spec.get("bias", 0.0)),
                "delay_steps": int(spec.get("delay_steps", 0)),
                "stuck_value": None,
            }
        elif target in self.action_runtime:
            spec = self._action_specs[target]
            self.action_runtime[target] = {
                "enabled": bool(spec.get("enabled", True)),
                "effect_scale": float(spec.get("effect_scale", 1.0)),
                "remap_target": None,
            }
        elif target in self.edge_states:
            self.edge_states[target] = bool(self._edge_specs[target].get("enabled", True))
        elif target in self.drift_overrides:
            del self.drift_overrides[target]
        return cleared

    def _inspect(self, target: str) -> dict[str, Any]:
        fault_ids = [
            fault_id
            for fault_id, fault in self.active_faults.items()
            if str(fault.get("target")) == target
        ]
        result: dict[str, Any] = {"target": target, "faults": fault_ids}
        if target in self.sensor_runtime:
            result["component_type"] = "sensor"
            result["status"] = copy.deepcopy(self.sensor_runtime[target])
        elif target in self.action_runtime:
            result["component_type"] = "actuator"
            result["status"] = copy.deepcopy(self.action_runtime[target])
        elif target in self.edge_states:
            result["component_type"] = "edge"
            result["status"] = {"enabled": self.edge_states[target]}
        else:
            value = self.resolve_target(target)
            result["component_type"] = "state"
            result["value"] = copy.deepcopy(value)
        return result

    def _read_sensor(self, sensor_id: str) -> dict[str, Any]:
        if sensor_id in self._sensor_reading_cache:
            return copy.deepcopy(self._sensor_reading_cache[sensor_id])
        spec = self._sensor_specs[sensor_id]
        runtime = self.sensor_runtime[sensor_id]
        target = str(spec["target"])
        result: dict[str, Any] = {
            "target": target,
            "status": "ok",
            "age_steps": int(runtime["delay_steps"]),
        }
        if not runtime["available"]:
            result.update({"status": "unavailable", "value": None})
            self._sensor_reading_cache[sensor_id] = copy.deepcopy(result)
            return result

        history = self.sensor_history[sensor_id]
        delay = int(runtime["delay_steps"])
        value = runtime["stuck_value"]
        if value is None:
            if delay == 0:
                value = self.resolve_target(target)
            else:
                index = max(0, len(history) - 1 - delay)
                value = history[index] if history else self.resolve_target(target)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value) + float(runtime["bias"])
            noise = float(spec.get("noise_std", 0.0))
            if noise:
                value += self._rng.gauss(0.0, noise)
            precision = int(spec.get("precision", 3))
            value = round(value, precision)
        if runtime["stuck_value"] is not None:
            result["status"] = "stale"
        elif delay > 0:
            result["status"] = "delayed"
        result["value"] = value
        self._sensor_reading_cache[sensor_id] = copy.deepcopy(result)
        return result

    def _record_sensor_history(self) -> None:
        self._sensor_reading_cache.clear()
        for sensor_id, spec in self._sensor_specs.items():
            self.sensor_history[sensor_id].append(
                copy.deepcopy(self.resolve_target(str(spec["target"])))
            )

    def _constraint_events(self) -> list[SimulatorEvent]:
        events: list[SimulatorEvent] = []
        for constraint in self.scenario.constraints:
            if self._constraint_violated(constraint):
                violation_key = (str(constraint["id"]), self.step)
                if violation_key in self._safety_violation_keys:
                    continue
                self._safety_violation_keys.add(violation_key)
                record = {
                    "id": constraint["id"],
                    "step": self.step,
                    "time": self.time,
                    "kind": constraint.get("kind", "condition"),
                }
                self.safety_violations.append(record)
                events.append(
                    self._event(
                        "safety_violation",
                        {**record, "constraint": copy.deepcopy(constraint)},
                        {"constraint_id": constraint["id"]},
                    )
                )
        return events

    def _constraint_violated(self, constraint: Mapping[str, Any]) -> bool:
        kind = constraint.get("kind", "condition")
        if kind == "bounds":
            value = self.resolve_target(str(constraint["target"]))
            minimum = constraint.get("min")
            maximum = constraint.get("max")
            return bool(
                (minimum is not None and value < minimum)
                or (maximum is not None and value > maximum)
            )
        if kind == "conditional":
            antecedent = self.evaluate_condition(constraint["if"])
            consequent = self.evaluate_condition(constraint["then"])
            return antecedent and not consequent
        if kind == "condition":
            return not self.evaluate_condition(constraint["condition"])
        raise ConfigError(f"Unsupported constraint kind {kind!r}")

    def _goal_met(self, goal: Mapping[str, Any]) -> bool:
        if "condition" in goal:
            return self.evaluate_condition(goal["condition"])
        if "all" in goal:
            return all(self.evaluate_condition(item) for item in goal["all"])
        if "any" in goal:
            return any(self.evaluate_condition(item) for item in goal["any"])
        raise ConfigError(f"Goal {goal.get('id')!r} must define condition, all, or any")

    def _evaluate_goals(self) -> None:
        for goal in self.scenario.goals:
            goal_id = str(goal["id"])
            if goal_id not in self.goal_first_met and self._goal_met(goal):
                self.goal_first_met[goal_id] = self.step
        current_status = self.goals_met
        if current_status != self._last_goal_status:
            self.goal_status_history.append(
                {"step": self.step, "time": self.time, "goals_met": current_status}
            )
            self._last_goal_status = current_status

    def _goal_events(self) -> list[SimulatorEvent]:
        before = set(self.goal_first_met)
        self._evaluate_goals()
        return [
            self._event(
                "task_condition",
                {"goal_id": goal_id},
                {"goal_id": goal_id},
            )
            for goal_id in self.goal_first_met
            if goal_id not in before
        ]

    def evaluate_condition(self, condition: Mapping[str, Any]) -> bool:
        target = str(condition["target"])
        actual = self.resolve_target(target)
        expected = condition.get("value")
        operator = str(condition.get("operator", "eq"))
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "gt":
            return actual > expected
        if operator == "ge":
            return actual >= expected
        if operator == "lt":
            return actual < expected
        if operator == "le":
            return actual <= expected
        if operator == "in":
            return actual in expected
        raise ConfigError(f"Unsupported condition operator {operator!r}")

    def resolve_target(self, target: str) -> Any:
        parts = target.split(".")
        if len(parts) != 2:
            raise ConfigError(f"State target must use '<node>.<variable>' or a resource/edge path: {target}")
        owner, key = parts
        if owner == "resource":
            if key not in self.resources:
                raise ConfigError(f"Unknown resource target {target!r}")
            return self.resources[key]
        if owner == "edge":
            if key not in self.edge_states:
                raise ConfigError(f"Unknown edge target {target!r}")
            return self.edge_states[key]
        if owner not in self.node_values or key not in self.node_values[owner]:
            raise ConfigError(f"Unknown state target {target!r}")
        return self.node_values[owner][key]

    def set_target(self, target: str, value: Any) -> None:
        owner, key = target.split(".", 1)
        self.resolve_target(target)
        if owner == "resource":
            self.resources[key] = value
        elif owner == "edge":
            self.edge_states[key] = bool(value)
        else:
            self.node_values[owner][key] = value

    def _clamp_variables(self) -> None:
        for node_id, node in self._node_specs.items():
            for variable_id in node.get("variables", {}):
                self._clamp_target(f"{node_id}.{variable_id}")

    def _clamp_target(self, target: str) -> None:
        owner, key = target.split(".", 1)
        if owner in {"resource", "edge"}:
            return
        spec = self._node_specs[owner]["variables"][key]
        value = self.node_values[owner][key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        if spec.get("hard_min") is not None:
            value = max(float(spec["hard_min"]), value)
        if spec.get("hard_max") is not None:
            value = min(float(spec["hard_max"]), value)
        self.node_values[owner][key] = value

    def _variable_change_events(
        self, before: Mapping[str, Mapping[str, Any]], source: str
    ) -> list[SimulatorEvent]:
        events: list[SimulatorEvent] = []
        for node_id, variables in self.node_values.items():
            for variable_id, new in variables.items():
                old = before[node_id][variable_id]
                if old != new:
                    events.extend(
                        self._target_change_event(
                            f"{node_id}.{variable_id}", old, new, source
                        )
                    )
        return events

    def _target_change_event(
        self, target: str, old: Any, new: Any, source: str
    ) -> list[SimulatorEvent]:
        if target.startswith("resource."):
            return [self._resource_event(target.split(".", 1)[1], old, new)]
        if target.startswith("edge."):
            edge_id = target.split(".", 1)[1]
            return [
                self._event(
                    "edge_changed",
                    {"edge_id": edge_id, "old": old, "new": new, "source": source},
                    {"edge_id": edge_id, "enabled": new},
                )
            ]
        public: dict[str, Any] = {"target": target}
        node_id, variable_id = target.split(".", 1)
        variable_spec = self._node_specs[node_id]["variables"][variable_id]
        if variable_spec.get("event_value_observable", False):
            public.update({"old": old, "new": new})
        return [
            self._event(
                "variable_changed",
                {"target": target, "old": old, "new": new, "source": source},
                public,
            )
        ]

    def _resource_event(self, resource_id: str, old: Any, new: Any) -> SimulatorEvent:
        public = {"target": resource_id}
        resource_spec = self.scenario.graph.get("resources", {}).get(resource_id, {})
        if resource_spec.get("observable", True):
            public.update({"old": old, "new": new})
        return self._event(
            "resource_changed",
            {"target": resource_id, "old": old, "new": new},
            public,
        )

    def _event(
        self,
        event_type: str,
        data: dict[str, Any],
        public_data: dict[str, Any],
        *,
        publicly_observable: bool = True,
    ) -> SimulatorEvent:
        return SimulatorEvent(
            event_type,
            self.step,
            self.time,
            data,
            public_data,
            publicly_observable,
        )
