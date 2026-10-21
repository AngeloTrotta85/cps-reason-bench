"""Human-readable scenario loading and validation."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from cps_reason_bench.versions import SCENARIO_SCHEMA_VERSION


class ConfigError(ValueError):
    """Raised when a scenario or run configuration is invalid."""


INTERACTION_MODES = {
    "periodic",
    "scheduled",
    "event_driven",
    "periodic_event",
    "scheduled_event",
}
ACTION_BATCH_SEMANTICS = {"sequential_per_command", "atomic"}
CONTEXT_MODES = {"current", "previous", "history", "summary"}
ACTION_OPERATIONS = {
    "noop",
    "adjust",
    "set",
    "toggle_edge",
    "inspect",
    "repair",
    "communicate",
}
FAULT_KINDS = {
    "sensor_bias",
    "sensor_stuck",
    "sensor_unavailable",
    "sensor_delay",
    "actuator_disabled",
    "actuator_scale",
    "actuator_remap",
    "edge_disabled",
    "drift_change",
}


def load_document(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")

    try:
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            document = json.loads(text)
        elif source.suffix.lower() in {".yaml", ".yml"}:
            document = yaml.safe_load(text)
        else:
            raise ConfigError("Configuration files must use .json, .yaml, or .yml")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load {source}: {exc}") from exc

    if not isinstance(document, dict):
        raise ConfigError(f"Top-level configuration in {source} must be a mapping")
    return document


@dataclass(frozen=True)
class InteractionConfig:
    mode: str = "periodic"
    interval_steps: int = 1
    initial_invocation: bool = True
    min_callback_steps: int = 1
    max_callback_steps: int = 20
    max_actions_per_interaction: int = 1
    action_batch_semantics: str = "sequential_per_command"
    allowed_event_types: tuple[str, ...] = ()
    include_event_values: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "InteractionConfig":
        data = dict(value or {})
        mode = str(data.get("mode", "periodic"))
        if mode not in INTERACTION_MODES:
            raise ConfigError(
                f"interaction.mode must be one of {sorted(INTERACTION_MODES)}, got {mode!r}"
            )
        interval = _positive_int(data.get("interval_steps", 1), "interaction.interval_steps")
        minimum = _positive_int(
            data.get("min_callback_steps", 1), "interaction.min_callback_steps"
        )
        maximum = _positive_int(
            data.get("max_callback_steps", 20), "interaction.max_callback_steps"
        )
        if maximum < minimum:
            raise ConfigError("interaction.max_callback_steps cannot be below the minimum")
        batch_semantics = str(
            data.get("action_batch_semantics", "sequential_per_command")
        )
        if batch_semantics not in ACTION_BATCH_SEMANTICS:
            raise ConfigError(
                "interaction.action_batch_semantics must be one of "
                f"{sorted(ACTION_BATCH_SEMANTICS)}, got {batch_semantics!r}"
            )
        return cls(
            mode=mode,
            interval_steps=interval,
            initial_invocation=bool(data.get("initial_invocation", True)),
            min_callback_steps=minimum,
            max_callback_steps=maximum,
            max_actions_per_interaction=_positive_int(
                data.get("max_actions_per_interaction", 1),
                "interaction.max_actions_per_interaction",
            ),
            action_batch_semantics=batch_semantics,
            allowed_event_types=tuple(str(item) for item in data.get("allowed_event_types", [])),
            include_event_values=bool(data.get("include_event_values", False)),
        )

    @property
    def allows_periodic(self) -> bool:
        return self.mode in {"periodic", "periodic_event"}

    @property
    def allows_callbacks(self) -> bool:
        return self.mode in {"scheduled", "scheduled_event"}

    @property
    def allows_subscriptions(self) -> bool:
        return self.mode in {"event_driven", "periodic_event", "scheduled_event"}


@dataclass(frozen=True)
class ContextConfig:
    mode: str = "current"
    history_limit: int = 3
    max_context_requests: int = 4
    include_action_outcomes: bool = True
    include_public_events: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ContextConfig":
        data = dict(value or {})
        mode = str(data.get("mode", "current"))
        if mode not in CONTEXT_MODES:
            raise ConfigError(f"context.mode must be one of {sorted(CONTEXT_MODES)}")
        return cls(
            mode=mode,
            history_limit=_positive_int(data.get("history_limit", 3), "context.history_limit"),
            max_context_requests=_positive_int(
                data.get("max_context_requests", 4), "context.max_context_requests"
            ),
            include_action_outcomes=bool(data.get("include_action_outcomes", True)),
            include_public_events=bool(data.get("include_public_events", False)),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    schema_version: int
    name: str
    description: str
    task_family: str
    difficulty: int
    difficulty_level: str
    difficulty_profile: dict[str, Any]
    seed: int
    max_steps: int
    time_step: float
    task: dict[str, Any]
    graph: dict[str, Any]
    sensors: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    faults: tuple[dict[str, Any], ...]
    constraints: tuple[dict[str, Any], ...]
    goals: tuple[dict[str, Any], ...]
    observability: dict[str, Any]
    interaction: InteractionConfig
    context: ContextConfig
    evaluation: dict[str, Any]
    baseline: dict[str, Any]
    reference: dict[str, Any]
    generation: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScenarioConfig":
        data = dict(value)
        version = int(data.get("schema_version", SCENARIO_SCHEMA_VERSION))
        if version != SCENARIO_SCHEMA_VERSION:
            raise ConfigError(f"Unsupported scenario schema_version: {version}")

        name = _required_string(data, "name")
        graph = _mapping(data.get("graph"), "graph")
        nodes = _mapping(graph.get("nodes"), "graph.nodes")
        if not nodes:
            raise ConfigError("graph.nodes must define at least one node")
        for node_id, node in nodes.items():
            _validate_path_component(str(node_id), f"graph.nodes.{node_id}")
            node_data = _mapping(node, f"graph.nodes.{node_id}")
            variables = _mapping(node_data.get("variables"), f"graph.nodes.{node_id}.variables")
            if not variables:
                raise ConfigError(f"Node {node_id!r} must define at least one variable")
            for variable_id, variable in variables.items():
                _validate_path_component(
                    str(variable_id), f"graph.nodes.{node_id}.variables.{variable_id}"
                )
                variable_data = _mapping(
                    variable, f"graph.nodes.{node_id}.variables.{variable_id}"
                )
                if "initial" not in variable_data:
                    raise ConfigError(f"Variable {node_id}.{variable_id} requires an initial value")

        resources = _mapping(graph.get("resources", {}), "graph.resources")
        for resource_id, resource in resources.items():
            _validate_path_component(str(resource_id), f"graph.resources.{resource_id}")
            resource_data = _mapping(resource, f"graph.resources.{resource_id}")
            if "initial" not in resource_data:
                raise ConfigError(f"Resource {resource_id!r} requires an initial value")
        graph["resources"] = resources

        edges = tuple(_mapping_list(graph.get("edges", []), "graph.edges"))
        _ensure_unique_ids(edges, "graph.edges")
        for edge in edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in nodes or target not in nodes:
                raise ConfigError(
                    f"Edge {edge['id']!r} must reference existing source and target nodes"
                )
            variable_id = edge.get("variable")
            if variable_id is not None and (
                variable_id not in nodes[source]["variables"]
                or variable_id not in nodes[target]["variables"]
            ):
                raise ConfigError(
                    f"Edge {edge['id']!r} couples unknown variable {variable_id!r}"
                )
        graph["edges"] = list(edges)

        sensors = tuple(_mapping_list(data.get("sensors", []), "sensors"))
        actions = tuple(_mapping_list(data.get("actions", []), "actions"))
        if not actions:
            raise ConfigError("A scenario must define at least one action")
        _ensure_unique_ids(sensors, "sensors")
        _ensure_unique_ids(actions, "actions")

        _validate_references(graph, sensors, actions)

        faults = tuple(_mapping_list(data.get("faults", []), "faults"))
        constraints = tuple(_mapping_list(data.get("constraints", []), "constraints"))
        goals = tuple(_mapping_list(data.get("goals", []), "goals"))
        _ensure_unique_ids(faults, "faults")
        _ensure_unique_ids(constraints, "constraints")
        _ensure_unique_ids(goals, "goals")
        _validate_faults_and_rules(graph, sensors, actions, faults, constraints, goals)

        time_step = float(data.get("time_step", 1.0))
        if time_step <= 0:
            raise ConfigError("time_step must be positive")

        difficulty = _positive_int(data.get("difficulty", 1), "difficulty")
        difficulty_level = str(data.get("difficulty_level", f"level_{difficulty}")).strip()
        if not difficulty_level:
            raise ConfigError("difficulty_level must be a non-empty string")
        evaluation = _mapping(data.get("evaluation", {}), "evaluation")
        observability = _mapping(data.get("observability", {}), "observability")
        interaction = InteractionConfig.from_mapping(data.get("interaction"))
        explicit_profile = _mapping(data.get("difficulty_profile", {}), "difficulty_profile")
        computed_profile = _compute_difficulty_profile(
            graph,
            sensors,
            actions,
            faults,
            constraints,
            goals,
            observability,
            interaction,
            evaluation,
        )
        for key, computed_value in computed_profile.items():
            if key in explicit_profile and explicit_profile[key] != computed_value:
                raise ConfigError(
                    f"difficulty_profile.{key} is {explicit_profile[key]!r}, "
                    f"but the scenario structure implies {computed_value!r}"
                )
        difficulty_profile = {**explicit_profile, **computed_profile}

        reference = _mapping(data.get("reference", {}), "reference")
        if reference:
            reference_kind = reference.get("kind", "demonstration")
            if reference_kind not in {
                "demonstration",
                "feasible_reference",
                "oracle",
                "optimal",
            }:
                raise ConfigError(f"Unsupported reference.kind: {reference_kind!r}")
            if reference_kind == "optimal" and not reference.get("evidence"):
                raise ConfigError("An optimal reference claim requires explicit evidence")

        normalized = copy.deepcopy(data)
        normalized["schema_version"] = version
        normalized["graph"] = copy.deepcopy(graph)
        normalized["difficulty"] = difficulty
        normalized["difficulty_level"] = difficulty_level
        normalized["difficulty_profile"] = copy.deepcopy(difficulty_profile)

        return cls(
            schema_version=version,
            name=name,
            description=str(data.get("description", "")),
            task_family=str(data.get("task_family", "general")),
            difficulty=difficulty,
            difficulty_level=difficulty_level,
            difficulty_profile=difficulty_profile,
            seed=int(data.get("seed", 0)),
            max_steps=_positive_int(data.get("max_steps", 10), "max_steps"),
            time_step=time_step,
            task=_mapping(data.get("task", {}), "task"),
            graph=graph,
            sensors=sensors,
            actions=actions,
            faults=faults,
            constraints=constraints,
            goals=goals,
            observability=observability,
            interaction=interaction,
            context=ContextConfig.from_mapping(data.get("context")),
            evaluation=evaluation,
            baseline=_mapping(data.get("baseline", {}), "baseline"),
            reference=reference,
            generation=_mapping(data.get("generation", {}), "generation"),
            raw=normalized,
        )

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(str(action["id"]) for action in self.actions)

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(str(sensor["id"]) for sensor in self.sensors)

    def to_mapping(self) -> dict[str, Any]:
        """Return a complete, serializable scenario definition for replay/export."""

        value = copy.deepcopy(self.raw)
        value["difficulty"] = self.difficulty
        value["difficulty_level"] = self.difficulty_level
        value["difficulty_profile"] = copy.deepcopy(self.difficulty_profile)
        if self.reference:
            value["reference"] = copy.deepcopy(self.reference)
        if self.generation:
            value["generation"] = copy.deepcopy(self.generation)
        return value


def load_scenario(path: str | Path) -> ScenarioConfig:
    return ScenarioConfig.from_mapping(load_document(path))


def dump_scenario(scenario: ScenarioConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".json":
        destination.write_text(
            json.dumps(scenario.to_mapping(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    elif destination.suffix.lower() in {".yaml", ".yml"}:
        destination.write_text(
            yaml.safe_dump(scenario.to_mapping(), sort_keys=False),
            encoding="utf-8",
        )
    else:
        raise ConfigError("Exported scenarios must use .json, .yaml, or .yml")
    return destination


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return dict(value)


def _mapping_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list")
    return [_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ConfigError(f"{name} must be positive")
    return result


def _ensure_unique_ids(items: tuple[dict[str, Any], ...], name: str) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        item_id = _required_string(item, "id")
        if item_id in seen:
            raise ConfigError(f"Duplicate id {item_id!r} in {name}")
        seen.add(item_id)


def _validate_path_component(value: str, name: str) -> None:
    if not value or "." in value:
        raise ConfigError(f"{name} must be non-empty and cannot contain '.'")


def _known_targets(graph: Mapping[str, Any]) -> set[str]:
    targets = _node_targets(graph)
    targets.update(f"resource.{resource_id}" for resource_id in graph["resources"])
    targets.update(f"edge.{edge['id']}" for edge in graph["edges"])
    return targets


def _node_targets(graph: Mapping[str, Any]) -> set[str]:
    return {
        f"{node_id}.{variable_id}"
        for node_id, node in graph["nodes"].items()
        for variable_id in node["variables"]
    }


def _validate_references(
    graph: Mapping[str, Any],
    sensors: tuple[dict[str, Any], ...],
    actions: tuple[dict[str, Any], ...],
) -> None:
    targets = _known_targets(graph)
    sensor_ids = {str(sensor["id"]) for sensor in sensors}
    action_ids = {str(action["id"]) for action in actions}
    edge_ids = {str(edge["id"]) for edge in graph["edges"]}
    resources = set(graph["resources"])

    for sensor in sensors:
        target = sensor.get("target")
        if target not in targets:
            raise ConfigError(f"Sensor {sensor['id']!r} references unknown target {target!r}")

    for action in actions:
        operation = str(action.get("operation", "noop"))
        if operation not in ACTION_OPERATIONS:
            raise ConfigError(f"Action {action['id']!r} has unsupported operation {operation!r}")
        if action.get("failure_feedback", "full") not in {"full", "generic"}:
            raise ConfigError(
                f"Action {action['id']!r} failure_feedback must be 'full' or 'generic'"
            )
        effect_delay = action.get("effect_delay_steps", 0)
        if (
            isinstance(effect_delay, bool)
            or not isinstance(effect_delay, int)
            or effect_delay < 0
        ):
            raise ConfigError(
                f"Action {action['id']!r} effect_delay_steps must be a non-negative integer"
            )
        if effect_delay and operation not in {"set", "adjust"}:
            raise ConfigError(
                f"Action {action['id']!r} delays an unsupported operation {operation!r}"
            )
        target = action.get("target")
        if operation in {"adjust", "set"} and target not in targets:
            raise ConfigError(f"Action {action['id']!r} references unknown target {target!r}")
        if operation == "toggle_edge" and target not in edge_ids:
            raise ConfigError(f"Action {action['id']!r} references unknown edge {target!r}")
        inspectable = targets | sensor_ids | action_ids | edge_ids
        if operation in {"inspect", "repair"} and target not in inspectable:
            raise ConfigError(f"Action {action['id']!r} references unknown component {target!r}")
        for resource_id in action.get("costs", {}):
            if resource_id not in resources:
                raise ConfigError(
                    f"Action {action['id']!r} consumes unknown resource {resource_id!r}"
                )
        parameters = _mapping(action.get("parameters", {}), f"action {action['id']} parameters")
        for parameter_id, parameter_schema in parameters.items():
            _mapping(
                parameter_schema,
                f"action {action['id']} parameter {parameter_id}",
            )
        for condition in action.get("preconditions", []):
            _validate_condition(condition, targets, f"action {action['id']} precondition")
        for effect in action.get("additional_effects", []):
            effect_data = _mapping(effect, f"action {action['id']} additional effect")
            effect_operation = effect_data.get("operation")
            if effect_operation not in {"set", "adjust"}:
                raise ConfigError(
                    f"Action {action['id']!r} has unsupported additional effect "
                    f"operation {effect_operation!r}"
                )
            if effect_data.get("target") not in targets:
                raise ConfigError(
                    f"Action {action['id']!r} has an effect on an unknown target"
                )
            if effect_operation == "adjust" and (
                isinstance(effect_data.get("amount", 0.0), bool)
                or not isinstance(effect_data.get("amount", 0.0), (int, float))
            ):
                raise ConfigError(
                    f"Action {action['id']!r} has a non-numeric additional adjustment"
                )


def _validate_faults_and_rules(
    graph: Mapping[str, Any],
    sensors: tuple[dict[str, Any], ...],
    actions: tuple[dict[str, Any], ...],
    faults: tuple[dict[str, Any], ...],
    constraints: tuple[dict[str, Any], ...],
    goals: tuple[dict[str, Any], ...],
) -> None:
    targets = _known_targets(graph)
    node_targets = _node_targets(graph)
    sensor_ids = {str(sensor["id"]) for sensor in sensors}
    action_ids = {str(action["id"]) for action in actions}
    edge_ids = {str(edge["id"]) for edge in graph["edges"]}
    for fault in faults:
        kind = str(fault.get("kind"))
        target = str(fault.get("target"))
        if kind not in FAULT_KINDS:
            raise ConfigError(f"Fault {fault['id']!r} has unsupported kind {kind!r}")
        expected = (
            sensor_ids
            if kind.startswith("sensor_")
            else action_ids
            if kind.startswith("actuator_")
            else edge_ids
            if kind == "edge_disabled"
            else node_targets
            if kind == "drift_change"
            else targets
        )
        if target not in expected:
            raise ConfigError(f"Fault {fault['id']!r} references unknown target {target!r}")
        if kind == "actuator_remap" and fault.get("value") not in targets:
            raise ConfigError(
                f"Fault {fault['id']!r} remaps to unknown state target {fault.get('value')!r}"
            )
        if int(fault.get("at_step", 0)) < 0:
            raise ConfigError(f"Fault {fault['id']!r} cannot occur before step 0")

    for constraint in constraints:
        kind = constraint.get("kind", "condition")
        if kind == "bounds":
            if constraint.get("target") not in targets:
                raise ConfigError(f"Constraint {constraint['id']!r} has an unknown target")
        elif kind == "conditional":
            _validate_condition(constraint.get("if"), targets, f"constraint {constraint['id']} if")
            _validate_condition(
                constraint.get("then"), targets, f"constraint {constraint['id']} then"
            )
        elif kind == "condition":
            _validate_condition(
                constraint.get("condition"), targets, f"constraint {constraint['id']}"
            )
        else:
            raise ConfigError(f"Constraint {constraint['id']!r} has unsupported kind {kind!r}")

    for goal in goals:
        if "condition" in goal:
            _validate_condition(goal["condition"], targets, f"goal {goal['id']}")
        elif "all" in goal and isinstance(goal["all"], list):
            for condition in goal["all"]:
                _validate_condition(condition, targets, f"goal {goal['id']}")
        elif "any" in goal and isinstance(goal["any"], list) and goal["any"]:
            for condition in goal["any"]:
                _validate_condition(condition, targets, f"goal {goal['id']}")
        else:
            raise ConfigError(f"Goal {goal['id']!r} must define condition, all, or any")


def _validate_condition(value: Any, targets: set[str], name: str) -> None:
    condition = _mapping(value, name)
    if condition.get("target") not in targets:
        raise ConfigError(f"{name} references unknown target {condition.get('target')!r}")
    if condition.get("operator", "eq") not in {"eq", "ne", "gt", "ge", "lt", "le", "in"}:
        raise ConfigError(f"{name} uses an unsupported operator")
    if "value" not in condition:
        raise ConfigError(f"{name} requires a value")


def _compute_difficulty_profile(
    graph: Mapping[str, Any],
    sensors: tuple[dict[str, Any], ...],
    actions: tuple[dict[str, Any], ...],
    faults: tuple[dict[str, Any], ...],
    constraints: tuple[dict[str, Any], ...],
    goals: tuple[dict[str, Any], ...],
    observability: Mapping[str, Any],
    interaction: InteractionConfig,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    node_count = len(graph["nodes"])
    state_targets = {
        f"{node_id}.{variable_id}"
        for node_id, node in graph["nodes"].items()
        for variable_id in node["variables"]
    }
    observable_targets = {
        str(sensor["target"])
        for sensor in sensors
        if sensor.get("observable", True)
    }
    hidden_count = 0 if observability.get("full_state", False) else len(
        state_targets - observable_targets
    )
    valid_actions = [action for action in actions if action.get("role") != "distractor"]
    distractor_actions = [action for action in actions if action.get("role") == "distractor"]
    callback_window = evaluation.get("callback_target_window")
    callback_tolerance = (
        int(callback_window[1]) - int(callback_window[0])
        if isinstance(callback_window, list) and len(callback_window) == 2
        else None
    )
    reference_depth = evaluation.get("reference_action_count")
    if reference_depth is None:
        reference_depth = len(evaluation.get("expected_action_sequence", []))
    misleading_count = int(evaluation.get("misleading_observation_count", 0))
    if not misleading_count:
        misleading_count = sum(
            fault.get("kind") in {"sensor_stuck", "sensor_bias", "sensor_delay"}
            for fault in faults
        )
    resource_profile = {
        resource_id: spec.get("initial")
        for resource_id, spec in graph.get("resources", {}).items()
    }
    reference_action_ids = list(evaluation.get("expected_action_sequence", []))
    if not reference_action_ids:
        reference_action_ids = [
            *evaluation.get("required_inspections", []),
            *evaluation.get("required_actions", []),
        ]
    action_by_id = {str(action["id"]): action for action in actions}
    reference_costs = {
        resource_id: sum(
            float(action_by_id.get(action_id, {}).get("costs", {}).get(resource_id, 0.0))
            for action_id in reference_action_ids
        )
        for resource_id in resource_profile
    }
    resource_scarcity = {
        resource_id: {
            "initial": initial,
            "reference_cost": reference_costs[resource_id],
            "margin": (
                float(initial) - reference_costs[resource_id]
                if isinstance(initial, (int, float))
                else None
            ),
            "initial_to_reference_cost_ratio": (
                float(initial) / reference_costs[resource_id]
                if isinstance(initial, (int, float))
                and reference_costs[resource_id] > 0
                else None
            ),
        }
        for resource_id, initial in resource_profile.items()
    }
    return {
        "number_of_nodes": node_count,
        "number_of_state_variables": len(state_targets),
        "number_of_valid_actions": len(valid_actions),
        "number_of_distractor_actions": len(distractor_actions),
        "minimum_required_planning_depth": int(reference_depth or 0),
        "number_of_prerequisites": sum(
            len(action.get("preconditions", [])) for action in actions
        ),
        "number_of_interacting_goals": len(goals),
        "number_of_constraints": len(constraints),
        "number_of_hidden_state_variables": hidden_count,
        "maximum_observation_delay_steps": max(
            (int(sensor.get("delay_steps", 0)) for sensor in sensors), default=0
        ),
        "maximum_sensor_noise_std": max(
            (float(sensor.get("noise_std", 0.0)) for sensor in sensors), default=0.0
        ),
        "number_of_faults": len(faults),
        "initial_resources": resource_profile,
        "resource_scarcity": resource_scarcity,
        "callback_timing_tolerance_steps": callback_tolerance,
        "number_of_candidate_event_subscriptions": len(interaction.allowed_event_types),
        "number_of_misleading_observations": misleading_count,
        "number_of_possible_safety_rule_violations": len(constraints),
    }
