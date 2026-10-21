"""Deterministic graph-dependent scenarios for CPS Reason Bench 0.5.0."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from cps_reason_bench.config import ConfigError, ScenarioConfig
from cps_reason_bench.versions import content_hash


GRAPH_V2_GENERATOR_ID = "cps_reason_graph_v2_generator"
GRAPH_V2_GENERATOR_VERSION = "1.0.0"
GRAPH_V2_SUITE_NAME = "cps_reason_graph_v2"
GRAPH_V2_SUITE_VERSION = "1.0.0"

GRAPH_V2_FAMILIES = (
    "prospective_reasoning",
    "planning",
    "interference",
    "conditional_safety",
    "adaptation",
    "temporal_tracking",
)

GRAPH_V2_LEVELS = (
    "graph_v2_entry_1",
    "graph_v2_entry_2",
    "graph_v2_entry_3",
    "graph_v2_bridge",
    "graph_v2_core_1",
    "graph_v2_core_2",
    "graph_v2_core_3",
    "graph_v2_core_4",
)

GRAPH_V2_LEVEL_ALIASES = {
    "graph_v2_entry_1": "Foundation 1",
    "graph_v2_entry_2": "Foundation 2",
    "graph_v2_entry_3": "Foundation 3",
    "graph_v2_bridge": "Foundation Bridge",
    "graph_v2_core_1": "Core 1",
    "graph_v2_core_2": "Core 2",
    "graph_v2_core_3": "Core 3",
    "graph_v2_core_4": "Core 4",
}

GRAPH_V2_PAIR_TYPES = (
    "edge_direction",
    "coupling_sign",
    "edge_state",
    "coupling_strength",
)

DOMAIN_BY_FAMILY = {
    "prospective_reasoning": "thermal_network",
    "planning": "process_chain",
    "interference": "communication_network",
    "conditional_safety": "power_service_dependency",
    "adaptation": "reconfigurable_process_network",
    "temporal_tracking": "delayed_transport_network",
}

CONSTRUCT_BY_FAMILY = {
    "prospective_reasoning": "retain and execute a future graph-dependent intention",
    "planning": "order prerequisite-sensitive actions across connected components",
    "interference": "resolve unreliable evidence using topology and remote corroboration",
    "conditional_safety": "verify a hidden safety consequent before graph-dependent actuation",
    "adaptation": "revise a route strategy after an actuator or edge regime change",
    "temporal_tracking": "attribute a delayed downstream response to an upstream action",
}


@dataclass(frozen=True)
class GraphBlueprint:
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    sink_id: str
    reference_graph_actions: tuple[str, ...]
    wrong_graph_actions: tuple[str, ...]
    pulse_actions: tuple[str, ...]
    goal_threshold: float
    topology_type: str
    directionality: str
    coupling_type: str
    topology_state: str
    decision_property: str
    pair_id: str | None
    pair_member: str | None
    pair_type: str | None


def graph_v2_cell_index(family: str, level: str) -> int:
    try:
        return GRAPH_V2_FAMILIES.index(family) * len(GRAPH_V2_LEVELS) + GRAPH_V2_LEVELS.index(level)
    except ValueError as exc:
        raise ConfigError(
            f"Unsupported graph-v2 family/level: {family!r}/{level!r}"
        ) from exc


def generate_graph_v2_scenario(
    task_family: str,
    level: str,
    generation_seed: int,
    parameters: Mapping[str, Any] | None = None,
    template_path: str | None = None,
) -> ScenarioConfig:
    """Generate one graph-v2 semantic condition without touching old generators."""

    if task_family not in GRAPH_V2_FAMILIES:
        raise ConfigError(f"Unknown graph-v2 family {task_family!r}")
    if level not in GRAPH_V2_LEVELS:
        raise ConfigError(f"Unknown graph-v2 level {level!r}")
    if template_path is not None:
        raise ConfigError("graph-v2 scenarios are self-contained and do not use templates")
    data = dict(parameters or {})
    instance_index = int(data.get("instance_index", -1))
    if instance_index not in range(5):
        raise ConfigError("graph-v2 instance_index must be in 0..4")
    cell_index = graph_v2_cell_index(task_family, level)
    level_index = GRAPH_V2_LEVELS.index(level)
    blueprint = _build_blueprint(task_family, cell_index, instance_index)
    scenario = _base_scenario(
        task_family,
        level,
        level_index,
        instance_index,
        cell_index,
        generation_seed,
        blueprint,
    )
    family_builder = {
        "prospective_reasoning": _configure_prospective,
        "planning": _configure_planning,
        "interference": _configure_interference,
        "conditional_safety": _configure_conditional,
        "adaptation": _configure_adaptation,
        "temporal_tracking": _configure_temporal,
    }[task_family]
    family_builder(scenario, blueprint, level_index, instance_index, cell_index)
    _configure_observability(scenario, blueprint, level_index, instance_index)
    _finalize_reference_and_resources(scenario)
    _record_generation_metadata(
        scenario,
        blueprint,
        task_family,
        level,
        level_index,
        instance_index,
        cell_index,
        generation_seed,
    )
    normalized = ScenarioConfig.from_mapping(scenario)
    value = normalized.to_mapping()
    semantic_source = copy.deepcopy(value)
    semantic_source.get("generation", {}).pop("semantic_scenario_hash", None)
    semantic_source.get("generation", {}).pop("physical_scenario_hash", None)
    value["generation"]["semantic_scenario_hash"] = content_hash(semantic_source)
    value["generation"]["physical_scenario_hash"] = _physical_hash(value)
    return ScenarioConfig.from_mapping(value)


def graph_v2_factor_record(
    scenario: ScenarioConfig,
    *,
    suite_entry_id: str,
    simulator_seed: int,
) -> dict[str, Any]:
    """Return the machine-readable graph coverage row for a materialized scenario."""

    profile = dict(scenario.difficulty_profile.get("graph_v2", {}))
    edges = list(scenario.graph.get("edges", []))
    topology_change_actions = [
        str(action["id"])
        for action in scenario.actions
        if action.get("operation") in {"toggle_edge", "repair"}
    ]
    edge_faults = [
        str(fault["id"])
        for fault in scenario.faults
        if fault.get("kind") == "edge_disabled"
    ]
    delayed_effects = sum(
        int(action.get("effect_delay_steps", 0)) > 0 for action in scenario.actions
    )
    return {
        "suite_entry_id": suite_entry_id,
        "scenario_name": scenario.name,
        "family": scenario.task_family,
        "difficulty": scenario.difficulty_level,
        "logical_position": int(profile["logical_position"]),
        "difficulty_alias": str(profile["difficulty_alias"]),
        "semantic_instance": str(profile["semantic_instance"]),
        "simulator_seed": int(simulator_seed),
        "node_count": len(scenario.graph["nodes"]),
        "edge_count": len(edges),
        "topology_type": str(profile["topology_type"]),
        "directionality": str(profile["directionality"]),
        "coupling_type": str(profile["coupling_type"]),
        "topology_state": str(profile["topology_state"]),
        "observability_type": str(profile["observability_type"]),
        "fault_locality": str(profile["fault_locality"]),
        "graph_dependent_decision_property": str(profile["decision_property"]),
        "counterfactual_pair_id": profile.get("counterfactual_pair_id") or "",
        "counterfactual_pair_member": profile.get("counterfactual_pair_member") or "",
        "counterfactual_pair_type": profile.get("counterfactual_pair_type") or "",
        "enabled_edge_count": sum(bool(edge.get("enabled", True)) for edge in edges),
        "directed_edge_count": sum(bool(edge.get("directed", False)) for edge in edges),
        "undirected_edge_count": sum(not bool(edge.get("directed", False)) for edge in edges),
        "coupled_edge_count": sum(float(edge.get("coupling", 0.0)) != 0.0 for edge in edges),
        "topology_change_action_count": len(topology_change_actions),
        "edge_fault_count": len(edge_faults),
        "hidden_edge_state_count": sum(
            not bool(edge.get("state_observable", True)) for edge in edges
        ),
        "sensor_count": len(scenario.sensors),
        "action_count": len(scenario.actions),
        "max_actions_per_interaction": scenario.interaction.max_actions_per_interaction,
        "action_batch_semantics": getattr(
            scenario.interaction, "action_batch_semantics", "sequential_per_command"
        ),
        "delayed_effect_action_count": delayed_effects,
        "fault_count": len(scenario.faults),
        "constraint_count": len(scenario.constraints),
        "goal_count": len(scenario.goals),
        "horizon": scenario.max_steps,
        "construct": str(profile["family_construct"]),
        "graph_demand": str(profile["graph_demand"]),
        "difficulty_manipulation": str(profile["difficulty_manipulation"]),
        "potential_confound": str(profile["potential_confound"]),
    }


def _build_blueprint(
    family: str, cell_index: int, instance_index: int
) -> GraphBlueprint:
    labels = _domain_labels(family)
    if instance_index == 0:
        nodes = {"sink": _node(labels["sink"])}
        actions = [
            _set_action("pulse_a", "sink.signal", 8.0, "Apply the useful local command."),
            _set_action("pulse_b", "sink.signal", -8.0, "Apply the opposite local command."),
        ]
        return GraphBlueprint(
            nodes,
            [],
            actions,
            "sink",
            ("pulse_a",),
            ("pulse_b",),
            ("pulse_a",),
            4.0,
            "single_component",
            "none",
            "none",
            "static",
            "topology_irrelevant_baseline",
            None,
            None,
            None,
        )

    if instance_index in {1, 2}:
        pair_type = GRAPH_V2_PAIR_TYPES[cell_index % len(GRAPH_V2_PAIR_TYPES)]
        member = "A" if instance_index == 1 else "B"
        return _counterfactual_blueprint(family, cell_index, pair_type, member, labels)

    if instance_index == 3:
        if cell_index % 2 == 0:
            nodes = {
                "source_a": _node(labels["source_a"]),
                "relay": _node(labels["relay"]),
                "sink": _node(labels["sink"]),
            }
            edges = [
                _edge("e_primary", "source_a", "relay", 0.75, directed=True),
                _edge("e_secondary", "relay", "sink", 0.75, directed=True),
            ]
            actions = [
                _set_action("pulse_a", "source_a.signal", 10.0, "Energize source A."),
                _set_action("pulse_b", "relay.signal", -10.0, "Apply an opposing relay pulse."),
            ]
            return GraphBlueprint(
                nodes,
                edges,
                actions,
                "sink",
                ("pulse_a",),
                ("pulse_b",),
                ("pulse_a",),
                4.0,
                "chain",
                "directed",
                "short_propagation_chain",
                "static",
                "topology_changes_action",
                None,
                None,
                None,
            )
        nodes = {
            "source_a": _node(labels["source_a"]),
            "source_b": _node(labels["source_b"]),
            "sink": _node(labels["sink"]),
        }
        edges = [
            _edge("e_primary", "source_a", "sink", 0.6, directed=True),
            _edge("e_secondary", "source_b", "sink", 0.6, directed=True),
        ]
        actions = [
            _set_action("pulse_a", "source_a.signal", 10.0, "Energize source A."),
            _set_action("pulse_b", "source_b.signal", 10.0, "Energize source B."),
        ]
        return GraphBlueprint(
            nodes,
            edges,
            actions,
            "sink",
            ("pulse_a", "pulse_b"),
            (),
            ("pulse_a", "pulse_b"),
            10.0,
            "branch_tree",
            "directed",
            "multiple_incoming",
            "static",
            "topology_changes_action",
            None,
            None,
            None,
        )

    nodes = {
        "source_a": _node(labels["source_a"]),
        "source_b": _node(labels["source_b"]),
        "relay_a": _node(f"{labels['relay']} A"),
        "relay_b": _node(f"{labels['relay']} B"),
        "sink": _node(labels["sink"]),
        "aux": _node(labels["aux"], signal=-4.0),
    }
    restore_path = cell_index % 2 == 0
    edges = [
        _edge("e_primary", "source_a", "relay_a", 0.7, directed=True),
        _edge(
            "e_path_a_out",
            "relay_a",
            "sink",
            0.7,
            directed=True,
            enabled=not restore_path,
        ),
        _edge("e_path_b_in", "source_b", "relay_b", 0.7, directed=True),
        _edge("e_path_b_out", "relay_b", "sink", 0.7, directed=True),
        _edge("e_cross", "relay_a", "relay_b", 0.15, directed=False),
        _edge("e_aux", "aux", "sink", 0.1, directed=True),
    ]
    actions = [
        _set_action("pulse_a", "source_a.signal", 12.0, "Energize route A."),
        _set_action("pulse_b", "source_b.signal", 12.0, "Energize route B."),
        _toggle_action("restore_path_a", "e_path_a_out", True, "Restore route A."),
        _toggle_action("disable_cross_link", "e_cross", False, "Isolate the cross link."),
    ]
    reference = ("restore_path_a", "pulse_a") if restore_path else ("pulse_b",)
    wrong = ("pulse_a",) if not restore_path else ("pulse_b",)
    return GraphBlueprint(
        nodes,
        edges,
        actions,
        "sink",
        reference,
        wrong,
        (reference[-1],),
        4.0,
        "alternative_path" if restore_path else "sparse_mesh",
        "mixed",
        "competing_influences",
        "edge_restored" if restore_path else "alternative_route",
        "edge_failure_changes_action" if restore_path else "topology_changes_action",
        None,
        None,
        None,
    )


def _counterfactual_blueprint(
    family: str,
    cell_index: int,
    pair_type: str,
    member: str,
    labels: Mapping[str, str],
) -> GraphBlueprint:
    pair_id = f"graph_v2_cf_{family}_{GRAPH_V2_LEVELS[cell_index % 8]}"
    if pair_type == "edge_direction":
        nodes = {
            "source_a": _node(labels["source_a"]),
            "source_b": _node(labels["source_b"]),
            "sink": _node(labels["sink"]),
        }
        if member == "A":
            edges = [
                _edge("e_primary", "source_a", "sink", 0.6, directed=True),
                _edge("e_secondary", "sink", "source_b", 0.6, directed=True),
            ]
            correct, wrong = "pulse_a", "pulse_b"
        else:
            edges = [
                _edge("e_primary", "sink", "source_a", 0.6, directed=True),
                _edge("e_secondary", "source_b", "sink", 0.6, directed=True),
            ]
            correct, wrong = "pulse_b", "pulse_a"
        actions = [
            _set_action("pulse_a", "source_a.signal", 10.0, "Energize source A."),
            _set_action("pulse_b", "source_b.signal", 10.0, "Energize source B."),
        ]
        return GraphBlueprint(
            nodes,
            edges,
            actions,
            "sink",
            (correct,),
            (wrong,),
            ("pulse_a", "pulse_b"),
            4.0,
            "chain",
            "directed",
            "short_propagation_chain",
            "static",
            "direction_changes_action",
            pair_id,
            member,
            pair_type,
        )

    nodes = {
        "source": _node(labels["source_a"]),
        "sink": _node(labels["sink"]),
    }
    if pair_type == "coupling_sign":
        coupling = 0.5 if member == "A" else -0.5
        edges = [_edge("e_primary", "source", "sink", coupling, directed=False)]
        actions = [
            _set_action("pulse_positive", "source.signal", 10.0, "Apply a positive source pulse."),
            _set_action("pulse_negative", "source.signal", -10.0, "Apply a negative source pulse."),
        ]
        correct = "pulse_positive" if member == "A" else "pulse_negative"
        wrong = "pulse_negative" if member == "A" else "pulse_positive"
        return GraphBlueprint(
            nodes,
            edges,
            actions,
            "sink",
            (correct,),
            (wrong,),
            ("pulse_positive", "pulse_negative"),
            4.0,
            "chain",
            "undirected",
            "single_edge",
            "static",
            "coupling_changes_action",
            pair_id,
            member,
            pair_type,
        )

    if pair_type == "edge_state":
        enabled = member == "A"
        edges = [
            _edge("e_primary", "source", "sink", 0.6, directed=True, enabled=enabled)
        ]
        actions = [
            _set_action("pulse_source", "source.signal", 10.0, "Energize the source."),
            _toggle_action("restore_primary", "e_primary", True, "Restore the transfer edge."),
        ]
        reference = ("pulse_source",) if enabled else ("restore_primary", "pulse_source")
        wrong = ("restore_primary", "pulse_source") if enabled else ("pulse_source",)
        return GraphBlueprint(
            nodes,
            edges,
            actions,
            "sink",
            reference,
            wrong,
            ("pulse_source",),
            4.0,
            "chain",
            "directed",
            "single_edge",
            "static" if enabled else "edge_disabled",
            "edge_failure_changes_action",
            pair_id,
            member,
            pair_type,
        )

    coupling = 0.6 if member == "A" else 0.2
    edges = [_edge("e_primary", "source", "sink", coupling, directed=True)]
    actions = [
        _set_action("pulse_small", "source.signal", 10.0, "Apply the smaller source pulse."),
        _set_action("pulse_large", "source.signal", 30.0, "Apply the larger source pulse."),
    ]
    correct = "pulse_small" if member == "A" else "pulse_large"
    wrong = "pulse_large" if member == "A" else "pulse_small"
    return GraphBlueprint(
        nodes,
        edges,
        actions,
        "sink",
        (correct,),
        (wrong,),
        ("pulse_small", "pulse_large"),
        4.0,
        "chain",
        "directed",
        "single_edge",
        "static",
        "coupling_changes_action",
        pair_id,
        member,
        pair_type,
    )


def _base_scenario(
    family: str,
    level: str,
    level_index: int,
    instance_index: int,
    cell_index: int,
    generation_seed: int,
    blueprint: GraphBlueprint,
) -> dict[str, Any]:
    max_actions, batch_semantics = _batch_configuration(
        level_index, instance_index, cell_index
    )
    actions = copy.deepcopy(blueprint.actions)
    for action in actions:
        if action.get("operation") != "noop":
            action.setdefault("costs", {})["energy"] = 0.5
    actions.append({"id": "wait", "operation": "noop", "description": "Take no action."})
    difficulty_manipulations = (
        "visible direct construct",
        "one hidden fact or dependency",
        "longer dependency or conflicting evidence",
        "combined family and modest graph demand",
        "controlled graph-dependent decision",
        "remote or delayed evidence with reduced slack",
        "hidden edge state or interacting influence",
        "overlapping temporal/fault demand and tight recovery",
    )
    graph_demand = {
        "topology_irrelevant_baseline": "local control without a relational dependency",
        "topology_changes_action": "choose the component/path that can influence the sink",
        "direction_changes_action": "identify the upstream source from edge direction",
        "coupling_changes_action": "select action magnitude or sign from coupling",
        "edge_failure_changes_action": "restore or reroute before applying control",
    }[blueprint.decision_property]
    sink_upper_limit = 40.0 if blueprint.pair_type in {
        "coupling_sign",
        "coupling_strength",
    } else 20.0
    return {
        "schema_version": 1,
        "name": f"graph_v2_{family}_{level}_i{instance_index + 1:02d}",
        "description": (
            f"Graph-v2 {GRAPH_V2_LEVEL_ALIASES[level]} condition "
            f"{instance_index + 1} for {family}."
        ),
        "task_family": family,
        "difficulty": level_index + 1,
        "difficulty_level": level,
        "difficulty_profile": {
            "graph_v2": {
                "benchmark_release": "0.5.0",
                "logical_position": level_index + 1,
                "difficulty_alias": GRAPH_V2_LEVEL_ALIASES[level],
                "semantic_instance": f"i{instance_index + 1:02d}",
                "family_construct": CONSTRUCT_BY_FAMILY[family],
                "graph_demand": graph_demand,
                "difficulty_manipulation": difficulty_manipulations[level_index],
                "potential_confound": (
                    "graph search may dominate unless family-specific timing, evidence, "
                    "rule, adaptation, or attribution behavior is also required"
                ),
                "topology_type": blueprint.topology_type,
                "directionality": blueprint.directionality,
                "coupling_type": blueprint.coupling_type,
                "topology_state": blueprint.topology_state,
                "decision_property": blueprint.decision_property,
                "counterfactual_pair_id": blueprint.pair_id,
                "counterfactual_pair_member": blueprint.pair_member,
                "counterfactual_pair_type": blueprint.pair_type,
            }
        },
        "seed": generation_seed,
        "max_steps": 7 + level_index // 2,
        "time_step": 1.0,
        "task": {
            "instructions": _task_instructions(family),
            "objectives": [
                "Reach the declared downstream service goal.",
                "Use the observed graph and public evidence when selecting actions.",
                "Respect the safety limit and resource budget.",
            ],
            "terminate_on_goals": family != "adaptation",
        },
        "graph": {
            "nodes": copy.deepcopy(blueprint.nodes),
            "edges": copy.deepcopy(blueprint.edges),
            "resources": {
                "energy": {"initial": 10.0, "min": 0.0, "observable": True}
            },
        },
        "sensors": _base_sensors(blueprint.nodes),
        "actions": actions,
        "faults": [],
        "constraints": [
            {
                "id": "sink_upper_limit",
                "kind": "condition",
                "condition": {
                    "target": f"{blueprint.sink_id}.signal",
                    "operator": "le",
                    "value": sink_upper_limit,
                },
            }
        ],
        "goals": [
            {
                "id": "downstream_service",
                "status_observable": True,
                "condition": {
                    "target": f"{blueprint.sink_id}.signal",
                    "operator": "ge",
                    "value": blueprint.goal_threshold,
                },
            }
        ],
        "observability": {"full_state": False},
        "interaction": {
            "mode": "periodic",
            "interval_steps": 1,
            "max_actions_per_interaction": max_actions,
            "action_batch_semantics": batch_semantics,
        },
        "context": {
            "mode": "history" if level_index >= 4 else "previous",
            "history_limit": min(6, 2 + level_index),
            "include_action_outcomes": True,
            "include_public_events": level_index >= 2,
        },
        "evaluation": {
            "success_requires_safety": True,
            "success_requires_graph_decision": True,
            "graph_v2": {
                "expected_exact_action_sequence": list(blueprint.reference_graph_actions),
                "counterfactual_wrong_action_sequence": list(blueprint.wrong_graph_actions),
                "correct_graph_actions": list(blueprint.reference_graph_actions),
                "decision_property": blueprint.decision_property,
                "counterfactual_pair_id": blueprint.pair_id,
                "counterfactual_pair_member": blueprint.pair_member,
                "counterfactual_pair_type": blueprint.pair_type,
            },
        },
        "baseline": {"script": [], "default_decision": _decision([])},
        "reference": {
            "kind": "feasible_reference",
            "evidence": "deterministic graph-v2 scripted replay",
            "optimality": "not_claimed",
        },
        "generation": {},
    }


def _configure_prospective(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
    cell_index: int,
) -> None:
    trigger_step = 1 if level_index == 0 else 2 + (level_index >= 6)
    sink = blueprint.sink_id
    scenario["graph"]["nodes"][sink]["variables"]["clock"] = {
        "initial": 0.0,
        "drift": 1.0,
        "hard_min": 0.0,
        "hard_max": 20.0,
    }
    scenario["sensors"].append(
        {"id": "clock_sensor", "target": f"{sink}.clock", "observable": True}
    )
    for action in scenario["actions"]:
        if action["id"] in set(blueprint.pulse_actions):
            action.setdefault("preconditions", []).append(
                {"target": f"{sink}.clock", "operator": "ge", "value": trigger_step}
            )
    scenario["task"]["instructions"] = (
        "Arrange a future invocation, then use the graph to activate the component "
        "that can deliver the downstream service when the clock is ready."
    )
    scenario["interaction"].update(
        {
            "mode": "scheduled_event",
            "min_callback_steps": 1,
            "max_callback_steps": max(4, trigger_step + 1),
            "allowed_event_types": ["variable_threshold"],
        }
    )
    graph_actions = list(blueprint.reference_graph_actions)
    first_decision: dict[str, Any]
    uses_subscription = level_index in {1, 2, 5, 7}
    if uses_subscription:
        first_decision = _decision(
            [],
            subscriptions=[
                {
                    "operation": "add",
                    "id": "watch_graph_clock",
                    "event_type": "variable_threshold",
                    "filters": {
                        "target": f"{sink}.clock",
                        "operator": "ge",
                        "threshold": trigger_step,
                    },
                }
            ],
            context_requests=["active_subscriptions"],
        )
    else:
        first_decision = _decision([], callback={"after_steps": trigger_step})
    script = [{"at_step": 0, "decision": first_decision}]
    max_actions = int(scenario["interaction"]["max_actions_per_interaction"])
    for offset, chunk in enumerate(_chunks(graph_actions, max_actions)):
        subscriptions = (
            [{"operation": "remove", "id": "watch_graph_clock"}]
            if uses_subscription and offset == 0
            else []
        )
        callback = (
            {"after_steps": 1}
            if offset < len(_chunks(graph_actions, max_actions)) - 1
            else None
        )
        script.append(
            {
                "at_step": trigger_step + offset,
                "decision": _decision(chunk, callback=callback, subscriptions=subscriptions),
            }
        )
    scenario["baseline"]["script"] = script
    scenario["evaluation"].update(
        {
            "required_actions": graph_actions,
            "success_requires_required_actions": True,
            "relevant_event_step": trigger_step,
            "relevant_action": next(
                action_id
                for action_id in reversed(graph_actions)
                if action_id in set(blueprint.pulse_actions)
            ),
            "callback_target_window": [trigger_step, trigger_step],
        }
    )
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = graph_actions


def _configure_planning(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
    cell_index: int,
) -> None:
    pulse_ids = set(blueprint.pulse_actions)
    prepare_by_pulse: dict[str, str] = {}
    additions: list[dict[str, Any]] = []
    for action in scenario["actions"]:
        if action["id"] not in pulse_ids:
            continue
        target = str(action["target"])
        node_id = target.split(".", 1)[0]
        armed_id = f"armed_{action['id']}"
        scenario["graph"]["nodes"][node_id]["variables"][armed_id] = {"initial": False}
        prepare_id = f"prepare_{action['id']}"
        prepare_by_pulse[str(action["id"])] = prepare_id
        additions.append(
            {
                "id": prepare_id,
                "operation": "set",
                "target": f"{node_id}.{armed_id}",
                "value": True,
                "costs": {"energy": 0.25},
                "description": f"Prepare the component used by {action['id']}.",
            }
        )
        action.setdefault("preconditions", []).append(
            {"target": f"{node_id}.{armed_id}", "operator": "eq", "value": True}
        )
    wait = scenario["actions"].pop()
    scenario["actions"].extend(additions)
    scenario["actions"].append(wait)
    reference: list[str] = []
    for action_id in blueprint.reference_graph_actions:
        if action_id in prepare_by_pulse:
            reference.append(prepare_by_pulse[action_id])
        reference.append(action_id)
    scenario["task"]["instructions"] = (
        "Prepare and activate the viable graph path in a prerequisite-valid order "
        "without wasting the resource budget."
    )
    scenario["baseline"]["script"] = _periodic_script(
        reference, int(scenario["interaction"]["max_actions_per_interaction"])
    )
    scenario["evaluation"].update(
        {
            "required_actions": reference,
            "expected_action_sequence": reference,
            "reference_action_count": len(reference),
            "success_requires_required_actions": True,
            "success_requires_expected_sequence": True,
        }
    )
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = reference


def _configure_interference(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
    cell_index: int,
) -> None:
    sink = blueprint.sink_id
    scenario["sensors"].extend(
        [
            {"id": "reliable_sink", "target": f"{sink}.signal", "observable": True},
            {"id": "salient_alarm", "target": f"{sink}.signal", "observable": True},
        ]
    )
    scenario["faults"].append(
        {
            "id": "stale_salient_alarm",
            "kind": "sensor_stuck",
            "target": "salient_alarm",
            "at_step": 0,
            "value": 11.0,
            "announced": False,
        }
    )
    reference = list(blueprint.reference_graph_actions)
    if level_index >= 3:
        inspect_target = "e_primary" if blueprint.edges else "salient_alarm"
        inspect_id = "inspect_graph_evidence"
        wait = scenario["actions"].pop()
        scenario["actions"].append(
            {
                "id": inspect_id,
                "operation": "inspect",
                "target": inspect_target,
                "costs": {"energy": 0.25},
                "description": "Inspect the relevant graph or sensor evidence.",
            }
        )
        scenario["actions"].append(wait)
        reference.insert(0, inspect_id)
        scenario["evaluation"]["required_inspections"] = [inspect_id]
        scenario["evaluation"]["success_requires_required_inspections"] = True
        scenario["evaluation"]["evidence_gathering_actions"] = [inspect_id]
    scenario["task"]["instructions"] = (
        "Resolve the salient alarm against reliable remote evidence and graph "
        "structure, then activate only the component that can serve the sink."
    )
    scenario["baseline"]["script"] = _periodic_script(
        reference, int(scenario["interaction"]["max_actions_per_interaction"])
    )
    scenario["evaluation"].update(
        {
            "required_actions": reference,
            "forbidden_actions": list(blueprint.wrong_graph_actions),
            "emergency_actions": list(blueprint.wrong_graph_actions),
            "success_requires_required_actions": True,
        }
    )
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = reference


def _configure_conditional(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
    cell_index: int,
) -> None:
    sink = blueprint.sink_id
    initial_guard = level_index == 0
    scenario["graph"]["nodes"][sink]["variables"]["safeguard"] = {
        "initial": initial_guard
    }
    scenario["constraints"].append(
        {
            "id": "service_requires_safeguard",
            "kind": "conditional",
            "if": {
                "target": f"{sink}.signal",
                "operator": "ge",
                "value": blueprint.goal_threshold,
            },
            "then": {
                "target": f"{sink}.safeguard",
                "operator": "eq",
                "value": True,
            },
        }
    )
    scenario["sensors"].append(
        {
            "id": "safeguard_sensor",
            "target": f"{sink}.safeguard",
            "observable": initial_guard,
        }
    )
    reference: list[str] = []
    if not initial_guard:
        wait = scenario["actions"].pop()
        scenario["actions"].extend(
            [
                {
                    "id": "inspect_safeguard",
                    "operation": "inspect",
                    "target": f"{sink}.safeguard",
                    "costs": {"energy": 0.25},
                    "description": "Inspect the hidden downstream safeguard.",
                },
                {
                    "id": "enable_safeguard",
                    "operation": "set",
                    "target": f"{sink}.safeguard",
                    "value": True,
                    "costs": {"energy": 0.25},
                    "description": "Enable the downstream safeguard.",
                },
            ]
        )
        scenario["actions"].append(wait)
        reference.extend(["inspect_safeguard", "enable_safeguard"])
        scenario["evaluation"].update(
            {
                "required_inspections": ["inspect_safeguard"],
                "success_requires_required_inspections": True,
            }
        )
    reference.extend(blueprint.reference_graph_actions)
    scenario["task"]["instructions"] = (
        "Verify the safety consequent when hidden, establish it if necessary, "
        "then use graph structure to select the safe upstream actuation."
    )
    scenario["baseline"]["script"] = _periodic_script(
        reference, int(scenario["interaction"]["max_actions_per_interaction"])
    )
    scenario["evaluation"].update(
        {"required_actions": reference, "success_requires_required_actions": True}
    )
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = reference


def _configure_adaptation(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
    cell_index: int,
) -> None:
    change_step = 2
    reference_before = list(blueprint.reference_graph_actions)
    if blueprint.edges:
        fault_target = _reference_edge_id(scenario, blueprint)
        scenario["faults"].append(
            {
                "id": "primary_edge_failure",
                "kind": "edge_disabled",
                "target": fault_target,
                "at_step": change_step,
                "announced": level_index <= 1,
            }
        )
        repair_action = {
            "id": "repair_primary_path",
            "operation": "repair",
            "target": fault_target,
            "costs": {"energy": 0.5},
            "description": "Repair the failed primary graph edge.",
        }
    else:
        pulse = blueprint.pulse_actions[0]
        scenario["faults"].append(
            {
                "id": "local_actuator_failure",
                "kind": "actuator_disabled",
                "target": pulse,
                "at_step": change_step,
                "announced": level_index <= 1,
            }
        )
        repair_action = {
            "id": "repair_primary_path",
            "operation": "repair",
            "target": pulse,
            "costs": {"energy": 0.5},
            "description": "Repair the failed local actuator.",
        }
    wait = scenario["actions"].pop()
    scenario["actions"].append(repair_action)
    scenario["actions"].append(wait)
    sink = blueprint.sink_id
    scenario["graph"]["nodes"][sink]["variables"]["signal"]["drift"] = -0.5
    scenario["graph"]["nodes"][sink]["variables"]["signal"]["hard_min"] = 0.0
    if level_index >= 6 and instance_index == 4:
        remote_sensor = next(
            sensor["id"]
            for sensor in scenario["sensors"]
            if not str(sensor["target"]).startswith(f"{sink}.")
        )
        scenario["faults"].append(
            {
                "id": "simultaneous_remote_sensor_delay",
                "kind": "sensor_delay",
                "target": remote_sensor,
                "at_step": change_step,
                "value": 1,
                "announced": False,
            }
        )
    reference_after = ["repair_primary_path", *blueprint.reference_graph_actions]
    max_actions = int(scenario["interaction"]["max_actions_per_interaction"])
    script = _periodic_script(reference_before, max_actions, start_step=0)
    post = _periodic_script(reference_after, max_actions, start_step=change_step)
    script.extend(post)
    script.sort(key=lambda item: int(item["at_step"]))
    scenario["baseline"]["script"] = script
    exact = [*reference_before, *reference_after]
    scenario["task"]["instructions"] = (
        "Use the initially effective graph path, detect its regime change, then "
        "repair or revise the strategy and restore downstream service."
    )
    scenario["task"]["terminate_on_goals"] = False
    scenario["evaluation"].update(
        {
            "required_actions": exact,
            "success_requires_required_actions": True,
            "success_requires_adaptation": True,
            "adaptation": {
                "regime_change_step": change_step,
                "sufficient_evidence_step": change_step,
                "original_actions": list(blueprint.pulse_actions),
                "replacement_actions": ["repair_primary_path"],
                "safe_replacement_actions": ["repair_primary_path"],
                "unnecessary_exploration_actions": list(blueprint.wrong_graph_actions),
                "evidence_clarity": "clear" if level_index <= 1 else "physical",
                "recovery_resource_ids": ["energy"],
            },
        }
    )
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = exact


def _configure_temporal(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
    cell_index: int,
) -> None:
    delay = min(3, 1 + level_index // 2)
    pulse_ids = set(blueprint.pulse_actions)
    action_by_id = {str(action["id"]): action for action in scenario["actions"]}
    attribution_actions: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    sink = blueprint.sink_id
    scenario["graph"]["nodes"][sink]["variables"]["attributed"] = {"initial": 0.0}
    for pulse_id in sorted(pulse_ids):
        pulse = action_by_id[pulse_id]
        pulse["effect_delay_steps"] = delay
        node_id = str(pulse["target"]).split(".", 1)[0]
        marker = f"credited_{pulse_id}"
        ready = f"response_ready_{pulse_id}"
        scenario["graph"]["nodes"][node_id]["variables"][marker] = {"initial": False}
        scenario["graph"]["nodes"][node_id]["variables"][ready] = {"initial": False}
        pulse.setdefault("additional_effects", []).append(
            {
                "operation": "set",
                "target": f"{node_id}.{ready}",
                "value": True,
            }
        )
        attribution_id = f"attribute_{pulse_id}"
        mapping[pulse_id] = attribution_id
        attribution_actions.append(
            {
                "id": attribution_id,
                "operation": "adjust",
                "target": f"{sink}.attributed",
                "amount": 1.0,
                "costs": {"energy": 0.25},
                "preconditions": [
                    {
                        "target": f"{node_id}.{ready}",
                        "operator": "eq",
                        "value": True,
                    },
                    {
                        "target": f"{node_id}.{marker}",
                        "operator": "eq",
                        "value": False,
                    },
                ],
                "additional_effects": [
                    {"operation": "set", "target": f"{node_id}.{marker}", "value": True}
                ],
                "description": f"Attribute the downstream response to {pulse_id}.",
            }
        )
    wait = scenario["actions"].pop()
    scenario["actions"].extend(attribution_actions)
    scenario["actions"].append(wait)
    initiations = list(blueprint.reference_graph_actions)
    restore_actions = [item for item in initiations if item not in pulse_ids]
    pulse_sequence = [item for item in initiations if item in pulse_ids]
    attributes = [mapping[item] for item in pulse_sequence]
    max_actions = int(scenario["interaction"]["max_actions_per_interaction"])
    script = _periodic_script(restore_actions + pulse_sequence, max_actions, start_step=0)
    attribution_step = delay + max(1, len(_chunks(restore_actions + pulse_sequence, max_actions)))
    script.extend(_periodic_script(attributes, max_actions, start_step=attribution_step))
    script.sort(key=lambda item: int(item["at_step"]))
    exact = [*restore_actions, *pulse_sequence, *attributes]
    scenario["baseline"]["script"] = script
    scenario["goals"] = [
        {
            "id": "downstream_response_attributed",
            "status_observable": True,
            "all": [
                {
                    "target": f"{sink}.signal",
                    "operator": "ge",
                    "value": blueprint.goal_threshold,
                },
                {
                    "target": f"{sink}.attributed",
                    "operator": "ge",
                    "value": float(len(attributes)),
                },
            ],
        }
    ]
    scenario["task"]["instructions"] = (
        "Initiate the viable upstream action, retain its identity across the delayed "
        "effect, and attribute the downstream response to the correct source."
    )
    scenario["evaluation"].update(
        {
            "required_actions": exact,
            "success_requires_required_actions": True,
            "success_requires_temporal_tracking": True,
            "temporal": {
                "initiation_to_response": mapping,
                "attribution_actions": attributes,
                "relevant_history_items": len(attributes),
                "required_temporal_distance": delay,
                "anomaly_actions": [
                    f"attribute_{item}" for item in blueprint.wrong_graph_actions if item in mapping
                ],
            },
        }
    )
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = exact


def _configure_observability(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    level_index: int,
    instance_index: int,
) -> None:
    sensors = list(scenario["sensors"])
    if instance_index == 0:
        observability = "full_local"
    elif instance_index in {1, 2}:
        hidden = level_index >= 5
        observability = "edge_hidden_inferred" if hidden else "edge_visible"
        for edge in scenario["graph"]["edges"]:
            edge["state_observable"] = not hidden
    elif instance_index == 3:
        delayed = level_index >= 3
        observability = "delayed_remote_sensor" if delayed else "remote_sensor"
        if delayed:
            for sensor in sensors:
                if not str(sensor["target"]).startswith(f"{blueprint.sink_id}."):
                    sensor["delay_steps"] = 1
    else:
        observability = "hidden_neighbor"
        for edge in scenario["graph"]["edges"]:
            if edge["id"] in {"e_path_a_out", "e_cross"}:
                edge["state_observable"] = False
        sensors = [
            sensor
            for sensor in sensors
            if not str(sensor["target"]).startswith("aux.")
        ]
        if level_index >= 4:
            for sensor in sensors:
                if str(sensor["target"]).startswith(("source_", "relay_")):
                    sensor["delay_steps"] = 1
            observability = "delayed_remote_sensor"
    scenario["sensors"] = sensors
    profile = scenario["difficulty_profile"]["graph_v2"]
    profile["observability_type"] = observability
    fault_kinds = {str(fault.get("kind")) for fault in scenario["faults"]}
    if len(scenario["faults"]) > 1:
        fault_locality = "multiple_simultaneous"
    elif "edge_disabled" in fault_kinds:
        fault_locality = "edge"
    elif any(kind.startswith("sensor_") for kind in fault_kinds):
        fault_locality = "remote" if instance_index else "local"
    elif scenario["faults"]:
        fault_locality = "local"
    elif instance_index >= 3 and blueprint.edges:
        fault_locality = "propagated"
    else:
        fault_locality = "none"
    profile["fault_locality"] = fault_locality


def _finalize_reference_and_resources(scenario: dict[str, Any]) -> None:
    action_by_id = {str(action["id"]): action for action in scenario["actions"]}
    scripted_ids = [
        str(command["action_id"])
        for item in scenario["baseline"]["script"]
        for command in item.get("decision", {}).get("actions", [])
    ]
    reference_cost = sum(
        float(action_by_id.get(action_id, {}).get("costs", {}).get("energy", 0.0))
        for action_id in scripted_ids
    )
    # A fixed budget keeps counterfactual members observationally matched. It is
    # still binding for wasteful repeated actions over the configured horizons.
    scenario["graph"]["resources"]["energy"]["initial"] = round(
        max(6.0, reference_cost + 0.75), 6
    )
    scenario["evaluation"]["reference_action_count"] = len(scripted_ids)
    scenario["evaluation"]["graph_v2"]["expected_exact_action_sequence"] = scripted_ids


def _record_generation_metadata(
    scenario: dict[str, Any],
    blueprint: GraphBlueprint,
    family: str,
    level: str,
    level_index: int,
    instance_index: int,
    cell_index: int,
    generation_seed: int,
) -> None:
    scenario["generation"] = {
        "generator_id": GRAPH_V2_GENERATOR_ID,
        "generator_version": GRAPH_V2_GENERATOR_VERSION,
        "benchmark_release": "0.5.0",
        "task_family": family,
        "level": level,
        "logical_position": level_index + 1,
        "generation_seed": generation_seed,
        "parameters": {
            "instance_index": instance_index,
            "cell_index": cell_index,
            "domain_abstraction": DOMAIN_BY_FAMILY[family],
            "counterfactual_pair_type": blueprint.pair_type,
        },
        "construct_mapping": {
            "family_construct": CONSTRUCT_BY_FAMILY[family],
            "graph_demand": scenario["difficulty_profile"]["graph_v2"]["graph_demand"],
            "difficulty_manipulation": scenario["difficulty_profile"]["graph_v2"][
                "difficulty_manipulation"
            ],
            "potential_confound": scenario["difficulty_profile"]["graph_v2"][
                "potential_confound"
            ],
        },
    }


def _batch_configuration(level_index: int, instance_index: int, cell_index: int) -> tuple[int, str]:
    if level_index <= 2 or instance_index == 0:
        return 1, "sequential_per_command"
    if level_index == 3:
        maximum = 2 if instance_index >= 3 else 1
    elif level_index == 4:
        maximum = 2 if instance_index == 4 else 1
    elif level_index == 5:
        maximum = 2 if instance_index >= 2 else 1
    elif level_index == 6:
        maximum = 2 if instance_index >= 1 else 1
    else:
        maximum = 3 if instance_index >= 3 else 2
    if maximum == 1:
        return maximum, "sequential_per_command"
    semantics = "atomic" if (cell_index + instance_index) % 2 == 0 or maximum == 3 else "sequential_per_command"
    return maximum, semantics


def _reference_edge_id(
    scenario: Mapping[str, Any], blueprint: GraphBlueprint
) -> str:
    action_by_id = {
        str(action["id"]): action for action in scenario["actions"]
    }
    pulse_ids = set(blueprint.pulse_actions)
    pulse_id = next(
        action_id
        for action_id in blueprint.reference_graph_actions
        if action_id in pulse_ids
    )
    source = str(action_by_id[pulse_id]["target"]).split(".", 1)[0]
    candidates = [
        edge
        for edge in scenario["graph"]["edges"]
        if str(edge["source"]) == source
    ]
    if not candidates:
        candidates = list(scenario["graph"]["edges"])
    return str(candidates[0]["id"])


def _task_instructions(family: str) -> str:
    return {
        "prospective_reasoning": "Arrange a future graph-dependent intervention.",
        "planning": "Plan a prerequisite-valid sequence across graph components.",
        "interference": "Resolve conflicting evidence before graph-dependent control.",
        "conditional_safety": "Verify a conditional safety rule before graph-dependent control.",
        "adaptation": "Recover after a graph or actuator regime change.",
        "temporal_tracking": "Attribute delayed downstream effects to upstream actions.",
    }[family]


def _domain_labels(family: str) -> dict[str, str]:
    domain = DOMAIN_BY_FAMILY[family]
    if domain == "thermal_network":
        return {"source_a": "Heater A", "source_b": "Heater B", "relay": "Room", "sink": "Occupied zone", "aux": "Outdoor boundary"}
    if domain == "process_chain":
        return {"source_a": "Pump A", "source_b": "Pump B", "relay": "Intermediate tank", "sink": "Product tank", "aux": "Waste branch"}
    if domain == "communication_network":
        return {"source_a": "UAV A", "source_b": "UAV B", "relay": "Relay", "sink": "Gateway", "aux": "Interference source"}
    if domain == "power_service_dependency":
        return {"source_a": "Battery A", "source_b": "Battery B", "relay": "Compute service", "sink": "Communication service", "aux": "Background load"}
    if domain == "reconfigurable_process_network":
        return {"source_a": "Primary actuator", "source_b": "Backup actuator", "relay": "Process stage", "sink": "Service output", "aux": "Disturbance"}
    return {"source_a": "Upstream channel A", "source_b": "Upstream channel B", "relay": "Transport stage", "sink": "Response monitor", "aux": "Unrelated channel"}


def _node(label: str, *, signal: float = 0.0) -> dict[str, Any]:
    return {
        "label": label,
        "variables": {
            "signal": {
                "initial": float(signal),
                "hard_min": -40.0,
                "hard_max": 40.0,
            }
        },
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    coupling: float,
    *,
    directed: bool,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "variable": "signal",
        "coupling": float(coupling),
        "directed": bool(directed),
        "enabled": bool(enabled),
        "observable": True,
        "state_observable": True,
    }


def _set_action(action_id: str, target: str, value: float, description: str) -> dict[str, Any]:
    return {
        "id": action_id,
        "operation": "set",
        "target": target,
        "value": float(value),
        "description": description,
    }


def _toggle_action(action_id: str, edge_id: str, enabled: bool, description: str) -> dict[str, Any]:
    return {
        "id": action_id,
        "operation": "toggle_edge",
        "target": edge_id,
        "edge_enabled": bool(enabled),
        "description": description,
    }


def _base_sensors(nodes: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"sensor_{node_id}",
            "target": f"{node_id}.signal",
            "observable": True,
            "precision": 3,
        }
        for node_id in nodes
    ]


def _decision(
    actions: list[str],
    *,
    callback: dict[str, Any] | None = None,
    subscriptions: list[dict[str, Any]] | None = None,
    context_requests: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "actions": [{"action_id": action_id} for action_id in actions],
        "callback": callback,
        "subscriptions": list(subscriptions or []),
        "context_requests": list(context_requests or []),
    }


def _periodic_script(
    actions: list[str] | tuple[str, ...],
    max_actions: int,
    *,
    start_step: int = 0,
) -> list[dict[str, Any]]:
    return [
        {"at_step": start_step + index, "decision": _decision(chunk)}
        for index, chunk in enumerate(_chunks(list(actions), max_actions))
    ]


def _chunks(actions: list[str], maximum: int) -> list[list[str]]:
    return [actions[index : index + maximum] for index in range(0, len(actions), maximum)]


def _physical_hash(value: Mapping[str, Any]) -> str:
    return content_hash(
        {
            key: copy.deepcopy(value[key])
            for key in (
                "max_steps",
                "time_step",
                "graph",
                "sensors",
                "actions",
                "faults",
                "constraints",
                "goals",
                "observability",
                "interaction",
            )
        }
    )
