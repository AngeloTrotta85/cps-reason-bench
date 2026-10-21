"""Additive robustness and factor-design utilities.

Released generators remain authoritative for Foundation v1 and Core-v2.  This
module wraps them only when a new suite explicitly requests a new generator ID.
"""

from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Mapping

from cps_reason_bench.config import ConfigError, ScenarioConfig
from cps_reason_bench.generators import generate_scenario
from cps_reason_bench.versions import content_hash


ROBUSTNESS_GENERATOR_ID = "cps_reason_robustness_generator"
ROBUSTNESS_GENERATOR_VERSION = "1.0.0"
FACTORIAL_GENERATOR_ID = "cps_reason_factorial_generator"
FACTORIAL_GENERATOR_VERSION = "1.0.0"
FACTOR_SCHEMA_ID = "cps-reason-factor-schema"
FACTOR_SCHEMA_VERSION = "1.0.0"

FAMILIES = (
    "prospective_reasoning",
    "planning",
    "interference",
    "conditional_safety",
    "adaptation",
    "temporal_tracking",
)

FOUNDATION_LEVELS = ("entry_1", "entry_2", "entry_3", "bridge")
CORE_LEVELS: dict[str, tuple[str, ...]] = {
    family: (
        ("controlled_1", "controlled_2", "controlled_3", "controlled_4")
        if family in {"adaptation", "temporal_tracking"}
        else ("basic", "controlled_2", "controlled_3", "controlled_4")
    )
    for family in FAMILIES
}

# Binary-exact quarter increments avoid changing an equality/threshold result only
# through floating-point representation while still varying physical units.
_INSTANCE_SCALES = (0.75, 1.0, 1.25, 1.5, 2.0)
_NONBEHAVIORAL_TOP_LEVEL = {
    "name",
    "description",
    "seed",
    "generation",
    "difficulty",
    "difficulty_profile",
    "baseline",
    "reference",
}
_ORDER_NORMALIZED_LISTS = ("actions", "sensors", "faults", "constraints", "goals")


def semantic_scenario_payload(
    scenario: ScenarioConfig | Mapping[str, Any], *, normalize_declaration_order: bool = False
) -> dict[str, Any]:
    """Return analysis-only behavioral content without provenance metadata."""

    source = scenario.to_mapping() if isinstance(scenario, ScenarioConfig) else scenario
    value = copy.deepcopy(dict(source))
    for key in _NONBEHAVIORAL_TOP_LEVEL:
        value.pop(key, None)
    if normalize_declaration_order:
        for key in _ORDER_NORMALIZED_LISTS:
            items = value.get(key)
            if isinstance(items, list):
                value[key] = sorted(items, key=lambda item: str(item.get("id", "")))
    return value


def semantic_scenario_hash(
    scenario: ScenarioConfig | Mapping[str, Any], *, normalize_declaration_order: bool = False
) -> str:
    return content_hash(
        semantic_scenario_payload(
            scenario, normalize_declaration_order=normalize_declaration_order
        )
    )


def physical_scenario_hash(scenario: ScenarioConfig | Mapping[str, Any]) -> str:
    """Hash physical/task semantics while ignoring declaration-list rotation."""

    return semantic_scenario_hash(scenario, normalize_declaration_order=True)


def physical_plant_hash(scenario: ScenarioConfig | Mapping[str, Any]) -> str:
    """Hash the paired plant while excluding context-treatment metadata."""

    payload = semantic_scenario_payload(
        scenario, normalize_declaration_order=True
    )
    payload.pop("context", None)
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("context_negative_control", None)
    return content_hash(payload)


def public_trajectory_hash(events_path: str | Path) -> str:
    """Hash only model-visible trajectory records from one completed trace."""

    values: list[dict[str, Any]] = []
    for line in Path(events_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        kind = row.get("record_type")
        if kind == "run_started":
            values.append({"kind": kind, "observation": row.get("visible_observation")})
        elif kind == "model_interaction":
            values.append(
                {
                    "kind": kind,
                    "observation": row.get("explicit_context", {}).get("observation"),
                    "outcomes": [
                        outcome.get("public") for outcome in row.get("action_outcomes", [])
                    ],
                    "triggers": row.get("invocation_triggers", []),
                }
            )
        elif kind == "simulation_transition":
            values.append(
                {
                    "kind": kind,
                    "observation": row.get("visible_observation_after"),
                    "events": [
                        {
                            "event_type": event.get("event_type"),
                            "public_data": event.get("public_data"),
                            "step": event.get("step"),
                        }
                        for event in row.get("events", [])
                        if event.get("public_data") is not None
                    ],
                }
            )
        elif kind == "run_completed":
            values.append(
                {"kind": kind, "observation": row.get("final_visible_observation")}
            )
    return content_hash(values)


def simulator_stochasticity(scenario: ScenarioConfig) -> dict[str, Any]:
    noisy = tuple(
        str(sensor["id"])
        for sensor in scenario.sensors
        if float(sensor.get("noise_std", 0.0)) > 0.0
    )
    return {
        "classification": "stochastic_sensor_noise" if noisy else "deterministic",
        "sensor_noise": bool(noisy),
        "noisy_sensor_ids": list(noisy),
        "stochastic_action_effect": False,
        "stochastic_dynamics": False,
        "other_stochasticity": False,
    }


def generate_robustness_scenario(
    task_family: str,
    level: str,
    generation_seed: int,
    parameters: Mapping[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> ScenarioConfig:
    """Generate a new semantic instance while preserving a released level design."""

    provided = dict(parameters or {})
    instance_index = int(provided.pop("instance_index", 0))
    if instance_index < 0:
        raise ConfigError("instance_index cannot be negative")
    base_layer = str(provided.pop("base_layer", "core_v2"))
    if base_layer not in {"foundation", "core_v2"}:
        raise ConfigError("base_layer must be foundation or core_v2")
    provided["content_revision"] = (
        "foundation_v1" if base_layer == "foundation" else "core_v2"
    )
    base = generate_scenario(
        task_family,
        level,
        generation_seed,
        provided,
        template_path,
    )
    value = base.to_mapping()
    base_profile = dict(value.get("difficulty_profile", {}))
    reasoning_profile = copy.deepcopy(base_profile.get("reasoning_profile"))
    scale = _INSTANCE_SCALES[instance_index % len(_INSTANCE_SCALES)]
    suffix = f"r{instance_index + 1:02d}"

    if task_family == "planning":
        _vary_planning_resource_costs(value, instance_index)
    _scale_numeric_plant(value, scale)
    _vary_irrelevant_boolean_evidence(value, instance_index)
    _rename_operational_ids(value, suffix)
    _refresh_public_text(value, task_family)

    family_parameters = copy.deepcopy(base.generation.get("parameters", {}))
    family_parameters.update(
        {
            "robustness_instance_index": instance_index,
            "robustness_numeric_scale": scale,
            "robustness_namespace": suffix,
        }
    )
    value["name"] = f"robustness_{base_layer}_{task_family}_{level}_{suffix}"
    value["description"] = (
        f"Additive robustness instance {instance_index + 1} for "
        f"{base_layer} {task_family} {level}."
    )
    value["seed"] = generation_seed
    value["generation"] = {
        "generator_id": ROBUSTNESS_GENERATOR_ID,
        "generator_version": ROBUSTNESS_GENERATOR_VERSION,
        "factor_schema_version": None,
        "task_family": task_family,
        "level": level,
        "generation_seed": generation_seed,
        "parameters": family_parameters,
        "template": Path(template_path).name if template_path else None,
        "base_generator": copy.deepcopy(base.generation),
    }
    value["difficulty_profile"] = {
        "family_parameters": family_parameters,
        **(
            {"reasoning_profile": reasoning_profile}
            if reasoning_profile is not None
            else {}
        ),
        "robustness_profile": {
            "base_layer": base_layer,
            "base_level": level,
            "instance_index": instance_index,
            "numeric_scale": scale,
            "namespace": suffix,
            "topology_changed": False,
            "variation_dimensions": _robustness_variation_dimensions(task_family),
            "selection_requirement": "unique_semantic_scenario_hash",
        },
    }
    scenario = ScenarioConfig.from_mapping(value)
    profile = scenario.to_mapping()["difficulty_profile"]
    profile["robustness_profile"].update(
        {
            "semantic_scenario_hash": semantic_scenario_hash(scenario),
            "physical_scenario_hash": physical_scenario_hash(scenario),
        }
    )
    final = scenario.to_mapping()
    final["difficulty_profile"] = profile
    final["generation"]["semantic_scenario_hash"] = semantic_scenario_hash(scenario)
    final["generation"]["physical_scenario_hash"] = physical_scenario_hash(scenario)
    return ScenarioConfig.from_mapping(final)


def factorial_rows(task_family: str) -> tuple[dict[str, Any], ...]:
    """Return every valid row in factor schema v1 for one family."""

    rows: list[dict[str, Any]]
    if task_family == "prospective_reasoning":
        rows = [
            dict(zip(("monitoring_mode", "sensor_delay_steps", "sensor_noise_std", "distractor"), values))
            for values in product(
                ("callback_interval", "event_subscription", "hybrid"),
                (0, 1),
                (0.0, 0.1),
                (False, True),
            )
        ]
    elif task_family == "planning":
        rows = [
            dict(zip(("planning_depth", "distractor_actions", "branching_factor", "resource_slack"), values))
            for values in product((3, 4), (0, 1, 2), (1, 2), (0, 1))
        ]
    elif task_family == "interference":
        rows = [
            {
                "misleading_sensor_count": misleading,
                "distractor_actions": distractors,
                "downstream_consequence_delay_steps": delay,
            }
            for misleading in (1, 2, 3)
            for distractors in (1, 2, 3)
            for delay in ((0, 2) if distractors == 3 else (0,))
        ]
    elif task_family == "conditional_safety":
        rows = [
            {
                "rule_count": rules,
                "candidate_inspection_count": candidates,
                "inspection_budget": budget,
                "observation_delay_steps": delay,
                "distractor_actions": distractor,
            }
            for rules in (1, 2)
            for candidates in ((1, 2, 3) if rules == 1 else (2, 3))
            for budget in ((1, 2) if rules == 1 else (2,))
            for delay in (0, 1)
            for distractor in (0, 1)
        ]
    elif task_family == "adaptation":
        rows = [
            {
                "adaptation_profile": "disabled_clear",
                "regime_change_mechanism": "actuator_disabled",
                "feedback": "clear",
                "evidence_required_observations": 1,
                "inspection_required": False,
                "replacement_strategy_count": 1,
            },
            {
                "adaptation_profile": "scaled_physical",
                "regime_change_mechanism": "actuator_scale",
                "feedback": "physical",
                "evidence_required_observations": 2,
                "inspection_required": False,
                "replacement_strategy_count": 1,
            },
            {
                "adaptation_profile": "remap_inspection",
                "regime_change_mechanism": "actuator_remap",
                "feedback": "partial",
                "evidence_required_observations": 2,
                "inspection_required": True,
                "replacement_strategy_count": 1,
            },
            {
                "adaptation_profile": "remap_safe_choice",
                "regime_change_mechanism": "actuator_remap",
                "feedback": "clear_unsafe",
                "evidence_required_observations": 1,
                "inspection_required": False,
                "replacement_strategy_count": 2,
            },
        ]
    elif task_family == "temporal_tracking":
        structures = (
            (1, 2, 0, 1, "previous"),
            (2, 3, 0, 1, "history"),
            (3, 4, 0, 1, "history"),
            (3, 5, 2, 2, "history"),
        )
        rows = [
            {
                "temporal_distance": distance,
                "action_effect_delay_steps": distance,
                "history_limit": history,
                "temporal_distractor_count": distractors,
                "overlapping_causal_actions": overlap,
                "relevant_history_items": relevant,
                "context_mode": context,
            }
            for distance, history, overlap, relevant, context in structures
            for distractors in (0, 2)
        ]
    else:
        raise ConfigError(f"Unknown factorial family {task_family!r}")
    return tuple(
        {"factor_row_id": f"{task_family}_f{index + 1:03d}", **row}
        for index, row in enumerate(rows)
    )


def pairwise_factorial_rows(task_family: str) -> tuple[dict[str, Any], ...]:
    rows = factorial_rows(task_family)
    factor_names = _pairwise_factor_names(task_family, rows[0])
    pair_sets: list[set[tuple[str, Any, str, Any]]] = []
    universe: set[tuple[str, Any, str, Any]] = set()
    for row in rows:
        pairs = {
            (left, _hashable(row[left]), right, _hashable(row[right]))
            for left, right in combinations(factor_names, 2)
        }
        pair_sets.append(pairs)
        universe.update(pairs)
    uncovered = set(universe)
    chosen: list[int] = []
    while uncovered:
        candidates = [index for index in range(len(rows)) if index not in chosen]
        best = max(
            candidates,
            key=lambda index: (len(pair_sets[index] & uncovered), -index),
        )
        covered = pair_sets[best] & uncovered
        if not covered:
            break
        chosen.append(best)
        uncovered.difference_update(covered)
    return tuple(rows[index] for index in chosen)


def pairwise_coverage(task_family: str) -> dict[str, Any]:
    full = factorial_rows(task_family)
    selected = pairwise_factorial_rows(task_family)
    names = _pairwise_factor_names(task_family, full[0])

    def pairs(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, Any, str, Any]]:
        return {
            (left, _hashable(row[left]), right, _hashable(row[right]))
            for row in rows
            for left, right in combinations(names, 2)
        }

    available = pairs(full)
    covered = pairs(selected)
    return {
        "family": task_family,
        "full_row_count": len(full),
        "selected_row_count": len(selected),
        "valid_pair_count": len(available),
        "covered_pair_count": len(covered & available),
        "coverage_rate": len(covered & available) / len(available),
        "selected_row_ids": [row["factor_row_id"] for row in selected],
        "uncovered_pairs": [list(item) for item in sorted(available - covered, key=repr)],
    }


def generate_factorial_scenario(
    task_family: str,
    factor_row_id: str,
    generation_seed: int,
    parameters: Mapping[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> ScenarioConfig:
    row = dict(parameters or {})
    row.pop("content_revision", None)
    expected = {
        item["factor_row_id"]: item for item in factorial_rows(task_family)
    }.get(factor_row_id)
    if expected is None:
        raise ConfigError(
            f"Unknown factor row {factor_row_id!r} for family {task_family!r}"
        )
    for key, value in expected.items():
        if key != "factor_row_id" and key in row and row[key] != value:
            raise ConfigError(
                f"Factor row {factor_row_id} requires {key}={value!r}, got {row[key]!r}"
            )
    factors = {key: value for key, value in expected.items() if key != "factor_row_id"}
    level, generator_parameters = _factorial_generator_parameters(task_family, factors)
    base = generate_scenario(
        task_family,
        level,
        generation_seed,
        {"content_revision": "core_v2", **generator_parameters},
        template_path,
    )
    value = base.to_mapping()
    base_profile = dict(value.get("difficulty_profile", {}))
    reasoning_profile = copy.deepcopy(base_profile.get("reasoning_profile"))
    value["name"] = f"factorial_{factor_row_id}"
    value["description"] = (
        f"Factor schema {FACTOR_SCHEMA_VERSION} condition {factor_row_id}."
    )
    value["difficulty_level"] = factor_row_id
    value["generation"] = {
        "generator_id": FACTORIAL_GENERATOR_ID,
        "generator_version": FACTORIAL_GENERATOR_VERSION,
        "factor_schema_id": FACTOR_SCHEMA_ID,
        "factor_schema_version": FACTOR_SCHEMA_VERSION,
        "factor_row_id": factor_row_id,
        "task_family": task_family,
        "level": factor_row_id,
        "base_level": level,
        "generation_seed": generation_seed,
        "parameters": factors,
        "template": Path(template_path).name if template_path else None,
        "base_generator": copy.deepcopy(base.generation),
    }
    value["difficulty_profile"] = {
        "family_parameters": copy.deepcopy(base.generation.get("parameters", {})),
        **(
            {"reasoning_profile": reasoning_profile}
            if reasoning_profile is not None
            else {}
        ),
        "factor_profile": {
            "factor_schema_id": FACTOR_SCHEMA_ID,
            "factor_schema_version": FACTOR_SCHEMA_VERSION,
            "factor_row_id": factor_row_id,
            "factors": factors,
        },
    }
    scenario = ScenarioConfig.from_mapping(value)
    semantic_hash = semantic_scenario_hash(scenario)
    final = scenario.to_mapping()
    final["generation"]["semantic_scenario_hash"] = semantic_hash
    final["generation"]["physical_scenario_hash"] = physical_scenario_hash(scenario)
    final["difficulty_profile"]["factor_profile"][
        "semantic_scenario_hash"
    ] = semantic_hash
    return ScenarioConfig.from_mapping(final)


def analysis_identifiers(
    scenario: ScenarioConfig,
    *,
    suite_name: str,
    suite_entry_id: str,
    simulator_seed: int,
    context_condition: str,
    model_configuration: str | None = None,
    requested_model_seed: int | None = None,
    repetition_index: int | None = None,
) -> dict[str, str]:
    semantic_hash = semantic_scenario_hash(scenario)
    physical_hash = physical_scenario_hash(scenario)
    plant_hash = physical_plant_hash(scenario)
    physical_instance_id = content_hash(
        {
            "family": scenario.task_family,
            "physical_plant_hash": plant_hash,
            "simulator_seed": simulator_seed,
        }
    )[:20]
    paired_condition_id = content_hash(
        {
            "suite_entry_id": suite_entry_id,
            "semantic_scenario_hash": semantic_hash,
            "simulator_seed": simulator_seed,
            "context_condition": context_condition,
        }
    )[:20]
    analysis_unit_id = content_hash(
        {
            "suite_name": suite_name,
            "paired_condition_id": paired_condition_id,
        }
    )[:20]
    replicate_id = content_hash(
        {
            "analysis_unit_id": analysis_unit_id,
            "model_configuration": model_configuration,
            "requested_model_seed": requested_model_seed,
            "repetition_index": repetition_index,
        }
    )[:20]
    return {
        "semantic_scenario_hash": semantic_hash,
        "physical_scenario_hash": physical_hash,
        "physical_plant_hash": plant_hash,
        "physical_instance_id": physical_instance_id,
        "paired_condition_id": paired_condition_id,
        "analysis_unit_id": analysis_unit_id,
        "replicate_id": replicate_id,
    }


def _factorial_generator_parameters(
    task_family: str, factors: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    if task_family == "prospective_reasoning":
        mode = str(factors["monitoring_mode"])
        return "basic", {
            "difficulty": 1,
            "sensor_delay_steps": int(factors["sensor_delay_steps"]),
            "sensor_noise_std": float(factors["sensor_noise_std"]),
            "add_distractor_action": bool(factors["distractor"]),
            "monitoring_mode": mode,
            "event_candidate_count": 3 if mode != "callback_interval" else 2,
            "callback_tolerance_steps": 1,
            "interaction_cost_pressure": mode == "hybrid",
        }
    if task_family == "planning":
        branching = int(factors["branching_factor"])
        return "basic", {
            "difficulty": 1,
            "planning_depth": int(factors["planning_depth"]),
            "distractor_actions": int(factors["distractor_actions"]),
            "branching_factor": branching,
            "valid_alternative_sequence_count": branching,
            "resource_slack": int(factors["resource_slack"]),
        }
    if task_family == "interference":
        return "basic", {
            "difficulty": 1,
            "misleading_sensor_count": int(factors["misleading_sensor_count"]),
            "distractor_actions": int(factors["distractor_actions"]),
            "evidence_conflict_count": int(factors["misleading_sensor_count"]),
            "downstream_consequence_delay_steps": int(
                factors["downstream_consequence_delay_steps"]
            ),
        }
    if task_family == "conditional_safety":
        return "basic", {
            "difficulty": 1,
            "hidden_rule_variables": int(factors["candidate_inspection_count"]),
            "rule_count": int(factors["rule_count"]),
            "candidate_inspection_count": int(factors["candidate_inspection_count"]),
            "inspection_budget": int(factors["inspection_budget"]),
            "observation_delay_steps": int(factors["observation_delay_steps"]),
            "distractor_actions": int(factors["distractor_actions"]),
        }
    if task_family == "adaptation":
        profile = str(factors["adaptation_profile"])
        levels = {
            "disabled_clear": "controlled_1",
            "scaled_physical": "controlled_2",
            "remap_inspection": "controlled_3",
            "remap_safe_choice": "controlled_4",
        }
        return levels[profile], {
            "difficulty": list(levels).index(profile) + 1,
            "regime_change_mechanism": factors["regime_change_mechanism"],
            "regime_change_visibility": factors["feedback"],
            "change_magnitude": profile,
            "plausible_explanation_count": 1 + int(factors["evidence_required_observations"]),
            "replacement_strategy_count": int(factors["replacement_strategy_count"]),
            "feedback_clarity": factors["feedback"],
            "evidence_required_observations": int(factors["evidence_required_observations"]),
            "resource_slack": 0,
        }
    if task_family == "temporal_tracking":
        overlap = int(factors["overlapping_causal_actions"])
        distance = int(factors["temporal_distance"])
        level = (
            "controlled_4"
            if overlap
            else {1: "controlled_1", 2: "controlled_2", 3: "controlled_3"}[distance]
        )
        return level, {"difficulty": distance + (1 if overlap else 0), **dict(factors)}
    raise ConfigError(f"Unknown factorial family {task_family!r}")


def _scale_numeric_plant(value: dict[str, Any], scale: float) -> None:
    numeric_targets: set[str] = set()
    for node_id, node in value["graph"]["nodes"].items():
        for variable_id, spec in node["variables"].items():
            initial = spec.get("initial")
            if isinstance(initial, (int, float)) and not isinstance(initial, bool):
                target = f"{node_id}.{variable_id}"
                numeric_targets.add(target)
                for key in ("initial", "drift", "hard_min", "hard_max"):
                    if key in spec:
                        spec[key] = _scaled(spec[key], scale)

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            target = item.get("target")
            if target in numeric_targets:
                for key in ("value", "amount", "min", "max", "threshold", "noise_std"):
                    if key in item:
                        item[key] = _scaled(item[key], scale)
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    for key in ("actions", "constraints", "goals", "sensors", "baseline"):
        walk(value.get(key, []))

    sensor_targets = {
        str(sensor["id"]): str(sensor.get("target"))
        for sensor in value.get("sensors", [])
    }
    for fault in value.get("faults", []):
        if fault.get("kind") in {"sensor_bias", "sensor_stuck"}:
            sensor_target = sensor_targets.get(str(fault.get("target")))
            if sensor_target in numeric_targets and "value" in fault:
                fault["value"] = _scaled(fault["value"], scale)


def _vary_irrelevant_boolean_evidence(value: dict[str, Any], instance_index: int) -> None:
    if value.get("task_family") != "conditional_safety":
        return
    irrelevant_targets = {
        str(action.get("target"))
        for action in value.get("actions", [])
        if action.get("operation") == "inspect" and action.get("role") == "distractor"
    }
    for node_id, node in value["graph"]["nodes"].items():
        for variable_id in node["variables"]:
            if any(marker in variable_id for marker in ("auxiliary", "bypass")):
                irrelevant_targets.add(f"{node_id}.{variable_id}")
    for target in irrelevant_targets:
        if "." not in target:
            continue
        node_id, variable_id = target.split(".", 1)
        spec = value["graph"]["nodes"][node_id]["variables"].get(variable_id)
        if spec and isinstance(spec.get("initial"), bool):
            spec["initial"] = bool(instance_index % 2)


def _vary_planning_resource_costs(value: dict[str, Any], instance_index: int) -> None:
    resource_id = "energy"
    resource = value["graph"].get("resources", {}).get(resource_id)
    if not isinstance(resource, dict):
        return
    actions = {
        str(action["id"]): action
        for action in value.get("actions", [])
    }
    script_actions = [
        str(command["action_id"])
        for item in value.get("baseline", {}).get("script", [])
        for command in item.get("decision", {}).get("actions", [])
    ]
    old_reference_cost = sum(
        float(actions[action_id].get("costs", {}).get(resource_id, 0.0))
        for action_id in script_actions
        if action_id in actions
    )
    original_initial = float(resource["initial"])
    margin = original_initial - old_reference_cost
    cost_actions = sorted(
        [
            action
            for action in value.get("actions", [])
            if resource_id in action.get("costs", {})
        ],
        key=lambda action: str(action["id"]),
    )
    for action_index, action in enumerate(cost_actions):
        multiplier = _INSTANCE_SCALES[
            (instance_index + action_index) % len(_INSTANCE_SCALES)
        ]
        action["costs"][resource_id] = _scaled(
            action["costs"][resource_id], multiplier
        )
        action["description"] = (
            f"{str(action.get('description', '')).rstrip()} "
            f"Listed energy cost: {_format_number(action['costs'][resource_id])}."
        ).strip()
    new_reference_cost = sum(
        float(actions[action_id].get("costs", {}).get(resource_id, 0.0))
        for action_id in script_actions
        if action_id in actions
    )
    resource["initial"] = round(new_reference_cost + margin, 6)


def _robustness_variation_dimensions(task_family: str) -> list[str]:
    common = ["operational_namespace", "action_declaration_order"]
    if task_family == "planning":
        return [*common, "per_action_resource_costs_with_fixed_reference_margin"]
    if task_family == "conditional_safety":
        return [*common, "rule_irrelevant_boolean_evidence"]
    return [*common, "numeric_plant_unit_scale"]


def _rename_operational_ids(value: dict[str, Any], suffix: str) -> None:
    token_map: dict[str, str] = {}
    nodes = value["graph"]["nodes"]
    for node_id in nodes:
        token_map[str(node_id)] = f"{node_id}_{suffix}"
    # Resource IDs remain stable because family invariants intentionally address
    # diagnostic and recovery budgets by their declared semantic names.
    # Goal, constraint, and fault IDs can intentionally match local variable IDs.
    # Keep those diagnostic IDs stable so exact-token replacement cannot rename a
    # graph variable key without also changing its dotted target.
    for collection in ("actions", "sensors"):
        for item in value.get(collection, []):
            if item.get("id") is not None:
                identifier = str(item["id"])
                token_map[identifier] = f"{identifier}_{suffix}"
    for edge in value["graph"].get("edges", []):
        identifier = str(edge["id"])
        token_map[identifier] = f"{identifier}_{suffix}"
    for script_item in value.get("baseline", {}).get("script", []):
        for subscription in script_item.get("decision", {}).get("subscriptions", []):
            if subscription.get("id"):
                identifier = str(subscription["id"])
                token_map.setdefault(identifier, f"{identifier}_{suffix}")

    node_map = {
        old: new for old, new in token_map.items() if old in value["graph"]["nodes"]
    }

    def replace(item: Any) -> Any:
        if isinstance(item, dict):
            return {replace(key): replace(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace(child) for child in item]
        if isinstance(item, tuple):
            return [replace(child) for child in item]
        if isinstance(item, str):
            if item in token_map:
                return token_map[item]
            if ":" in item:
                namespace, identifier = item.split(":", 1)
                if namespace in {"sensor", "resource"} and identifier in token_map:
                    return f"{namespace}:{token_map[identifier]}"
            if "." in item:
                first, remainder = item.split(".", 1)
                if first in node_map:
                    return f"{node_map[first]}.{remainder}"
        return item

    renamed = replace(value)
    value.clear()
    value.update(renamed)
    for node in value["graph"]["nodes"].values():
        label = str(node.get("label", "CPS component"))
        node["label"] = f"{label} [{suffix}]"


def _refresh_public_text(value: dict[str, Any], task_family: str) -> None:
    if task_family == "prospective_reasoning":
        constraint = value.get("constraints", [{}])[0] if value.get("constraints") else {}
        if constraint.get("max") is not None:
            limit = _format_number(constraint["max"])
            if value.get("generation", {}).get("parameters", {}).get(
                "content_revision"
            ) != "foundation_v1":
                value["task"]["instructions"] = (
                    f"Keep process load at or below {limit}. Use the enabled future "
                    "interaction mechanisms and conserve intervention resources."
                )
                value["task"]["objectives"] = [
                    f"Finish with process load at or below {limit}.",
                    "Respect the listed intervention resource budget.",
                ]
                value["task"]["safety_rules"] = [
                    f"Process load must never exceed {limit}."
                ]
        for action in value.get("actions", []):
            if action.get("operation") == "adjust" and action.get("amount") is not None:
                action["description"] = (
                    f"Adjust process load by {_format_number(action['amount'])} using "
                    "the listed resource cost."
                )
            elif str(action.get("id", "")).startswith("acknowledge_cycle"):
                action["description"] = (
                    "Acknowledge the cycle when its declared readiness precondition is satisfied."
                )
    elif task_family == "interference" and value.get("constraints"):
        maximum = next(
            (item.get("max") for item in value["constraints"] if item.get("max") is not None),
            None,
        )
        if maximum is not None:
            limit = _format_number(maximum)
            value["task"]["instructions"] = (
                f"Keep the sector online and temperature at or below {limit}. Compare "
                "sensor status, redundant measurements, and inspection evidence before acting."
            )
            value["task"]["objectives"] = [
                "Keep the sector online.",
                f"Keep temperature at or below {limit}.",
            ]
            for action in value.get("actions", []):
                if (
                    action.get("operation") == "adjust"
                    and action.get("amount") is not None
                ):
                    action["description"] = (
                        f"Adjust sector temperature by {_format_number(action['amount'])}."
                    )
    elif task_family == "adaptation" and value.get("goals"):
        conditions = value["goals"][0].get("condition", value["goals"][0])
        target = conditions.get("value")
        if target is not None:
            threshold = _format_number(target)
            value["task"]["instructions"] = (
                f"Maintain process output at or above {threshold} through the operating "
                "window. Infer actuator changes from public outcomes and plant response."
            )
            for action in value.get("actions", []):
                if (
                    action.get("operation") == "adjust"
                    and action.get("amount") is not None
                ):
                    action["description"] = (
                        f"Adjust its declared target by {_format_number(action['amount'])} "
                        "using the listed recovery cost."
                    )


def _scaled(value: Any, scale: float) -> Any:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    return round(float(value) * scale, 6)


def _format_number(value: Any) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def _hashable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    return value


def _pairwise_factor_names(
    task_family: str, row: Mapping[str, Any]
) -> tuple[str, ...]:
    excluded = {"factor_row_id"}
    if task_family == "adaptation":
        excluded.add("adaptation_profile")
    if task_family == "temporal_tracking":
        # Delay is intentionally equal to temporal distance in schema v1; counting
        # it as another independent factor would inflate pair coverage.
        excluded.add("action_effect_delay_steps")
    return tuple(key for key in row if key not in excluded)
