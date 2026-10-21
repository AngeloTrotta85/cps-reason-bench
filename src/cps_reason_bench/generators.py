"""Deterministic, inspectable generators for the initial task families."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

from cps_reason_bench.config import ConfigError, ScenarioConfig, dump_scenario, load_scenario
from cps_reason_bench.paths import (
    bundled_foundation_scenario_path,
    bundled_scenario_path,
)
from cps_reason_bench.versions import (
    GENERATOR_VERSION,
    GENERATOR_VERSION_FOUNDATION,
    GENERATOR_VERSION_V2,
)


GENERATOR_LEVELS: dict[str, dict[str, dict[str, Any]]] = {
    "prospective_reasoning": {
        "basic": {"difficulty": 1, "sensor_delay_steps": 0, "sensor_noise_std": 0.0},
        "controlled_2": {
            "difficulty": 2,
            "sensor_delay_steps": 1,
            "sensor_noise_std": 0.1,
            "add_distractor_action": True,
        },
        "controlled_3": {
            "difficulty": 3,
            "sensor_delay_steps": 0,
            "sensor_noise_std": 0.0,
            "add_distractor_action": True,
            "monitoring_mode": "event_subscription",
            "event_candidate_count": 3,
            "callback_tolerance_steps": 0,
            "interaction_cost_pressure": False,
            "content_revision": "core_v2",
        },
        "controlled_4": {
            "difficulty": 4,
            "sensor_delay_steps": 1,
            "sensor_noise_std": 0.1,
            "add_distractor_action": True,
            "monitoring_mode": "hybrid",
            "event_candidate_count": 3,
            "callback_tolerance_steps": 1,
            "interaction_cost_pressure": True,
            "content_revision": "core_v2",
        },
    },
    "planning": {
        "basic": {"difficulty": 1, "planning_depth": 3, "distractor_actions": 0},
        "controlled_2": {"difficulty": 2, "planning_depth": 4, "distractor_actions": 1},
        "controlled_3": {
            "difficulty": 3,
            "planning_depth": 3,
            "distractor_actions": 2,
            "branching_factor": 2,
            "valid_alternative_sequence_count": 2,
            "resource_slack": 1,
            "content_revision": "core_v2",
        },
        "controlled_4": {
            "difficulty": 4,
            "planning_depth": 4,
            "distractor_actions": 2,
            "branching_factor": 2,
            "valid_alternative_sequence_count": 2,
            "resource_slack": 0,
            "content_revision": "core_v2",
        },
    },
    "interference": {
        "basic": {"difficulty": 1, "misleading_sensor_count": 1, "distractor_actions": 1},
        "controlled_2": {
            "difficulty": 2,
            "misleading_sensor_count": 2,
            "distractor_actions": 2,
        },
        "controlled_3": {
            "difficulty": 3,
            "misleading_sensor_count": 3,
            "distractor_actions": 2,
            "evidence_conflict_count": 3,
            "downstream_consequence_delay_steps": 0,
            "content_revision": "core_v2",
        },
        "controlled_4": {
            "difficulty": 4,
            "misleading_sensor_count": 3,
            "distractor_actions": 3,
            "evidence_conflict_count": 3,
            "downstream_consequence_delay_steps": 2,
            "content_revision": "core_v2",
        },
    },
    "conditional_safety": {
        "basic": {"difficulty": 1, "hidden_rule_variables": 1, "distractor_actions": 0},
        "controlled_2": {
            "difficulty": 2,
            "hidden_rule_variables": 1,
            "observation_delay_steps": 1,
            "distractor_actions": 1,
        },
        "controlled_3": {
            "difficulty": 3,
            "hidden_rule_variables": 3,
            "observation_delay_steps": 1,
            "distractor_actions": 1,
            "rule_count": 2,
            "candidate_inspection_count": 3,
            "inspection_budget": 2,
            "content_revision": "core_v2",
        },
        "controlled_4": {
            "difficulty": 4,
            "hidden_rule_variables": 3,
            "observation_delay_steps": 1,
            "distractor_actions": 2,
            "rule_count": 1,
            "candidate_inspection_count": 3,
            "inspection_budget": 1,
            "content_revision": "core_v2",
        },
    },
    "adaptation": {
        "controlled_1": {
            "difficulty": 1,
            "regime_change_mechanism": "actuator_disabled",
            "regime_change_visibility": "immediate_action_rejection",
            "change_magnitude": "complete_loss",
            "plausible_explanation_count": 1,
            "replacement_strategy_count": 1,
            "feedback_clarity": "clear",
            "content_revision": "core_v2",
        },
        "controlled_2": {
            "difficulty": 2,
            "regime_change_mechanism": "actuator_scale",
            "regime_change_visibility": "delayed_physical_response",
            "change_magnitude": "severe_reduction",
            "plausible_explanation_count": 2,
            "replacement_strategy_count": 1,
            "feedback_clarity": "physical_only",
            "evidence_required_observations": 2,
            "content_revision": "core_v2",
        },
        "controlled_3": {
            "difficulty": 3,
            "regime_change_mechanism": "actuator_remap",
            "regime_change_visibility": "partial_then_inspection",
            "change_magnitude": "redirected_effect",
            "plausible_explanation_count": 3,
            "replacement_strategy_count": 1,
            "feedback_clarity": "partial",
            "evidence_required_observations": 2,
            "content_revision": "core_v2",
        },
        "controlled_4": {
            "difficulty": 4,
            "regime_change_mechanism": "actuator_remap",
            "regime_change_visibility": "observable_unsafe_effect",
            "change_magnitude": "redirected_unsafe_effect",
            "plausible_explanation_count": 3,
            "replacement_strategy_count": 2,
            "feedback_clarity": "clear",
            "evidence_required_observations": 1,
            "resource_slack": 0,
            "content_revision": "core_v2",
        },
    },
    "temporal_tracking": {
        "controlled_1": {
            "difficulty": 1,
            "temporal_distance": 1,
            "action_effect_delay_steps": 1,
            "relevant_history_items": 1,
            "temporal_distractor_count": 0,
            "overlapping_causal_actions": 0,
            "context_mode": "previous",
            "history_limit": 2,
            "content_revision": "core_v2",
        },
        "controlled_2": {
            "difficulty": 2,
            "temporal_distance": 2,
            "action_effect_delay_steps": 2,
            "relevant_history_items": 1,
            "temporal_distractor_count": 0,
            "overlapping_causal_actions": 0,
            "context_mode": "history",
            "history_limit": 3,
            "content_revision": "core_v2",
        },
        "controlled_3": {
            "difficulty": 3,
            "temporal_distance": 3,
            "action_effect_delay_steps": 3,
            "relevant_history_items": 1,
            "temporal_distractor_count": 2,
            "overlapping_causal_actions": 0,
            "context_mode": "history",
            "history_limit": 4,
            "content_revision": "core_v2",
        },
        "controlled_4": {
            "difficulty": 4,
            "temporal_distance": 3,
            "action_effect_delay_steps": 3,
            "relevant_history_items": 2,
            "temporal_distractor_count": 2,
            "overlapping_causal_actions": 2,
            "context_mode": "history",
            "history_limit": 5,
            "content_revision": "core_v2",
        },
    },
}


FOUNDATION_GENERATOR_LEVELS: dict[str, dict[str, dict[str, Any]]] = {
    "prospective_reasoning": {
        "entry_1": {
            "difficulty": 1,
            "content_revision": "foundation_v1",
            "monitoring_mode": "callback",
            "target_step": 1,
            "event_candidate_count": 0,
            "primary_manipulation": "one_step_callback",
        },
        "entry_2": {
            "difficulty": 2,
            "content_revision": "foundation_v1",
            "monitoring_mode": "event_subscription",
            "target_step": 1,
            "event_candidate_count": 1,
            "primary_manipulation": "single_reachable_event",
        },
        "entry_3": {
            "difficulty": 3,
            "content_revision": "foundation_v1",
            "monitoring_mode": "callback_or_event",
            "target_step": 1,
            "event_candidate_count": 1,
            "primary_manipulation": "monitoring_choice",
        },
        "bridge": {
            "difficulty": 4,
            "content_revision": "foundation_v1",
            "monitoring_mode": "hybrid",
            "target_step": 2,
            "event_candidate_count": 2,
            "interaction_cost_pressure": True,
            "primary_manipulation": "timing_with_monitoring_choice",
        },
    },
    "planning": {
        "entry_1": {
            "difficulty": 1,
            "content_revision": "foundation_v1",
            "planning_depth": 1,
            "distractor_actions": 0,
            "primary_manipulation": "direct_action",
        },
        "entry_2": {
            "difficulty": 2,
            "content_revision": "foundation_v1",
            "planning_depth": 2,
            "distractor_actions": 0,
            "primary_manipulation": "two_action_chain",
        },
        "entry_3": {
            "difficulty": 3,
            "content_revision": "foundation_v1",
            "planning_depth": 3,
            "distractor_actions": 0,
            "primary_manipulation": "three_action_chain",
        },
        "bridge": {
            "difficulty": 4,
            "content_revision": "foundation_v1",
            "planning_depth": 3,
            "distractor_actions": 1,
            "primary_manipulation": "linear_chain_with_distractor",
        },
    },
    "interference": {
        "entry_1": {
            "difficulty": 1,
            "content_revision": "foundation_v1",
            "evidence_structure": "consistent_high",
            "misleading_sensor_count": 0,
            "primary_manipulation": "consistent_evidence_baseline",
        },
        "entry_2": {
            "difficulty": 2,
            "content_revision": "foundation_v1",
            "evidence_structure": "explicit_stale_high",
            "misleading_sensor_count": 1,
            "primary_manipulation": "marked_stale_signal",
        },
        "entry_3": {
            "difficulty": 3,
            "content_revision": "foundation_v1",
            "evidence_structure": "stale_vs_two_current",
            "misleading_sensor_count": 1,
            "primary_manipulation": "one_vs_two_evidence_conflict",
        },
        "bridge": {
            "difficulty": 4,
            "content_revision": "foundation_v1",
            "evidence_structure": "inspect_then_control",
            "misleading_sensor_count": 1,
            "primary_manipulation": "conflict_with_useful_inspection",
        },
    },
    "conditional_safety": {
        "entry_1": {
            "difficulty": 1,
            "content_revision": "foundation_v1",
            "hidden_consequent": False,
            "candidate_inspection_count": 0,
            "inspection_budget": 2,
            "primary_manipulation": "fully_visible_rule",
        },
        "entry_2": {
            "difficulty": 2,
            "content_revision": "foundation_v1",
            "hidden_consequent": True,
            "candidate_inspection_count": 1,
            "inspection_budget": 1,
            "primary_manipulation": "single_relevant_inspection",
        },
        "entry_3": {
            "difficulty": 3,
            "content_revision": "foundation_v1",
            "hidden_consequent": True,
            "candidate_inspection_count": 2,
            "inspection_budget": 2,
            "primary_manipulation": "relevant_vs_irrelevant_inspection",
        },
        "bridge": {
            "difficulty": 4,
            "content_revision": "foundation_v1",
            "hidden_consequent": True,
            "candidate_inspection_count": 2,
            "inspection_budget": 1,
            "primary_manipulation": "inspection_choice_with_sufficient_limit",
        },
    },
    "adaptation": {
        "entry_1": {
            "difficulty": 1,
            "content_revision": "foundation_v1",
            "regime_change_mechanism": "actuator_disabled",
            "effect_scale": 0.0,
            "requires_inspection": False,
            "primary_manipulation": "explicit_rejection_then_backup",
        },
        "entry_2": {
            "difficulty": 2,
            "content_revision": "foundation_v1",
            "regime_change_mechanism": "actuator_scale",
            "effect_scale": 0.0,
            "requires_inspection": False,
            "primary_manipulation": "accepted_without_effect_then_backup",
        },
        "entry_3": {
            "difficulty": 3,
            "content_revision": "foundation_v1",
            "regime_change_mechanism": "actuator_scale",
            "effect_scale": 0.5,
            "requires_inspection": False,
            "primary_manipulation": "reduced_effect_then_backup",
        },
        "bridge": {
            "difficulty": 4,
            "content_revision": "foundation_v1",
            "regime_change_mechanism": "actuator_scale",
            "effect_scale": 0.0,
            "requires_inspection": True,
            "primary_manipulation": "partial_evidence_inspection_then_backup",
        },
    },
    "temporal_tracking": {
        "entry_1": {
            "difficulty": 1,
            "content_revision": "foundation_v1",
            "temporal_distance": 1,
            "action_effect_delay_steps": 1,
            "context_mode": "previous",
            "history_limit": 2,
            "temporal_distractor_count": 0,
            "action_choice_count": 1,
            "primary_manipulation": "single_one_step_effect",
        },
        "entry_2": {
            "difficulty": 2,
            "content_revision": "foundation_v1",
            "temporal_distance": 2,
            "action_effect_delay_steps": 2,
            "context_mode": "history",
            "history_limit": 3,
            "temporal_distractor_count": 0,
            "action_choice_count": 1,
            "primary_manipulation": "single_two_step_effect",
        },
        "entry_3": {
            "difficulty": 3,
            "content_revision": "foundation_v1",
            "temporal_distance": 2,
            "action_effect_delay_steps": 2,
            "context_mode": "history",
            "history_limit": 3,
            "temporal_distractor_count": 1,
            "action_choice_count": 1,
            "primary_manipulation": "delayed_effect_with_irrelevant_change",
        },
        "bridge": {
            "difficulty": 4,
            "content_revision": "foundation_v1",
            "temporal_distance": 2,
            "action_effect_delay_steps": 2,
            "context_mode": "history",
            "history_limit": 3,
            "temporal_distractor_count": 0,
            "action_choice_count": 2,
            "primary_manipulation": "two_initiators_one_pending_effect",
        },
    },
}


def available_generator_levels() -> dict[str, tuple[str, ...]]:
    return {
        family: tuple(levels)
        for family, levels in GENERATOR_LEVELS.items()
    }


def available_foundation_generator_levels() -> dict[str, tuple[str, ...]]:
    return {
        family: tuple(levels)
        for family, levels in FOUNDATION_GENERATOR_LEVELS.items()
    }


def generate_scenario(
    task_family: str,
    level: str,
    generation_seed: int,
    parameters: Mapping[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> ScenarioConfig:
    """Generate one scenario by applying a small, recorded family transformation."""

    legacy_levels = GENERATOR_LEVELS.get(task_family, {})
    foundation_levels = FOUNDATION_GENERATOR_LEVELS.get(task_family, {})
    is_foundation = level in foundation_levels
    if level in legacy_levels:
        level_defaults = legacy_levels[level]
    elif is_foundation:
        level_defaults = foundation_levels[level]
    else:
        available = tuple(legacy_levels) + tuple(foundation_levels)
        raise ConfigError(
            f"Unknown generator family/level {task_family!r}/{level!r}; "
            f"available levels: {available}"
        )

    default_template = (
        bundled_foundation_scenario_path(task_family)
        if is_foundation
        else bundled_scenario_path(task_family)
    )
    selected_template = Path(template_path) if template_path else default_template
    template = load_scenario(selected_template)
    if template.task_family != task_family:
        raise ConfigError(
            f"Generator template family {template.task_family!r} does not match {task_family!r}"
        )
    config = copy.deepcopy(template.to_mapping())
    merged_parameters = {**level_defaults, **dict(parameters or {})}
    rng = random.Random(generation_seed)
    action_order_variant = int(
        merged_parameters.get("action_order_variant", rng.randrange(3))
    )
    merged_parameters["action_order_variant"] = action_order_variant
    if task_family == "temporal_tracking":
        merged_parameters.setdefault("request_variant", generation_seed % 2)

    if task_family == "prospective_reasoning":
        _generate_prospective(config, merged_parameters)
    elif task_family == "planning":
        _generate_planning(config, merged_parameters)
    elif task_family == "interference":
        _generate_interference(config, merged_parameters)
    elif task_family == "conditional_safety":
        _generate_conditional(config, merged_parameters)
    elif task_family == "adaptation":
        _generate_adaptation(config, merged_parameters)
    elif task_family == "temporal_tracking":
        _generate_temporal_tracking(config, merged_parameters)

    _rotate(config["actions"], action_order_variant)
    config["name"] = f"{task_family}_{level}_g{generation_seed}"
    level_kind = "foundation" if is_foundation else "controlled"
    config["description"] = (
        f"Generated {task_family} scenario at {level_kind} level {level}; "
        f"generation seed {generation_seed}."
    )
    config["difficulty"] = int(merged_parameters["difficulty"])
    config["difficulty_level"] = level
    config["seed"] = generation_seed
    config["generation"] = {
        "generator_id": f"{task_family}_generator",
        "generator_version": _generator_version(
            task_family, level, merged_parameters
        ),
        "task_family": task_family,
        "level": level,
        "generation_seed": generation_seed,
        "parameters": copy.deepcopy(merged_parameters),
        "template": selected_template.name,
    }
    profile = {"family_parameters": copy.deepcopy(merged_parameters)}
    reasoning_profile = _reasoning_profile(task_family, level, merged_parameters)
    if reasoning_profile is not None:
        profile["reasoning_profile"] = reasoning_profile
    config["difficulty_profile"] = profile
    config["reference"] = {
        "kind": "feasible_reference",
        "evidence": "scripted policy replay",
        "optimality": "not_claimed",
    }
    return ScenarioConfig.from_mapping(config)


def export_generated_scenario(
    task_family: str,
    level: str,
    generation_seed: int,
    output_path: str | Path,
    parameters: Mapping[str, Any] | None = None,
    template_path: str | Path | None = None,
) -> ScenarioConfig:
    scenario = generate_scenario(
        task_family,
        level,
        generation_seed,
        parameters,
        template_path,
    )
    dump_scenario(scenario, output_path)
    return scenario


def _generate_prospective(config: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    if parameters.get("content_revision") == "foundation_v1":
        _generate_foundation_prospective(config, parameters)
        return
    config["sensors"][0]["delay_steps"] = int(parameters["sensor_delay_steps"])
    config["sensors"][0]["noise_std"] = float(parameters["sensor_noise_std"])
    if parameters.get("add_distractor_action"):
        config["actions"].append(
            {
                "id": "small_adjustment",
                "operation": "adjust",
                "target": "reactor.load",
                "amount": -1.0,
                "costs": {"intervention_energy": 1.0},
                "role": "distractor",
                "description": "Use all intervention energy for an insufficient load reduction.",
            }
        )
    if parameters.get("content_revision") == "core_v2":
        config["task"]["instructions"] = (
            "Keep reactor load at or below 7 while minimizing intervention and policy "
            "invocations. Use the enabled callback and event-monitoring mechanisms when "
            "they improve timely control."
        )

    monitoring_mode = parameters.get("monitoring_mode")
    if monitoring_mode == "event_subscription":
        config["interaction"] = {
            "mode": "event_driven",
            "initial_invocation": True,
            "max_actions_per_interaction": 1,
            "allowed_event_types": [
                "variable_threshold",
                "resource_threshold",
                "safety_violation",
            ],
        }
        config["context"].update(
            {"mode": "current", "include_public_events": True}
        )
        config["evaluation"].pop("callback_target_window", None)
        config["evaluation"].update(
            {
                "relevant_event_step": 3,
                "relevant_action": "relieve_load",
                "required_subscription_event_type": "variable_threshold",
                "success_requires_required_actions": True,
            }
        )
        config["baseline"]["script"] = [
            {
                "at_step": 0,
                "decision": {
                    "actions": [],
                    "callback": None,
                    "subscriptions": [
                        {
                            "operation": "add",
                            "id": "watch_load",
                            "event_type": "variable_threshold",
                            "filters": {
                                "target": "reactor.load",
                                "operator": "ge",
                                "threshold": 5.0,
                            },
                        }
                    ],
                    "context_requests": ["active_subscriptions"],
                },
            },
            {
                "at_step": 3,
                "decision": {
                    "actions": [{"action_id": "relieve_load"}],
                    "callback": None,
                    "subscriptions": [
                        {"operation": "remove", "id": "watch_load"}
                    ],
                    "context_requests": [],
                },
            },
        ]
    elif monitoring_mode == "hybrid":
        variables = config["graph"]["nodes"]["reactor"]["variables"]
        variables["coolant_demand"] = {
            "initial": 1.0,
            "drift": 0.5,
            "hard_min": 0.0,
            "hard_max": 8.0,
        }
        config["sensors"].append(
            {
                "id": "coolant_demand_sensor",
                "target": "reactor.coolant_demand",
                "observable": True,
            }
        )
        config["interaction"].update(
            {
                "mode": "scheduled_event",
                "initial_invocation": True,
                "min_callback_steps": 1,
                "max_callback_steps": 6,
                "allowed_event_types": [
                    "variable_threshold",
                    "resource_threshold",
                    "safety_violation",
                ],
            }
        )
        config["context"].update(
            {"mode": "history", "history_limit": 3, "include_public_events": True}
        )
        config["evaluation"].update(
            {
                "callback_target_window": [3, 4],
                "relevant_event_step": 3,
                "relevant_action": "relieve_load",
                "required_subscription_event_type": "variable_threshold",
                "success_requires_required_actions": True,
            }
        )
        config["baseline"]["script"] = [
            {
                "at_step": 0,
                "decision": {
                    "actions": [],
                    "callback": {"after_steps": 4},
                    "subscriptions": [
                        {
                            "operation": "add",
                            "id": "watch_load",
                            "event_type": "variable_threshold",
                            "filters": {
                                "target": "reactor.load",
                                "operator": "ge",
                                "threshold": 5.0,
                            },
                        }
                    ],
                    "context_requests": ["active_subscriptions"],
                },
            },
            {
                "at_step": 3,
                "decision": {
                    "actions": [{"action_id": "relieve_load"}],
                    "callback": {"cancel": True},
                    "subscriptions": [
                        {"operation": "remove", "id": "watch_load"}
                    ],
                    "context_requests": [],
                },
            },
        ]


def _generate_planning(config: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    if parameters.get("content_revision") == "foundation_v1":
        _generate_foundation_planning(config, parameters)
        return
    if int(parameters["planning_depth"]) >= 4:
        variables = config["graph"]["nodes"]["plant"]["variables"]
        variables["calibration_ready"] = {"initial": False}
        repair = next(action for action in config["actions"] if action["id"] == "repair_output")
        repair["preconditions"].insert(
            2,
            {"target": "plant.calibration_ready", "operator": "eq", "value": True},
        )
        calibration = {
            "id": "calibrate_output",
            "operation": "set",
            "target": "plant.calibration_ready",
            "value": True,
            "costs": {"energy": 1.0},
            "preconditions": [
                {"target": "plant.power_ready", "operator": "eq", "value": True},
                {"target": "plant.maintenance_ready", "operator": "eq", "value": True},
                {"target": "plant.calibration_ready", "operator": "eq", "value": False},
            ],
            "description": "Calibrate the output after maintenance is positioned.",
        }
        repair_index = config["actions"].index(repair)
        config["actions"].insert(repair_index, calibration)
        config["graph"]["resources"]["energy"]["initial"] = 4.0
        config["max_steps"] = 7
        config["evaluation"]["required_actions"].insert(2, "calibrate_output")
        config["evaluation"]["expected_action_sequence"].insert(2, "calibrate_output")
        config["evaluation"]["reference_action_count"] = 4
        config["baseline"]["script"] = [
            _script_action(0, "restore_power"),
            _script_action(1, "position_maintenance"),
            _script_action(2, "calibrate_output"),
            _script_action(3, "repair_output"),
        ]
    if (
        int(parameters.get("planning_depth", 3)) == 3
        and int(parameters.get("branching_factor", 1)) >= 2
    ):
        remote = {
            "id": "deploy_remote_maintenance",
            "operation": "set",
            "target": "plant.maintenance_ready",
            "value": True,
            "costs": {"energy": 1.0},
            "preconditions": [
                {"target": "plant.power_ready", "operator": "eq", "value": True},
                {
                    "target": "plant.maintenance_ready",
                    "operator": "eq",
                    "value": False,
                },
            ],
            "description": "Deploy remote maintenance after local power is restored.",
        }
        repair = next(
            action for action in config["actions"] if action["id"] == "repair_output"
        )
        config["actions"].insert(config["actions"].index(repair), remote)
        config["evaluation"]["required_actions"] = [
            "restore_power",
            "repair_output",
        ]
        config["evaluation"]["valid_action_sequences"] = [
            ["restore_power", "position_maintenance", "repair_output"],
            ["restore_power", "deploy_remote_maintenance", "repair_output"],
        ]

    if (
        int(parameters.get("planning_depth", 3)) >= 4
        and int(parameters.get("branching_factor", 1)) >= 2
    ):
        calibration = next(
            action for action in config["actions"] if action["id"] == "calibrate_output"
        )
        calibration["preconditions"] = [
            condition
            for condition in calibration["preconditions"]
            if condition.get("target") != "plant.maintenance_ready"
        ]
        config["evaluation"]["valid_action_sequences"] = [
            [
                "restore_power",
                "position_maintenance",
                "calibrate_output",
                "repair_output",
            ],
            [
                "restore_power",
                "calibrate_output",
                "position_maintenance",
                "repair_output",
            ],
        ]

    distractor_count = int(parameters.get("distractor_actions", 0))
    if distractor_count:
        config["actions"].append(
            {
                "id": "reset_power",
                "operation": "set",
                "target": "plant.power_ready",
                "value": False,
                "role": "distractor",
                "description": "Disable restored power and invalidate downstream prerequisites.",
            }
        )
        config["evaluation"].setdefault("forbidden_actions", []).append("reset_power")
    if distractor_count >= 2:
        config["actions"].append(
            {
                "id": "diagnostic_cycle",
                "operation": "noop",
                "costs": {"energy": 1.0},
                "role": "distractor",
                "description": "Consume one energy unit on a nonessential diagnostic cycle.",
            }
        )
        config["evaluation"].setdefault("forbidden_actions", []).append(
            "diagnostic_cycle"
        )
        config["evaluation"].setdefault("unnecessary_actions", []).append(
            "diagnostic_cycle"
        )

    if parameters.get("content_revision") == "core_v2":
        depth = int(parameters.get("planning_depth", 3))
        reference_cost = float(depth)
        slack = float(parameters.get("resource_slack", 0))
        config["graph"]["resources"]["energy"]["initial"] = reference_cost + slack
        config["evaluation"]["success_requires_required_actions"] = True
        config["task"]["instructions"] = (
            "Restore output integrity using the listed action prerequisites. Choose a "
            "valid ordering and preserve enough energy to complete the repair."
        )


def _generate_interference(config: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    if parameters.get("content_revision") == "foundation_v1":
        _generate_foundation_interference(config, parameters)
        return
    if int(parameters["misleading_sensor_count"]) >= 2:
        config["sensors"].append(
            {
                "id": "secondary_alarm",
                "target": "sector.temperature",
                "observable": True,
            }
        )
        config["faults"].append(
            {
                "id": "secondary_high_bias",
                "at_step": 0,
                "kind": "sensor_bias",
                "target": "secondary_alarm",
                "value": 4.0,
            }
        )
    if int(parameters["misleading_sensor_count"]) >= 3:
        config["sensors"].append(
            {
                "id": "tertiary_alarm",
                "target": "sector.temperature",
                "observable": True,
            }
        )
        config["faults"].append(
            {
                "id": "tertiary_low_bias",
                "at_step": 0,
                "kind": "sensor_bias",
                "target": "tertiary_alarm",
                "value": -2.0,
            }
        )
    if int(parameters.get("distractor_actions", 1)) >= 2:
        config["actions"].append(
            {
                "id": "boost_output",
                "operation": "adjust",
                "target": "sector.temperature",
                "amount": 1.0,
                "role": "distractor",
                "description": "Increase output and temperature despite the alarm conflict.",
            }
        )
        config["evaluation"].setdefault("forbidden_actions", []).append("boost_output")
        config["evaluation"].setdefault("unnecessary_actions", []).append("boost_output")
    if int(parameters.get("distractor_actions", 1)) >= 3:
        delay = int(parameters.get("downstream_consequence_delay_steps", 2))
        config["actions"].append(
            {
                "id": "increase_throughput",
                "operation": "adjust",
                "target": "sector.temperature",
                "amount": 5.0,
                "effect_delay_steps": delay,
                "role": "distractor",
                "description": (
                    "Increase throughput; its thermal response appears after "
                    f"{delay} simulator steps."
                ),
            }
        )
        config["evaluation"].setdefault("forbidden_actions", []).append(
            "increase_throughput"
        )
        config["evaluation"].setdefault("unnecessary_actions", []).append(
            "increase_throughput"
        )
    config["evaluation"]["misleading_observation_count"] = int(
        parameters["misleading_sensor_count"]
    )
    if parameters.get("content_revision") == "core_v2":
        config["task"]["instructions"] = (
            "Keep the sector online and temperature at or below 8. Compare available "
            "sensor status, redundant measurements, and inspection evidence before "
            "selecting an intervention."
        )
        config["evaluation"].update(
            {
                "evidence_gathering_actions": ["inspect_primary"],
                "emergency_actions": ["emergency_shutdown"],
                "downstream_consequence_delay_steps": int(
                    parameters.get("downstream_consequence_delay_steps", 0)
                ),
                "success_requires_required_actions": True,
                "success_requires_required_inspections": True,
            }
        )


def _generate_conditional(config: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    if parameters.get("content_revision") == "foundation_v1":
        _generate_foundation_conditional(config, parameters)
        return
    delay = int(parameters.get("observation_delay_steps", 0))
    config["sensors"][0]["delay_steps"] = delay
    if int(parameters.get("distractor_actions", 0)) >= 1:
        config["actions"].append(
            {
                "id": "close_valve",
                "operation": "set",
                "target": "transfer.valve_open",
                "value": False,
                "role": "distractor",
                "description": "Close the valve, directly violating the active-pump rule.",
            }
        )
        config["evaluation"].setdefault("forbidden_actions", []).append("close_valve")

    rule_count = int(parameters.get("rule_count", 1))
    candidate_count = int(parameters.get("candidate_inspection_count", 1))
    if rule_count >= 2:
        variables = config["graph"]["nodes"]["transfer"]["variables"]
        variables.update(
            {
                "backup_active": {"initial": True},
                "vent_open": {"initial": False},
            }
        )
        config["sensors"].append(
            {
                "id": "backup_status",
                "target": "transfer.backup_active",
                "observable": True,
            }
        )
        config["actions"].extend(
            [
                {
                    "id": "inspect_vent",
                    "operation": "inspect",
                    "target": "transfer.vent_open",
                    "costs": {"diagnostic_budget": 1.0},
                    "description": "Inspect the otherwise hidden backup vent state.",
                },
                {
                    "id": "open_vent",
                    "operation": "set",
                    "target": "transfer.vent_open",
                    "value": True,
                    "description": "Open the backup vent.",
                },
                {
                    "id": "stop_backup",
                    "operation": "set",
                    "target": "transfer.backup_active",
                    "value": False,
                    "description": "Stop the backup transfer path.",
                },
            ]
        )
        if candidate_count >= 3:
            variables["auxiliary_ready"] = {"initial": True}
            config["actions"].append(
                {
                    "id": "inspect_auxiliary",
                    "operation": "inspect",
                    "target": "transfer.auxiliary_ready",
                    "costs": {"diagnostic_budget": 1.0},
                    "role": "distractor",
                    "description": "Inspect a state not used by either target rule.",
                }
            )
            config["evaluation"].setdefault("unnecessary_actions", []).append(
                "inspect_auxiliary"
            )
            config["evaluation"].setdefault("forbidden_actions", []).append(
                "inspect_auxiliary"
            )
        config["constraints"].append(
            {
                "id": "active_backup_requires_open_vent",
                "kind": "conditional",
                "if": {
                    "target": "transfer.backup_active",
                    "operator": "eq",
                    "value": True,
                },
                "then": {
                    "target": "transfer.vent_open",
                    "operator": "eq",
                    "value": True,
                },
            }
        )
        config["goals"] = [
            {
                "id": "both_conditional_rules_satisfied",
                "all": [
                    {
                        "target": "transfer.valve_open",
                        "operator": "eq",
                        "value": True,
                    },
                    {
                        "target": "transfer.vent_open",
                        "operator": "eq",
                        "value": True,
                    },
                ],
            }
        ]
        config["graph"]["resources"]["diagnostic_budget"]["initial"] = 2.0
        config["max_steps"] = 6
        config["context"].update({"mode": "history", "history_limit": 4})
        config["evaluation"].update(
            {
                "required_inspections": ["inspect_valve", "inspect_vent"],
                "required_actions": ["open_valve", "open_vent"],
                "expected_action_sequence": [
                    "inspect_valve",
                    "open_valve",
                    "inspect_vent",
                    "open_vent",
                ],
                "reference_action_count": 4,
                "allowed_safety_violations": 6,
            }
        )
        config["baseline"]["script"] = [
            _script_action(0, "inspect_valve"),
            _script_action(1, "open_valve"),
            _script_action(2, "inspect_vent"),
            _script_action(3, "open_vent"),
        ]
    elif candidate_count >= 2:
        variables = config["graph"]["nodes"]["transfer"]["variables"]
        candidate_specs = [
            (
                "bypass_ready",
                False,
                "inspect_bypass",
                "Inspect a bypass state not used by the target rule.",
            ),
            (
                "auxiliary_ready",
                True,
                "inspect_auxiliary",
                "Inspect an auxiliary state not used by the target rule.",
            ),
        ]
        for variable_id, initial, action_id, description in candidate_specs[
            : candidate_count - 1
        ]:
            variables[variable_id] = {"initial": initial}
            config["actions"].append(
                {
                    "id": action_id,
                    "operation": "inspect",
                    "target": f"transfer.{variable_id}",
                    "costs": {"diagnostic_budget": 1.0},
                    "role": "distractor",
                    "description": description,
                }
            )
            config["evaluation"].setdefault("unnecessary_actions", []).append(
                action_id
            )
            config["evaluation"].setdefault("forbidden_actions", []).append(action_id)
        config["context"].update({"mode": "history", "history_limit": 3})

    if parameters.get("content_revision") == "core_v2":
        config["graph"]["resources"]["diagnostic_budget"]["initial"] = float(
            parameters.get("inspection_budget", 1)
        )
        config["evaluation"].update(
            {
                "success_requires_required_actions": True,
                "success_requires_required_inspections": True,
                "candidate_inspection_actions": [
                    str(action["id"])
                    for action in config["actions"]
                    if action.get("operation") == "inspect"
                ],
            }
        )
        config["task"]["instructions"] = (
            "Enforce every listed conditional safety rule. Use the available inspection "
            "budget to obtain relevant evidence and then restore a valid physical state."
        )


def _generate_adaptation(config: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    if parameters.get("content_revision") == "foundation_v1":
        _generate_foundation_adaptation(config, parameters)
        return
    mode = str(parameters["regime_change_mechanism"])
    fault = config["faults"][0]
    fault.update(
        {
            "at_step": 2,
            "target": "primary_boost",
            "announced": False,
        }
    )
    config["task"]["instructions"] = (
        "Maintain process output at or above 7 through the end of the operating "
        "window. Use observed plant response and public action outcomes to select an "
        "effective actuator."
    )
    config["evaluation"].update(
        {
            "success_requires_required_actions": True,
            "success_requires_adaptation": True,
        }
    )

    if mode == "actuator_disabled":
        fault.update(
            {
                "id": "primary_drive_loss",
                "kind": "actuator_disabled",
            }
        )
        fault.pop("value", None)
        config["evaluation"]["adaptation"].update(
            {
                "regime_change_step": 2,
                "sufficient_evidence_step": 3,
                "evidence_clarity": "clear",
                "evidence_required_observations": 1,
            }
        )
        return

    config["evaluation"].pop("reference_expected_rejections", None)
    if mode == "actuator_scale":
        fault.update(
            {
                "id": "primary_effectiveness_loss",
                "kind": "actuator_scale",
                "value": 0.25,
            }
        )
        config["sensors"][0]["delay_steps"] = 1
        config["graph"]["resources"]["recovery_energy"]["initial"] = 6.0
        config["max_steps"] = 7
        config["context"].update(
            {
                "mode": "history",
                "history_limit": 5,
                "include_action_outcomes": False,
            }
        )
        config["evaluation"].update(
            {
                "expected_action_sequence": [
                    "primary_boost",
                    "primary_boost",
                    "primary_boost",
                    "primary_boost",
                    "backup_boost",
                ],
                "reference_action_count": 5,
            }
        )
        config["evaluation"]["adaptation"].update(
            {
                "regime_change_step": 2,
                "sufficient_evidence_step": 4,
                "evidence_clarity": "omitted_action_outcome_with_delayed_sensor",
                "evidence_required_observations": 2,
            }
        )
        config["baseline"]["script"] = [
            _script_action(0, "primary_boost"),
            _script_action(1, "primary_boost"),
            _script_action(2, "primary_boost"),
            _script_action(3, "primary_boost"),
            _script_action(4, "backup_boost"),
        ]
        return

    fault.update(
        {
            "id": "primary_target_redirected",
            "kind": "actuator_remap",
            "value": "process.waste_heat",
        }
    )
    config["graph"]["resources"]["recovery_energy"]["initial"] = 5.0
    config["max_steps"] = 6
    config["context"].update(
        {"mode": "history", "history_limit": 5, "include_action_outcomes": True}
    )

    if int(parameters.get("replacement_strategy_count", 1)) == 1:
        config["sensors"][0]["delay_steps"] = 1
        config["sensors"][1]["delay_steps"] = 1
        config["graph"]["resources"]["diagnostic_budget"] = {
            "initial": 1.0,
            "min": 0.0,
            "observable": True,
        }
        backup = next(
            action for action in config["actions"] if action["id"] == "backup_boost"
        )
        config["actions"].insert(
            config["actions"].index(backup),
            {
                "id": "inspect_primary",
                "operation": "inspect",
                "target": "primary_boost",
                "costs": {"diagnostic_budget": 1.0},
                "description": "Inspect the primary actuator's current operating state.",
            },
        )
        config["evaluation"].update(
            {
                "required_inspections": ["inspect_primary"],
                "success_requires_required_inspections": True,
                "expected_action_sequence": [
                    "primary_boost",
                    "primary_boost",
                    "primary_boost",
                    "inspect_primary",
                    "backup_boost",
                ],
                "reference_action_count": 5,
            }
        )
        config["evaluation"]["adaptation"].update(
            {
                "regime_change_step": 2,
                "sufficient_evidence_step": 4,
                "evidence_clarity": "partial_then_inspection",
                "evidence_required_observations": 2,
                "unnecessary_exploration_actions": [],
            }
        )
        config["baseline"]["script"] = [
            _script_action(0, "primary_boost"),
            _script_action(1, "primary_boost"),
            _script_action(2, "primary_boost"),
            _script_action(3, "inspect_primary"),
            _script_action(4, "backup_boost"),
        ]
        return

    backup = next(
        action for action in config["actions"] if action["id"] == "backup_boost"
    )
    backup["additional_effects"] = [
        {"operation": "adjust", "target": "process.waste_heat", "amount": -3.0}
    ]
    config["actions"].append(
        {
            "id": "rapid_bypass",
            "operation": "adjust",
            "target": "process.output",
            "amount": 6.0,
            "costs": {"recovery_energy": 2.0},
            "additional_effects": [
                {"operation": "adjust", "target": "process.waste_heat", "amount": 4.0}
            ],
            "role": "distractor",
            "description": (
                "Use the rapid bypass for strong output recovery with substantial waste heat."
            ),
        }
    )
    config["constraints"].append(
        {
            "id": "waste_heat_limit",
            "kind": "bounds",
            "target": "process.waste_heat",
            "max": 2.0,
        }
    )
    config["evaluation"].update(
        {
            "forbidden_actions": ["rapid_bypass"],
            "unnecessary_actions": ["rapid_bypass"],
            "expected_action_sequence": [
                "primary_boost",
                "primary_boost",
                "primary_boost",
                "backup_boost",
            ],
            "reference_action_count": 4,
            "allowed_safety_violations": 2,
        }
    )
    config["evaluation"]["adaptation"].update(
        {
            "regime_change_step": 2,
            "replacement_actions": ["backup_boost", "rapid_bypass"],
            "safe_replacement_actions": ["backup_boost"],
            "sufficient_evidence_step": 3,
            "evidence_clarity": "clear",
            "evidence_required_observations": 1,
            "unnecessary_exploration_actions": ["rapid_bypass"],
        }
    )
    config["baseline"]["script"] = [
        _script_action(0, "primary_boost"),
        _script_action(1, "primary_boost"),
        _script_action(2, "primary_boost"),
        _script_action(3, "backup_boost"),
    ]


def _generate_temporal_tracking(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    if parameters.get("content_revision") == "foundation_v1":
        _generate_foundation_temporal(config, parameters)
        return
    delay = int(parameters["action_effect_delay_steps"])
    variant = int(parameters.get("request_variant", 0))
    first_channel = "A" if variant % 2 == 0 else "B"
    second_channel = "B" if first_channel == "A" else "A"
    first_start = f"start_channel_{first_channel.lower()}"
    first_route = f"route_channel_{first_channel.lower()}"
    config["task"]["instructions"] = (
        "Route each completed actuator response to the channel requested when its "
        "command was issued. Command responses appear after their documented delay; "
        "use only explicit benchmark context to associate responses with earlier requests."
    )

    if int(parameters.get("overlapping_causal_actions", 0)) >= 2:
        variables = config["graph"]["nodes"]["actuator_bank"]["variables"]
        variables.clear()
        variables.update(
            {
                "requested_channel_one": {"initial": first_channel},
                "requested_channel_two": {"initial": second_channel},
                "response_ready": {"initial": False},
                "response_source": {"initial": "none"},
                "processed_count": {"initial": 0, "hard_min": 0, "hard_max": 2},
                "ambient_load": {
                    "initial": 1.0,
                    "drift": 0.5,
                    "hard_min": 0.0,
                    "hard_max": 8.0,
                },
            }
        )
        config["graph"]["resources"]["command_budget"]["initial"] = 2.0
        config["sensors"] = [
            {
                "id": "request_one_sensor",
                "target": "actuator_bank.requested_channel_one",
                "observable": True,
            },
            {
                "id": "request_two_sensor",
                "target": "actuator_bank.requested_channel_two",
                "observable": True,
            },
            {
                "id": "response_sensor",
                "target": "actuator_bank.response_ready",
                "observable": True,
            },
            {
                "id": "ambient_sensor",
                "target": "actuator_bank.ambient_load",
                "observable": True,
            },
        ]
        config["faults"] = [
            {
                "id": "request_one_display_expires",
                "at_step": 2,
                "kind": "sensor_unavailable",
                "target": "request_one_sensor",
                "announced": False,
            },
            {
                "id": "request_two_display_expires",
                "at_step": 2,
                "kind": "sensor_unavailable",
                "target": "request_two_sensor",
                "announced": False,
            },
        ]
        for action in config["actions"]:
            if str(action["id"]).startswith("start_channel_"):
                action["effect_delay_steps"] = delay
                action["description"] = (
                    f"Issue {action['id'].removeprefix('start_channel_').upper()} channel "
                    f"command; its response appears after {delay} steps."
                )
        config["goals"][0]["condition"]["value"] = 2
        config["max_steps"] = 6
        config["context"].update(
            {"mode": "history", "history_limit": int(parameters["history_limit"])}
        )
        second_start = f"start_channel_{second_channel.lower()}"
        second_route = f"route_channel_{second_channel.lower()}"
        expected = [first_start, second_start, first_route, second_route]
        config["evaluation"].update(
            {
                "required_actions": expected,
                "expected_action_sequence": expected,
                "expected_actions_by_step": {
                    0: first_start,
                    1: second_start,
                    3: first_route,
                    4: second_route,
                },
                "reference_action_count": 4,
            }
        )
        config["baseline"]["script"] = [
            _script_action(0, first_start),
            _script_action(1, second_start),
            _script_action(3, first_route),
            _script_action(4, second_route),
        ]
        selector_sensors = ["request_one_sensor", "request_two_sensor"]
    else:
        variables = config["graph"]["nodes"]["actuator_bank"]["variables"]
        variables["requested_channel"]["initial"] = first_channel
        for action in config["actions"]:
            if str(action["id"]).startswith("start_channel_"):
                action["effect_delay_steps"] = delay
                action["description"] = (
                    f"Issue {action['id'].removeprefix('start_channel_').upper()} channel "
                    f"command; its response appears after {delay} steps."
                )
        config["faults"][0]["at_step"] = 1
        config["max_steps"] = delay + 2
        config["context"].update(
            {
                "mode": str(parameters["context_mode"]),
                "history_limit": int(parameters["history_limit"]),
            }
        )
        if int(parameters.get("temporal_distractor_count", 0)):
            variables.update(
                {
                    "ambient_load": {
                        "initial": 1.0,
                        "drift": 0.5,
                        "hard_min": 0.0,
                        "hard_max": 8.0,
                    },
                    "vibration": {
                        "initial": 0.0,
                        "drift": 0.2,
                        "hard_min": 0.0,
                        "hard_max": 4.0,
                    },
                }
            )
            config["sensors"].extend(
                [
                    {
                        "id": "ambient_sensor",
                        "target": "actuator_bank.ambient_load",
                        "observable": True,
                    },
                    {
                        "id": "vibration_sensor",
                        "target": "actuator_bank.vibration",
                        "observable": True,
                    },
                ]
            )
            config["actions"].append(
                {
                    "id": "inspect_ambient",
                    "operation": "inspect",
                    "target": "actuator_bank.ambient_load",
                    "role": "distractor",
                    "description": "Inspect an ambient variable unrelated to response routing.",
                }
            )
            config["evaluation"].setdefault("unnecessary_actions", []).append(
                "inspect_ambient"
            )
        config["evaluation"].update(
            {
                "required_actions": [first_start, first_route],
                "expected_action_sequence": [first_start, first_route],
                "expected_actions_by_step": {0: first_start, delay: first_route},
                "reference_action_count": 2,
            }
        )
        config["baseline"]["script"] = [
            _script_action(0, first_start),
            _script_action(delay, first_route),
        ]
        selector_sensors = ["request_sensor"]

    config["evaluation"].update(
        {
            "success_requires_required_actions": True,
            "success_requires_temporal_tracking": True,
            "temporal": {
                "required_temporal_distance": int(parameters["temporal_distance"]),
                "action_effect_delay_steps": delay,
                "relevant_history_items": int(parameters["relevant_history_items"]),
                "temporal_distractor_count": int(
                    parameters.get("temporal_distractor_count", 0)
                ),
                "overlapping_causal_actions": int(
                    parameters.get("overlapping_causal_actions", 0)
                ),
                "selector_sensors": selector_sensors,
                "response_sensor": "response_sensor",
                "initiation_to_response": {
                    "start_channel_a": "route_channel_a",
                    "start_channel_b": "route_channel_b",
                },
                "attribution_actions": ["route_channel_a", "route_channel_b"],
                "current_only_expected_insufficient": True,
            },
        }
    )


def _generate_foundation_prospective(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    target_step = int(parameters["target_step"])
    mode = str(parameters["monitoring_mode"])
    config["max_steps"] = target_step + 2
    config["task"]["instructions"] = (
        f"The cycle becomes ready after {target_step} simulator step"
        f"{'s' if target_step != 1 else ''}. Arrange a future invocation for that "
        "time, then acknowledge the ready cycle."
    )
    acknowledge = next(
        action for action in config["actions"] if action["id"] == "acknowledge_cycle"
    )
    acknowledge["preconditions"][0]["value"] = float(target_step)
    acknowledge["description"] = (
        f"Acknowledge the cycle after progress reaches {target_step}."
    )
    config["evaluation"] = {
        "relevant_event_step": target_step,
        "relevant_action": "acknowledge_cycle",
        "required_actions": ["acknowledge_cycle"],
        "reference_action_count": 1,
        "success_requires_required_actions": True,
        "success_requires_safety": True,
        "foundation": {
            "monitoring_mode": mode,
            "event_reachable": True,
            "interaction_cost_pressure": bool(
                parameters.get("interaction_cost_pressure", False)
            ),
        },
    }
    config["context"].update(
        {"mode": "current", "history_limit": 2, "include_public_events": True}
    )

    if mode == "callback":
        config["interaction"] = {
            "mode": "scheduled",
            "initial_invocation": True,
            "min_callback_steps": 1,
            "max_callback_steps": target_step + 1,
            "max_actions_per_interaction": 1,
        }
        config["evaluation"]["callback_target_window"] = [
            target_step,
            target_step,
        ]
        config["baseline"]["script"] = [
            {
                "at_step": 0,
                "decision": {
                    "actions": [],
                    "callback": {"after_steps": target_step},
                    "subscriptions": [],
                    "context_requests": [],
                },
            },
            _script_action(target_step, "acknowledge_cycle"),
        ]
        return

    config["interaction"] = {
        "mode": (
            "event_driven" if mode == "event_subscription" else "scheduled_event"
        ),
        "initial_invocation": True,
        "min_callback_steps": 1,
        "max_callback_steps": target_step + 1,
        "max_actions_per_interaction": 1,
        "allowed_event_types": ["variable_threshold"],
    }
    if int(parameters.get("event_candidate_count", 1)) > 1:
        config["interaction"]["allowed_event_types"].append("resource_threshold")
    if mode != "event_subscription":
        config["evaluation"]["callback_target_window"] = [
            target_step,
            target_step,
        ]
    config["evaluation"]["required_subscription_event_type"] = (
        "variable_threshold"
    )
    config["baseline"]["script"] = [
        {
            "at_step": 0,
            "decision": {
                "actions": [],
                "callback": None,
                "subscriptions": [
                    {
                        "operation": "add",
                        "id": "watch_progress",
                        "event_type": "variable_threshold",
                        "filters": {
                            "target": "process.progress",
                            "operator": "ge",
                            "threshold": float(target_step),
                        },
                    }
                ],
                "context_requests": ["active_subscriptions"],
            },
        },
        {
            "at_step": target_step,
            "decision": {
                "actions": [{"action_id": "acknowledge_cycle"}],
                "callback": None,
                "subscriptions": [
                    {"operation": "remove", "id": "watch_progress"}
                ],
                "context_requests": [],
            },
        },
    ]


def _generate_foundation_planning(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    depth = int(parameters["planning_depth"])
    activate = next(
        action for action in config["actions"] if action["id"] == "activate_output"
    )
    sequence: list[str] = []
    if depth >= 2:
        config["actions"].insert(
            0,
            {
                "id": "enable_power",
                "operation": "set",
                "target": "plant.power_ready",
                "value": True,
                "costs": {"energy": 1.0},
                "preconditions": [
                    {
                        "target": "plant.power_ready",
                        "operator": "eq",
                        "value": False,
                    }
                ],
                "description": "Enable plant power before dependent operations.",
            },
        )
        sequence.append("enable_power")
        activate.setdefault("preconditions", []).append(
            {"target": "plant.power_ready", "operator": "eq", "value": True}
        )
    if depth >= 3:
        activate_index = config["actions"].index(activate)
        config["actions"].insert(
            activate_index,
            {
                "id": "prepare_component",
                "operation": "set",
                "target": "plant.component_ready",
                "value": True,
                "costs": {"energy": 1.0},
                "preconditions": [
                    {"target": "plant.power_ready", "operator": "eq", "value": True},
                    {
                        "target": "plant.component_ready",
                        "operator": "eq",
                        "value": False,
                    },
                ],
                "description": "Prepare the component after power is enabled.",
            },
        )
        sequence.append("prepare_component")
        activate.setdefault("preconditions", []).append(
            {"target": "plant.component_ready", "operator": "eq", "value": True}
        )
    sequence.append("activate_output")
    if depth >= 2:
        config["constraints"] = [
            {
                "id": "enabled_output_requires_power",
                "kind": "conditional",
                "if": {
                    "target": "plant.output_enabled",
                    "operator": "eq",
                    "value": True,
                },
                "then": {
                    "target": "plant.power_ready",
                    "operator": "eq",
                    "value": True,
                },
            }
        ]
    if int(parameters.get("distractor_actions", 0)):
        config["actions"].append(
            {
                "id": "test_indicator",
                "operation": "noop",
                "costs": {"energy": 1.0},
                "role": "distractor",
                "description": "Consume energy on an indicator test unrelated to output enablement.",
            }
        )
    config["graph"]["resources"]["energy"]["initial"] = float(depth + 3)
    config["max_steps"] = depth + 3
    config["task"]["instructions"] = (
        "Enable process output. Follow the visible action prerequisites in a valid "
        "order; the available energy and horizon allow recovery from one mistake."
    )
    config["evaluation"] = {
        "required_actions": sequence,
        "expected_action_sequence": sequence,
        "reference_action_count": len(sequence),
        "success_requires_required_actions": True,
        "success_requires_expected_sequence": True,
        "success_requires_safety": True,
        "foundation": {
            "planning_depth": depth,
            "recovery_step_margin": 3,
            "unintended_bypass": False,
        },
    }
    if int(parameters.get("distractor_actions", 0)):
        config["evaluation"].update(
            {
                "forbidden_actions": ["test_indicator"],
                "unnecessary_actions": ["test_indicator"],
            }
        )
    config["baseline"]["script"] = [
        _script_action(step, action_id)
        for step, action_id in enumerate(sequence)
    ]


def _generate_foundation_interference(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    structure = str(parameters["evidence_structure"])
    variables = config["graph"]["nodes"]["sector"]["variables"]
    config["evaluation"]["foundation"] = {
        "evidence_structure": structure,
        "reliability_cues_observable": True,
        "hidden_fault_knowledge_required": False,
    }
    if structure == "consistent_high":
        config["task"]["instructions"] = (
            "Keep the sector online and restore temperature to at most 6 using "
            "the consistent current sensor evidence."
        )
        return

    variables["temperature"]["initial"] = 4.0
    config["faults"] = [
        {
            "id": "primary_stale_high",
            "at_step": 0,
            "kind": "sensor_stuck",
            "target": "primary_temperature",
            "value": 9.0,
            "announced": False,
        }
    ]
    config["task"]["terminate_on_goals"] = False
    config["max_steps"] = 1
    config["evaluation"].update(
        {
            "required_actions": ["wait"],
            "forbidden_actions": ["cool_sector", "emergency_shutdown"],
            "unnecessary_actions": ["cool_sector", "emergency_shutdown"],
            "emergency_actions": ["emergency_shutdown"],
            "evidence_gathering_actions": [],
            "reference_action_count": 1,
            "success_requires_required_actions": True,
        }
    )
    config["baseline"]["script"] = [_script_action(0, "wait")]
    config["task"]["instructions"] = (
        "Keep the online sector unchanged when current reliable temperature is "
        "normal. Sensor status identifies readings that are stale."
    )

    if structure == "stale_vs_two_current":
        config["sensors"].append(
            {
                "id": "secondary_temperature",
                "target": "sector.temperature",
                "observable": True,
            }
        )
        return
    if structure != "inspect_then_control":
        return

    variables["temperature"]["initial"] = 7.0
    config["task"]["terminate_on_goals"] = True
    config["max_steps"] = 4
    config["actions"].insert(
        0,
        {
            "id": "inspect_primary",
            "operation": "inspect",
            "target": "primary_temperature",
            "description": "Inspect the primary temperature sensor and its current status.",
        },
    )
    config["evaluation"].update(
        {
            "required_actions": ["inspect_primary", "cool_sector"],
            "required_inspections": ["inspect_primary"],
            "forbidden_actions": ["emergency_shutdown"],
            "unnecessary_actions": ["emergency_shutdown"],
            "evidence_gathering_actions": ["inspect_primary"],
            "reference_action_count": 2,
            "success_requires_required_actions": True,
            "success_requires_required_inspections": True,
        }
    )
    config["baseline"]["script"] = [
        _script_action(0, "inspect_primary"),
        _script_action(1, "cool_sector"),
    ]
    config["task"]["instructions"] = (
        "Keep the sector online and reduce temperature to at most 6. A salient "
        "reading conflicts with a current reliable reading; inspect the uncertain "
        "sensor before applying a consequential control."
    )


def _generate_foundation_conditional(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    hidden = bool(parameters["hidden_consequent"])
    candidate_count = int(parameters["candidate_inspection_count"])
    budget = int(parameters["inspection_budget"])
    sequence: list[str] = []
    if hidden:
        config["sensors"] = [
            sensor for sensor in config["sensors"] if sensor["id"] != "valve_sensor"
        ]
        config["actions"].insert(
            0,
            {
                "id": "inspect_valve",
                "operation": "inspect",
                "target": "transfer.valve_open",
                "costs": {"diagnostic_budget": 1.0},
                "description": "Inspect the hidden valve state used by the safety rule.",
            },
        )
        sequence.append("inspect_valve")
    if candidate_count >= 2:
        config["actions"].insert(
            1,
            {
                "id": "inspect_auxiliary",
                "operation": "inspect",
                "target": "transfer.auxiliary_ready",
                "costs": {"diagnostic_budget": 1.0},
                "role": "distractor",
                "description": "Inspect an auxiliary state not used by the target rule.",
            },
        )
    sequence.append("open_valve")
    config["graph"]["resources"]["diagnostic_budget"]["initial"] = float(budget)
    config["max_steps"] = len(sequence) + 3
    config["task"]["instructions"] = (
        "Enforce the rule that an active pump requires an open valve. Use only "
        "inspection evidence relevant to that rule when the valve state is hidden."
    )
    required_inspections = ["inspect_valve"] if hidden else []
    config["evaluation"] = {
        "required_actions": ["open_valve"],
        "required_inspections": required_inspections,
        "expected_action_sequence": sequence,
        "reference_action_count": len(sequence),
        "allowed_safety_violations": len(sequence) + 1,
        "success_requires_required_actions": True,
        "success_requires_required_inspections": hidden,
        "success_requires_safety": True,
        "candidate_inspection_actions": [
            action["id"]
            for action in config["actions"]
            if action.get("operation") == "inspect"
        ],
        "foundation": {
            "rule_state_visible_or_inspectable": True,
            "inspection_budget_sufficient": budget >= len(required_inspections),
        },
    }
    if candidate_count >= 2:
        config["evaluation"].update(
            {
                "forbidden_actions": ["inspect_auxiliary"],
                "unnecessary_actions": ["inspect_auxiliary"],
            }
        )
    config["baseline"]["script"] = [
        _script_action(step, action_id)
        for step, action_id in enumerate(sequence)
    ]


def _generate_foundation_adaptation(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    mode = str(parameters["regime_change_mechanism"])
    fault = config["faults"][0]
    fault.update({"at_step": 1, "target": "primary_boost", "announced": False})
    config["evaluation"].pop("reference_expected_rejections", None)
    if mode == "actuator_disabled":
        fault.update({"id": "primary_loss", "kind": "actuator_disabled"})
        fault.pop("value", None)
        config["evaluation"]["reference_expected_rejections"] = [
            {"action_id": "primary_boost", "at_step": 1}
        ]
        clarity = "clear_rejection"
    else:
        fault.update(
            {
                "id": "primary_effectiveness_change",
                "kind": "actuator_scale",
                "value": float(parameters["effect_scale"]),
            }
        )
        clarity = "visible_physical_response"

    sequence = ["primary_boost", "primary_boost"]
    required_inspections: list[str] = []
    if parameters.get("requires_inspection"):
        backup = next(
            action for action in config["actions"] if action["id"] == "backup_boost"
        )
        config["actions"].insert(
            config["actions"].index(backup),
            {
                "id": "inspect_primary",
                "operation": "inspect",
                "target": "primary_boost",
                "costs": {"diagnostic_budget": 1.0},
                "description": "Inspect primary actuator effectiveness after the changed response.",
            },
        )
        sequence.append("inspect_primary")
        required_inspections = ["inspect_primary"]
        clarity = "physical_response_then_inspection"
    sequence.append("backup_boost")
    evidence_step = 3 if required_inspections else 2
    config["max_steps"] = len(sequence) + 2
    config["task"]["instructions"] = (
        "Raise process output to at least 5. The primary actuator initially works; "
        "use visible output and public outcomes to recognize changed effectiveness "
        "and select the independent backup."
    )
    config["evaluation"].update(
        {
            "required_actions": ["primary_boost", "backup_boost"],
            "required_inspections": required_inspections,
            "expected_action_sequence": sequence,
            "reference_action_count": len(sequence),
            "success_requires_required_actions": True,
            "success_requires_required_inspections": bool(required_inspections),
            "success_requires_adaptation": True,
            "success_requires_safety": True,
            "adaptation": {
                "regime_change_step": 1,
                "effect_target": "process.output",
                "original_actions": ["primary_boost"],
                "replacement_actions": ["backup_boost"],
                "safe_replacement_actions": ["backup_boost"],
                "sufficient_evidence_step": evidence_step,
                "evidence_clarity": clarity,
                "evidence_required_observations": 1,
                "unnecessary_exploration_actions": [],
                "recovery_resource_ids": ["recovery_energy"],
            },
            "foundation": {
                "backup_strategy_effective": True,
                "recovery_step_margin": 2,
                "one_pre_evidence_repeat_allowed": True,
            },
        }
    )
    config["baseline"]["script"] = [
        _script_action(step, action_id)
        for step, action_id in enumerate(sequence)
    ]


def _generate_foundation_temporal(
    config: dict[str, Any], parameters: Mapping[str, Any]
) -> None:
    delay = int(parameters["action_effect_delay_steps"])
    choices = int(parameters["action_choice_count"])
    variant = int(parameters.get("request_variant", 0))
    selected_channel = "A" if choices == 1 or variant % 2 == 0 else "B"
    selected_start = f"start_channel_{selected_channel.lower()}"
    selected_route = f"route_channel_{selected_channel.lower()}"
    start_a = next(
        action for action in config["actions"] if action["id"] == "start_channel_a"
    )
    start_a["effect_delay_steps"] = delay
    start_a["description"] = (
        f"Issue the requested channel A command; its response appears after {delay} steps."
    )
    mapping = {"start_channel_a": "route_channel_a"}
    if choices >= 2:
        config["graph"]["nodes"]["actuator_bank"]["variables"][
            "requested_channel"
        ]["initial"] = selected_channel
        config["actions"].extend(
            [
                {
                    "id": "start_channel_b",
                    "operation": "set",
                    "target": "actuator_bank.response_ready",
                    "value": True,
                    "effect_delay_steps": delay,
                    "costs": {"command_budget": 1.0},
                    "preconditions": [
                        {
                            "target": "actuator_bank.response_ready",
                            "operator": "eq",
                            "value": False,
                        }
                    ],
                    "additional_effects": [
                        {
                            "operation": "set",
                            "target": "actuator_bank.response_source",
                            "value": "B",
                        }
                    ],
                    "description": (
                        f"Issue the requested channel B command; its response appears after {delay} steps."
                    ),
                },
                {
                    "id": "route_channel_b",
                    "operation": "set",
                    "target": "actuator_bank.response_ready",
                    "value": False,
                    "preconditions": [
                        {
                            "target": "actuator_bank.response_ready",
                            "operator": "eq",
                            "value": True,
                        },
                        {
                            "target": "actuator_bank.response_source",
                            "operator": "eq",
                            "value": "B",
                        },
                    ],
                    "additional_effects": [
                        {
                            "operation": "adjust",
                            "target": "actuator_bank.processed_count",
                            "amount": 1,
                        }
                    ],
                    "description": "Route the available channel B response.",
                },
            ]
        )
        mapping["start_channel_b"] = "route_channel_b"
        config["faults"] = [
            {
                "id": "request_display_expires",
                "at_step": 1,
                "kind": "sensor_unavailable",
                "target": "request_sensor",
                "announced": False,
            }
        ]
    if int(parameters.get("temporal_distractor_count", 0)):
        config["graph"]["nodes"]["actuator_bank"]["variables"]["ambient_load"] = {
            "initial": 1.0,
            "drift": 0.5,
            "hard_min": 0.0,
            "hard_max": 5.0,
        }
        config["sensors"].append(
            {
                "id": "ambient_sensor",
                "target": "actuator_bank.ambient_load",
                "observable": True,
            }
        )
    config["max_steps"] = delay + 3
    config["context"].update(
        {
            "mode": str(parameters["context_mode"]),
            "history_limit": int(parameters["history_limit"]),
            "include_action_outcomes": True,
        }
    )
    config["task"]["instructions"] = (
        "Issue the requested channel command, then route its response after the "
        "documented delay using only the explicit interaction context."
    )
    config["evaluation"] = {
        "required_actions": [selected_start, selected_route],
        "expected_action_sequence": [selected_start, selected_route],
        "expected_actions_by_step": {0: selected_start, delay: selected_route},
        "reference_action_count": 2,
        "success_requires_required_actions": True,
        "success_requires_temporal_tracking": True,
        "success_requires_safety": True,
        "temporal": {
            "required_temporal_distance": delay,
            "action_effect_delay_steps": delay,
            "relevant_history_items": 1,
            "temporal_distractor_count": int(
                parameters.get("temporal_distractor_count", 0)
            ),
            "overlapping_causal_actions": 0,
            "selector_sensors": ["request_sensor"],
            "response_sensor": "response_sensor",
            "initiation_to_response": mapping,
            "attribution_actions": list(mapping.values()),
            "current_only_expected_insufficient": choices >= 2,
        },
        "foundation": {
            "explicit_history_sufficient": True,
            "overlapping_pending_effects": False,
            "hidden_state_required": False,
        },
    }
    config["baseline"]["script"] = [
        _script_action(0, selected_start),
        _script_action(delay, selected_route),
    ]


def _generator_version(
    task_family: str, level: str, parameters: Mapping[str, Any]
) -> str:
    if parameters.get("content_revision") == "foundation_v1":
        return GENERATOR_VERSION_FOUNDATION
    if (
        task_family in {"adaptation", "temporal_tracking"}
        or level in {"controlled_3", "controlled_4"}
        or parameters.get("content_revision") == "core_v2"
    ):
        return GENERATOR_VERSION_V2
    return GENERATOR_VERSION


def _reasoning_profile(
    task_family: str, level: str, parameters: Mapping[str, Any]
) -> dict[str, Any] | None:
    content_revision = parameters.get("content_revision")
    if content_revision == "foundation_v1":
        return {
            "intended_progression": level,
            "benchmark_layer": "foundation",
            "primary_manipulation": parameters.get("primary_manipulation"),
        }
    if content_revision != "core_v2":
        return None
    common = {"intended_progression": level}
    if task_family == "adaptation":
        return {
            **common,
            "regime_change_visibility": parameters["regime_change_visibility"],
            "change_magnitude": parameters["change_magnitude"],
            "plausible_explanation_count": int(
                parameters["plausible_explanation_count"]
            ),
            "obsolete_action_count": 1,
            "replacement_strategy_count": int(
                parameters["replacement_strategy_count"]
            ),
            "feedback_clarity": parameters["feedback_clarity"],
        }
    if task_family == "temporal_tracking":
        return {
            **common,
            "temporal_distance": int(parameters["temporal_distance"]),
            "action_effect_delay": int(parameters["action_effect_delay_steps"]),
            "relevant_historical_observations": int(
                parameters["relevant_history_items"]
            ),
            "temporal_distractor_count": int(
                parameters["temporal_distractor_count"]
            ),
            "overlapping_causal_actions": int(
                parameters["overlapping_causal_actions"]
            ),
            "required_context_mode": parameters["context_mode"],
            "required_history_limit": int(parameters["history_limit"]),
        }
    if task_family == "prospective_reasoning":
        return {
            **common,
            "monitoring_mode": parameters.get("monitoring_mode", "callback_interval"),
            "callback_tolerance_steps": int(
                parameters.get("callback_tolerance_steps", 1)
            ),
            "event_candidate_count": int(parameters.get("event_candidate_count", 2)),
            "monitoring_alternative_count": 2,
            "interaction_cost_pressure": bool(
                parameters.get("interaction_cost_pressure", False)
            ),
        }
    if task_family == "planning":
        return {
            **common,
            "prerequisite_depth": int(parameters["planning_depth"]),
            "branching_factor": int(parameters.get("branching_factor", 1)),
            "valid_alternative_sequence_count": int(
                parameters.get("valid_alternative_sequence_count", 1)
            ),
            "distractor_count": int(parameters.get("distractor_actions", 0)),
            "resource_slack": float(parameters.get("resource_slack", 0)),
        }
    if task_family == "interference":
        return {
            **common,
            "salience_strength": "high",
            "evidence_conflict_count": int(
                parameters.get(
                    "evidence_conflict_count", parameters["misleading_sensor_count"]
                )
            ),
            "distractor_count": int(parameters.get("distractor_actions", 1)),
            "downstream_consequence_delay": int(
                parameters.get("downstream_consequence_delay_steps", 0)
            ),
        }
    if task_family == "conditional_safety":
        return {
            **common,
            "rule_count": int(parameters.get("rule_count", 1)),
            "candidate_inspection_count": int(
                parameters.get("candidate_inspection_count", 1)
            ),
            "falsifying_state_count": int(parameters.get("rule_count", 1)),
            "inspection_budget": int(parameters.get("inspection_budget", 1)),
        }
    return common


def _rotate(items: list[Any], offset: int) -> None:
    if not items:
        return
    offset %= len(items)
    items[:] = items[offset:] + items[:offset]


def _script_action(step: int, action_id: str) -> dict[str, Any]:
    return {
        "at_step": step,
        "decision": {
            "actions": [{"action_id": action_id}],
            "callback": None,
            "subscriptions": [],
            "context_requests": [],
        },
    }
