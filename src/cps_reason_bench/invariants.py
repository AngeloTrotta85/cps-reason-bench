"""Conservative checks that a scenario contains its intended reasoning challenge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.protocol import ActionCommand
from cps_reason_bench.references import ReferenceReport
from cps_reason_bench.simulator import GraphCPSSimulator


INVARIANT_STATUSES = {"verified", "warning", "unverified", "failed"}


@dataclass(frozen=True)
class InvariantCheck:
    code: str
    status: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in INVARIANT_STATUSES:
            raise ValueError(f"Unknown invariant status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class FamilyInvariantReport:
    task_family: str
    status: str
    checks: tuple[InvariantCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
        }

    def check(self, code: str) -> InvariantCheck | None:
        return next((item for item in self.checks if item.code == code), None)


def validate_family_invariants(
    scenario: ScenarioConfig,
    reference: ReferenceReport | None = None,
) -> FamilyInvariantReport:
    if scenario.generation.get("generator_id") == "cps_reason_graph_v2_generator":
        checks = _graph_v2_checks(scenario, reference)
        return FamilyInvariantReport(
            scenario.task_family,
            _overall_status(checks),
            tuple(checks),
        )
    if scenario.task_family == "protocol_sanity":
        checks = _protocol_sanity_checks(scenario, reference)
        return FamilyInvariantReport(
            scenario.task_family,
            _overall_status(checks),
            tuple(checks),
        )
    if _is_foundation_scenario(scenario):
        checks = _foundation_checks(scenario, reference)
        return FamilyInvariantReport(
            scenario.task_family,
            _overall_status(checks),
            tuple(checks),
        )
    validators: dict[
        str,
        Callable[[ScenarioConfig, ReferenceReport | None], list[InvariantCheck]],
    ] = {
        "prospective_reasoning": _prospective_checks,
        "planning": _planning_checks,
        "interference": _interference_checks,
        "conditional_safety": _conditional_checks,
        "adaptation": _adaptation_checks,
        "temporal_tracking": _temporal_checks,
    }
    validator = validators.get(scenario.task_family)
    if validator is None:
        checks = [
            InvariantCheck(
                "family_supported",
                "unverified",
                "No family-specific reasoning invariants are registered.",
            )
        ]
    else:
        checks = validator(scenario, reference)
    return FamilyInvariantReport(
        scenario.task_family,
        _overall_status(checks),
        tuple(checks),
    )


def _graph_v2_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    profile = dict(scenario.difficulty_profile.get("graph_v2", {}))
    graph_evaluation = dict(scenario.evaluation.get("graph_v2", {}))
    expected = list(graph_evaluation.get("expected_exact_action_sequence", []))
    scripted = [
        str(command["action_id"])
        for item in scenario.baseline.get("script", [])
        for command in item.get("decision", {}).get("actions", [])
    ]
    nonbaseline = profile.get("decision_property") != "topology_irrelevant_baseline"
    relational = bool(
        not nonbaseline
        or (len(scenario.graph["nodes"]) > 1 and scenario.graph.get("edges"))
    )
    checks = [
        InvariantCheck(
            "graph_v2_metadata",
            "verified" if profile and graph_evaluation else "failed",
            "Versioned graph construct and evaluation metadata are present."
            if profile and graph_evaluation
            else "Graph-v2 metadata is missing.",
        ),
        InvariantCheck(
            "graph_v2_relational_structure",
            "verified" if relational else "failed",
            (
                "Relational graph conditions contain multiple nodes and at least one edge."
                if nonbaseline
                else "The one-node condition is an explicit graph-irrelevant control."
            ),
            {
                "node_count": len(scenario.graph["nodes"]),
                "edge_count": len(scenario.graph.get("edges", [])),
            },
        ),
        InvariantCheck(
            "graph_v2_reference_sequence",
            "verified" if expected and expected == scripted else "failed",
            "The machine-readable expected sequence matches the scripted control."
            if expected and expected == scripted
            else "Expected and scripted action sequences differ or are empty.",
            {"expected": expected, "scripted": scripted},
        ),
        InvariantCheck(
            "graph_v2_reference_feasible",
            "verified"
            if reference and reference.script_valid and reference.task_success
            else "failed",
            "The deterministic reference is protocol-valid, safe, and successful."
            if reference and reference.script_valid and reference.task_success
            else "The deterministic reference did not establish feasibility.",
        ),
    ]
    family_requirements = {
        "prospective_reasoning": bool(
            scenario.evaluation.get("callback_target_window")
            and (
                scenario.interaction.allows_callbacks
                or scenario.interaction.allows_subscriptions
            )
        ),
        "planning": bool(scenario.evaluation.get("expected_action_sequence")),
        "interference": bool(
            scenario.evaluation.get("forbidden_actions")
            or any(
                fault.get("kind") in {"sensor_stuck", "sensor_bias", "sensor_delay"}
                for fault in scenario.faults
            )
        ),
        "conditional_safety": any(
            constraint.get("kind") == "conditional"
            for constraint in scenario.constraints
        ),
        "adaptation": bool(scenario.evaluation.get("adaptation") and scenario.faults),
        "temporal_tracking": bool(
            scenario.evaluation.get("temporal")
            and any(
                int(action.get("effect_delay_steps", 0)) > 0
                for action in scenario.actions
            )
        ),
    }
    construct_present = family_requirements.get(scenario.task_family, False)
    checks.append(
        InvariantCheck(
            "graph_v2_family_construct",
            "verified" if construct_present else "failed",
            "The family-specific mechanism remains present alongside graph demand."
            if construct_present
            else "The family-specific mechanism is absent.",
        )
    )
    return checks


def _overall_status(checks: list[InvariantCheck]) -> str:
    statuses = {check.status for check in checks}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    if "unverified" in statuses:
        return "unverified"
    return "verified"


def _is_foundation_scenario(scenario: ScenarioConfig) -> bool:
    family_parameters = scenario.difficulty_profile.get("family_parameters", {})
    return bool(
        scenario.generation.get("generator_version") == "3.0.0"
        or family_parameters.get("content_revision") == "foundation_v1"
    )


def _protocol_sanity_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    metadata = dict(scenario.evaluation.get("protocol_sanity", {}))
    required = list(scenario.evaluation.get("required_actions", []))
    action_ids = set(scenario.action_ids)
    goal_targets = {
        str(condition["target"])
        for goal in scenario.goals
        for condition in (
            goal.get("all", [])
            or goal.get("any", [])
            or ([goal["condition"]] if "condition" in goal else [])
        )
    }
    visible_targets = {
        str(sensor.get("target"))
        for sensor in scenario.sensors
        if sensor.get("observable", True)
    }
    direct_action_exists = bool(
        len(required) == 1
        and required[0] in action_ids
        and metadata.get("correct_action") == required[0]
    )
    hidden_state_not_required = bool(
        not metadata.get("hidden_state_required", True)
        and (
            scenario.observability.get("full_state")
            or goal_targets <= visible_targets
        )
    )
    short_reference = bool(
        metadata.get("one_step_solution")
        and reference
        and reference.task_success
        and reference.accepted_action_count == 1
        and scenario.max_steps <= 2
    )
    simple_interaction = bool(
        scenario.interaction.allows_periodic
        and not scenario.interaction.allows_callbacks
        and not scenario.interaction.allows_subscriptions
    )
    return [
        InvariantCheck(
            "protocol_sanity_direct_action",
            "verified" if direct_action_exists else "failed",
            (
                "Exactly one declared correct operational action exists."
                if direct_action_exists
                else "The direct correct action is missing or ambiguous."
            ),
            {"required_actions": required},
        ),
        InvariantCheck(
            "protocol_sanity_public_evidence",
            "verified" if hidden_state_not_required else "failed",
            (
                "Goal-relevant state is public and no hidden state is required."
                if hidden_state_not_required
                else "The task may require hidden goal-relevant state."
            ),
            {"goal_targets": sorted(goal_targets), "visible_targets": sorted(visible_targets)},
        ),
        InvariantCheck(
            "protocol_sanity_one_step_reference",
            "verified" if short_reference else "failed",
            (
                "The scripted reference completes the task with one accepted decision."
                if short_reference
                else "A one-step successful reference was not established."
            ),
        ),
        InvariantCheck(
            "protocol_sanity_simple_interaction",
            "verified" if simple_interaction else "failed",
            (
                "Callbacks, subscriptions, and nonperiodic interaction are disabled."
                if simple_interaction
                else "The protocol-sanity task enables an unrelated interaction mechanism."
            ),
        ),
    ]


def _foundation_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    validators: dict[
        str,
        Callable[[ScenarioConfig, ReferenceReport | None], list[InvariantCheck]],
    ] = {
        "prospective_reasoning": _foundation_prospective_checks,
        "planning": _foundation_planning_checks,
        "interference": _foundation_interference_checks,
        "conditional_safety": _foundation_conditional_checks,
        "adaptation": _foundation_adaptation_checks,
        "temporal_tracking": _foundation_temporal_checks,
    }
    checks = [
        InvariantCheck(
            "foundation_reference_succeeds",
            "verified" if reference and reference.task_success else "failed",
            (
                "The transparent scripted reference completes the foundation task."
                if reference and reference.task_success
                else "The foundation reference does not complete the task."
            ),
        ),
        InvariantCheck(
            "foundation_reference_protocol_valid",
            "verified" if reference and reference.script_valid else "failed",
            (
                "The reference uses valid decisions and expected action outcomes."
                if reference and reference.script_valid
                else "The foundation reference contains an invalid decision or action outcome."
            ),
        ),
    ]
    checks.extend(validators[scenario.task_family](scenario, reference))
    return checks


def _foundation_prospective_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    target_step = int(scenario.evaluation.get("relevant_event_step", -1))
    subscription_targets = [
        item.get("filters", {}).get("target")
        for item in (reference.required_event_subscriptions if reference else ())
        if item.get("event_type") == "variable_threshold"
    ]
    mechanism = bool(
        (
            scenario.evaluation.get("callback_target_window")
            and scenario.interaction.allows_callbacks
        )
        or (subscription_targets and scenario.interaction.allows_subscriptions)
    )
    reachable = all(str(target) in _mutable_targets(scenario) for target in subscription_targets)
    return [
        InvariantCheck(
            "foundation_prospective_mechanism",
            "verified" if mechanism else "failed",
            "A callback or event subscription has a demonstrated reference role.",
        ),
        InvariantCheck(
            "foundation_prospective_event_reachable",
            "verified" if reachable else "failed",
            "Every reference event target has a known deterministic change mechanism.",
            {"subscription_targets": subscription_targets},
        ),
        InvariantCheck(
            "foundation_prospective_timing_within_horizon",
            "verified" if 0 < target_step < scenario.max_steps else "failed",
            "The required future interaction occurs within the episode horizon.",
            {"target_step": target_step, "max_steps": scenario.max_steps},
        ),
    ]


def _foundation_planning_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    expected = list(scenario.evaluation.get("expected_action_sequence", []))
    final_action = next(
        (action for action in scenario.actions if action["id"] == expected[-1]),
        None,
    ) if expected else None
    bypass_restricted = bool(
        len(expected) == 1 or (final_action and final_action.get("preconditions"))
    )
    visible_targets = {
        str(sensor.get("target"))
        for sensor in scenario.sensors
        if sensor.get("observable", True)
    }
    prerequisite_targets = {
        str(condition["target"])
        for action in scenario.actions
        for condition in action.get("preconditions", [])
    }
    margin = scenario.max_steps - len(expected)
    return [
        InvariantCheck(
            "foundation_planning_chain_length",
            "verified" if 1 <= len(expected) <= 3 else "failed",
            "The intended foundation plan contains one to three actions.",
            {"expected_action_count": len(expected)},
        ),
        InvariantCheck(
            "foundation_planning_no_bypass",
            "verified" if bypass_restricted else "failed",
            "The direct entry task is intentional or the final action is prerequisite guarded.",
        ),
        InvariantCheck(
            "foundation_planning_prerequisites_public",
            "verified" if prerequisite_targets <= visible_targets else "failed",
            "Every prerequisite target is represented by a public sensor.",
            {"prerequisite_targets": sorted(prerequisite_targets)},
        ),
        InvariantCheck(
            "foundation_planning_recovery_horizon",
            "verified" if margin >= 2 else "failed",
            "The horizon leaves at least two steps beyond the reference sequence.",
            {"recovery_step_margin": margin},
        ),
    ]


def _foundation_interference_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    metadata = dict(scenario.evaluation.get("foundation", {}))
    faulty_sensors = {
        str(fault["target"])
        for fault in scenario.faults
        if str(fault.get("kind", "")).startswith("sensor_")
    }
    observable_sensors = {
        str(sensor["id"])
        for sensor in scenario.sensors
        if sensor.get("observable", True)
    }
    reliability_observable = bool(
        metadata.get("reliability_cues_observable")
        and faulty_sensors <= observable_sensors
    )
    evidence_targets = [
        str(sensor.get("target"))
        for sensor in scenario.sensors
        if sensor.get("observable", True)
    ]
    evidence_available = bool(
        len(evidence_targets) >= 2
        and len(set(evidence_targets)) < len(evidence_targets)
    )
    required_inspections = set(scenario.evaluation.get("required_inspections", []))
    inspection_ids = {
        str(action["id"])
        for action in scenario.actions
        if action.get("operation") == "inspect"
    }
    inspection_available = required_inspections <= inspection_ids
    return [
        InvariantCheck(
            "foundation_interference_reliability_public",
            "verified" if reliability_observable else "failed",
            "Reliability status for any fault-affected sensor is publicly observable.",
            {"faulty_sensors": sorted(faulty_sensors)},
        ),
        InvariantCheck(
            "foundation_interference_evidence_available",
            "verified" if evidence_available else "failed",
            "At least two public observations provide directly comparable evidence.",
        ),
        InvariantCheck(
            "foundation_interference_inspection_available",
            "verified" if inspection_available else "failed",
            "Every required bridge inspection is available as an action.",
        ),
        InvariantCheck(
            "foundation_interference_no_hidden_fault_dependency",
            (
                "verified"
                if not metadata.get("hidden_fault_knowledge_required", True)
                else "failed"
            ),
            "Correct evidence use does not require hidden fault metadata.",
        ),
    ]


def _foundation_conditional_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    conditionals = [
        constraint
        for constraint in scenario.constraints
        if constraint.get("kind") == "conditional"
    ]
    required = set(scenario.evaluation.get("required_inspections", []))
    action_by_id = {str(action["id"]): action for action in scenario.actions}
    inspection_targets = {
        str(action_by_id[action_id].get("target"))
        for action_id in required
        if action_id in action_by_id
    }
    consequent_targets = {str(item["then"]["target"]) for item in conditionals}
    sensors = {
        str(sensor.get("target"))
        for sensor in scenario.sensors
        if sensor.get("observable", True)
    }
    evidence_available = consequent_targets <= sensors | inspection_targets
    initial_budget = float(
        scenario.graph.get("resources", {})
        .get("diagnostic_budget", {})
        .get("initial", 0)
    )
    return [
        InvariantCheck(
            "foundation_conditional_single_rule",
            "verified" if len(conditionals) == 1 else "failed",
            "Exactly one conditional safety rule is configured.",
        ),
        InvariantCheck(
            "foundation_conditional_evidence_available",
            "verified" if evidence_available else "failed",
            "The rule consequent is visible or exposed by the required inspection.",
        ),
        InvariantCheck(
            "foundation_conditional_budget_sufficient",
            "verified" if initial_budget >= len(required) else "failed",
            "Diagnostic budget is sufficient for every required inspection.",
            {"budget": initial_budget, "required_inspection_count": len(required)},
        ),
    ]


def _foundation_adaptation_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    metadata = dict(scenario.evaluation.get("adaptation", {}))
    change_step = int(metadata.get("regime_change_step", -1))
    original = list(metadata.get("original_actions", []))
    backup = list(metadata.get("safe_replacement_actions", []))
    target = str(metadata.get("effect_target", ""))
    before = _action_effect_at_step(scenario, original[0], target, 0) if original else None
    after = (
        _action_effect_at_step(scenario, original[0], target, change_step)
        if original and 0 <= change_step < scenario.max_steps
        else None
    )
    backup_effect = (
        _action_effect_at_step(scenario, backup[0], target, change_step)
        if backup and 0 <= change_step < scenario.max_steps
        else None
    )
    degraded = bool(
        before
        and after
        and (
            not after["accepted"]
            or abs(float(after["delta"])) < abs(float(before["delta"]))
        )
    )
    backup_works = bool(
        backup_effect and backup_effect["accepted"] and backup_effect["delta"] > 0
    )
    evidence_step = int(metadata.get("sufficient_evidence_step", scenario.max_steps))
    return [
        InvariantCheck(
            "foundation_adaptation_original_works",
            "verified" if before and before["accepted"] and before["delta"] > 0 else "failed",
            "The primary action is effective before the regime change.",
            {"pre_change_effect": before},
        ),
        InvariantCheck(
            "foundation_adaptation_change_clear",
            "verified" if degraded else "failed",
            "The primary action is rejected or visibly less effective after the change.",
            {"post_change_effect": after},
        ),
        InvariantCheck(
            "foundation_adaptation_backup_works",
            "verified" if backup_works else "failed",
            "The named backup has a positive post-change effect on the goal target.",
            {"backup_effect": backup_effect},
        ),
        InvariantCheck(
            "foundation_adaptation_recovery_horizon",
            "verified" if evidence_step + 1 < scenario.max_steps else "failed",
            "At least one action step remains after sufficient change evidence.",
            {"sufficient_evidence_step": evidence_step, "max_steps": scenario.max_steps},
        ),
    ]


def _foundation_temporal_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    metadata = dict(scenario.evaluation.get("temporal", {}))
    mapping = dict(metadata.get("initiation_to_response", {}))
    action_by_id = {str(action["id"]): action for action in scenario.actions}
    distance = int(metadata.get("required_temporal_distance", 0))
    delays_match = bool(
        mapping
        and all(
            int(action_by_id[action_id].get("effect_delay_steps", 0)) == distance
            for action_id in mapping
        )
    )
    context_sufficient = bool(
        (distance == 1 and scenario.context.mode in {"previous", "history"})
        or (
            scenario.context.mode == "history"
            and scenario.context.history_limit >= distance + 1
        )
    )
    expected_steps = [
        int(step)
        for step, action in scenario.evaluation.get("expected_actions_by_step", {}).items()
        if action in set(mapping.values())
    ]
    return [
        InvariantCheck(
            "foundation_temporal_delay_matches",
            "verified" if delays_match else "failed",
            "Every initiating action uses the declared one- or two-step delay.",
            {"declared_distance": distance},
        ),
        InvariantCheck(
            "foundation_temporal_context_sufficient",
            "verified" if context_sufficient else "failed",
            "Explicit previous/history context retains the required dependency.",
        ),
        InvariantCheck(
            "foundation_temporal_no_overlap",
            (
                "verified"
                if int(metadata.get("overlapping_causal_actions", 0)) == 0
                else "failed"
            ),
            "The foundation task has no overlapping pending causal actions.",
        ),
        InvariantCheck(
            "foundation_temporal_effect_within_horizon",
            (
                "verified"
                if expected_steps and max(expected_steps) < scenario.max_steps
                else "failed"
            ),
            "The delayed response and reference attribution occur within the horizon.",
        ),
    ]


def _prospective_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    has_callback = bool(scenario.evaluation.get("callback_target_window"))
    scripted_subscriptions = bool(
        reference and reference.required_event_subscriptions
    )
    mechanism_enabled = (
        has_callback and scenario.interaction.allows_callbacks
    ) or (scripted_subscriptions and scenario.interaction.allows_subscriptions)
    checks = [
        InvariantCheck(
            "prospective_future_mechanism",
            "verified" if mechanism_enabled else "failed",
            (
                "A configured callback or event subscription has a reference role."
                if mechanism_enabled
                else "No enabled callback or event subscription has a demonstrated role."
            ),
            {
                "callback_window": scenario.evaluation.get("callback_target_window"),
                "scripted_subscription_count": (
                    len(reference.required_event_subscriptions) if reference else 0
                ),
            },
        )
    ]
    mutable = _mutable_targets(scenario)
    threshold_targets = [
        item.get("filters", {}).get("target")
        for item in (reference.required_event_subscriptions if reference else ())
        if item.get("event_type") == "variable_threshold"
    ]
    if threshold_targets:
        reachable = all(str(target) in mutable for target in threshold_targets)
        checks.append(
            InvariantCheck(
                "prospective_event_reachable",
                "verified" if reachable else "failed",
                (
                    "Every reference threshold target has a known change mechanism."
                    if reachable
                    else "A reference threshold target has no known change mechanism."
                ),
                {"threshold_targets": threshold_targets},
            )
        )
    else:
        checks.append(
            InvariantCheck(
                "prospective_event_reachable",
                "unverified",
                "No reference threshold subscription is present for reachability analysis.",
            )
        )
    timing_matters = bool(
        (has_callback or scenario.evaluation.get("relevant_event_step") is not None)
        and scenario.constraints
        and scenario.evaluation.get("required_actions")
    )
    checks.append(
        InvariantCheck(
            "prospective_timing_material",
            "verified" if timing_matters else "unverified",
            (
                "Timing is linked to a required intervention and a physical constraint."
                if timing_matters
                else "The effect of timing on success or safety was not established."
            ),
        )
    )
    non_polling = not scenario.interaction.allows_periodic
    checks.append(
        InvariantCheck(
            "prospective_polling_control",
            "verified" if non_polling else "warning",
            (
                "The interaction mode prevents unrestricted periodic polling."
                if non_polling
                else "Periodic invocation is enabled; use interaction metrics to assess polling."
            ),
        )
    )
    return checks


def _planning_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    prerequisite_count = sum(
        len(action.get("preconditions", [])) for action in scenario.actions
    )
    checks = [
        InvariantCheck(
            "planning_prerequisites_present",
            "verified" if prerequisite_count else "failed",
            (
                f"The scenario declares {prerequisite_count} action prerequisites."
                if prerequisite_count
                else "The scenario has no action prerequisites."
            ),
            {"prerequisite_count": prerequisite_count},
        )
    ]
    expected = list(scenario.evaluation.get("expected_action_sequence", []))
    if len(expected) >= 2:
        simulator = GraphCPSSimulator(scenario, scenario.seed)
        rejected = 0
        for action_id in reversed(expected):
            outcomes, _ = simulator.apply_actions([ActionCommand(str(action_id))])
            rejected += int(not outcomes[0].accepted)
        checks.append(
            InvariantCheck(
                "planning_arbitrary_order_restricted",
                "verified" if rejected else "failed",
                (
                    "Reversing the reference sequence causes at least one rejected action."
                    if rejected
                    else "The reference actions can be executed in reverse without rejection."
                ),
                {"reverse_order_rejections": rejected},
            )
        )
        final_action = next(
            (action for action in scenario.actions if action["id"] == expected[-1]),
            None,
        )
        final_preconditions = (
            len(final_action.get("preconditions", [])) if final_action else 0
        )
        checks.append(
            InvariantCheck(
                "planning_goal_bypass_restricted",
                "verified" if final_preconditions else "warning",
                (
                    "The final reference action is guarded by intermediate conditions."
                    if final_preconditions
                    else "No intermediate precondition guards the final reference action."
                ),
                {"final_action_preconditions": final_preconditions},
            )
        )
    else:
        checks.extend(
            [
                InvariantCheck(
                    "planning_arbitrary_order_restricted",
                    "unverified",
                    "No multi-action reference sequence is available for an order check.",
                ),
                InvariantCheck(
                    "planning_goal_bypass_restricted",
                    "unverified",
                    "No final reference action is available for a bypass check.",
                ),
            ]
        )
    return checks


def _interference_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    sensor_by_id = {str(sensor["id"]): sensor for sensor in scenario.sensors}
    faulty_sensor_ids = {
        str(fault["target"])
        for fault in scenario.faults
        if str(fault.get("kind", "")).startswith("sensor_")
    }
    corroborated = False
    for sensor_id in faulty_sensor_ids:
        target = sensor_by_id.get(sensor_id, {}).get("target")
        if target and any(
            other_id not in faulty_sensor_ids and other.get("target") == target
            for other_id, other in sensor_by_id.items()
        ):
            corroborated = True
            break
    instructions = str(scenario.task.get("instructions", "")).lower()
    revealing_phrases = (
        "misleading",
        "stale alarm",
        "alarm can fail",
        "unreliable signal",
    )
    reveals_trap = any(phrase in instructions for phrase in revealing_phrases)
    forbidden = list(scenario.evaluation.get("forbidden_actions", []))
    measurable = _forbidden_action_has_cost(scenario, forbidden[0]) if forbidden else None
    return [
        InvariantCheck(
            "interference_conflicting_evidence",
            "verified" if corroborated else "failed",
            (
                "A fault-affected salient sensor has independent evidence for the same state."
                if corroborated
                else "No independent evidence counters the fault-affected signal."
            ),
            {"faulty_sensor_count": len(faulty_sensor_ids)},
        ),
        InvariantCheck(
            "interference_instruction_non_disclosure",
            "warning" if reveals_trap else "verified",
            (
                "Task instructions explicitly disclose signal unreliability."
                if reveals_trap
                else "Task instructions do not label a signal as misleading or stale."
            ),
        ),
        InvariantCheck(
            "interference_trap_has_consequence",
            "verified" if measurable else ("failed" if measurable is False else "unverified"),
            (
                "A declared tempting action produces measurable failure, cost, or safety impact."
                if measurable
                else "The measurable consequence of a tempting action was not established."
            ),
            {"checked_action": forbidden[0] if forbidden else None},
        ),
    ]


def _conditional_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    conditionals = [
        constraint
        for constraint in scenario.constraints
        if constraint.get("kind") == "conditional"
    ]
    simulator = GraphCPSSimulator(scenario, scenario.seed)
    falsified = sum(simulator._constraint_violated(item) for item in conditionals)  # noqa: SLF001
    inspect_targets = {
        str(action.get("target"))
        for action in scenario.actions
        if action.get("operation") == "inspect"
    }
    consequent_targets = {str(item["then"]["target"]) for item in conditionals}
    evidence_available = bool(inspect_targets & consequent_targets)
    required = set(scenario.evaluation.get("required_inspections", []))
    all_inspections = {
        str(action["id"])
        for action in scenario.actions
        if action.get("operation") == "inspect"
    }
    distinguishable = bool(required and all_inspections - required)
    return [
        InvariantCheck(
            "conditional_rule_present",
            "verified" if conditionals else "failed",
            (
                f"The scenario contains {len(conditionals)} conditional rule(s)."
                if conditionals
                else "No conditional safety rule is configured."
            ),
        ),
        InvariantCheck(
            "conditional_rule_falsifiable",
            "verified" if falsified else "unverified",
            (
                "At least one target rule is falsified in the initial reachable state."
                if falsified
                else "Automatic analysis did not exhibit a falsifying state."
            ),
            {"initially_falsified_rules": falsified},
        ),
        InvariantCheck(
            "conditional_evidence_observable",
            "verified" if evidence_available else "failed",
            (
                "An inspection can expose a consequent state used by a target rule."
                if evidence_available
                else "No allowed inspection exposes a target rule's consequent state."
            ),
        ),
        InvariantCheck(
            "conditional_inspection_relevance_distinguishable",
            "verified" if distinguishable else "unverified",
            (
                "Required and unnecessary inspection actions are both present."
                if distinguishable
                else "The scenario has no unnecessary inspection control condition."
            ),
        ),
    ]


def _adaptation_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    metadata = dict(scenario.evaluation.get("adaptation", {}))
    if not metadata:
        return [
            InvariantCheck(
                "adaptation_metadata",
                "failed",
                "The scenario does not declare adaptation evaluation metadata.",
            )
        ]
    change_step = int(metadata.get("regime_change_step", -1))
    original = [str(item) for item in metadata.get("original_actions", [])]
    replacements = [str(item) for item in metadata.get("replacement_actions", [])]
    target = str(metadata.get("effect_target", ""))
    pre_effect = _action_effect_at_step(scenario, original[0], target, 0) if original else None
    post_effect = (
        _action_effect_at_step(scenario, original[0], target, change_step)
        if original and 0 <= change_step <= scenario.max_steps
        else None
    )
    original_works = bool(
        pre_effect and pre_effect["accepted"] and pre_effect["delta"] != 0
    )
    post_insufficient = bool(
        pre_effect
        and post_effect
        and (
            not post_effect["accepted"]
            or abs(float(post_effect["delta"])) < abs(float(pre_effect["delta"]))
        )
    )
    evidence_step = metadata.get("sufficient_evidence_step")
    evidence_exists = bool(
        evidence_step is not None
        and int(evidence_step) <= scenario.max_steps
        and (
            scenario.context.include_action_outcomes
            or any(sensor.get("target") == target for sensor in scenario.sensors)
            or scenario.evaluation.get("required_inspections")
        )
    )
    script_actions = [
        (int(item.get("at_step", -1)), str(action["action_id"]))
        for item in scenario.baseline.get("script", [])
        for action in item.get("decision", {}).get("actions", [])
    ]
    reference_adapts = bool(
        reference
        and reference.task_success
        and any(step < change_step and action in original for step, action in script_actions)
        and any(step >= change_step and action in replacements for step, action in script_actions)
    )
    return [
        InvariantCheck(
            "adaptation_original_strategy_effective",
            "verified" if original_works else "failed",
            (
                "The original strategy changes its intended target before the regime change."
                if original_works
                else "The original strategy was not effective before the regime change."
            ),
            {"pre_change_effect": pre_effect},
        ),
        InvariantCheck(
            "adaptation_change_within_horizon",
            "verified" if 0 < change_step < scenario.max_steps else "failed",
            (
                "The regime change occurs within the episode horizon."
                if 0 < change_step < scenario.max_steps
                else "The regime change does not occur within the usable horizon."
            ),
            {"regime_change_step": change_step, "max_steps": scenario.max_steps},
        ),
        InvariantCheck(
            "adaptation_original_strategy_degraded",
            "verified" if post_insufficient else "failed",
            (
                "The original strategy becomes rejected, redirected, or less effective."
                if post_insufficient
                else "The original strategy remains equally effective after the change."
            ),
            {"post_change_effect": post_effect},
        ),
        InvariantCheck(
            "adaptation_evidence_public",
            "verified" if evidence_exists else "unverified",
            (
                "A configured public outcome, sensor, or inspection can expose change evidence."
                if evidence_exists
                else "Public evidence availability could not be established automatically."
            ),
            {"sufficient_evidence_step": evidence_step},
        ),
        InvariantCheck(
            "adaptation_pre_change_completion_blocked",
            (
                "verified"
                if scenario.evaluation.get("success_requires_adaptation")
                else "unverified"
            ),
            (
                "Task success explicitly requires demonstrated post-change adaptation."
                if scenario.evaluation.get("success_requires_adaptation")
                else "The physical goal may be sufficient before the intended change."
            ),
        ),
        InvariantCheck(
            "adaptation_reference_recovers",
            "verified" if reference_adapts else "failed",
            (
                "The feasible reference uses the original strategy before the change and a replacement afterward."
                if reference_adapts
                else "The reference does not demonstrate the intended strategy revision."
            ),
        ),
    ]


def _temporal_checks(
    scenario: ScenarioConfig, reference: ReferenceReport | None
) -> list[InvariantCheck]:
    metadata = dict(scenario.evaluation.get("temporal", {}))
    if not metadata:
        return [
            InvariantCheck(
                "temporal_metadata",
                "failed",
                "The scenario does not declare temporal evaluation metadata.",
            )
        ]
    declared_distance = int(metadata.get("required_temporal_distance", 0))
    mapping = dict(metadata.get("initiation_to_response", {}))
    action_by_id = {str(action["id"]): action for action in scenario.actions}
    delays = {
        action_id: int(action_by_id.get(str(action_id), {}).get("effect_delay_steps", 0))
        for action_id in mapping
    }
    dependency_exists = bool(mapping and all(delay > 0 for delay in delays.values()))
    distance_matches = bool(
        dependency_exists and all(delay == declared_distance for delay in delays.values())
    )
    context_mode = scenario.context.mode
    negative_control = bool(scenario.evaluation.get("context_negative_control"))
    context_supports = bool(
        (declared_distance <= 1 and context_mode in {"previous", "history"})
        or (
            context_mode == "history"
            and scenario.context.history_limit >= declared_distance
        )
    )
    selector_sensors = set(metadata.get("selector_sensors", []))
    expected_steps = [
        int(step)
        for step, action in scenario.evaluation.get("expected_actions_by_step", {}).items()
        if action in set(mapping.values())
    ]
    first_attribution_step = min(expected_steps) if expected_steps else None
    unavailable_selectors = {
        str(fault["target"])
        for fault in scenario.faults
        if fault.get("kind") == "sensor_unavailable"
        and first_attribution_step is not None
        and int(fault.get("at_step", 0)) <= first_attribution_step
    }
    source_exposed = any(
        sensor.get("target") == "actuator_bank.response_source"
        for sensor in scenario.sensors
    )
    current_only_insufficient = bool(
        len(mapping) >= 2
        and selector_sensors
        and selector_sensors <= unavailable_selectors
        and not source_exposed
    )
    delayed_within_horizon = bool(
        expected_steps and max(expected_steps) < scenario.max_steps
    )
    distractor_targets = {
        str(sensor.get("target"))
        for sensor in scenario.sensors
        if sensor["id"] not in selector_sensors
        and sensor["id"] != metadata.get("response_sensor")
    }
    distractors_reveal = "actuator_bank.response_source" in distractor_targets
    return [
        InvariantCheck(
            "temporal_dependency_exists",
            "verified" if dependency_exists else "failed",
            (
                "Initiating actions have delayed physical effects and mapped response actions."
                if dependency_exists
                else "No delayed action-response dependency was found."
            ),
            {"action_delays": delays},
        ),
        InvariantCheck(
            "temporal_distance_matches",
            "verified" if distance_matches else "failed",
            (
                "Declared temporal distance matches every initiating action delay."
                if distance_matches
                else "Declared temporal distance does not match the action delay."
            ),
            {"declared_distance": declared_distance},
        ),
        InvariantCheck(
            "temporal_context_sufficient",
            (
                "verified"
                if context_supports
                else "warning"
                if negative_control
                else "failed"
            ),
            (
                "Configured explicit context can retain the required dependency."
                if context_supports
                else "Context is intentionally insufficient as a declared negative control."
                if negative_control
                else "Configured context cannot retain the declared temporal distance."
            ),
            {
                "context_mode": context_mode,
                "history_limit": scenario.context.history_limit,
                "negative_control": negative_control,
            },
        ),
        InvariantCheck(
            "temporal_current_only_control",
            "verified" if current_only_insufficient else "unverified",
            (
                "At attribution time, selector sensors are unavailable and the hidden response source is not sensed."
                if current_only_insufficient
                else "Current-only bypassability could not be excluded automatically."
            ),
        ),
        InvariantCheck(
            "temporal_effect_within_horizon",
            "verified" if delayed_within_horizon else "failed",
            (
                "Reference attribution occurs within the episode horizon."
                if delayed_within_horizon
                else "The delayed dependency does not complete within the horizon."
            ),
        ),
        InvariantCheck(
            "temporal_distractor_non_disclosure",
            "failed" if distractors_reveal else "verified",
            (
                "A distractor sensor exposes the hidden response source."
                if distractors_reveal
                else "Distractor sensors do not expose the hidden response source."
            ),
        ),
        InvariantCheck(
            "temporal_reference_solves",
            "verified" if reference and reference.task_success else "failed",
            (
                "The feasible reference solves the delayed attribution task."
                if reference and reference.task_success
                else "The reference does not solve the temporal task."
            ),
        ),
    ]


def _mutable_targets(scenario: ScenarioConfig) -> set[str]:
    targets: set[str] = set()
    for action in scenario.actions:
        if action.get("operation") in {"set", "adjust"} and action.get("target"):
            targets.add(str(action["target"]))
        targets.update(
            str(effect["target"])
            for effect in action.get("additional_effects", [])
            if effect.get("target")
        )
    for node_id, node in scenario.graph["nodes"].items():
        for variable_id, variable in node["variables"].items():
            if float(variable.get("drift", 0.0)) != 0.0:
                targets.add(f"{node_id}.{variable_id}")
    return targets


def _forbidden_action_has_cost(scenario: ScenarioConfig, action_id: str) -> bool:
    simulator = GraphCPSSimulator(scenario, scenario.seed)
    initial_goals = simulator.goals_met
    initial_resources = dict(simulator.resources)
    outcomes, _ = simulator.apply_actions([ActionCommand(action_id)])
    for _ in range(
        max(
            1,
            int(
                next(
                    (
                        action.get("effect_delay_steps", 0)
                        for action in scenario.actions
                        if action["id"] == action_id
                    ),
                    0,
                )
            )
            + 1,
        )
    ):
        simulator.advance()
    return bool(
        not outcomes[0].accepted
        or simulator.goals_met != initial_goals
        or simulator.resources != initial_resources
        or simulator.safety_violations
    )


def _action_effect_at_step(
    scenario: ScenarioConfig,
    action_id: str,
    target: str,
    step: int,
) -> dict[str, Any]:
    simulator = GraphCPSSimulator(scenario, scenario.seed)
    while simulator.step < step and not simulator.done:
        simulator.advance()
    before = simulator.resolve_target(target)
    outcomes, _ = simulator.apply_actions([ActionCommand(action_id)])
    after = simulator.resolve_target(target)
    delta = (
        float(after) - float(before)
        if isinstance(before, (int, float))
        and not isinstance(before, bool)
        and isinstance(after, (int, float))
        and not isinstance(after, bool)
        else int(after != before)
    )
    return {
        "step": step,
        "accepted": outcomes[0].accepted,
        "delta": delta,
        "reason_code": outcomes[0].reason_code,
    }
