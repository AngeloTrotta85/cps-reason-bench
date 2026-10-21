"""General CPS and task-family metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.scheduling import CallbackScheduler
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.subscriptions import SubscriptionManager


@dataclass
class RunStatistics:
    interaction_count: int = 0
    model_call_count: int = 0
    invalid_output_count: int = 0
    invalid_output_attempt_count: int = 0
    output_retry_count: int = 0
    accepted_action_count: int = 0
    invalid_action_count: int = 0
    total_latency_seconds: float = 0.0
    token_usage: dict[str, int | float] = field(default_factory=dict)
    invocation_steps: list[int] = field(default_factory=list)
    action_records: list[dict[str, Any]] = field(default_factory=list)
    subscription_records: list[dict[str, Any]] = field(default_factory=list)
    triggered_subscription_ids: set[str] = field(default_factory=set)
    invocation_causes: list[list[str]] = field(default_factory=list)
    first_attempt_count: int = 0
    first_attempt_json_valid_count: int = 0
    first_attempt_schema_valid_count: int = 0
    first_attempt_decision_valid_count: int = 0
    validity_after_correction_count: int = 0
    final_executable_decision_count: int = 0
    correction_request_count: int = 0
    additional_model_call_count: int = 0
    additional_token_usage: dict[str, int | float] = field(default_factory=dict)
    additional_latency_seconds: float = 0.0
    backend_error_count: int = 0
    runtime_error_count: int = 0
    infrastructure_failure: dict[str, Any] | None = None

    def add_usage(self, usage: dict[str, int | float | None]) -> None:
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.token_usage[key] = self.token_usage.get(key, 0) + value

    def add_additional_usage(self, usage: dict[str, int | float | None]) -> None:
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.additional_token_usage[key] = (
                    self.additional_token_usage.get(key, 0) + value
                )


def compute_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
    scheduler: CallbackScheduler,
    subscriptions: SubscriptionManager,
) -> dict[str, Any]:
    initial_resources = {
        resource_id: spec["initial"]
        for resource_id, spec in scenario.graph.get("resources", {}).items()
    }
    resource_consumption = {
        resource_id: initial_resources[resource_id] - simulator.resources[resource_id]
        for resource_id in initial_resources
    }
    accepted_records = [
        record for record in stats.action_records if record.get("outcome", {}).get("accepted")
    ]
    accepted_action_ids = [
        str(record["command"]["action_id"]) for record in accepted_records
    ]
    evaluation = scenario.evaluation
    unnecessary_ids = set(evaluation.get("unnecessary_actions", []))
    unnecessary_actions = sum(item in unnecessary_ids for item in accepted_action_ids)
    allowed_safety_violations = int(evaluation.get("allowed_safety_violations", 0))
    success_requires_safety = bool(evaluation.get("success_requires_safety", True))
    physical_success = simulator.goals_met and (
        not success_requires_safety
        or len(simulator.safety_violations) <= allowed_safety_violations
    )
    task_specific = _task_metrics(
        scenario,
        simulator,
        stats,
        scheduler,
        accepted_action_ids,
    )
    behavior_requirements_met = _behavior_requirements_met(
        scenario, task_specific
    )
    task_success = physical_success and behavior_requirements_met
    if scenario.task_family in {"adaptation", "temporal_tracking"}:
        task_specific["final_task_success"] = task_success

    added_subscription_ids = {
        record["id"]
        for record in stats.subscription_records
        if record.get("operation") == "add"
    }
    general = {
        "task_success": task_success,
        "goals_met": simulator.goals_met,
        "simulation_steps": simulator.step,
        "simulated_time": simulator.time,
        "llm_or_policy_invocations": stats.interaction_count,
        "model_calls_including_retries": stats.model_call_count,
        "invalid_actions": stats.invalid_action_count,
        "invalid_structured_outputs": stats.invalid_output_count,
        "backend_errors": stats.backend_error_count,
        "runtime_errors": stats.runtime_error_count,
        "infrastructure_failures": int(stats.infrastructure_failure is not None),
        "invalid_output_attempts": stats.invalid_output_attempt_count,
        "output_retries": stats.output_retry_count,
        "safety_violations": len(simulator.safety_violations),
        "resource_consumption": resource_consumption,
        "unnecessary_actions": unnecessary_actions,
        "accepted_actions": stats.accepted_action_count,
        "interaction_cost": stats.interaction_count,
        "token_usage": stats.token_usage,
        "total_latency_seconds": round(stats.total_latency_seconds, 6),
        "callback_requests": len(scheduler.history),
        "callback_triggers": len(scheduler.triggered_history),
        "canceled_callbacks": scheduler.cancel_count,
        "replaced_callbacks": scheduler.replacement_count,
        "active_subscriptions_at_end": len(subscriptions.active),
        "useful_subscription_triggers": subscriptions.useful_trigger_count,
        "irrelevant_subscriptions": len(
            added_subscription_ids - stats.triggered_subscription_ids
        ),
        "final_system_state": simulator.true_state(),
    }
    if _has_behavior_success_requirements(evaluation):
        general["behavior_requirements_met"] = behavior_requirements_met
    first_count = stats.first_attempt_count
    protocol_compliance = {
        "first_attempt_count": first_count,
        "first_attempt_json_valid": stats.first_attempt_json_valid_count,
        "first_attempt_json_valid_rate": _rate(
            stats.first_attempt_json_valid_count, first_count
        ),
        "first_attempt_schema_valid": stats.first_attempt_schema_valid_count,
        "first_attempt_schema_valid_rate": _rate(
            stats.first_attempt_schema_valid_count, first_count
        ),
        "first_attempt_decision_valid": stats.first_attempt_decision_valid_count,
        "first_attempt_decision_valid_rate": _rate(
            stats.first_attempt_decision_valid_count, first_count
        ),
        "correction_requests": stats.correction_request_count,
        "valid_after_correction": stats.validity_after_correction_count,
        "final_executable_decisions": stats.final_executable_decision_count,
        "additional_model_calls": stats.additional_model_call_count,
        "additional_token_usage": stats.additional_token_usage,
        "additional_latency_seconds": round(stats.additional_latency_seconds, 6),
        "backend_errors": stats.backend_error_count,
    }
    return {
        "scenario": scenario.name,
        "task_family": scenario.task_family,
        "difficulty": scenario.difficulty,
        "difficulty_level": scenario.difficulty_level,
        "difficulty_profile": scenario.difficulty_profile,
        "general": general,
        "protocol_compliance": protocol_compliance,
        "task_specific": task_specific,
    }


def _task_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
    scheduler: CallbackScheduler,
    accepted_action_ids: list[str],
) -> dict[str, Any]:
    evaluation = scenario.evaluation
    required_actions = list(evaluation.get("required_actions", []))
    required_inspections = list(evaluation.get("required_inspections", []))
    forbidden_actions = set(evaluation.get("forbidden_actions", []))
    reference_count = evaluation.get("reference_action_count")
    expected_sequence = list(evaluation.get("expected_action_sequence", []))

    metrics: dict[str, Any] = {
        "required_action_coverage": _coverage(required_actions, accepted_action_ids),
        "required_inspection_coverage": _coverage(required_inspections, accepted_action_ids),
        "forbidden_action_count": sum(
            action_id in forbidden_actions for action_id in accepted_action_ids
        ),
        "prerequisite_violations": simulator.prerequisite_violations,
    }
    if expected_sequence:
        metrics["expected_sequence_satisfied"] = _is_subsequence(
            expected_sequence, accepted_action_ids
        )
    valid_sequences = [
        list(sequence) for sequence in evaluation.get("valid_action_sequences", [])
    ]
    if valid_sequences:
        metrics["valid_alternative_sequence_satisfied"] = any(
            _is_subsequence(sequence, accepted_action_ids)
            for sequence in valid_sequences
        )
    if reference_count is not None:
        metrics["excess_actions_vs_reference"] = max(
            0, len(accepted_action_ids) - int(reference_count)
        )

    callback_window = evaluation.get("callback_target_window")
    if callback_window:
        start, end = int(callback_window[0]), int(callback_window[1])
        due_steps = [item.due_step for item in scheduler.triggered_history]
        in_window = [step for step in due_steps if start <= step <= end]
        metrics.update(
            {
                "callback_in_target_window": bool(in_window),
                "missed_critical_callback": not bool(in_window),
                "unnecessary_callbacks": sum(
                    step < start or step > end for step in due_steps
                ),
                "callback_timing_error_steps": (
                    min(abs(step - ((start + end) / 2)) for step in due_steps)
                    if due_steps
                    else None
                ),
            }
        )

    expected_by_step = evaluation.get("expected_actions_by_step", {})
    if expected_by_step:
        checks = []
        for step, expected in expected_by_step.items():
            actual = [
                record["command"]["action_id"]
                for record in stats.action_records
                if record["step"] == int(step)
                and record.get("outcome", {}).get("accepted")
            ]
            checks.append(str(expected) in actual)
        metrics["temporal_tracking_accuracy"] = sum(checks) / len(checks)

    if evaluation.get("graph_v2"):
        metrics.update(_graph_v2_metrics(scenario, simulator, accepted_action_ids))

    metrics["planning_success"] = bool(
        scenario.task_family == "planning" and simulator.goals_met
    )
    metrics["misleading_signal_susceptibility"] = metrics["forbidden_action_count"]
    metrics["conditional_rule_verification_accuracy"] = metrics[
        "required_inspection_coverage"
    ]
    if scenario.task_family == "prospective_reasoning":
        metrics.update(_prospective_metrics(scenario, stats, scheduler))
    elif scenario.task_family == "planning":
        metrics.update(_planning_metrics(scenario, simulator, stats))
    elif scenario.task_family == "interference":
        metrics.update(_interference_metrics(scenario, simulator, stats))
    elif scenario.task_family == "conditional_safety":
        metrics.update(_conditional_metrics(scenario, simulator, stats))
    elif scenario.task_family == "adaptation":
        metrics.update(_adaptation_metrics(scenario, simulator, stats))
    elif scenario.task_family == "temporal_tracking":
        metrics.update(_temporal_metrics(scenario, simulator, stats))
    metrics.update(_rejected_action_repetition_metrics(scenario, stats))
    return metrics


def _has_behavior_success_requirements(evaluation: dict[str, Any]) -> bool:
    return any(
        bool(evaluation.get(key))
        for key in (
            "success_requires_required_actions",
            "success_requires_required_inspections",
            "success_requires_expected_sequence",
            "success_requires_valid_sequence",
            "success_requires_adaptation",
            "success_requires_temporal_tracking",
            "success_requires_graph_decision",
        )
    )


def _behavior_requirements_met(
    scenario: ScenarioConfig, task_metrics: dict[str, Any]
) -> bool:
    evaluation = scenario.evaluation
    checks: list[bool] = []
    if evaluation.get("success_requires_required_actions"):
        checks.append(task_metrics.get("required_action_coverage") == 1.0)
    if evaluation.get("success_requires_required_inspections"):
        checks.append(task_metrics.get("required_inspection_coverage") == 1.0)
    if evaluation.get("success_requires_expected_sequence"):
        checks.append(bool(task_metrics.get("expected_sequence_satisfied")))
    if evaluation.get("success_requires_valid_sequence"):
        checks.append(bool(task_metrics.get("valid_alternative_sequence_satisfied")))
    if evaluation.get("success_requires_adaptation"):
        checks.append(bool(task_metrics.get("adaptation_recovery_demonstrated")))
    if evaluation.get("success_requires_temporal_tracking"):
        checks.append(bool(task_metrics.get("temporal_dependency_handled")))
    if evaluation.get("success_requires_graph_decision"):
        checks.append(bool(task_metrics.get("graph_dependent_decision_correct")))
    return all(checks)


def _graph_v2_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    accepted_action_ids: list[str],
) -> dict[str, Any]:
    metadata = dict(scenario.evaluation.get("graph_v2", {}))
    expected = [
        str(item) for item in metadata.get("expected_exact_action_sequence", [])
    ]
    correct_graph_actions = [
        str(item) for item in metadata.get("correct_graph_actions", [])
    ]
    property_name = str(metadata.get("decision_property", ""))
    profile = dict(scenario.difficulty_profile.get("graph_v2", {}))
    coupling_type = str(profile.get("coupling_type", "none"))
    observability_type = str(profile.get("observability_type", "full_local"))
    exact = bool(expected) and accepted_action_ids == expected
    result: dict[str, Any] = {
        "graph_dependent_decision_correct": exact,
        "graph_dependent_action_accuracy": float(exact),
        "graph_exact_action_sequence_agreement": exact,
        "graph_correct_action_coverage": _coverage(
            correct_graph_actions, accepted_action_ids
        ),
        "graph_decision_property": property_name,
        "counterfactual_pair_id": metadata.get("counterfactual_pair_id"),
        "counterfactual_pair_member": metadata.get("counterfactual_pair_member"),
        "edge_direction_awareness": None,
        "edge_state_awareness": None,
        "coupling_awareness": None,
        "correct_neighbor_selection": None,
        "correct_path_or_branch_choice": None,
        "topology_change_adaptation": None,
        "propagated_effect_prediction": (
            bool(simulator.goals_met)
            if coupling_type
            in {
                "multiple_incoming",
                "short_propagation_chain",
                "competing_influences",
            }
            else None
        ),
        "remote_state_evidence_use": (
            float(exact)
            if observability_type
            in {
                "hidden_neighbor",
                "remote_sensor",
                "delayed_remote_sensor",
                "edge_hidden_inferred",
            }
            else None
        ),
        "rerouting_success": None,
    }
    if property_name == "direction_changes_action":
        result["edge_direction_awareness"] = exact
        result["correct_neighbor_selection"] = exact
    elif property_name == "coupling_changes_action":
        result["coupling_awareness"] = exact
    elif property_name == "edge_failure_changes_action":
        result["edge_state_awareness"] = exact
        result["rerouting_success"] = exact
    elif property_name == "topology_changes_action":
        result["correct_path_or_branch_choice"] = exact
        result["correct_neighbor_selection"] = exact
    if any(
        action.get("operation") in {"toggle_edge", "repair"}
        for action in scenario.actions
    ):
        result["topology_change_adaptation"] = exact
    return result


def _prospective_metrics(
    scenario: ScenarioConfig,
    stats: RunStatistics,
    scheduler: CallbackScheduler,
) -> dict[str, Any]:
    evaluation = scenario.evaluation
    relevant_step = evaluation.get("relevant_event_step")
    if relevant_step is None:
        window = evaluation.get("callback_target_window")
        relevant_step = int(window[0]) if window else None
    action_id = evaluation.get("relevant_action")
    if action_id is None:
        required = evaluation.get("required_actions", [])
        action_id = required[0] if required else None
    action_steps = [
        int(record["step"])
        for record in stats.action_records
        if record["command"]["action_id"] == action_id
        and record.get("outcome", {}).get("accepted")
    ]
    first_action = min(action_steps) if action_steps else None
    added_ids = {
        str(record["id"])
        for record in stats.subscription_records
        if record.get("operation") == "add"
    }
    reaction_delay = (
        first_action - int(relevant_step)
        if first_action is not None and relevant_step is not None
        else None
    )
    return {
        "callback_count": len(scheduler.history),
        "triggered_callback_count": len(scheduler.triggered_history),
        "reaction_delay_steps": reaction_delay,
        "delayed_intervention_steps": (
            max(0, reaction_delay) if reaction_delay is not None else None
        ),
        "missed_relevant_events": int(first_action is None),
        "redundant_subscriptions": len(
            added_ids - stats.triggered_subscription_ids
        ),
        "useful_subscription_count": len(stats.triggered_subscription_ids),
    }


def _planning_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
) -> dict[str, Any]:
    initial_resources = {
        resource_id: float(spec["initial"])
        for resource_id, spec in scenario.graph.get("resources", {}).items()
        if isinstance(spec.get("initial"), (int, float))
    }
    return {
        "planning_action_attempts": len(stats.action_records),
        "planning_accepted_actions": stats.accepted_action_count,
        "planning_resource_consumption": {
            resource_id: initial - float(simulator.resources[resource_id])
            for resource_id, initial in initial_resources.items()
        },
        "planning_safety_violations": len(simulator.safety_violations),
    }


def _interference_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
) -> dict[str, Any]:
    evaluation = scenario.evaluation
    forbidden = set(evaluation.get("forbidden_actions", []))
    emergency = set(evaluation.get("emergency_actions", []))
    evidence = set(evaluation.get("evidence_gathering_actions", []))
    accepted = [
        str(record["command"]["action_id"])
        for record in stats.action_records
        if record.get("outcome", {}).get("accepted")
    ]
    misleading = sum(action_id in forbidden for action_id in accepted)
    return {
        "misleading_action_count": misleading,
        "misleading_action_rate": misleading / len(accepted) if accepted else 0.0,
        "unnecessary_emergency_action_count": sum(
            action_id in emergency for action_id in accepted
        ),
        "unnecessary_emergency_action_rate": (
            sum(action_id in emergency for action_id in accepted) / len(accepted)
            if accepted
            else 0.0
        ),
        "evidence_gathering_action_count": sum(
            action_id in evidence for action_id in accepted
        ),
        "interference_safety_outcome": len(simulator.safety_violations) == 0,
    }


def _conditional_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
) -> dict[str, Any]:
    required = set(scenario.evaluation.get("required_inspections", []))
    inspection_ids = {
        str(action["id"])
        for action in scenario.actions
        if action.get("operation") == "inspect"
    }
    accepted = [
        str(record["command"]["action_id"])
        for record in stats.action_records
        if record.get("outcome", {}).get("accepted")
    ]
    accepted_inspections = [item for item in accepted if item in inspection_ids]
    return {
        "missed_falsifying_observations": len(required - set(accepted_inspections)),
        "unnecessary_inspection_count": sum(
            action_id not in required for action_id in accepted_inspections
        ),
        "final_rule_satisfaction": all(
            not simulator._constraint_violated(constraint)  # noqa: SLF001
            for constraint in scenario.constraints
        ),
    }


def _adaptation_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
) -> dict[str, Any]:
    metadata = dict(scenario.evaluation.get("adaptation", {}))
    if not metadata and scenario.evaluation.get("change_step") is not None:
        metadata = {
            "regime_change_step": scenario.evaluation["change_step"],
            "original_actions": scenario.evaluation.get("obsolete_actions", []),
            "replacement_actions": scenario.evaluation.get("adaptive_actions", []),
            "sufficient_evidence_step": scenario.evaluation["change_step"],
        }
    if not metadata:
        return {}

    change_step = int(metadata["regime_change_step"])
    evidence_step = int(metadata.get("sufficient_evidence_step", change_step))
    original = set(metadata.get("original_actions", []))
    replacements = set(metadata.get("replacement_actions", []))
    safe_replacements = set(metadata.get("safe_replacement_actions", replacements))
    unnecessary = set(metadata.get("unnecessary_exploration_actions", []))
    records = list(stats.action_records)
    original_before = [
        record
        for record in records
        if int(record["step"]) < change_step
        and record["command"]["action_id"] in original
        and record.get("outcome", {}).get("accepted")
    ]
    obsolete_after = [
        record
        for record in records
        if int(record["step"]) >= change_step
        and record["command"]["action_id"] in original
    ]
    effective = [
        record
        for record in records
        if int(record["step"]) >= change_step
        and record["command"]["action_id"] in safe_replacements
        and record.get("outcome", {}).get("accepted")
    ]
    revised = [
        record
        for record in records
        if int(record["step"]) >= change_step
        and record["command"]["action_id"] not in original | {"wait"}
    ]
    first_effective = effective[0] if effective else None
    first_revised = revised[0] if revised else None
    first_interaction_index = next(
        (
            index
            for index, step in enumerate(stats.invocation_steps)
            if step >= change_step
        ),
        None,
    )
    effective_interaction_index = (
        next(
            (
                index
                for index, step in enumerate(stats.invocation_steps)
                if first_effective is not None
                and step >= int(first_effective["step"])
            ),
            None,
        )
        if first_effective is not None
        else None
    )
    recovery_steps = [
        int(item["step"])
        for item in simulator.goal_status_history
        if item["goals_met"] and int(item["step"]) >= change_step
    ]
    recovery_step = min(recovery_steps) if recovery_steps else None
    recovery_end = recovery_step if recovery_step is not None else simulator.step
    post_evidence = [
        record for record in obsolete_after if int(record["step"]) >= evidence_step
    ]
    pre_evidence = [
        record for record in obsolete_after if int(record["step"]) < evidence_step
    ]
    clarity = str(metadata.get("evidence_clarity", "unverified"))
    if not scenario.context.include_action_outcomes:
        feedback_category = "omitted"
    elif clarity == "generic":
        feedback_category = "generic"
    else:
        feedback_category = "clear"
    recovery_cost: dict[str, float] = {}
    action_by_id = {str(action["id"]): action for action in scenario.actions}
    for resource_id in metadata.get("recovery_resource_ids", []):
        recovery_cost[str(resource_id)] = sum(
            float(
                action_by_id.get(str(record["command"]["action_id"]), {})
                .get("costs", {})
                .get(resource_id, 0.0)
            )
            for record in records
            if int(record["step"]) >= change_step
            and record.get("outcome", {}).get("accepted")
        )
    recovery_demonstrated = bool(original_before and first_effective)
    allowed = int(scenario.evaluation.get("allowed_safety_violations", 0))
    adaptation_success = bool(
        recovery_demonstrated
        and simulator.goals_met
        and (
            not scenario.evaluation.get("success_requires_safety", True)
            or len(simulator.safety_violations) <= allowed
        )
    )
    return {
        "adaptation_success": adaptation_success,
        "adaptation_recovery_demonstrated": recovery_demonstrated,
        "regime_change_step": change_step,
        "first_interaction_after_change": (
            stats.invocation_steps[first_interaction_index]
            if first_interaction_index is not None
            else None
        ),
        "first_revised_action": (
            first_revised["command"]["action_id"] if first_revised else None
        ),
        "first_revised_action_step": (
            int(first_revised["step"]) if first_revised else None
        ),
        "first_effective_post_change_action": (
            first_effective["command"]["action_id"] if first_effective else None
        ),
        "first_effective_post_change_action_step": (
            int(first_effective["step"]) if first_effective else None
        ),
        "adaptation_delay_steps": (
            int(first_effective["step"]) - change_step if first_effective else None
        ),
        "evidence_to_adaptation_delay_steps": (
            int(first_effective["step"]) - evidence_step if first_effective else None
        ),
        "adaptation_delay_interactions": (
            effective_interaction_index - first_interaction_index
            if effective_interaction_index is not None
            and first_interaction_index is not None
            else None
        ),
        "obsolete_actions_after_change": len(obsolete_after),
        "obsolete_actions_before_sufficient_evidence": len(pre_evidence),
        "obsolete_actions_after_sufficient_evidence": len(post_evidence),
        "obsolete_actions_after_clear_evidence": (
            len(post_evidence) if feedback_category == "clear" else 0
        ),
        "obsolete_actions_after_generic_evidence": (
            len(post_evidence) if feedback_category == "generic" else 0
        ),
        "obsolete_actions_with_omitted_outcomes": (
            len(post_evidence) if feedback_category == "omitted" else 0
        ),
        "repeated_obsolete_actions": max(0, len(obsolete_after) - 1),
        "unnecessary_exploration_count": sum(
            record["command"]["action_id"] in unnecessary for record in records
        ),
        "recovery_step": recovery_step,
        "recovery_time_steps": (
            recovery_step - change_step if recovery_step is not None else None
        ),
        "recovery_resource_cost": recovery_cost,
        "safety_violations_during_recovery": sum(
            change_step <= int(item["step"]) <= recovery_end
            for item in simulator.safety_violations
        ),
    }


def _temporal_metrics(
    scenario: ScenarioConfig,
    simulator: GraphCPSSimulator,
    stats: RunStatistics,
) -> dict[str, Any]:
    metadata = dict(scenario.evaluation.get("temporal", {}))
    if not metadata:
        return {}
    mapping = {
        str(action): str(response)
        for action, response in dict(
            metadata.get("initiation_to_response", {})
        ).items()
    }
    attribution_actions = set(metadata.get("attribution_actions", mapping.values()))
    completions = [
        item
        for item in simulator.completed_delayed_effects
        if item.get("success") and item.get("action_id") in mapping
    ]
    route_records = [
        record
        for record in stats.action_records
        if record["command"]["action_id"] in attribution_actions
    ]
    associated: set[int] = set()
    detected: set[int] = set()
    correct = 0
    false_associations = 0
    for record in route_records:
        candidates = [
            index
            for index, completion in enumerate(completions)
            if index not in associated
            and int(completion["completed_step"]) <= int(record["step"])
        ]
        if not candidates:
            false_associations += 1
            continue
        index = candidates[0]
        detected.add(index)
        expected = mapping[str(completions[index]["action_id"])]
        if (
            record["command"]["action_id"] == expected
            and record.get("outcome", {}).get("accepted")
        ):
            correct += 1
            associated.add(index)
        else:
            false_associations += 1

    accepted_initiations = sum(
        record["command"]["action_id"] in mapping
        and record.get("outcome", {}).get("accepted")
        for record in stats.action_records
    )
    expected_dependencies = max(
        int(metadata.get("relevant_history_items", 1)),
        accepted_initiations,
        len(completions),
    )
    missed = max(0, expected_dependencies - correct)
    anomaly_actions = set(metadata.get("anomaly_actions", []))
    false_anomalies = sum(
        record["command"]["action_id"] in anomaly_actions
        for record in stats.action_records
    )
    handled = correct >= expected_dependencies and false_associations == 0
    return {
        "temporal_reasoning_success": bool(handled and simulator.goals_met),
        "temporal_dependency_handled": handled,
        "temporal_comparison_accuracy": (
            correct / len(route_records) if route_records else 0.0
        ),
        "delayed_effect_detection_accuracy": (
            len(detected) / expected_dependencies if expected_dependencies else None
        ),
        "delayed_effect_attribution_accuracy": (
            correct / len(route_records) if route_records else 0.0
        ),
        "correct_previous_action_association": correct,
        "false_temporal_associations": false_associations,
        "false_anomaly_detections": false_anomalies,
        "missed_temporal_dependencies": missed,
        "completed_delayed_effects": len(completions),
        "required_temporal_distance": int(
            metadata.get("required_temporal_distance", 0)
        ),
    }


def _coverage(required: list[str], actual: list[str]) -> float | None:
    if not required:
        return None
    return len(set(required) & set(actual)) / len(set(required))


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    iterator = iter(actual)
    return all(any(candidate == item for candidate in iterator) for item in expected)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rejected_action_repetition_metrics(
    scenario: ScenarioConfig, stats: RunStatistics
) -> dict[str, int]:
    feedback_modes = {
        str(action["id"]): str(action.get("failure_feedback", "full"))
        for action in scenario.actions
    }
    clear = 0
    omitted = 0
    hidden = 0
    previous_rejected: str | None = None
    for record in stats.action_records:
        action_id = str(record["command"]["action_id"])
        if previous_rejected == action_id:
            if not scenario.context.include_action_outcomes:
                omitted += 1
            elif feedback_modes.get(action_id, "full") == "generic":
                hidden += 1
            else:
                clear += 1
        previous_rejected = (
            action_id if not record.get("outcome", {}).get("accepted") else None
        )
    return {
        "repeated_rejected_actions_with_clear_feedback": clear,
        "repeated_rejected_actions_with_omitted_feedback": omitted,
        "repeated_rejected_actions_with_hidden_cause": hidden,
    }
