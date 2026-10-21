"""Semantic validation beyond schema and reference checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.invariants import (
    FamilyInvariantReport,
    validate_family_invariants,
)
from cps_reason_bench.references import ReferenceReport, analyze_reference


@dataclass(frozen=True)
class SemanticValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    reference: ReferenceReport | None
    family_invariants: FamilyInvariantReport

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "reference": self.reference.to_dict() if self.reference else None,
            "family_invariants": self.family_invariants.to_dict(),
        }


def validate_scenario_semantics(
    scenario: ScenarioConfig,
    *,
    check_reference: bool = True,
) -> SemanticValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    interaction = scenario.interaction
    if not interaction.initial_invocation and not interaction.allows_periodic:
        errors.append(
            "interaction configuration can never make its first policy invocation"
        )
    if (
        scenario.evaluation.get("callback_target_window")
        and not interaction.allows_callbacks
    ):
        errors.append("callback evaluation is incompatible with the interaction mode")
    resource_maxima = sorted(
        str(resource_id)
        for resource_id, resource in scenario.graph.get("resources", {}).items()
        if resource.get("max") is not None
    )
    if resource_maxima:
        warnings.append(
            "resource maxima are declarative only and are not enforced by the runtime: "
            f"{resource_maxima}"
        )
    if interaction.max_actions_per_interaction > 1:
        if interaction.action_batch_semantics == "atomic":
            warnings.append(
                "multiple actions execute in order inside one atomic transaction; "
                "any rejected command rolls back the complete list, and safety/goal "
                "evaluation occurs after commit or rollback"
            )
        else:
            warnings.append(
                "multiple actions execute sequentially, with per-command rollback on "
                "exceptions and safety/goal evaluation after the complete action list"
            )

    action_ids = set(scenario.action_ids)
    for key in ("required_actions", "required_inspections", "forbidden_actions"):
        unknown = set(scenario.evaluation.get(key, [])) - action_ids
        if unknown:
            errors.append(
                f"evaluation.{key} references unknown actions: {sorted(unknown)}"
            )
    action_by_id = {str(action["id"]): action for action in scenario.actions}
    for action_id in scenario.evaluation.get("required_inspections", []):
        if action_by_id[action_id].get("operation") != "inspect":
            errors.append(f"required inspection {action_id!r} is not an inspect action")

    mutable_targets = _mutable_targets(scenario)
    simulator_initial = {
        f"{node_id}.{variable_id}": variable["initial"]
        for node_id, node in scenario.graph["nodes"].items()
        for variable_id, variable in node["variables"].items()
    }
    for goal in scenario.goals:
        if "any" in goal:
            if not any(
                _condition_reachable_or_holds(
                    condition, simulator_initial, mutable_targets
                )
                for condition in goal["any"]
            ):
                errors.append(
                    f"goal {goal['id']!r} has no reachable alternative condition"
                )
            continue
        conditions = [goal["condition"]] if "condition" in goal else goal.get("all", [])
        for condition in conditions:
            target = str(condition["target"])
            if (
                target in simulator_initial
                and not _condition_holds(simulator_initial[target], condition)
                and target not in mutable_targets
            ):
                errors.append(
                    f"goal {goal['id']!r} targets unreachable state {target!r}"
                )

    for script_item in scenario.baseline.get("script", []):
        step = int(script_item.get("at_step", -1))
        if step < 0 or step >= scenario.max_steps:
            errors.append(f"reference script contains out-of-horizon step {step}")
        decision = script_item.get("decision", {})
        for subscription in decision.get("subscriptions", []):
            if subscription.get("operation") not in {"add", "update"}:
                continue
            event_type = subscription.get("event_type")
            target = subscription.get("filters", {}).get("target")
            if event_type in {"variable_threshold", "resource_threshold"}:
                normalized = (
                    f"resource.{target}"
                    if event_type == "resource_threshold"
                    else target
                )
                if normalized not in mutable_targets:
                    warnings.append(
                        f"subscription {subscription.get('id')!r} targets state with no known change mechanism"
                    )

    reference = analyze_reference(scenario) if check_reference else None
    if reference and reference.script_present:
        if not reference.script_valid:
            errors.append(
                "the configured reference script produces invalid output or actions"
            )
        if not reference.task_success:
            errors.append(
                "the configured reference script does not complete its own scenario"
            )
    if (
        reference
        and reference.policy_kind == "optimal"
        and reference.solver.status != "optimal_established"
    ):
        errors.append(
            "optimal reference claim is not established by the available solver"
        )
    if reference and reference.solver.status in {"not_analyzed", "incomplete"}:
        warnings.append(
            f"automatic solvability not established: {reference.solver.reason}"
        )

    family_invariants = validate_family_invariants(scenario, reference)
    for check in family_invariants.checks:
        if check.status == "failed":
            errors.append(f"family invariant {check.code} failed: {check.message}")
        elif check.status == "warning":
            warnings.append(f"family invariant {check.code}: {check.message}")

    return SemanticValidationReport(
        tuple(errors), tuple(warnings), reference, family_invariants
    )


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
        targets.update(f"resource.{key}" for key in action.get("costs", {}))
    for node_id, node in scenario.graph["nodes"].items():
        for variable_id, variable in node["variables"].items():
            if float(variable.get("drift", 0.0)) != 0.0:
                targets.add(f"{node_id}.{variable_id}")
    for edge in scenario.graph.get("edges", []):
        variable = edge.get("variable")
        if variable and float(edge.get("coupling", 0.0)) != 0.0:
            targets.add(f"{edge['target']}.{variable}")
            if not edge.get("directed", False):
                targets.add(f"{edge['source']}.{variable}")
    for fault in scenario.faults:
        if fault.get("kind") == "drift_change":
            targets.add(str(fault["target"]))
    return targets


def _condition_holds(actual: Any, condition: dict[str, Any]) -> bool:
    expected = condition.get("value")
    operator = condition.get("operator", "eq")
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
    return False


def _condition_reachable_or_holds(
    condition: dict[str, Any],
    initial: dict[str, Any],
    mutable_targets: set[str],
) -> bool:
    target = str(condition["target"])
    return target in mutable_targets or (
        target in initial and _condition_holds(initial[target], condition)
    )
