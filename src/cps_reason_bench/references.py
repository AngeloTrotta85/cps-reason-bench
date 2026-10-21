"""Reference-policy replay and bounded exact solving for tractable scenarios."""

from __future__ import annotations

import itertools
import json
import tempfile
from dataclasses import dataclass
from typing import Any

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.policies import ScriptedPolicy
from cps_reason_bench.protocol import ActionCommand
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.simulator import GraphCPSSimulator


@dataclass(frozen=True)
class SolverResult:
    status: str
    reason: str
    solvable: bool | None
    minimum_action_count: int | None
    minimum_resource_use: dict[str, float] | None
    solution_sequences: tuple[tuple[str, ...], ...] = ()
    explored_sequences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "solvable": self.solvable,
            "minimum_action_count": self.minimum_action_count,
            "minimum_resource_use": self.minimum_resource_use,
            "solution_sequences": [list(item) for item in self.solution_sequences],
            "explored_sequences": self.explored_sequences,
        }


@dataclass(frozen=True)
class ReferenceReport:
    policy_kind: str
    claim: str
    script_present: bool
    script_valid: bool | None
    task_success: bool | None
    safety_violations: int | None
    accepted_action_count: int | None
    valid_action_sequence: tuple[str, ...]
    valid_callback_intervals: tuple[int, ...] | None
    required_event_subscriptions: tuple[dict[str, Any], ...]
    required_inspections: tuple[str, ...]
    known_rule_violations: int | None
    expected_rejection_count: int
    observed_expected_rejections: int
    unexpected_rejections: int
    solver: SolverResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "claim": self.claim,
            "script_present": self.script_present,
            "script_valid": self.script_valid,
            "task_success": self.task_success,
            "safety_violations": self.safety_violations,
            "accepted_action_count": self.accepted_action_count,
            "valid_action_sequence": list(self.valid_action_sequence),
            "valid_callback_intervals": (
                list(self.valid_callback_intervals)
                if self.valid_callback_intervals is not None
                else None
            ),
            "required_event_subscriptions": list(self.required_event_subscriptions),
            "required_inspections": list(self.required_inspections),
            "known_rule_violations": self.known_rule_violations,
            "expected_rejection_count": self.expected_rejection_count,
            "observed_expected_rejections": self.observed_expected_rejections,
            "unexpected_rejections": self.unexpected_rejections,
            "solver": self.solver.to_dict(),
        }


def analyze_reference(scenario: ScenarioConfig) -> ReferenceReport:
    script_present = bool(scenario.baseline.get("script"))
    policy_kind = str(scenario.reference.get("kind", "demonstration"))
    claim = str(
        scenario.reference.get(
            "optimality",
            "not_claimed" if policy_kind != "optimal" else "claimed",
        )
    )
    script_valid: bool | None = None
    task_success: bool | None = None
    safety_violations: int | None = None
    accepted_actions: int | None = None
    expected_rejections = {
        (str(item["action_id"]), int(item["at_step"]))
        for item in scenario.evaluation.get("reference_expected_rejections", [])
    }
    observed_expected_rejections = 0
    unexpected_rejections = 0
    if script_present:
        with tempfile.TemporaryDirectory(prefix="cps-reference-") as output_dir:
            result = BenchmarkRunner(
                scenario,
                ScriptedPolicy("reference_replay"),
                output_dir,
            ).run()
            rejected = _rejected_actions_from_trace(result.events_path)
        general = result.metrics["general"]
        observed = {(item["action_id"], item["step"]) for item in rejected}
        observed_expected_rejections = len(observed & expected_rejections)
        unexpected_rejections = len(observed - expected_rejections)
        missing_expected_rejections = len(expected_rejections - observed)
        script_valid = bool(
            general["invalid_structured_outputs"] == 0
            and unexpected_rejections == 0
            and missing_expected_rejections == 0
        )
        task_success = bool(general["task_success"])
        safety_violations = int(general["safety_violations"])
        accepted_actions = int(general["accepted_actions"])

    solver = solve_small_scenario(scenario)
    scripted_actions = tuple(
        str(action["action_id"])
        for item in scenario.baseline.get("script", [])
        for action in item.get("decision", {}).get("actions", [])
    )
    subscriptions = tuple(
        dict(subscription)
        for item in scenario.baseline.get("script", [])
        for subscription in item.get("decision", {}).get("subscriptions", [])
        if subscription.get("operation") == "add"
    )
    callback_window = scenario.evaluation.get("callback_target_window")
    callback_intervals = (
        tuple(range(int(callback_window[0]), int(callback_window[1]) + 1))
        if isinstance(callback_window, list) and len(callback_window) == 2
        else None
    )
    return ReferenceReport(
        policy_kind=policy_kind,
        claim=claim,
        script_present=script_present,
        script_valid=script_valid,
        task_success=task_success,
        safety_violations=safety_violations,
        accepted_action_count=accepted_actions,
        valid_action_sequence=scripted_actions,
        valid_callback_intervals=callback_intervals,
        required_event_subscriptions=subscriptions,
        required_inspections=tuple(
            str(item) for item in scenario.evaluation.get("required_inspections", [])
        ),
        known_rule_violations=safety_violations,
        expected_rejection_count=len(expected_rejections),
        observed_expected_rejections=observed_expected_rejections,
        unexpected_rejections=unexpected_rejections,
        solver=solver,
    )


def _rejected_actions_from_trace(path: Any) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("record_type") != "model_interaction":
            continue
        for outcome in record.get("action_outcomes", []):
            if not outcome.get("accepted"):
                rejected.append(
                    {
                        "action_id": str(outcome["action_id"]),
                        "step": int(record["step"]),
                    }
                )
    return rejected


def solve_small_scenario(
    scenario: ScenarioConfig,
    *,
    maximum_depth: int = 8,
    maximum_sequences: int = 5000,
) -> SolverResult:
    reason = _solver_inapplicability_reason(scenario, maximum_depth)
    if reason:
        return SolverResult("not_analyzed", reason, None, None, None)

    action_ids = tuple(
        str(action["id"])
        for action in scenario.actions
        if action.get("operation") != "noop" and action.get("role") != "distractor"
    )
    explored = 0
    for depth in range(0, min(maximum_depth, scenario.max_steps) + 1):
        solutions: list[tuple[str, ...]] = []
        resource_uses: list[dict[str, float]] = []
        for sequence in itertools.product(action_ids, repeat=depth):
            explored += 1
            if explored > maximum_sequences:
                return SolverResult(
                    "incomplete",
                    f"sequence limit {maximum_sequences} reached before exhaustive completion",
                    None,
                    None,
                    tuple(solutions),
                    explored,
                )
            solved, resources = _evaluate_sequence(scenario, sequence)
            if solved:
                if len(solutions) < 10:
                    solutions.append(sequence)
                resource_uses.append(resources)
        if resource_uses:
            minimum_resources = min(
                resource_uses,
                key=lambda item: (sum(item.values()), tuple(sorted(item.items()))),
            )
            return SolverResult(
                "optimal_established",
                "exhaustive breadth-by-depth enumeration of the bounded deterministic action space",
                True,
                depth,
                minimum_resources,
                tuple(solutions),
                explored,
            )
    return SolverResult(
        "unsolvable_within_horizon",
        "all supported action sequences within the episode horizon were exhausted",
        False,
        None,
        None,
        (),
        explored,
    )


def _solver_inapplicability_reason(
    scenario: ScenarioConfig, maximum_depth: int
) -> str | None:
    if scenario.interaction.mode != "periodic" or scenario.interaction.interval_steps != 1:
        return "exact solver currently requires periodic invocation at every step"
    if scenario.max_steps > maximum_depth:
        return f"episode horizon exceeds configured exact-solver depth {maximum_depth}"
    if scenario.faults:
        return "exact solver does not claim completeness for faulted scenarios"
    if any(int(action.get("effect_delay_steps", 0)) for action in scenario.actions):
        return "exact solver does not claim completeness for delayed action effects"
    if any(
        float(variable.get("drift", 0.0)) != 0.0
        for node in scenario.graph["nodes"].values()
        for variable in node["variables"].values()
        if isinstance(variable.get("initial"), (int, float))
        and not isinstance(variable.get("initial"), bool)
    ):
        return "exact solver does not claim completeness for autonomous continuous dynamics"
    unsupported = {
        str(action.get("operation", "noop"))
        for action in scenario.actions
        if action.get("operation", "noop") not in {"set", "adjust", "inspect", "noop"}
    }
    if unsupported:
        return f"unsupported solver action operations: {sorted(unsupported)}"
    if any(action.get("parameters") for action in scenario.actions):
        return "parameterized actions are outside the bounded solver"
    action_count = sum(action.get("operation") != "noop" for action in scenario.actions)
    if action_count > 7:
        return "action branching factor exceeds the bounded solver limit"
    return None


def _evaluate_sequence(
    scenario: ScenarioConfig, sequence: tuple[str, ...]
) -> tuple[bool, dict[str, float]]:
    simulator = GraphCPSSimulator(scenario, scenario.seed)
    initial_resources = {
        key: float(value) for key, value in simulator.resources.items()
    }
    if simulator.goals_met:
        return _safe_success(scenario, simulator), {
            key: 0.0 for key in initial_resources
        }
    for action_id in sequence:
        outcomes, _ = simulator.apply_actions([ActionCommand(action_id)])
        if not outcomes[0].accepted:
            return False, {}
        if simulator.goals_met:
            return _safe_success(scenario, simulator), _resource_use(
                initial_resources, simulator
            )
        simulator.advance()
    return simulator.goals_met and _safe_success(scenario, simulator), _resource_use(
        initial_resources, simulator
    )


def _safe_success(scenario: ScenarioConfig, simulator: GraphCPSSimulator) -> bool:
    if not scenario.evaluation.get("success_requires_safety", True):
        return True
    allowed = int(scenario.evaluation.get("allowed_safety_violations", 0))
    return len(simulator.safety_violations) <= allowed


def _resource_use(
    initial: dict[str, float], simulator: GraphCPSSimulator
) -> dict[str, float]:
    return {
        key: initial[key] - float(simulator.resources[key])
        for key in initial
    }
