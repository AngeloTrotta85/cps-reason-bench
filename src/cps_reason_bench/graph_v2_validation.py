"""Coverage and construct-validity checks for the versioned graph-v2 suite."""

from __future__ import annotations

import copy
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from cps_reason_bench.config import ConfigError, ScenarioConfig
from cps_reason_bench.graph_v2 import (
    GRAPH_V2_FAMILIES,
    GRAPH_V2_LEVELS,
    graph_v2_factor_record,
)
from cps_reason_bench.policies.scripted import ScriptedPolicy
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.suites import SuiteScenarioInstance
from cps_reason_bench.versions import content_hash


REQUIRED_COVERAGE = {
    "node_count_category": {"1", "2", "3", "4-8"},
    "topology_type": {
        "single_component",
        "chain",
        "branch_tree",
        "sparse_mesh",
        "alternative_path",
    },
    "directionality": {"undirected", "directed", "mixed"},
    "coupling_type": {
        "none",
        "single_edge",
        "multiple_incoming",
        "short_propagation_chain",
        "competing_influences",
    },
    "topology_state": {
        "static",
        "edge_disabled",
        "edge_restored",
        "alternative_route",
    },
    "observability_type": {
        "full_local",
        "hidden_neighbor",
        "remote_sensor",
        "delayed_remote_sensor",
        "edge_visible",
        "edge_hidden_inferred",
    },
    "fault_locality": {
        "local",
        "remote",
        "edge",
        "propagated",
        "multiple_simultaneous",
    },
    "graph_dependent_decision_property": {
        "topology_irrelevant_baseline",
        "topology_changes_action",
        "direction_changes_action",
        "coupling_changes_action",
        "edge_failure_changes_action",
    },
}


def coverage_rows(
    instances: Iterable[SuiteScenarioInstance],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        if len(instance.simulator_seeds) != 1:
            raise ConfigError(
                f"Graph-v2 entry {instance.suite_entry_id!r} must declare one simulator seed"
            )
        row = graph_v2_factor_record(
            instance.scenario,
            suite_entry_id=instance.suite_entry_id,
            simulator_seed=instance.simulator_seeds[0],
        )
        row["node_count_category"] = _node_count_category(int(row["node_count"]))
        row["semantic_scenario_hash"] = instance.scenario.generation.get(
            "semantic_scenario_hash"
        )
        row["physical_scenario_hash"] = instance.scenario.generation.get(
            "physical_scenario_hash"
        )
        rows.append(row)
    return rows


def graph_v2_coverage_report(
    instances: Iterable[SuiteScenarioInstance],
    *,
    complete: bool = True,
) -> dict[str, Any]:
    materialized = tuple(instances)
    rows = coverage_rows(materialized)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, evidence: Any) -> None:
        checks.append(
            {"id": check_id, "status": "pass" if passed else "fail", "evidence": evidence}
        )

    family_counts = Counter(str(row["family"]) for row in rows)
    level_counts = Counter(str(row["difficulty"]) for row in rows)
    cell_counts = Counter(
        (str(row["family"]), str(row["difficulty"])) for row in rows
    )
    check(
        "six_families",
        set(family_counts) == set(GRAPH_V2_FAMILIES),
        dict(sorted(family_counts.items())),
    )
    if complete:
        check(
            "eight_levels",
            set(level_counts) == set(GRAPH_V2_LEVELS),
            dict(sorted(level_counts.items())),
        )
        check("primary_condition_count", len(rows) == 240, len(rows))
        check(
            "five_instances_per_family_level",
            len(cell_counts) == 48 and set(cell_counts.values()) == {5},
            {f"{family}/{level}": count for (family, level), count in sorted(cell_counts.items())},
        )
    else:
        check("pilot_has_all_families", all(family_counts.values()), dict(family_counts))
        check("pilot_has_multiple_levels", len(level_counts) >= 3, dict(level_counts))

    distributions: dict[str, dict[str, int]] = {}
    for field, required in REQUIRED_COVERAGE.items():
        counts = Counter(str(row[field]) for row in rows)
        distributions[field] = dict(sorted(counts.items()))
        if complete:
            missing = sorted(required - set(counts))
            check(f"coverage_{field}", not missing, {"counts": dict(counts), "missing": missing})

    numeric_checks = {
        "multi_node": sum(int(row["node_count"]) > 1 for row in rows),
        "edges": sum(int(row["edge_count"]) > 0 for row in rows),
        "directed": sum(int(row["directed_edge_count"]) > 0 for row in rows),
        "undirected": sum(int(row["undirected_edge_count"]) > 0 for row in rows),
        "coupling": sum(int(row["coupled_edge_count"]) > 0 for row in rows),
        "topology_change": sum(int(row["topology_change_action_count"]) > 0 for row in rows),
        "edge_fault": sum(int(row["edge_fault_count"]) > 0 for row in rows),
        "hidden_edge_state": sum(int(row["hidden_edge_state_count"]) > 0 for row in rows),
        "multi_action": sum(int(row["max_actions_per_interaction"]) > 1 for row in rows),
        "three_action": sum(int(row["max_actions_per_interaction"]) == 3 for row in rows),
        "atomic_batch": sum(str(row["action_batch_semantics"]) == "atomic" for row in rows),
    }
    for label, count in numeric_checks.items():
        check(f"nonzero_{label}", count > 0, count)

    pair_counts = Counter(
        str(row["counterfactual_pair_id"])
        for row in rows
        if row["counterfactual_pair_id"]
    )
    check(
        "counterfactual_pairs",
        (len(pair_counts) == 48 and set(pair_counts.values()) == {2}) if complete else bool(pair_counts),
        {"pair_count": len(pair_counts), "member_counts": dict(Counter(pair_counts.values()))},
    )
    return {
        "benchmark": "cps_reason_graph_v2@1.0.0",
        "profile": "complete" if complete else "pilot",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "scenario_count": len(rows),
        "family_level_cell_count": len(cell_counts),
        "semantic_instance_count": len(rows),
        "distributions": distributions,
        "feature_counts": numeric_checks,
        "checks": checks,
        "coverage_matrix_hash": content_hash(rows),
    }


def assert_graph_v2_coverage(
    instances: Iterable[SuiteScenarioInstance], *, complete: bool = True
) -> dict[str, Any]:
    report = graph_v2_coverage_report(instances, complete=complete)
    if report["status"] != "pass":
        failed = [item["id"] for item in report["checks"] if item["status"] == "fail"]
        raise ConfigError(f"Graph-v2 coverage validation failed: {failed}")
    return report


def counterfactual_validity_report(
    instances: Iterable[SuiteScenarioInstance],
) -> dict[str, Any]:
    grouped: dict[str, list[SuiteScenarioInstance]] = defaultdict(list)
    for instance in instances:
        pair_id = instance.scenario.difficulty_profile.get("graph_v2", {}).get(
            "counterfactual_pair_id"
        )
        if pair_id:
            grouped[str(pair_id)].append(instance)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cps-graph-v2-counterfactual-") as output:
        output_root = Path(output)
        for pair_id, members in sorted(grouped.items()):
            members.sort(
                key=lambda item: str(
                    item.scenario.difficulty_profile["graph_v2"][
                        "counterfactual_pair_member"
                    ]
                )
            )
            if len(members) != 2:
                rows.append({"pair_id": pair_id, "status": "fail", "reason": "pair cardinality"})
                continue
            left, right = members
            left_scenario, right_scenario = left.scenario, right.scenario
            left_expected = list(
                left_scenario.evaluation["graph_v2"]["expected_exact_action_sequence"]
            )
            right_expected = list(
                right_scenario.evaluation["graph_v2"]["expected_exact_action_sequence"]
            )
            same_action_catalog = left_scenario.action_ids == right_scenario.action_ids
            same_goals = left_scenario.goals == right_scenario.goals
            same_local_state = _non_topology_observation(left_scenario) == _non_topology_observation(
                right_scenario
            )
            graph_changed = content_hash(left_scenario.graph) != content_hash(right_scenario.graph)
            correct_decision_changed = left_expected != right_expected

            left_result = BenchmarkRunner(
                left_scenario,
                ScriptedPolicy("graph_v2_pair_reference"),
                output_root / pair_id / "reference_left",
                seed=99173,
            ).run()
            right_result = BenchmarkRunner(
                right_scenario,
                ScriptedPolicy("graph_v2_pair_reference"),
                output_root / pair_id / "reference_right",
                seed=99173,
            ).run()
            same_behavior_right = _scenario_with_script(
                right_scenario, left_scenario.baseline
            )
            ignored_result = BenchmarkRunner(
                same_behavior_right,
                ScriptedPolicy("graph_ignoring_counterfactual"),
                output_root / pair_id / "same_behavior_right",
                seed=99173,
            ).run()
            trajectory_changed = _trajectory_hash(left_result.events_path) != _trajectory_hash(
                ignored_result.events_path
            )
            left_success = bool(left_result.metrics["general"]["task_success"])
            right_success = bool(right_result.metrics["general"]["task_success"])
            ignored_success = bool(ignored_result.metrics["general"]["task_success"])
            passed = all(
                (
                    same_action_catalog,
                    same_goals,
                    same_local_state,
                    graph_changed,
                    correct_decision_changed,
                    trajectory_changed,
                    left_success,
                    right_success,
                    not ignored_success,
                )
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": left_scenario.difficulty_profile["graph_v2"][
                        "counterfactual_pair_type"
                    ],
                    "family": left_scenario.task_family,
                    "difficulty": left_scenario.difficulty_level,
                    "same_action_catalog": same_action_catalog,
                    "same_goals": same_goals,
                    "same_non_topology_observation": same_local_state,
                    "graph_changed": graph_changed,
                    "correct_decision_changed": correct_decision_changed,
                    "same_behavior_changes_trajectory": trajectory_changed,
                    "left_reference_success": left_success,
                    "right_reference_success": right_success,
                    "graph_ignoring_behavior_succeeds_on_right": ignored_success,
                    "left_expected_actions": left_expected,
                    "right_expected_actions": right_expected,
                    "status": "pass" if passed else "fail",
                }
            )
    return {
        "benchmark": "cps_reason_graph_v2@1.0.0",
        "status": "pass" if rows and all(row["status"] == "pass" for row in rows) else "fail",
        "pair_count": len(rows),
        "pair_type_counts": dict(sorted(Counter(row.get("pair_type") for row in rows).items())),
        "checks": rows,
        "report_hash": content_hash(rows),
    }


def _node_count_category(count: int) -> str:
    if count in {1, 2, 3}:
        return str(count)
    if 4 <= count <= 8:
        return "4-8"
    return "outside_declared_range"


def _non_topology_observation(scenario: ScenarioConfig) -> dict[str, Any]:
    observation = GraphCPSSimulator(scenario, seed=99173).observation()
    observation.pop("topology", None)
    return observation


def _scenario_with_script(
    scenario: ScenarioConfig, baseline: Mapping[str, Any]
) -> ScenarioConfig:
    value = scenario.to_mapping()
    value["baseline"] = copy.deepcopy(dict(baseline))
    return ScenarioConfig.from_mapping(value)


def _trajectory_hash(path: Path) -> str:
    trajectory: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        states = [
            record.get("true_simulator_state"),
            record.get("true_state_before"),
            record.get("true_state_after"),
            record.get("final_system_state"),
        ]
        for state in states:
            if isinstance(state, Mapping):
                trajectory.append(
                    {
                        "step": state.get("step"),
                        "nodes": state.get("nodes"),
                        "edges": state.get("edges"),
                    }
                )
    return content_hash(trajectory)
