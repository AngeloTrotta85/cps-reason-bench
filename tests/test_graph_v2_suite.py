from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from cps_reason_bench.graph_v2 import (
    GRAPH_V2_FAMILIES,
    GRAPH_V2_GENERATOR_ID,
    GRAPH_V2_LEVELS,
)
from cps_reason_bench.graph_v2_validation import (
    assert_graph_v2_coverage,
    counterfactual_validity_report,
    coverage_rows,
)
from cps_reason_bench.suites import load_suite, materialize_suite, validate_suite
from cps_reason_bench.versions import PACKAGE_VERSION


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / "benchmark/suites/graph_v2_complete.yaml"
PILOT = ROOT / "benchmark/suites/graph_v2_pilot.yaml"


class GraphV2SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_suite(COMPLETE)
        cls.instances = materialize_suite(cls.suite)
        cls.rows = coverage_rows(cls.instances)

    def test_release_identity_and_complete_matrix(self) -> None:
        self.assertEqual("0.5.0", PACKAGE_VERSION)
        self.assertEqual("cps_reason_graph_v2", self.suite.name)
        self.assertEqual("1.0.0", self.suite.version)
        self.assertEqual(240, len(self.instances))
        self.assertEqual(set(GRAPH_V2_FAMILIES), {row["family"] for row in self.rows})
        self.assertEqual(set(GRAPH_V2_LEVELS), {row["difficulty"] for row in self.rows})
        cells = Counter((row["family"], row["difficulty"]) for row in self.rows)
        self.assertEqual(48, len(cells))
        self.assertEqual({5}, set(cells.values()))
        self.assertEqual(240, len({row["scenario_name"] for row in self.rows}))
        self.assertTrue(
            all(
                instance.scenario.generation.get("generator_id") == GRAPH_V2_GENERATOR_ID
                for instance in self.instances
            )
        )

    def test_declared_graph_coverage_has_no_zero_category(self) -> None:
        report = assert_graph_v2_coverage(self.instances, complete=True)
        self.assertEqual("pass", report["status"])
        features = report["feature_counts"]
        for name in (
            "multi_node",
            "edges",
            "directed",
            "undirected",
            "coupling",
            "topology_change",
            "edge_fault",
            "hidden_edge_state",
            "multi_action",
            "three_action",
            "atomic_batch",
        ):
            self.assertGreater(features[name], 0, name)
        self.assertEqual({"1", "2", "3", "4-8"}, set(report["distributions"]["node_count_category"]))

    def test_materialization_is_stable_across_python_hash_seeds(self) -> None:
        script = """
from pathlib import Path

from cps_reason_bench.graph_v2_validation import assert_graph_v2_coverage
from cps_reason_bench.suites import load_suite, materialize_suite
from cps_reason_bench.versions import content_hash

suite = load_suite(Path("benchmark/suites/graph_v2_complete.yaml"))
instances = materialize_suite(suite)
manifest = [
    {
        "suite_entry_id": item.suite_entry_id,
        "scenario_hash": content_hash(item.scenario.to_mapping()),
        "simulator_seeds": list(item.simulator_seeds),
        "generation_seed": item.generation_seed,
    }
    for item in instances
]
coverage = assert_graph_v2_coverage(instances, complete=True)
print(content_hash(manifest), coverage["coverage_matrix_hash"])
"""
        outputs: list[str] = []
        for hash_seed in ("0", "1"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_graph_size_and_reasoning_level_are_not_identical(self) -> None:
        by_level: dict[str, set[int]] = defaultdict(set)
        for row in self.rows:
            by_level[str(row["difficulty"])].add(int(row["node_count"]))
        self.assertTrue(
            all(
                {1, 6}.issubset(counts)
                and bool(counts.intersection({2, 3}))
                and len(counts) >= 3
                for counts in by_level.values()
            )
        )
        levels_by_size: dict[int, set[str]] = defaultdict(set)
        for row in self.rows:
            levels_by_size[int(row["node_count"])].add(str(row["difficulty"]))
        self.assertEqual(set(GRAPH_V2_LEVELS), levels_by_size[1])
        self.assertEqual(set(GRAPH_V2_LEVELS), levels_by_size[3])
        self.assertEqual(set(GRAPH_V2_LEVELS), levels_by_size[6])

    def test_all_240_scripted_references_are_semantically_valid(self) -> None:
        report = validate_suite(self.suite)
        self.assertTrue(report.valid, report.errors[:5])
        self.assertEqual(240, len(report.scenario_reports))
        self.assertTrue(
            all(
                row["reference"]["script_valid"]
                and row["reference"]["task_success"]
                and row["reference"]["safety_violations"] == 0
                for row in report.scenario_reports
            )
        )

    def test_counterfactual_pairs_change_trajectory_and_correct_behavior(self) -> None:
        report = counterfactual_validity_report(self.instances)
        self.assertEqual("pass", report["status"])
        self.assertEqual(48, report["pair_count"])
        self.assertEqual(
            {"coupling_sign": 12, "coupling_strength": 12, "edge_direction": 12, "edge_state": 12},
            report["pair_type_counts"],
        )
        self.assertTrue(
            all(
                row["same_action_catalog"]
                and row["same_goals"]
                and row["same_non_topology_observation"]
                and row["graph_changed"]
                and row["correct_decision_changed"]
                and row["same_behavior_changes_trajectory"]
                and not row["graph_ignoring_behavior_succeeds_on_right"]
                for row in report["checks"]
            )
        )

    def test_machine_reports_and_construct_documents_match_suite(self) -> None:
        coverage = json.loads((ROOT / "benchmark/graph_v2_coverage_report.json").read_text())
        validity = json.loads((ROOT / "benchmark/graph_v2_construct_validity.json").read_text())
        self.assertEqual("pass", coverage["status"])
        self.assertEqual(240, coverage["scenario_count"])
        self.assertEqual("pass", validity["status"])
        self.assertTrue((ROOT / "docs/GRAPHS_AND_SCENARIOS.md").is_file())
        self.assertTrue((ROOT / "PAPER_SOFTWARE_CONSISTENCY.md").is_file())

    def test_pilot_exercises_all_families_and_required_engine_paths(self) -> None:
        instances = materialize_suite(load_suite(PILOT))
        report = assert_graph_v2_coverage(instances, complete=False)
        self.assertEqual("pass", report["status"])
        self.assertEqual(9, len(instances))
        self.assertEqual(set(GRAPH_V2_FAMILIES), {item.scenario.task_family for item in instances})


if __name__ == "__main__":
    unittest.main()
