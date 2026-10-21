from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cps_reason_bench.config import dump_scenario, load_scenario
from cps_reason_bench.generators import generate_scenario
from cps_reason_bench.suites import load_suite, materialize_suite, validate_suite
from cps_reason_bench.versions import content_hash


ROOT = Path(__file__).resolve().parents[1]


class SuiteAndGenerationTests(unittest.TestCase):
    def test_core_suite_loads_and_is_semantically_valid(self) -> None:
        suite = load_suite(ROOT / "examples" / "suites" / "core_v1.yaml")
        self.assertEqual("cps_reason_core", suite.name)
        self.assertEqual("1.0.0", suite.version)
        self.assertEqual(0, suite.protocol.retry_policy.max_corrections)
        instances = materialize_suite(suite)
        self.assertEqual(4, len(instances))
        self.assertEqual(set(suite.task_families), {item.scenario.task_family for item in instances})
        report = validate_suite(suite)
        self.assertTrue(report.valid, report.errors)

    def test_generation_is_deterministic_and_seeded(self) -> None:
        first = generate_scenario("planning", "controlled_2", 77)
        second = generate_scenario("planning", "controlled_2", 77)
        different = generate_scenario("planning", "controlled_2", 78)
        self.assertEqual(first.to_mapping(), second.to_mapping())
        self.assertNotEqual(first.name, different.name)
        self.assertEqual(4, first.difficulty_profile["minimum_required_planning_depth"])
        self.assertEqual(1, first.difficulty_profile["number_of_distractor_actions"])

    def test_generated_scenario_serialization_replays_exactly(self) -> None:
        scenario = generate_scenario("prospective_reasoning", "controlled_2", 19)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.yaml"
            dump_scenario(scenario, path)
            reloaded = load_scenario(path)
        self.assertEqual(content_hash(scenario.to_mapping()), content_hash(reloaded.to_mapping()))
        self.assertEqual(19, reloaded.generation["generation_seed"])

    def test_authored_scenarios_expose_interpretable_profiles(self) -> None:
        for path in sorted((ROOT / "examples" / "scenarios").glob("*.yaml")):
            with self.subTest(path=path.name):
                scenario = load_scenario(path)
                self.assertIn(scenario.difficulty_level, {"basic", "controlled_1"})
                self.assertIn("number_of_state_variables", scenario.difficulty_profile)
                self.assertIn("family_parameters", scenario.difficulty_profile)
                self.assertEqual("feasible_reference", scenario.reference["kind"])


if __name__ == "__main__":
    unittest.main()
