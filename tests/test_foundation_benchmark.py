from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cps_reason_bench.catalog import render_catalog_markdown
from cps_reason_bench.generators import (
    available_foundation_generator_levels,
    generate_scenario,
)
from cps_reason_bench.policies import ScriptedPolicy
from cps_reason_bench.references import analyze_reference
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.suites import load_suite, materialize_suite, validate_suite
from cps_reason_bench.validation import validate_scenario_semantics
from cps_reason_bench.versions import (
    PROMPT_TEMPLATE_CONTENT_HASH,
    PROMPT_TEMPLATE_V2_CONTENT_HASH,
)


ROOT = Path(__file__).resolve().parents[1]
FOUNDATION_LEVELS = ("entry_1", "entry_2", "entry_3", "bridge")
FAMILIES = (
    "prospective_reasoning",
    "planning",
    "interference",
    "conditional_safety",
    "adaptation",
    "temporal_tracking",
)


class ProtocolSanityTests(unittest.TestCase):
    def test_protocol_sanity_suite_has_four_separate_verified_tasks(self) -> None:
        suite = load_suite(ROOT / "examples" / "suites" / "protocol_sanity_v1.yaml")
        instances = materialize_suite(suite)

        self.assertEqual("cps_reason_protocol_sanity_v1", suite.name)
        self.assertEqual(4, len(instances))
        self.assertEqual(("protocol_sanity",), suite.task_families)
        self.assertTrue(validate_suite(suite).valid)
        for instance in instances:
            report = validate_scenario_semantics(instance.scenario)
            self.assertEqual("verified", report.family_invariants.status)
            self.assertEqual(0, suite.protocol.retry_policy.max_corrections)

    def test_direct_action_parameter_and_explicit_noop_references_succeed(self) -> None:
        suite = load_suite(ROOT / "examples" / "suites" / "protocol_sanity_v1.yaml")
        with tempfile.TemporaryDirectory() as directory:
            for instance in materialize_suite(suite):
                with self.subTest(instance=instance.suite_entry_id):
                    result = BenchmarkRunner(
                        instance.scenario,
                        ScriptedPolicy(),
                        Path(directory) / instance.suite_entry_id,
                    ).run()
                    self.assertTrue(result.metrics["general"]["task_success"])
                    self.assertEqual(1, result.metrics["general"]["accepted_actions"])
                    self.assertEqual(
                        0, result.metrics["general"]["unnecessary_actions"]
                    )


class FoundationGeneratorTests(unittest.TestCase):
    def test_four_foundation_levels_are_additive_and_deterministic(self) -> None:
        self.assertEqual(
            {family: FOUNDATION_LEVELS for family in FAMILIES},
            available_foundation_generator_levels(),
        )
        for family_index, family in enumerate(FAMILIES):
            for level_index, level in enumerate(FOUNDATION_LEVELS, start=1):
                seed = 8000 + family_index * 10 + level_index
                with self.subTest(family=family, level=level):
                    first = generate_scenario(family, level, seed)
                    second = generate_scenario(family, level, seed)
                    self.assertEqual(first.to_mapping(), second.to_mapping())
                    self.assertEqual("3.0.0", first.generation["generator_version"])
                    self.assertEqual(level_index, first.difficulty)
                    self.assertEqual(
                        "foundation",
                        first.difficulty_profile["reasoning_profile"][
                            "benchmark_layer"
                        ],
                    )

    def test_every_foundation_reference_and_invariant_is_valid(self) -> None:
        for family_index, family in enumerate(FAMILIES):
            for level_index, level in enumerate(FOUNDATION_LEVELS):
                with self.subTest(family=family, level=level):
                    scenario = generate_scenario(
                        family, level, 8200 + family_index * 10 + level_index
                    )
                    report = validate_scenario_semantics(scenario)
                    self.assertTrue(report.valid, report.errors)
                    self.assertEqual("verified", report.family_invariants.status)
                    assert report.reference is not None
                    self.assertTrue(report.reference.script_valid)
                    self.assertTrue(report.reference.task_success)

    def test_planning_has_no_shorter_bypass_and_generous_horizon(self) -> None:
        for expected_depth, level in enumerate(FOUNDATION_LEVELS, start=1):
            scenario = generate_scenario("planning", level, 8300 + expected_depth)
            reference = analyze_reference(scenario)
            intended_depth = min(expected_depth, 3)
            self.assertEqual(intended_depth, reference.solver.minimum_action_count)
            self.assertGreaterEqual(scenario.max_steps - intended_depth, 2)

    def test_entry_tasks_do_not_add_noise_or_delayed_sensing(self) -> None:
        for family_index, family in enumerate(FAMILIES):
            scenario = generate_scenario(family, "entry_1", 8400 + family_index)
            with self.subTest(family=family):
                self.assertTrue(
                    all(
                        float(sensor.get("noise_std", 0.0)) == 0.0
                        for sensor in scenario.sensors
                    )
                )
                self.assertTrue(
                    all(
                        int(sensor.get("delay_steps", 0)) == 0
                        for sensor in scenario.sensors
                    )
                )
                if family != "adaptation":
                    self.assertEqual(0, len(scenario.faults))


class FoundationSuiteTests(unittest.TestCase):
    def test_released_core_and_prompt_identities_remain_frozen(self) -> None:
        expected = {
            "core_v1.yaml": "f63f560b11f79e4b00011e0fcc4d7f5e53cf6bdebeaf0ce8f0b6963b912ea478",
            "core_v1_correction_assisted.yaml": "8e904ba8c5f465e4ea5fd87660bdbe89ec40cff6e1fe4c0a014886e98e556c03",
            "core_v2_smoke.yaml": "de834116639e27ca1b98a0210d07ed952abe269ccd63d0325ae0335066216772",
            "core_v2.yaml": "608d1e5e810da8b811bd9240c597910c85847d948754b397a599da00e4daa5c2",
        }
        for filename, identity in expected.items():
            with self.subTest(suite=filename):
                suite = load_suite(ROOT / "examples" / "suites" / filename)
                self.assertEqual(identity, suite.identity["content_hash"])
        self.assertEqual(
            "148b55721925af4a71a7a764e6d2482f53f4310502c635827809834756d73035",
            PROMPT_TEMPLATE_CONTENT_HASH,
        )
        self.assertEqual(
            "47552b82b406fc0514cc13c7951d88012da1d1175b731455b815faf8ccc731f1",
            PROMPT_TEMPLATE_V2_CONTENT_HASH,
        )

    def test_smoke_and_full_suites_validate(self) -> None:
        smoke = load_suite(ROOT / "examples" / "suites" / "foundation_v1_smoke.yaml")
        full = load_suite(ROOT / "examples" / "suites" / "foundation_v1.yaml")
        self.assertEqual(6, len(materialize_suite(smoke)))
        self.assertEqual(24, len(materialize_suite(full)))
        self.assertTrue(validate_suite(smoke).valid)
        self.assertTrue(validate_suite(full).valid)
        self.assertEqual("2.0.0", full.protocol.prompt_template_version)
        self.assertEqual(0, full.protocol.retry_policy.max_corrections)

    def test_foundation_smoke_scripted_execution_succeeds(self) -> None:
        suite = load_suite(ROOT / "examples" / "suites" / "foundation_v1_smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            results = [
                BenchmarkRunner(
                    instance.scenario,
                    ScriptedPolicy(),
                    Path(directory) / instance.suite_entry_id,
                    seed=instance.simulator_seeds[0],
                ).run()
                for instance in materialize_suite(suite)
            ]
        self.assertEqual(6, len(results))
        self.assertTrue(
            all(result.metrics["general"]["task_success"] for result in results)
        )

    def test_public_catalog_does_not_expose_reference_answers(self) -> None:
        suite = load_suite(ROOT / "examples" / "suites" / "foundation_v1.yaml")
        catalog = render_catalog_markdown(suite)
        self.assertNotIn("expected_action_sequence", catalog)
        self.assertNotIn("reference_action_sequence", catalog)
        self.assertNotIn("solver_solution_sequences", catalog)


if __name__ == "__main__":
    unittest.main()
