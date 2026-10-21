from __future__ import annotations

import copy
import json
import tempfile
import unittest

from cps_reason_bench.config import ScenarioConfig
from cps_reason_bench.generators import generate_scenario
from cps_reason_bench.policies import ScriptedPolicy
from cps_reason_bench.protocol import ActionCommand
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.validation import validate_scenario_semantics


class AdaptationFamilyTests(unittest.TestCase):
    def test_adaptation_generation_is_deterministic_across_four_levels(self) -> None:
        levels = [f"controlled_{index}" for index in range(1, 5)]
        for index, level in enumerate(levels, start=1):
            with self.subTest(level=level):
                first = generate_scenario("adaptation", level, 300 + index)
                second = generate_scenario("adaptation", level, 300 + index)
                self.assertEqual(first.to_mapping(), second.to_mapping())
                self.assertEqual(index, first.difficulty)
                profile = first.difficulty_profile["reasoning_profile"]
                self.assertEqual(level, profile["intended_progression"])
                self.assertEqual("2.0.0", first.generation["generator_version"])

    def test_regime_change_preserves_initial_effect_then_degrades_it(self) -> None:
        obvious = generate_scenario("adaptation", "controlled_1", 310)
        simulator = GraphCPSSimulator(obvious)
        before = simulator.resolve_target("process.output")
        outcome, _ = simulator.apply_actions([ActionCommand("primary_boost")])
        self.assertTrue(outcome[0].accepted)
        self.assertEqual(3.0, simulator.resolve_target("process.output") - before)
        while simulator.step < 2:
            simulator.advance()
        rejected, _ = simulator.apply_actions([ActionCommand("primary_boost")])
        self.assertFalse(rejected[0].accepted)
        public = rejected[0].to_public_dict()
        self.assertTrue(public["details"]["failure_cause_visible"])

        reduced = generate_scenario("adaptation", "controlled_2", 311)
        reduced_simulator = GraphCPSSimulator(reduced)
        while reduced_simulator.step < 2:
            reduced_simulator.advance()
        before_reduced = reduced_simulator.resolve_target("process.output")
        scaled, _ = reduced_simulator.apply_actions([ActionCommand("primary_boost")])
        self.assertTrue(scaled[0].accepted)
        self.assertEqual(
            0.75,
            reduced_simulator.resolve_target("process.output") - before_reduced,
        )

    def test_adaptation_metrics_respect_evidence_timing_and_recovery(self) -> None:
        scenario = generate_scenario("adaptation", "controlled_2", 312)
        with tempfile.TemporaryDirectory() as directory:
            result = BenchmarkRunner(
                scenario, ScriptedPolicy(), directory
            ).run()
        metrics = result.metrics["task_specific"]
        self.assertTrue(metrics["adaptation_success"])
        self.assertEqual(2, metrics["adaptation_delay_steps"])
        self.assertEqual(2, metrics["obsolete_actions_before_sufficient_evidence"])
        self.assertEqual(0, metrics["obsolete_actions_after_sufficient_evidence"])
        self.assertEqual(1, metrics["repeated_obsolete_actions"])
        self.assertEqual({"recovery_energy": 4.0}, metrics["recovery_resource_cost"])

    def test_every_adaptation_reference_is_valid_and_recovers(self) -> None:
        for index in range(1, 5):
            level = f"controlled_{index}"
            with self.subTest(level=level):
                report = validate_scenario_semantics(
                    generate_scenario("adaptation", level, 320 + index)
                )
                self.assertTrue(report.valid, report.errors)
                self.assertEqual("verified", report.family_invariants.status)
                assert report.reference is not None
                self.assertTrue(report.reference.script_valid)
                self.assertTrue(report.reference.task_success)


class TemporalTrackingFamilyTests(unittest.TestCase):
    def test_temporal_generation_is_deterministic_across_four_levels(self) -> None:
        for index in range(1, 5):
            level = f"controlled_{index}"
            with self.subTest(level=level):
                first = generate_scenario("temporal_tracking", level, 400 + index)
                second = generate_scenario("temporal_tracking", level, 400 + index)
                self.assertEqual(first.to_mapping(), second.to_mapping())
                self.assertEqual(index, first.difficulty)
                self.assertEqual(
                    first.evaluation["temporal"]["required_temporal_distance"],
                    first.difficulty_profile["reasoning_profile"]["temporal_distance"],
                )

    def test_delayed_effect_occurs_at_declared_distance(self) -> None:
        scenario = generate_scenario("temporal_tracking", "controlled_2", 402)
        start = scenario.baseline["script"][0]["decision"]["actions"][0][
            "action_id"
        ]
        simulator = GraphCPSSimulator(scenario)
        outcomes, _ = simulator.apply_actions([ActionCommand(start)])
        self.assertTrue(outcomes[0].accepted)
        self.assertEqual("scheduled", outcomes[0].message)
        simulator.advance()
        self.assertFalse(simulator.resolve_target("actuator_bank.response_ready"))
        simulator.advance()
        self.assertTrue(simulator.resolve_target("actuator_bank.response_ready"))
        self.assertEqual(1, len(simulator.completed_delayed_effects))
        self.assertEqual(2, simulator.completed_delayed_effects[0]["completed_step"])

    def test_history_context_contains_dependency_and_current_control_does_not(self) -> None:
        scenario = generate_scenario("temporal_tracking", "controlled_2", 404)
        with tempfile.TemporaryDirectory() as directory:
            history_result = BenchmarkRunner(
                scenario, ScriptedPolicy(), directory
            ).run()
            history_records = [
                json.loads(line)
                for line in history_result.events_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if '"record_type": "model_interaction"' in line
            ]
            attribution = next(item for item in history_records if item["step"] == 2)
            self.assertIn("interaction_history", attribution["explicit_context"])
            self.assertEqual(
                0, attribution["explicit_context"]["interaction_history"][0]["step"]
            )

            current_data = copy.deepcopy(scenario.to_mapping())
            current_data.pop("difficulty_profile", None)
            current_data["context"]["mode"] = "current"
            current_data["evaluation"]["context_negative_control"] = True
            current = ScenarioConfig.from_mapping(current_data)
            current_result = BenchmarkRunner(
                current, ScriptedPolicy(), directory
            ).run()
            current_records = [
                json.loads(line)
                for line in current_result.events_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if '"record_type": "model_interaction"' in line
            ]
        current_attribution = next(item for item in current_records if item["step"] == 2)
        self.assertNotIn("interaction_history", current_attribution["explicit_context"])
        self.assertNotIn("previous_interaction", current_attribution["explicit_context"])

    def test_wrong_delayed_effect_attribution_is_counted(self) -> None:
        scenario = generate_scenario("temporal_tracking", "controlled_1", 406)
        raw = copy.deepcopy(scenario.to_mapping())
        raw.pop("difficulty_profile", None)
        correct_route = raw["baseline"]["script"][1]["decision"]["actions"][0][
            "action_id"
        ]
        wrong_route = (
            "route_channel_b" if correct_route == "route_channel_a" else "route_channel_a"
        )
        raw["baseline"]["script"][1]["decision"]["actions"] = [
            {"action_id": wrong_route}
        ]
        wrong = ScenarioConfig.from_mapping(raw)
        with tempfile.TemporaryDirectory() as directory:
            result = BenchmarkRunner(wrong, ScriptedPolicy(), directory).run()
        metrics = result.metrics["task_specific"]
        self.assertEqual(1, metrics["false_temporal_associations"])
        self.assertEqual(1, metrics["missed_temporal_dependencies"])
        self.assertFalse(metrics["temporal_reasoning_success"])
        self.assertFalse(result.metrics["general"]["task_success"])

    def test_overlapping_reference_attributes_both_effects(self) -> None:
        scenario = generate_scenario("temporal_tracking", "controlled_4", 408)
        with tempfile.TemporaryDirectory() as directory:
            result = BenchmarkRunner(
                scenario, ScriptedPolicy(), directory
            ).run()
        metrics = result.metrics["task_specific"]
        self.assertEqual(2, metrics["completed_delayed_effects"])
        self.assertEqual(2, metrics["correct_previous_action_association"])
        self.assertEqual(1.0, metrics["delayed_effect_attribution_accuracy"])
        self.assertTrue(metrics["temporal_reasoning_success"])


if __name__ == "__main__":
    unittest.main()
