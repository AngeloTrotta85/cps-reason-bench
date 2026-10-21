from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cps_reason_bench.config import ConfigError, ScenarioConfig, load_scenario
from cps_reason_bench.policies import ScriptedPolicy
from cps_reason_bench.protocol import ActionCommand
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.simulator import GraphCPSSimulator
from cps_reason_bench.validation import validate_scenario_semantics


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "examples" / "scenarios"


class ConfigAndSimulatorTests(unittest.TestCase):
    def test_all_examples_load(self) -> None:
        paths = sorted(SCENARIOS.glob("*.yaml"))
        self.assertEqual(6, len(paths))
        for path in paths:
            with self.subTest(path=path.name):
                scenario = load_scenario(path)
                self.assertGreater(scenario.max_steps, 0)
                self.assertTrue(scenario.action_ids)

    def test_partial_observation_does_not_include_true_state(self) -> None:
        scenario = load_scenario(SCENARIOS / "conditional_safety_basic.yaml")
        simulator = GraphCPSSimulator(scenario)
        observation = simulator.observation()
        self.assertNotIn("full_state", observation)
        self.assertNotIn("valve_open", str(observation["sensors"]))
        self.assertFalse(simulator.true_state()["nodes"]["transfer"]["valve_open"])

    def test_stuck_sensor_is_deterministic_and_marked_stale(self) -> None:
        scenario = load_scenario(SCENARIOS / "interference_basic.yaml")
        first = GraphCPSSimulator(scenario).observation()
        second = GraphCPSSimulator(scenario).observation()
        self.assertEqual(first, second)
        primary = first["sensors"]["primary_temperature"]
        redundant = first["sensors"]["redundant_temperature"]
        self.assertEqual("stale", primary["status"])
        self.assertEqual(9.0, primary["value"])
        self.assertEqual(4.0, redundant["value"])

    def test_unknown_sensor_reference_is_rejected_during_loading(self) -> None:
        source = load_scenario(SCENARIOS / "planning_basic.yaml")
        invalid = copy.deepcopy(source.raw)
        invalid["sensors"][0]["target"] = "plant.does_not_exist"
        with self.assertRaisesRegex(ConfigError, "unknown target"):
            ScenarioConfig.from_mapping(invalid)

    def test_drift_change_rejects_resource_and_edge_paths_during_loading(self) -> None:
        source = load_scenario(SCENARIOS / "planning_basic.yaml")
        variants = []

        resource = copy.deepcopy(source.raw)
        resource["faults"] = [
            {
                "id": "bad_resource_drift",
                "kind": "drift_change",
                "target": "resource.energy",
                "value": 1.0,
                "at_step": 1,
            }
        ]
        variants.append(resource)

        edge = copy.deepcopy(source.raw)
        edge["graph"]["edges"] = [
            {
                "id": "loop",
                "source": "plant",
                "target": "plant",
                "variable": "power_ready",
            }
        ]
        edge["faults"] = [
            {
                "id": "bad_edge_drift",
                "kind": "drift_change",
                "target": "edge.loop",
                "value": 1.0,
                "at_step": 1,
            }
        ]
        variants.append(edge)

        for raw in variants:
            with self.subTest(target=raw["faults"][0]["target"]), self.assertRaisesRegex(
                ConfigError, "unknown target"
            ):
                ScenarioConfig.from_mapping(raw)

    def test_resource_maximum_is_reported_as_unenforced(self) -> None:
        raw = copy.deepcopy(load_scenario(SCENARIOS / "planning_basic.yaml").raw)
        raw["graph"]["resources"]["energy"]["max"] = 3.0
        report = validate_scenario_semantics(ScenarioConfig.from_mapping(raw))
        self.assertTrue(report.valid)
        self.assertTrue(
            any("resource maxima" in warning for warning in report.warnings)
        )

    def test_unsupported_additional_effect_is_rejected_during_loading(self) -> None:
        raw = copy.deepcopy(load_scenario(SCENARIOS / "planning_basic.yaml").raw)
        raw["actions"][0]["additional_effects"] = [
            {
                "operation": "unsupported",
                "target": "plant.output_integrity",
            }
        ]
        with self.assertRaisesRegex(ConfigError, "unsupported additional effect"):
            ScenarioConfig.from_mapping(raw)

    def test_execution_exception_rolls_back_and_invalidates_the_episode(self) -> None:
        scenario = load_scenario(SCENARIOS / "planning_basic.yaml")
        simulator = GraphCPSSimulator(scenario)
        simulator._action_specs["restore_power"]["additional_effects"] = [
            {
                "operation": "unsupported",
                "target": "plant.output_integrity",
            }
        ]
        before = simulator.true_state()
        outcomes, events = simulator.apply_actions(
            [ActionCommand("restore_power")]
        )

        self.assertEqual("runtime_execution_failure", outcomes[0].reason_code)
        self.assertTrue(outcomes[0].details["rollback_succeeded"])
        self.assertEqual(before, simulator.true_state())
        self.assertEqual("action_runtime_failure", events[0].event_type)
        self.assertFalse(events[0].publicly_observable)

        class MalformedSimulator(GraphCPSSimulator):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self._action_specs["restore_power"]["additional_effects"] = [
                    {
                        "operation": "unsupported",
                        "target": "plant.output_integrity",
                    }
                ]

        with tempfile.TemporaryDirectory() as directory, patch(
            "cps_reason_bench.runner.GraphCPSSimulator", MalformedSimulator
        ):
            result = BenchmarkRunner(
                scenario, ScriptedPolicy(), directory
            ).run()
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

        self.assertEqual("infrastructure_failed", result.completion_status)
        self.assertEqual("runtime_execution_failure", summary["infrastructure_failure"]["category"])
        self.assertIsNone(summary["metrics"]["general"]["task_success"])
        self.assertEqual(0, summary["metrics"]["general"]["simulation_steps"])


if __name__ == "__main__":
    unittest.main()
