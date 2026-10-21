from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from cps_reason_bench.config import ConfigError, ScenarioConfig
from cps_reason_bench.graph_v2 import generate_graph_v2_scenario
from cps_reason_bench.protocol import ActionCommand
from cps_reason_bench.policies import ScriptedPolicy
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.simulator import GraphCPSSimulator


def engine_scenario(
    *,
    nodes: dict[str, float],
    edges: list[dict[str, object]],
    actions: list[dict[str, object]] | None = None,
    faults: list[dict[str, object]] | None = None,
    hard_max: float = 100.0,
    batch: str = "sequential_per_command",
    max_actions: int = 3,
) -> ScenarioConfig:
    graph_nodes = {
        node_id: {
            "label": node_id,
            "variables": {
                "signal": {
                    "initial": value,
                    "hard_min": -100.0,
                    "hard_max": hard_max,
                }
            },
        }
        for node_id, value in nodes.items()
    }
    return ScenarioConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "graph_v2_engine_test",
            "task_family": "general",
            "difficulty": 1,
            "difficulty_level": "test",
            "seed": 7,
            "max_steps": 5,
            "time_step": 1.0,
            "task": {"terminate_on_goals": False},
            "graph": {"nodes": graph_nodes, "edges": edges, "resources": {}},
            "sensors": [
                {"id": f"sensor_{node}", "target": f"{node}.signal"}
                for node in nodes
            ],
            "actions": actions or [{"id": "wait", "operation": "noop"}],
            "faults": faults or [],
            "constraints": [],
            "goals": [
                {
                    "id": "never",
                    "condition": {"target": f"{next(iter(nodes))}.signal", "operator": "ge", "value": 999},
                }
            ],
            "observability": {"full_state": False},
            "interaction": {
                "mode": "periodic",
                "max_actions_per_interaction": max_actions,
                "action_batch_semantics": batch,
            },
            "context": {"mode": "current"},
            "evaluation": {},
            "baseline": {},
            "reference": {},
            "generation": {},
        }
    )


def edge(
    edge_id: str,
    source: str,
    target: str,
    coupling: float,
    *,
    directed: bool,
    enabled: bool = True,
    state_observable: bool = True,
) -> dict[str, object]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "variable": "signal",
        "coupling": coupling,
        "directed": directed,
        "enabled": enabled,
        "observable": True,
        "state_observable": state_observable,
    }


class GraphV2EngineTests(unittest.TestCase):
    def test_directed_and_undirected_two_node_coupling(self) -> None:
        directed = GraphCPSSimulator(
            engine_scenario(nodes={"a": 10.0, "b": 0.0}, edges=[edge("ab", "a", "b", 0.5, directed=True)])
        )
        directed.advance()
        self.assertEqual(10.0, directed.node_values["a"]["signal"])
        self.assertEqual(5.0, directed.node_values["b"]["signal"])

        undirected = GraphCPSSimulator(
            engine_scenario(nodes={"a": 10.0, "b": 0.0}, edges=[edge("ab", "a", "b", 0.5, directed=False)])
        )
        undirected.advance()
        self.assertEqual(5.0, undirected.node_values["a"]["signal"])
        self.assertEqual(5.0, undirected.node_values["b"]["signal"])

    def test_three_node_coupling_is_synchronous_and_propagates_next_step(self) -> None:
        simulator = GraphCPSSimulator(
            engine_scenario(
                nodes={"a": 10.0, "b": 0.0, "c": 0.0},
                edges=[
                    edge("ab", "a", "b", 0.5, directed=True),
                    edge("bc", "b", "c", 0.5, directed=True),
                ],
            )
        )
        simulator.advance()
        self.assertEqual(5.0, simulator.node_values["b"]["signal"])
        self.assertEqual(0.0, simulator.node_values["c"]["signal"])
        simulator.advance()
        self.assertEqual(2.5, simulator.node_values["c"]["signal"])

    def test_multiple_incoming_edges_sum_then_hard_bounds_clamp(self) -> None:
        simulator = GraphCPSSimulator(
            engine_scenario(
                nodes={"a": 10.0, "b": 10.0, "sink": 0.0},
                edges=[
                    edge("a_sink", "a", "sink", 0.5, directed=True),
                    edge("b_sink", "b", "sink", 0.5, directed=True),
                ],
                hard_max=7.0,
            )
        )
        simulator.advance()
        self.assertEqual(7.0, simulator.node_values["sink"]["signal"])

    def test_edge_disable_restore_fault_and_observation_projection(self) -> None:
        scenario = engine_scenario(
            nodes={"a": 10.0, "b": 0.0},
            edges=[edge("ab", "a", "b", 0.5, directed=True, state_observable=False)],
            actions=[
                {
                    "id": "disable",
                    "operation": "toggle_edge",
                    "target": "ab",
                    "edge_enabled": False,
                },
                {
                    "id": "restore",
                    "operation": "toggle_edge",
                    "target": "ab",
                    "edge_enabled": True,
                },
            ],
            faults=[
                {"id": "edge_fault", "kind": "edge_disabled", "target": "ab", "at_step": 1}
            ],
        )
        simulator = GraphCPSSimulator(scenario)
        self.assertNotIn("enabled", simulator.observation()["topology"]["edges"][0])
        simulator.advance()
        self.assertFalse(simulator.edge_states["ab"])
        simulator.apply_actions([ActionCommand("restore")])
        self.assertTrue(simulator.edge_states["ab"])
        simulator.apply_actions([ActionCommand("disable")])
        before = simulator.node_values["b"]["signal"]
        simulator.advance()
        self.assertEqual(before, simulator.node_values["b"]["signal"])

    def test_multi_node_sensors_actions_and_delayed_cross_node_effect(self) -> None:
        scenario = engine_scenario(
            nodes={"a": 0.0, "b": 0.0},
            edges=[edge("ab", "a", "b", 0.5, directed=True)],
            actions=[
                {"id": "set_a", "operation": "set", "target": "a.signal", "value": 10.0, "effect_delay_steps": 1},
                {"id": "set_b", "operation": "set", "target": "b.signal", "value": 2.0},
            ],
        )
        simulator = GraphCPSSimulator(scenario)
        self.assertEqual({"sensor_a", "sensor_b"}, set(simulator.observation()["sensors"]))
        outcomes, _ = simulator.apply_actions([ActionCommand("set_a"), ActionCommand("set_b")])
        self.assertTrue(all(item.accepted for item in outcomes))
        self.assertEqual(0.0, simulator.node_values["a"]["signal"])
        simulator.advance()
        self.assertEqual(10.0, simulator.node_values["a"]["signal"])
        self.assertEqual(6.0, simulator.node_values["b"]["signal"])

    def test_atomic_batch_supports_prerequisites_and_rolls_back_partial_failure(self) -> None:
        actions = [
            {"id": "prepare", "operation": "set", "target": "a.signal", "value": 1.0},
            {
                "id": "act",
                "operation": "set",
                "target": "b.signal",
                "value": 5.0,
                "preconditions": [{"target": "a.signal", "operator": "eq", "value": 1.0}],
            },
        ]
        scenario = engine_scenario(
            nodes={"a": 0.0, "b": 0.0}, edges=[], actions=actions, batch="atomic"
        )
        simulator = GraphCPSSimulator(scenario)
        outcomes, _ = simulator.apply_actions([ActionCommand("prepare"), ActionCommand("act")])
        self.assertTrue(all(item.accepted for item in outcomes))
        self.assertEqual(5.0, simulator.node_values["b"]["signal"])

        simulator.reset()
        before = simulator.true_state()
        outcomes, events = simulator.apply_actions(
            [ActionCommand("prepare"), ActionCommand("missing"), ActionCommand("act")]
        )
        self.assertEqual(before, simulator.true_state())
        self.assertEqual(3, len(outcomes))
        self.assertEqual("batch_rolled_back", outcomes[0].reason_code)
        self.assertEqual("invalid_action", outcomes[1].reason_code)
        self.assertEqual("batch_not_executed", outcomes[2].reason_code)
        self.assertTrue(any(item.event_type == "action_batch_rolled_back" for item in events))

    def test_legacy_batches_commit_prior_commands_and_validate_batch_mode(self) -> None:
        scenario = engine_scenario(
            nodes={"a": 0.0},
            edges=[],
            actions=[{"id": "set_a", "operation": "set", "target": "a.signal", "value": 4.0}],
        )
        simulator = GraphCPSSimulator(scenario)
        outcomes, _ = simulator.apply_actions([ActionCommand("set_a"), ActionCommand("missing")])
        self.assertTrue(outcomes[0].accepted)
        self.assertEqual(4.0, simulator.node_values["a"]["signal"])

        raw = copy.deepcopy(scenario.raw)
        raw["interaction"]["action_batch_semantics"] = "unknown"
        with self.assertRaisesRegex(ConfigError, "action_batch_semantics"):
            ScenarioConfig.from_mapping(raw)

    def test_safety_is_evaluated_after_the_complete_committed_list(self) -> None:
        scenario = engine_scenario(
            nodes={"a": 0.0},
            edges=[],
            actions=[
                {"id": "high", "operation": "set", "target": "a.signal", "value": 20.0},
                {"id": "safe", "operation": "set", "target": "a.signal", "value": 5.0},
            ],
            batch="atomic",
            max_actions=2,
        )
        raw = copy.deepcopy(scenario.raw)
        raw.pop("difficulty_profile", None)
        raw["constraints"] = [
            {
                "id": "upper",
                "kind": "condition",
                "condition": {"target": "a.signal", "operator": "le", "value": 10.0},
            }
        ]
        simulator = GraphCPSSimulator(ScenarioConfig.from_mapping(raw))
        outcomes, _ = simulator.apply_actions([ActionCommand("high"), ActionCommand("safe")])
        self.assertTrue(all(item.accepted for item in outcomes))
        self.assertEqual(5.0, simulator.node_values["a"]["signal"])
        self.assertEqual([], simulator.safety_violations)

    def test_same_simulator_encodes_four_domain_metaphors(self) -> None:
        cases = (
            ("prospective_reasoning", "Heater"),
            ("planning", "Pump"),
            ("interference", "UAV"),
            ("conditional_safety", "Battery"),
        )
        for family, label in cases:
            with self.subTest(family=family):
                scenario = generate_graph_v2_scenario(
                    family, "graph_v2_entry_1", 42, {"instance_index": 1}
                )
                simulator = GraphCPSSimulator(scenario)
                self.assertIsInstance(simulator, GraphCPSSimulator)
                self.assertIn(label, json_labels(scenario))

    def test_graph_metrics_are_grounded_in_the_declared_reference(self) -> None:
        scenario = generate_graph_v2_scenario(
            "planning",
            "graph_v2_entry_1",
            42,
            {"instance_index": 1},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = BenchmarkRunner(
                scenario,
                ScriptedPolicy("graph_metric_reference"),
                Path(directory),
            ).run()
        metrics = result.metrics["task_specific"]
        self.assertTrue(metrics["graph_dependent_decision_correct"])
        self.assertEqual(1.0, metrics["graph_dependent_action_accuracy"])
        self.assertIn(
            metrics["correct_neighbor_selection"],
            {None, True},
        )


def json_labels(scenario: ScenarioConfig) -> str:
    return " ".join(str(node.get("label", "")) for node in scenario.graph["nodes"].values())


if __name__ == "__main__":
    unittest.main()
