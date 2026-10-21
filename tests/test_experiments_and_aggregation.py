from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cps_reason_bench.aggregation import (
    aggregate_episode_records,
    write_experiment_exports,
)
from cps_reason_bench.experiments import (
    expand_experiment,
    load_experiment,
    replay_run_summary,
    run_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


def aggregate_record(run_id: str, success: bool, actions: int) -> dict[str, object]:
    return {
        "run_id": run_id,
        "model_configuration": "test-model",
        "metrics": {
            "general": {"task_success": success, "accepted_actions": actions},
            "protocol_compliance": {"first_attempt_json_valid_rate": 1.0},
            "task_specific": {},
        },
        "run_metadata": {},
        "policy": {},
    }


class ExperimentAndAggregationTests(unittest.TestCase):
    def test_matrix_expansion_is_stable_and_complete(self) -> None:
        config = load_experiment(
            ROOT / "examples" / "experiments" / "baseline_comparison.yaml"
        )
        first = expand_experiment(config)
        second = expand_experiment(config)
        self.assertEqual(16, len(first))
        self.assertEqual(
            [item.episode_id for item in first], [item.episode_id for item in second]
        )
        self.assertEqual({0, 1}, {item.repetition_index for item in first})

    def test_aggregation_and_csv_values_are_correct(self) -> None:
        records = [aggregate_record("a", True, 1), aggregate_record("b", False, 3)]
        groups = aggregate_episode_records(records, ("model_configuration",))
        self.assertEqual(1, len(groups))
        group = groups[0]
        self.assertEqual(2, group["run_count"])
        self.assertEqual(0.5, group["success_rate"])
        stats = group["metrics"]["general.accepted_actions"]
        self.assertEqual(2.0, stats["mean"])
        self.assertEqual(2.0, stats["median"])
        self.assertEqual(1.0, stats["stddev"])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_experiment_exports(
                records, directory, ("model_configuration",)
            )
            with paths["aggregates_csv"].open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row["metric"] == "general.accepted_actions" for row in rows))
            self.assertEqual(2, len(paths["episodes_jsonl"].read_text().splitlines()))

    def test_small_experiment_exports_and_replays_episode(self) -> None:
        source = load_experiment(
            ROOT / "examples" / "experiments" / "baseline_comparison.yaml"
        )
        scripted = (ROOT / "examples" / "models" / "scripted_reference.yaml",)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                source,
                model_paths=scripted,
                task_families=("planning",),
                repetitions=1,
                output_dir=Path(directory) / "experiment",
            )
            result = run_experiment(config)
            self.assertEqual(1, len(result.run_results))
            self.assertTrue(result.export_paths["episodes_csv"].is_file())
            self.assertTrue(result.export_paths["aggregates_json"].is_file())
            generated = list((config.output_dir / "generated_scenarios").glob("*.yaml"))
            self.assertEqual(1, len(generated))
            original = result.run_results[0]
            replay = replay_run_summary(
                original.summary_path, Path(directory) / "replay"
            )
            self.assertEqual(
                original.metrics["general"]["final_system_state"],
                replay.metrics["general"]["final_system_state"],
            )
            summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(original.run_id, summary["run_metadata"]["parent_run_id"])


if __name__ == "__main__":
    unittest.main()

