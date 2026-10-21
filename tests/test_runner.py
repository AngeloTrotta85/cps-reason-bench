from __future__ import annotations

import json
import tempfile
import unittest

from cps_reason_bench.config import load_scenario
from cps_reason_bench.policies import ScriptedPolicy
from cps_reason_bench.runner import BenchmarkRunner
from tests.test_config_and_simulator import SCENARIOS


class RunnerTests(unittest.TestCase):
    def test_all_scripted_examples_complete_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            for path in sorted(SCENARIOS.glob("*.yaml")):
                with self.subTest(path=path.name):
                    result = BenchmarkRunner(
                        load_scenario(path), ScriptedPolicy(), output_dir
                    ).run()
                    general = result.metrics["general"]
                    self.assertTrue(general["task_success"])
                    self.assertEqual(0, general["invalid_structured_outputs"])
                    self.assertEqual(
                        len(
                            load_scenario(path).evaluation.get(
                                "reference_expected_rejections", []
                            )
                        ),
                        general["invalid_actions"],
                    )
                    self.assertTrue(result.events_path.is_file())
                    self.assertTrue(result.summary_path.is_file())
                    records = [
                        json.loads(line)
                        for line in result.events_path.read_text(encoding="utf-8").splitlines()
                    ]
                    self.assertEqual("run_started", records[0]["record_type"])
                    self.assertEqual("run_completed", records[-1]["record_type"])
                    self.assertTrue(
                        any(record["record_type"] == "model_interaction" for record in records)
                    )
                    interaction_steps = [
                        record["step"]
                        for record in records
                        if record["record_type"] == "model_interaction"
                    ]
                    self.assertTrue(interaction_steps)
                    self.assertLess(max(interaction_steps), load_scenario(path).max_steps)
                    interaction = next(
                        record
                        for record in records
                        if record["record_type"] == "model_interaction"
                    )
                    self.assertIn("true_state_at_invocation", interaction)
                    self.assertIn("true_state_after_actions", interaction)


if __name__ == "__main__":
    unittest.main()
