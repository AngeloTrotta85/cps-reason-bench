from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from cps_reason_bench.plotting import (
    FAMILY_ORDER,
    PlottingUnavailable,
    build_plot_data,
    wilson_interval,
    write_result_plots,
)


def _episode(
    run_id: str,
    *,
    suite: str,
    suite_version: str,
    family: str,
    level: str,
    success: bool,
    model: str = "mock-small",
) -> dict[str, Any]:
    task_specific = {
        "callback_timing_error_steps": 0,
        "missed_relevant_events": 0,
        "prerequisite_violations": 0,
        "excess_actions_vs_reference": 0,
        "misleading_action_rate": 0.0,
        "evidence_gathering_action_count": 1,
        "required_inspection_coverage": 1.0,
        "unnecessary_inspection_count": 0,
        "adaptation_success": 1.0,
        "adaptation_delay_steps": 1,
        "obsolete_actions_after_sufficient_evidence": 0,
        "delayed_effect_detection_accuracy": 1.0,
        "delayed_effect_attribution_accuracy": 1.0,
        "false_temporal_associations": 0,
    }
    return {
        "run_id": run_id,
        "suite_name": suite,
        "suite_version": suite_version,
        "model_configuration": model,
        "backend": "mock",
        "model": model,
        "task_family": family,
        "difficulty_level": level,
        "metrics": {
            "task_family": family,
            "difficulty_level": level,
            "general": {
                "task_success": success,
                "safety_violations": 0,
                "accepted_actions": 1,
                "invalid_actions": 0,
                "unnecessary_actions": 0,
                "llm_or_policy_invocations": 1,
                "token_usage": {"total_tokens": 24},
                "total_latency_seconds": 0.25,
                "useful_subscription_triggers": 1,
                "final_system_state": {"secret": "must not leak"},
            },
            "protocol_compliance": {
                "first_attempt_count": 1,
                "first_attempt_json_valid": 1,
                "first_attempt_schema_valid": 1,
                "first_attempt_decision_valid": 1,
                "final_executable_decisions": 1,
            },
            "task_specific": task_specific,
        },
        "scenario_configuration": {"hidden": "must not leak"},
    }


def _records() -> list[dict[str, Any]]:
    records = [
        _episode(
            "protocol-1",
            suite="cps_reason_protocol_sanity_v1",
            suite_version="1.0.0",
            family="protocol_sanity",
            level="sanity_1",
            success=True,
        )
    ]
    for index, family in enumerate(FAMILY_ORDER):
        records.append(
            _episode(
                f"foundation-{index}",
                suite="cps_reason_foundation_v1",
                suite_version="1.0.0",
                family=family,
                level="entry_1",
                success=index % 2 == 0,
            )
        )
    records.append(
        _episode(
            "core-1",
            suite="cps_reason_core_v2",
            suite_version="2.0.0",
            family="planning",
            level="basic",
            success=False,
        )
    )
    return records


class PlotAggregationTests(unittest.TestCase):
    def test_wilson_interval_and_protocol_aggregation(self) -> None:
        lower, upper = wilson_interval(1, 1)
        self.assertAlmostEqual(0.2065493144, lower)
        self.assertEqual(1.0, upper)

        data = build_plot_data(_records())
        self.assertEqual(8, data["record_count"])
        planning_groups = [
            group
            for group in data["success_by_model_family"]
            if group["family"] == "planning"
        ]
        self.assertEqual(2, len(planning_groups))
        self.assertEqual([1, 1], sorted(group["n"] for group in planning_groups))
        protocol = data["protocol_sanity_by_model"][0]["metrics"]
        self.assertEqual(1.0, protocol["json_validity"]["rate"])
        self.assertEqual(1, protocol["executable_decision"]["denominator"])
        serialized = json.dumps(data)
        self.assertNotIn("final_system_state", serialized)
        self.assertNotIn("scenario_configuration", serialized)
        self.assertNotIn("must not leak", serialized)

    def test_missing_matplotlib_is_a_clear_optional_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episodes.jsonl"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in _records()),
                encoding="utf-8",
            )

            def unavailable() -> tuple[Any, Any]:
                raise ModuleNotFoundError("matplotlib")

            with self.assertRaisesRegex(PlottingUnavailable, r"\[plots\]"):
                write_result_plots(
                    [source],
                    Path(directory) / "plots",
                    matplotlib_loader=unavailable,
                )


@unittest.skipUnless(
    importlib.util.find_spec("matplotlib") is not None,
    "optional Matplotlib dependency is unavailable",
)
class MatplotlibPlotTests(unittest.TestCase):
    def test_generates_png_svg_manifest_report_and_layer_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episodes.jsonl"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in _records()),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "provenance": {
                            "provenance_id": "test-provenance",
                            "runtime": {
                                "cpu_model": "test-cpu",
                                "operating_system": "test-os",
                                "final_system_state": "must not leak",
                            },
                            "ollama": {
                                "version": "test",
                                "selected_models": [
                                    {
                                        "identifier": "mock-small",
                                        "digest": "abc",
                                        "parameter_size": "small",
                                        "secret": "must not leak",
                                    }
                                ],
                            },
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = write_result_plots([source], root / "plots")
            manifest = json.loads(result["manifest"].read_text(encoding="utf-8"))

            plot_ids = {plot["plot_id"] for plot in manifest["plots"]}
            self.assertIn("success_by_model_family", plot_ids)
            self.assertIn("protocol_compliance", plot_ids)
            self.assertIn("suite_layer_comparison", plot_ids)
            self.assertEqual(
                {"1.0.0", "2.0.0"},
                {item.rsplit("@", 1)[1] for item in manifest["suite_versions"]},
            )
            for plot in manifest["plots"]:
                self.assertTrue(Path(plot["files"]["png"]).is_file())
                self.assertTrue(Path(plot["files"]["svg"]).is_file())
                self.assertTrue(plot["source_run_ids"])
            self.assertIsNotNone(result["foundation_report"])
            self.assertTrue(result["foundation_report"].is_file())
            source_metadata = manifest["source_episode_files"][0]
            self.assertEqual(
                "test-cpu", source_metadata["runtime_provenance"]["cpu_model"]
            )
            self.assertEqual(
                "mock-small",
                source_metadata["provider_provenance"]["selected_models"][0][
                    "identifier"
                ],
            )

            serialized = json.dumps(manifest)
            self.assertNotIn("final_system_state", serialized)
            self.assertNotIn("scenario_configuration", serialized)
            self.assertNotIn("must not leak", serialized)


if __name__ == "__main__":
    unittest.main()
