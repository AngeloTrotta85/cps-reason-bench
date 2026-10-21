from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cps_reason_bench.pilot_analysis import (
    _context_ablation,
    build_pilot_summary,
    classify_episode,
    empirical_level_summary,
    write_pilot_reports,
)
from cps_reason_bench.provenance import (
    CommandResult,
    collect_experiment_provenance,
)
from cps_reason_bench.suites import load_suite


ROOT = Path(__file__).resolve().parents[1]


def _runner(args: tuple[str, ...]) -> CommandResult:
    outputs = {
        ("git", "rev-parse", "--is-inside-work-tree"): CommandResult(0, "true\n"),
        ("git", "rev-parse", "HEAD"): CommandResult(0, "abc123\n"),
        ("git", "status", "--porcelain"): CommandResult(0, " M file.py\n"),
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ): CommandResult(0, "Test GPU, 4096, 3072\n"),
        ("ollama", "--version"): CommandResult(0, "ollama version is 9.9.9\n"),
    }
    return outputs.get(args, CommandResult(127, "", "unavailable"))


def _http(method: str, path: str, payload: dict | None) -> dict:
    if path == "/api/version":
        return {"version": "0.31.1"}
    if path == "/api/tags":
        return {
            "models": [
                {
                    "name": "qwen3:test",
                    "digest": "sha256:full-digest",
                    "size": 1234,
                    "details": {
                        "format": "gguf",
                        "family": "qwen3",
                        "families": ["qwen3"],
                        "parameter_size": "2.0B",
                        "quantization_level": "Q4_K_M",
                    },
                }
            ]
        }
    if path == "/api/show" and payload == {"model": "qwen3:test"}:
        return {
            "details": {
                "format": "gguf",
                "family": "qwen3",
                "families": ["qwen3"],
                "parameter_size": "2.0B",
                "quantization_level": "Q4_K_M",
            },
            "capabilities": ["completion", "thinking"],
            "model_info": {
                "qwen3.context_length": 40960,
                "qwen3.embedding_length": 2048,
            },
            "template": "intentionally not retained",
            "license": "intentionally not retained",
        }
    raise AssertionError((method, path, payload))


def _record(
    run_id: str,
    *,
    model: str = "qwen3:test",
    family: str = "planning",
    level: str = "basic",
    success: bool = False,
    task: dict | None = None,
    context_mode: str = "history",
    suite_name: str = "cps_reason_core_v2",
) -> dict:
    return {
        "run_id": run_id,
        "episode_id": f"episode-{run_id}",
        "experiment_name": "pilot-test",
        "suite_name": suite_name,
        "suite_version": "2.0.0",
        "suite_entry_id": f"{family}-{level}",
        "model_configuration": "qwen3_test_local",
        "backend": "ollama",
        "model": model,
        "task_family": family,
        "difficulty_level": level,
        "scenario": f"{family}_{level}",
        "simulator_seed": 10,
        "repetition_index": 0,
        "interaction_mode": "periodic",
        "context_mode": context_mode,
        "summary_path": "summary.json",
        "events_path": None,
        "policy": {
            "provider": "ollama",
            "model": model,
            "decoding": {"temperature": 0.0, "seed": 7},
        },
        "run_metadata": {
            "suite_name": suite_name,
            "suite_version": "2.0.0",
            "benchmark_versions": {
                "package_version": "0.3.0",
                "prompt_template_id": "cps-reason-json-control",
                "prompt_template_version": "2.0.0",
                "prompt_template_hash": "prompt-hash",
                "protocol_schema_id": "cps-reason-decision",
                "protocol_schema_version": "1.0.0",
            },
        },
        "metrics": {
            "scenario": f"{family}_{level}",
            "task_family": family,
            "difficulty_level": level,
            "general": {
                "task_success": success,
                "backend_errors": 0,
                "safety_violations": 0,
                "llm_or_policy_invocations": 2,
                "model_calls_including_retries": 2,
                "accepted_actions": int(success),
                "invalid_actions": 0,
                "unnecessary_actions": 0,
                "interaction_cost": 2,
                "token_usage": {"total_tokens": 20},
                "total_latency_seconds": 1.5,
            },
            "protocol_compliance": {
                "first_attempt_count": 2,
                "first_attempt_json_valid": 2,
                "first_attempt_schema_valid": 2,
                "first_attempt_decision_valid": 2,
                "final_executable_decisions": 2,
            },
            "task_specific": task or {},
        },
    }


class ProvenanceTests(unittest.TestCase):
    def test_collects_available_runtime_and_mocked_ollama_metadata(self) -> None:
        provenance = collect_experiment_provenance(
            ["qwen3:test"], command_runner=_runner, http_client=_http
        )
        runtime = provenance["runtime"]
        model = provenance["ollama"]["selected_models"][0]
        self.assertEqual("abc123", runtime["source_revision"]["git_commit"])
        self.assertTrue(runtime["source_revision"]["working_tree_dirty"])
        self.assertEqual(
            4096 * 1024 * 1024, runtime["gpu_models"][0]["memory_total_bytes"]
        )
        self.assertEqual("sha256:full-digest", model["digest"])
        self.assertEqual("Q4_K_M", model["quantization_level"])
        self.assertEqual(40960, model["context_length"])
        self.assertNotIn("template", model)
        self.assertNotIn("license", model)

    def test_unavailable_commands_and_service_are_nonfatal_and_privacy_bounded(
        self,
    ) -> None:
        def unavailable(_: tuple[str, ...]) -> CommandResult:
            return CommandResult(127, "", "missing")

        def offline(*_: object) -> dict:
            raise OSError("offline")

        provenance = collect_experiment_provenance(
            ["missing:model"], command_runner=unavailable, http_client=offline
        )
        self.assertFalse(provenance["ollama"]["running"])
        self.assertIsNone(provenance["ollama"]["version"])
        self.assertIsNone(provenance["runtime"]["source_revision"]["git_commit"])
        serialized = json.dumps(provenance).lower()
        for forbidden in (
            "hostname",
            "home_directory",
            "environment_variables",
            "username",
        ):
            self.assertNotIn(forbidden, serialized)


class PilotAnalysisTests(unittest.TestCase):
    def test_failure_taxonomy_uses_protocol_and_behavioral_evidence(self) -> None:
        record = _record(
            "r1",
            task={"prerequisite_violations": 2, "expected_sequence_satisfied": False},
        )
        trace = [
            {
                "record_type": "model_interaction",
                "interaction_index": 0,
                "step": 0,
                "visible_observation": {"sensors": {}},
                "raw_response": "not-json",
                "validation_errors": ["invalid JSON"],
                "output_attempts": [
                    {
                        "raw_response": "not-json",
                        "json_valid": False,
                        "schema_valid": False,
                        "decision_valid": False,
                        "validation_errors": ["invalid JSON"],
                    }
                ],
                "public_action_outcomes": [],
            }
        ]
        result = classify_episode(record, trace)
        categories = {item["category"] for item in result["categories"]}
        self.assertIn("invalid_json", categories)
        self.assertIn("unmet_prerequisite", categories)
        self.assertIn("incorrect_action_order", categories)
        self.assertNotIn("hidden_reasoning_failure", categories)

    def test_empirical_level_summary_preserves_counts_and_nonmonotonic_values(
        self,
    ) -> None:
        records = [
            _record("a", level="basic", success=True),
            _record("b", level="basic", success=False),
            _record("c", level="controlled_2", success=False),
        ]
        groups = empirical_level_summary(records)
        self.assertEqual(2, len(groups))
        basic = next(
            item for item in groups if item["dimensions"]["difficulty_level"] == "basic"
        )
        self.assertEqual(2, basic["run_count"])
        self.assertEqual(1, basic["success_count"])
        self.assertEqual(0.5, basic["success_rate"])
        self.assertEqual(1.0, basic["first_attempt_decision_valid_rate"])

    def test_context_ablation_requires_dedicated_paired_suites(self) -> None:
        records = [
            _record("core", family="temporal_tracking", context_mode="history"),
            _record(
                "current",
                family="temporal_tracking",
                context_mode="current",
                suite_name="cps_reason_temporal_current_control",
            ),
            _record(
                "history",
                family="temporal_tracking",
                context_mode="history",
                suite_name="cps_reason_temporal_history",
            ),
        ]

        groups = _context_ablation(records)

        self.assertEqual(2, len(groups))
        self.assertEqual(
            {
                "current_only_negative_control",
                "explicit_history",
            },
            {item["comparison_role"] for item in groups},
        )
        self.assertNotIn(
            "cps_reason_core_v2",
            {item["dimensions"]["suite_name"] for item in groups},
        )

    def test_pilot_summary_and_markdown_reconstruct_from_episode_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            experiment.mkdir()
            provenance = collect_experiment_provenance(
                ["qwen3:test"], command_runner=_runner, http_client=_http
            )
            (experiment / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiment_name": "pilot-test",
                        "experiment_content_hash": "experiment-hash",
                        "run_count": 2,
                        "provenance": provenance,
                    }
                ),
                encoding="utf-8",
            )
            records = [
                _record("success", success=True),
                _record("failure", success=False),
            ]
            (experiment / "episodes.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            (experiment / "failures.json").write_text("[]\n", encoding="utf-8")
            summary = build_pilot_summary([experiment])
            self.assertEqual(2, summary["episode_count"])
            self.assertEqual(1, summary["results_by_model"][0]["success_count"])
            self.assertEqual(
                1, summary["results_by_model"][0]["model_task_failure_count"]
            )
            self.assertTrue(summary["artifact_integrity"]["stored_outputs_consistent"])
            self.assertEqual(1, len(summary["representative_episodes"]))
            self.assertEqual([], summary["temporal_context_ablation"])
            self.assertEqual(
                ["success", "failure"], summary["traceability"]["all_run_ids"]
            )
            paths = write_pilot_reports(
                [experiment], root / "pilot.json", root / "pilot.md"
            )
            self.assertTrue(paths["pilot_summary_json"].is_file())
            markdown = paths["pilot_report_markdown"].read_text(encoding="utf-8")
            self.assertIn("## 32. Observable Failure Taxonomy", markdown)
            self.assertIn("## 34. Benchmark-Calibration Concerns", markdown)
            self.assertIn(
                "no context-ablation result is reported",
                markdown,
            )

    def test_pilot_summary_merges_model_metadata_from_multiple_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_dirs = []
            for index, model in enumerate(("qwen3:a", "qwen3:b"), start=1):
                experiment = root / f"experiment-{index}"
                experiment.mkdir()
                provenance = {
                    "provenance_id": f"provenance-{index}",
                    "runtime": {},
                    "ollama": {
                        "running": True,
                        "version": "test",
                        "errors": [],
                        "installed_models": [],
                        "selected_models": [
                            {
                                "identifier": model,
                                "available": True,
                                "digest": f"digest-{index}",
                                "quantization_level": "Q4_K_M",
                            }
                        ],
                    },
                }
                (experiment / "manifest.json").write_text(
                    json.dumps(
                        {
                            "experiment_name": f"pilot-{index}",
                            "experiment_content_hash": f"hash-{index}",
                            "run_count": 1,
                            "provenance": provenance,
                        }
                    ),
                    encoding="utf-8",
                )
                (experiment / "episodes.jsonl").write_text(
                    json.dumps(_record(f"run-{index}", model=model)) + "\n",
                    encoding="utf-8",
                )
                (experiment / "failures.json").write_text("[]\n", encoding="utf-8")
                experiment_dirs.append(experiment)

            summary = build_pilot_summary(experiment_dirs)
            models = {item["model"]: item for item in summary["models"]}
            self.assertEqual("digest-1", models["qwen3:a"]["digest"])
            self.assertEqual("digest-2", models["qwen3:b"]["digest"])
            self.assertEqual(
                ["provenance-1", "provenance-2"],
                summary["provenance"]["all_provenance_ids"],
            )

    def test_frozen_core_v2_identity_is_unchanged(self) -> None:
        suite = load_suite(ROOT / "examples" / "suites" / "core_v2.yaml")
        self.assertEqual(
            "608d1e5e810da8b811bd9240c597910c85847d948754b397a599da00e4daa5c2",
            suite.identity["content_hash"],
        )


if __name__ == "__main__":
    unittest.main()
