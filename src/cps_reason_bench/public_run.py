"""Release orchestration; physical execution and scoring stay in the existing runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from cps_reason_bench.aggregation import DEFAULT_GROUP_BY
from cps_reason_bench.config import ConfigError, dump_scenario
from cps_reason_bench.experiments import (
    ExperimentConfig, ExperimentResult, ExperimentRunSpec, expand_experiment,
    load_episode_record, prepare_experiment, run_experiment_episode, write_experiment_outputs,
)
from cps_reason_bench.provenance import collect_runtime_provenance
from cps_reason_bench.public_config import RunConfig, list_conditions, require_credentials
from cps_reason_bench.run_logging import atomic_write_json, atomic_write_text
from cps_reason_bench.validation import validate_scenario_semantics
from cps_reason_bench.versions import PACKAGE_VERSION, content_hash


def _experiment(config: RunConfig) -> ExperimentConfig:
    return ExperimentConfig(1, config.name, (), (), (), (), config.simulator_seeds,
        config.repetitions, config.output_dir, DEFAULT_GROUP_BY, False,
        config.source_path, config.to_dict())


def plan_experiment(config: RunConfig) -> tuple[ExperimentRunSpec, ...]:
    """Use the historical expansion algorithm, including model repetition seeds."""
    return expand_experiment(_experiment(config),
        suite_instances=((config.suite, config.conditions),), model_configs=config.models)


def validate_config(config: RunConfig, *, check_credentials: bool = True) -> None:
    """Validate references and scenario semantics offline, before any provider request."""
    if check_credentials:
        require_credentials(config)
    for item in config.conditions:
        report = validate_scenario_semantics(item.scenario)
        if not report.valid:
            raise ConfigError(f"Invalid condition {item.suite_entry_id}: {report.errors}")


def run_experiment(
    config: RunConfig, *, authorize_cloud: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Run once into a fresh directory. Failed or interrupted runs are never overwritten.

    Cloud authorization is explicit even for Python callers. Infrastructure failures
    stop the matrix; retrying a charged/ambiguous attempt is never implicit.
    """
    if config.requires_confirmation and not authorize_cloud:
        raise ConfigError("Cloud execution requires authorize_cloud=True (CLI: confirm or --yes)")
    validate_config(config)
    specs = plan_experiment(config)
    experiment = _experiment(config)
    if config.output_dir.exists():
        raise ConfigError("output_dir already exists; choose a new directory to preserve previous evidence")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    provenance = {"runtime": collect_runtime_provenance(), "ollama": {
        "collection_status": "not_probed", "reason": "Release orchestration makes no discovery calls"},
        "collected_at_utc": now()}
    provenance["provenance_id"] = content_hash(provenance)
    workspace = prepare_experiment(experiment, specs, provenance=provenance)
    atomic_write_text(config.output_dir / "config.resolved.yaml", yaml.safe_dump(config.to_dict(), sort_keys=False))
    snapshots = config.output_dir / "condition_snapshots"
    snapshots.mkdir()
    for item in config.conditions:
        dump_scenario(item.scenario, snapshots / f"{content_hash(item.scenario.to_mapping())}.yaml")
    metadata = {"package_version": PACKAGE_VERSION, "release_config_version": 1,
        "started_at_utc": now(), "status": "running", "planned_episodes": len(specs),
        "suite": config.suite.identity, "conditions": list_conditions(config),
        "models": [model.identity for model in config.models], "completed_episodes": 0,
        "infrastructure_failures": 0, "config_hash": content_hash(config.to_dict())}
    atomic_write_json(config.output_dir / "metadata.json", metadata)
    results, records, failures = [], [], []
    exports: dict[str, Path] = {}
    try:
        for index, spec in enumerate(specs, 1):
            if progress:
                progress(f"[{index}/{len(specs)}] model={spec.model_config.name} condition={spec.suite_instance.suite_entry_id} repetition={spec.repetition_index + 1}")
            result = run_experiment_episode(workspace, spec, model_call_delay_seconds=config.model_call_delay_seconds)
            if result.completion_status == "infrastructure_failed":
                summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
                failures.append({"episode_id": spec.episode_id, "summary_path": str(result.summary_path),
                    "error_classification": summary.get("infrastructure_failure"),
                    "retry_policy": "Stopped; inspect retained evidence before authorizing another attempt"})
                metadata["status"] = "infrastructure_failed"
                break
            results.append(result)
            records.append(load_episode_record(result.summary_path))
            if progress:
                progress(f"Completed {len(results)}/{len(specs)}; summary={result.summary_path.name}")
        else:
            metadata["status"] = "completed"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise
    except Exception:
        metadata["status"] = "error"
        raise
    finally:
        # Preserve partial evidence even on interruption; unexpected errors still propagate.
        metadata.update(finished_at_utc=now(), completed_episodes=len(results), infrastructure_failures=len(failures))
        atomic_write_json(config.output_dir / "metadata.json", metadata)
        exports = write_experiment_outputs(experiment, records, failures)
    return ExperimentResult(config.name, config.output_dir, tuple(results), exports,
        workspace.manifest_path, tuple(failures))


def load_results(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read completed episode records from the native aggregate JSONL export."""
    path = Path(output_dir) / "episodes.jsonl"
    if not path.is_file():
        raise ConfigError("No episodes.jsonl found; use a completed or partially exported run directory")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
