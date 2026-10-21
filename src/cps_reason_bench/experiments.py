"""Sequential experiment matrices over suites, models, seeds, and repetitions."""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from cps_reason_bench.aggregation import (
    DEFAULT_GROUP_BY,
    episode_record,
    write_experiment_exports,
)
from cps_reason_bench.config import (
    ConfigError,
    ScenarioConfig,
    dump_scenario,
    load_document,
)
from cps_reason_bench.family_reports import write_family_reports
from cps_reason_bench.models.config import (
    ModelConfig,
    build_policy,
    load_model_config,
    resolve_model_seed,
)
from cps_reason_bench.paths import examples_root
from cps_reason_bench.provenance import (
    collect_experiment_provenance,
    compact_episode_provenance,
)
from cps_reason_bench.run_logging import atomic_write_json
from cps_reason_bench.runner import BenchmarkRunner, RunResult
from cps_reason_bench.suites import (
    BenchmarkSuiteConfig,
    SuiteScenarioInstance,
    load_suite,
    materialize_suite,
)
from cps_reason_bench.validation import validate_scenario_semantics
from cps_reason_bench.versions import EXPERIMENT_SCHEMA_VERSION, content_hash


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    suite_paths: tuple[Path, ...]
    model_paths: tuple[Path, ...]
    task_families: tuple[str, ...]
    difficulty_levels: tuple[str, ...]
    simulator_seeds: tuple[int, ...]
    repetitions: int
    output_dir: Path
    group_by: tuple[str, ...]
    continue_on_error: bool
    source_path: Path
    raw: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class ExperimentRunSpec:
    episode_id: str
    suite: BenchmarkSuiteConfig
    suite_instance: SuiteScenarioInstance
    model_config: ModelConfig
    simulator_seed: int
    random_baseline_seed: int
    requested_model_seed: int | None
    repetition_index: int


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    output_dir: Path
    run_results: tuple[RunResult, ...]
    export_paths: dict[str, Path]
    manifest_path: Path
    failed_runs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExperimentWorkspace:
    config: ExperimentConfig
    specs: tuple[ExperimentRunSpec, ...]
    episode_dir: Path
    generated_dir: Path
    manifest_path: Path
    provenance: dict[str, Any]


def load_experiment(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    data = load_document(source)
    version = int(data.get("experiment_schema_version", EXPERIMENT_SCHEMA_VERSION))
    if version != EXPERIMENT_SCHEMA_VERSION:
        raise ConfigError(f"Unsupported experiment_schema_version: {version}")
    name = _required(data.get("name"), "experiment name")
    suites = _resolve_path_list(source, data.get("suites"), "suites")
    models = _resolve_path_list(source, data.get("models"), "models")
    filters = dict(data.get("filters", {}))
    repetitions = int(data.get("repetitions", 1))
    if repetitions <= 0:
        raise ConfigError("repetitions must be positive")
    output_raw = Path(str(data.get("output_dir", f"../../runs/{name}")))
    output_dir = (
        output_raw
        if output_raw.is_absolute()
        else (source.parent / output_raw).resolve()
    )
    aggregation = dict(data.get("aggregation", {}))
    return ExperimentConfig(
        schema_version=version,
        name=name,
        suite_paths=suites,
        model_paths=models,
        task_families=tuple(str(item) for item in filters.get("task_families", [])),
        difficulty_levels=tuple(
            str(item) for item in filters.get("difficulty_levels", [])
        ),
        simulator_seeds=tuple(int(item) for item in data.get("simulator_seeds", [])),
        repetitions=repetitions,
        output_dir=output_dir,
        group_by=tuple(aggregation.get("group_by", DEFAULT_GROUP_BY)),
        continue_on_error=bool(data.get("continue_on_error", False)),
        source_path=source,
        raw=data,
    )


def expand_experiment(
    config: ExperimentConfig,
    *,
    suite_instances: tuple[tuple[BenchmarkSuiteConfig, tuple[SuiteScenarioInstance, ...]], ...] | None = None,
    model_configs: tuple[ModelConfig, ...] | None = None,
) -> tuple[ExperimentRunSpec, ...]:
    """Expand file-based or already materialized inputs using identical seed rules."""
    prepared = suite_instances if suite_instances is not None else tuple(
        (suite, materialize_suite(suite))
        for suite in (load_suite(path) for path in config.suite_paths)
    )
    models = model_configs if model_configs is not None else tuple(
        load_model_config(path) for path in config.model_paths
    )
    specs: list[ExperimentRunSpec] = []
    for suite, instances in prepared:
        for instance in instances:
            scenario = instance.scenario
            if (
                config.task_families
                and scenario.task_family not in config.task_families
            ):
                continue
            if (
                config.difficulty_levels
                and scenario.difficulty_level not in config.difficulty_levels
            ):
                continue
            seeds = config.simulator_seeds or instance.simulator_seeds
            for model_config in models:
                for simulator_seed in seeds:
                    for repetition in range(config.repetitions):
                        campaign_identity_hash = _campaign_identity_hash(config.raw)
                        identity = {
                            "experiment": config.name,
                            "suite": suite.identity,
                            "entry": instance.suite_entry_id,
                            "scenario_hash": content_hash(scenario.to_mapping()),
                            "model": model_config.identity,
                            "simulator_seed": simulator_seed,
                            "repetition_index": repetition,
                        }
                        if campaign_identity_hash is not None:
                            identity["campaign_identity_hash"] = campaign_identity_hash
                        digest = content_hash(identity)
                        random_seed = int(digest[:8], 16)
                        model_seed_digest = digest
                        if (
                            suite.evaluation.get("model_repetition_seed_policy")
                            == "fixed_per_model_condition"
                        ):
                            model_seed_digest = content_hash(
                                {
                                    key: value
                                    for key, value in identity.items()
                                    if key != "repetition_index"
                                }
                            )
                        model_seed = resolve_model_seed(
                            model_config, int(model_seed_digest[8:16], 16)
                        )
                        specs.append(
                            ExperimentRunSpec(
                                episode_id=digest[:16],
                                suite=suite,
                                suite_instance=instance,
                                model_config=model_config,
                                simulator_seed=simulator_seed,
                                random_baseline_seed=random_seed,
                                requested_model_seed=model_seed,
                                repetition_index=repetition,
                            )
                        )
    if not specs:
        raise ConfigError("Experiment filters produced an empty run matrix")
    return tuple(specs)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    specs = expand_experiment(config)
    workspace = prepare_experiment(config, specs)
    run_results: list[RunResult] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        try:
            result = run_experiment_episode(workspace, spec)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            if summary.get("completion_status") == "infrastructure_failed":
                failures.append(
                    {
                        "episode_id": spec.episode_id,
                        "exception_type": "InfrastructureEpisodeFailure",
                        "message": str(
                            summary.get("infrastructure_failure", {}).get(
                                "message", "episode terminated as infrastructure failure"
                            )
                        ),
                        "error_classification": summary.get(
                            "infrastructure_failure"
                        ),
                        "summary_path": str(result.summary_path),
                    }
                )
                continue
            run_results.append(result)
            records.append(episode_record(summary, result.summary_path))
        except Exception as exc:
            failure = experiment_failure(spec, exc)
            failures.append(failure)
            if not config.continue_on_error:
                raise

    export_paths = write_experiment_outputs(config, records, failures)
    return ExperimentResult(
        config.name,
        config.output_dir,
        tuple(run_results),
        export_paths,
        workspace.manifest_path,
        tuple(failures),
    )


def prepare_experiment(
    config: ExperimentConfig,
    specs: tuple[ExperimentRunSpec, ...] | None = None,
    *,
    preserve_existing_manifest: bool = False,
    provenance: dict[str, Any] | None = None,
) -> ExperimentWorkspace:
    selected_specs = specs or expand_experiment(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = config.output_dir / "episodes"
    generated_dir = config.output_dir / "generated_scenarios"
    episode_dir.mkdir(exist_ok=True)
    generated_dir.mkdir(exist_ok=True)
    manifest_path = config.output_dir / "manifest.json"
    ollama_models = tuple(
        spec.model_config.model
        for spec in selected_specs
        if spec.model_config.backend == "ollama"
    )
    existing_manifest: dict[str, Any] | None = None
    if preserve_existing_manifest and manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_manifest = loaded
        except (OSError, json.JSONDecodeError):
            existing_manifest = None
    provenance = provenance if provenance is not None else (
        dict(existing_manifest.get("provenance", {}))
        if existing_manifest and existing_manifest.get("provenance")
        else collect_experiment_provenance(ollama_models)
    )
    manifest = {
        "experiment": config.raw,
        "experiment_name": config.name,
        "experiment_content_hash": content_hash(config.raw),
        "run_count": len(selected_specs),
        "episodes": [_spec_manifest(spec) for spec in selected_specs],
        "provenance": provenance,
    }
    if existing_manifest:
        _validate_existing_manifest(existing_manifest, manifest)
    else:
        atomic_write_json(manifest_path, manifest)
    return ExperimentWorkspace(
        config,
        selected_specs,
        episode_dir,
        generated_dir,
        manifest_path,
        provenance,
    )


def run_experiment_episode(
    workspace: ExperimentWorkspace,
    spec: ExperimentRunSpec,
    *,
    model_call_delay_seconds: float = 0.0,
    model_timeout_seconds: float | None = None,
    local_workers: int = 1,
) -> RunResult:
    if spec.episode_id not in {item.episode_id for item in workspace.specs}:
        raise ConfigError(
            f"Episode {spec.episode_id} is not in the prepared experiment"
        )
    config = workspace.config
    scenario = spec.suite_instance.scenario
    validation = validate_scenario_semantics(scenario)
    if not validation.valid:
        raise ConfigError(
            f"Suite scenario {spec.suite_instance.suite_entry_id!r} is invalid: "
            + "; ".join(validation.errors)
        )
    if spec.suite_instance.source_kind == "generator":
        scenario_filename = (
            f"{spec.suite.name}_{spec.suite_instance.suite_entry_id}.yaml"
        )
        destination = workspace.generated_dir / scenario_filename
        if not destination.is_file():
            dump_scenario(scenario, destination)
    effective_model_config = spec.model_config
    if model_timeout_seconds is not None:
        if model_timeout_seconds <= 0:
            raise ConfigError("Launcher model_timeout_seconds must be positive")
        effective_model_config = replace(
            spec.model_config, timeout_seconds=float(model_timeout_seconds)
        )
    policy = build_policy(
        effective_model_config,
        random_seed=spec.random_baseline_seed,
        requested_model_seed=spec.requested_model_seed,
        retry_policy=spec.suite.protocol.retry_policy,
        prompt_template_id=spec.suite.protocol.prompt_template_id,
        prompt_template_version=spec.suite.protocol.prompt_template_version,
    )
    analysis = _analysis_identifiers(spec)
    run_metadata = {
        "episode_id": spec.episode_id,
        "experiment_name": config.name,
        "experiment_content_hash": content_hash(config.raw),
        "suite_name": spec.suite.name,
        "suite_version": spec.suite.version,
        "suite_content_hash": spec.suite.identity["content_hash"],
        "suite_entry_id": spec.suite_instance.suite_entry_id,
        "repetition_index": spec.repetition_index,
        "random_baseline_seed": spec.random_baseline_seed,
        "requested_model_seed": spec.requested_model_seed,
        "model_configuration": effective_model_config.safe_dict(),
        "retry_policy": spec.suite.protocol.retry_policy.to_dict(),
        "output_validation_policy": spec.suite.protocol.output_validation_policy,
        "prompt_template_id": spec.suite.protocol.prompt_template_id,
        "prompt_template_version": spec.suite.protocol.prompt_template_version,
        "protocol_schema_id": spec.suite.protocol.protocol_schema_id,
        "protocol_schema_version": spec.suite.protocol.protocol_schema_version,
        "semantic_validation": validation.to_dict(),
        "analysis": analysis,
        "provenance": compact_episode_provenance(
            workspace.provenance,
            spec.model_config.model if spec.model_config.backend == "ollama" else None,
        ),
        "launcher_execution": {
            "model_call_delay_seconds": max(0.0, float(model_call_delay_seconds)),
            "model_timeout_seconds": model_timeout_seconds,
            "local_workers": local_workers,
        },
    }
    campaign_identity = config.raw.get("campaign_identity")
    if isinstance(campaign_identity, Mapping):
        run_metadata["campaign_identity"] = copy.deepcopy(dict(campaign_identity))
    return BenchmarkRunner(
        scenario,
        policy,
        workspace.episode_dir,
        seed=spec.simulator_seed,
        run_metadata=run_metadata,
        model_call_delay_seconds=model_call_delay_seconds,
    ).run()


def load_episode_record(summary_path: str | Path) -> dict[str, Any]:
    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    return episode_record(summary, source)


def write_experiment_outputs(
    config: ExperimentConfig,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Path]:
    from cps_reason_bench.cloud_reporting import (
        enrich_episode_records,
        write_cloud_reports,
    )

    enriched_records = enrich_episode_records(records)
    export_paths = write_experiment_exports(
        enriched_records, config.output_dir, config.group_by
    )
    export_paths.update(write_family_reports(enriched_records, config.output_dir))
    export_paths.update(write_cloud_reports(enriched_records, config.output_dir))
    failure_path = config.output_dir / "failures.json"
    atomic_write_json(failure_path, failures)
    export_paths["failures_json"] = failure_path
    return export_paths


def experiment_failure(spec: ExperimentRunSpec, exc: BaseException) -> dict[str, Any]:
    failure = {
        "episode_id": spec.episode_id,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    if hasattr(exc, "to_dict"):
        failure["error_classification"] = exc.to_dict()
    return failure


def _validate_existing_manifest(
    existing: dict[str, Any], expected: dict[str, Any]
) -> None:
    if existing.get("experiment_content_hash") != expected["experiment_content_hash"]:
        raise ConfigError(
            "Existing manifest belongs to a different experiment configuration; "
            "choose a new output directory"
        )
    existing_ids = {
        item.get("episode_id")
        for item in existing.get("episodes", [])
        if isinstance(item, dict)
    }
    expected_ids = {item["episode_id"] for item in expected["episodes"]}
    if existing_ids != expected_ids:
        raise ConfigError(
            "Existing manifest episode identities differ from the selected matrix"
        )


def replay_run_summary(summary_path: str | Path, output_dir: str | Path) -> RunResult:
    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    reproduction = summary.get("reproduction", {})
    scenario_data = reproduction.get("scenario_configuration")
    model_data = reproduction.get("model_configuration")
    if not isinstance(scenario_data, Mapping) or not isinstance(model_data, Mapping):
        raise ConfigError("Summary does not contain complete replay configuration")
    scenario = ScenarioConfig.from_mapping(scenario_data)
    model_config = ModelConfig.from_mapping(model_data)
    metadata = summary.get("run_metadata", {})
    seeds = metadata.get("seeds", {})
    retry = metadata.get("retry_policy")
    from cps_reason_bench.suites import RetryPolicyConfig

    policy = build_policy(
        model_config,
        random_seed=int(seeds.get("random_baseline_seed") or 0),
        requested_model_seed=seeds.get("requested_model_seed"),
        retry_policy=RetryPolicyConfig.from_mapping(retry),
        prompt_template_id=metadata.get("benchmark_versions", {}).get(
            "prompt_template_id", "cps-reason-json-control"
        ),
        prompt_template_version=metadata.get("benchmark_versions", {}).get(
            "prompt_template_version", "1.0.0"
        ),
    )
    return BenchmarkRunner(
        scenario,
        policy,
        output_dir,
        seed=int(reproduction["simulator_seed"]),
        reasoning_enabled=bool(reproduction.get("reasoning_enabled", False)),
        run_metadata={
            **metadata,
            "parent_run_id": summary["run_id"],
            "repetition_index": seeds.get("repetition_index", 0),
        },
    ).run()


def _spec_manifest(spec: ExperimentRunSpec) -> dict[str, Any]:
    return {
        "episode_id": spec.episode_id,
        "suite": spec.suite.identity,
        "suite_entry_id": spec.suite_instance.suite_entry_id,
        "scenario": spec.suite_instance.scenario.name,
        "task_family": spec.suite_instance.scenario.task_family,
        "difficulty_level": spec.suite_instance.scenario.difficulty_level,
        "model_configuration": spec.model_config.identity,
        "simulator_seed": spec.simulator_seed,
        "scenario_generation_seed": spec.suite_instance.generation_seed,
        "random_baseline_seed": spec.random_baseline_seed,
        "requested_model_seed": spec.requested_model_seed,
        "repetition_index": spec.repetition_index,
        "analysis": _analysis_identifiers(spec),
    }


def _campaign_identity_hash(raw: Mapping[str, Any]) -> str | None:
    value = raw.get("campaign_identity")
    if not isinstance(value, Mapping):
        return None
    identity_hash = value.get("campaign_identity_hash") or value.get("identity_hash")
    return str(identity_hash) if identity_hash else content_hash(dict(value))


def _analysis_identifiers(spec: ExperimentRunSpec) -> dict[str, str]:
    from cps_reason_bench.design import analysis_identifiers

    context_condition = (
        "current_only"
        if spec.suite.evaluation.get("context_negative_control")
        or "current_control" in spec.suite.name
        else (
            "history"
            if "temporal" in spec.suite.name
            and spec.suite_instance.scenario.context.mode == "history"
            else spec.suite_instance.scenario.context.mode
        )
    )
    return analysis_identifiers(
        spec.suite_instance.scenario,
        suite_name=spec.suite.name,
        suite_entry_id=spec.suite_instance.suite_entry_id,
        simulator_seed=spec.simulator_seed,
        context_condition=context_condition,
        model_configuration=spec.model_config.name,
        requested_model_seed=spec.requested_model_seed,
        repetition_index=spec.repetition_index,
    )


def _resolve_path_list(source: Path, value: Any, name: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Experiment {name} must be a non-empty list")
    return tuple(_resolve_experiment_path(source, item) for item in value)


def _resolve_experiment_path(source: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    primary = (source.parent / path).resolve()
    if primary.exists():
        return primary
    parts = path.parts
    if "examples" in parts:
        index = parts.index("examples")
        installed = (examples_root() / Path(*parts[index + 1 :])).resolve()
        if installed.exists():
            return installed
    return primary


def _required(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()
