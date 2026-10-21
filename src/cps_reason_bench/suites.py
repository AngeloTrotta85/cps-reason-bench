"""Versioned benchmark-suite definitions and scenario materialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cps_reason_bench.config import (
    ConfigError,
    ScenarioConfig,
    load_document,
    load_scenario,
)
from cps_reason_bench.generators import generate_scenario
from cps_reason_bench.versions import (
    PROMPT_TEMPLATE_ID,
    PROMPT_TEMPLATE_VERSION,
    PROTOCOL_SCHEMA_ID,
    PROTOCOL_SCHEMA_VERSION,
    SUITE_SCHEMA_VERSION,
    content_hash,
    get_prompt_template,
)


@dataclass(frozen=True)
class RetryPolicyConfig:
    name: str = "strict_zero_correction"
    max_corrections: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RetryPolicyConfig":
        data = dict(value or {})
        maximum = int(data.get("max_corrections", 0))
        if maximum < 0:
            raise ConfigError("retry_policy.max_corrections cannot be negative")
        return cls(str(data.get("name", "strict_zero_correction")), maximum)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "max_corrections": self.max_corrections}


@dataclass(frozen=True)
class SuiteProtocolConfig:
    prompt_template_id: str = PROMPT_TEMPLATE_ID
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    protocol_schema_id: str = PROTOCOL_SCHEMA_ID
    protocol_schema_version: str = PROTOCOL_SCHEMA_VERSION
    output_validation_policy: str = "strict"
    retry_policy: RetryPolicyConfig = field(default_factory=RetryPolicyConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SuiteProtocolConfig":
        data = dict(value or {})
        prompt = dict(data.get("prompt", {}))
        schema = dict(data.get("schema", {}))
        validation = str(data.get("output_validation_policy", "strict"))
        if validation != "strict":
            raise ConfigError(
                "Only the documented strict output-validation policy is supported"
            )
        prompt_id = str(prompt.get("id", PROMPT_TEMPLATE_ID))
        prompt_version = str(prompt.get("version", PROMPT_TEMPLATE_VERSION))
        schema_id = str(schema.get("id", PROTOCOL_SCHEMA_ID))
        schema_version = str(schema.get("version", PROTOCOL_SCHEMA_VERSION))
        try:
            get_prompt_template(prompt_id, prompt_version)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if (schema_id, schema_version) != (PROTOCOL_SCHEMA_ID, PROTOCOL_SCHEMA_VERSION):
            raise ConfigError(
                f"Unsupported protocol schema {schema_id}@{schema_version}"
            )
        return cls(
            prompt_template_id=prompt_id,
            prompt_template_version=prompt_version,
            protocol_schema_id=schema_id,
            protocol_schema_version=schema_version,
            output_validation_policy=validation,
            retry_policy=RetryPolicyConfig.from_mapping(data.get("retry_policy")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": {
                "id": self.prompt_template_id,
                "version": self.prompt_template_version,
            },
            "schema": {
                "id": self.protocol_schema_id,
                "version": self.protocol_schema_version,
            },
            "output_validation_policy": self.output_validation_policy,
            "retry_policy": self.retry_policy.to_dict(),
        }


@dataclass(frozen=True)
class SuiteScenarioEntry:
    entry_id: str
    scenario_path: Path | None
    generator: dict[str, Any] | None
    simulator_seeds: tuple[int, ...]
    declared_generation_seed: int | None = None


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    schema_version: int
    name: str
    version: str
    description: str
    task_families: tuple[str, ...]
    entries: tuple[SuiteScenarioEntry, ...]
    interaction: dict[str, Any]
    context: dict[str, Any]
    protocol: SuiteProtocolConfig
    evaluation: dict[str, Any]
    difficulty_configurations: dict[str, Any]
    metric_requirements: dict[str, tuple[str, ...]]
    source_path: Path
    raw: dict[str, Any] = field(repr=False)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "content_hash": content_hash(self.raw),
        }


@dataclass(frozen=True)
class SuiteScenarioInstance:
    suite_entry_id: str
    scenario: ScenarioConfig
    simulator_seeds: tuple[int, ...]
    source_kind: str
    source_path: Path | None
    generation_seed: int | None


@dataclass(frozen=True)
class SuiteValidationReport:
    valid: bool
    suite_identity: dict[str, Any]
    scenario_reports: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "suite": self.suite_identity,
            "scenario_reports": list(self.scenario_reports),
            "errors": list(self.errors),
        }


def load_suite(path: str | Path) -> BenchmarkSuiteConfig:
    source = Path(path).resolve()
    data = load_document(source)
    version = int(data.get("suite_schema_version", SUITE_SCHEMA_VERSION))
    if version != SUITE_SCHEMA_VERSION:
        raise ConfigError(f"Unsupported suite_schema_version: {version}")
    name = _non_empty(data.get("name"), "suite name")
    suite_version = _non_empty(data.get("version"), "suite version")
    families = tuple(str(item) for item in data.get("task_families", []))
    if not families:
        raise ConfigError("A benchmark suite must list task_families")
    raw_entries = data.get("scenarios")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ConfigError("A benchmark suite must define at least one scenario entry")

    entries: list[SuiteScenarioEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ConfigError(f"scenarios[{index}] must be a mapping")
        entry = dict(raw_entry)
        entry_id = _non_empty(entry.get("id"), f"scenarios[{index}].id")
        if entry_id in seen:
            raise ConfigError(f"Duplicate suite scenario id {entry_id!r}")
        seen.add(entry_id)
        has_path = "path" in entry
        has_generator = "generator" in entry
        if has_path == has_generator:
            raise ConfigError(
                f"Suite entry {entry_id!r} must define exactly one of path or generator"
            )
        scenario_path = (
            (source.parent / str(entry["path"])).resolve() if has_path else None
        )
        generator = dict(entry["generator"]) if has_generator else None
        seeds = tuple(int(item) for item in entry.get("simulator_seeds", []))
        if not seeds:
            seeds = (0,)
        declared_generation_seed = (
            int(entry["generation_seed"])
            if entry.get("generation_seed") is not None
            else None
        )
        entries.append(
            SuiteScenarioEntry(
                entry_id,
                scenario_path,
                generator,
                seeds,
                declared_generation_seed,
            )
        )

    return BenchmarkSuiteConfig(
        schema_version=version,
        name=name,
        version=suite_version,
        description=str(data.get("description", "")),
        task_families=families,
        entries=tuple(entries),
        interaction=dict(data.get("interaction", {"source": "scenario"})),
        context=dict(data.get("context", {"source": "scenario"})),
        protocol=SuiteProtocolConfig.from_mapping(data.get("protocol")),
        evaluation=dict(data.get("evaluation", {})),
        difficulty_configurations=dict(data.get("difficulty_configurations", {})),
        metric_requirements={
            str(family): tuple(str(metric) for metric in metrics)
            for family, metrics in dict(data.get("metric_requirements", {})).items()
        },
        source_path=source,
        raw=data,
    )


def materialize_suite(suite: BenchmarkSuiteConfig) -> tuple[SuiteScenarioInstance, ...]:
    instances: list[SuiteScenarioInstance] = []
    observed_families: set[str] = set()
    for entry in suite.entries:
        if entry.scenario_path is not None:
            scenario = load_scenario(entry.scenario_path)
            source_kind = "file"
            generation_seed = entry.declared_generation_seed
        else:
            assert entry.generator is not None
            family = _non_empty(entry.generator.get("family"), "generator.family")
            level = _non_empty(entry.generator.get("level"), "generator.level")
            generation_seed = int(entry.generator.get("generation_seed", 0))
            template = entry.generator.get("template")
            template_path = (
                (suite.source_path.parent / str(template)).resolve()
                if template
                else None
            )
            family_levels = suite.difficulty_configurations.get(family, {})
            if family_levels and not isinstance(family_levels, Mapping):
                raise ConfigError(
                    f"difficulty_configurations.{family} must be a mapping"
                )
            suite_parameters = (
                dict(family_levels.get(level, {}))
                if isinstance(family_levels, Mapping)
                else {}
            )
            suite_parameters.update(dict(entry.generator.get("parameters", {})))
            generator_id = entry.generator.get("generator_id")
            if generator_id is None:
                scenario = generate_scenario(
                    family,
                    level,
                    generation_seed,
                    suite_parameters,
                    template_path,
                )
            else:
                from cps_reason_bench.design import (
                    FACTORIAL_GENERATOR_ID,
                    ROBUSTNESS_GENERATOR_ID,
                    generate_factorial_scenario,
                    generate_robustness_scenario,
                )

                if generator_id == ROBUSTNESS_GENERATOR_ID:
                    scenario = generate_robustness_scenario(
                        family,
                        level,
                        generation_seed,
                        suite_parameters,
                        template_path,
                    )
                elif generator_id == FACTORIAL_GENERATOR_ID:
                    scenario = generate_factorial_scenario(
                        family,
                        level,
                        generation_seed,
                        suite_parameters,
                        template_path,
                    )
                elif generator_id == "cps_reason_graph_v2_generator":
                    from cps_reason_bench.graph_v2 import generate_graph_v2_scenario

                    scenario = generate_graph_v2_scenario(
                        family,
                        level,
                        generation_seed,
                        suite_parameters,
                        template_path,
                    )
                else:
                    raise ConfigError(
                        f"Suite entry {entry.entry_id!r} requests unknown generator "
                        f"ID {generator_id!r}"
                    )
            declared_generator_version = entry.generator.get("version")
            if declared_generator_version is not None and str(
                declared_generator_version
            ) != str(scenario.generation.get("generator_version")):
                raise ConfigError(
                    f"Suite entry {entry.entry_id!r} requires generator version "
                    f"{declared_generator_version}, but materialization used "
                    f"{scenario.generation.get('generator_version')}"
                )
            source_kind = "generator"
        scenario = _apply_suite_scenario_overrides(scenario, suite)
        if scenario.task_family not in suite.task_families:
            raise ConfigError(
                f"Suite entry {entry.entry_id!r} has unlisted family {scenario.task_family!r}"
            )
        observed_families.add(scenario.task_family)
        instances.append(
            SuiteScenarioInstance(
                entry.entry_id,
                scenario,
                entry.simulator_seeds,
                source_kind,
                entry.scenario_path,
                generation_seed,
            )
        )
    missing = set(suite.task_families) - observed_families
    if missing:
        raise ConfigError(
            f"Suite has no scenario entries for families: {sorted(missing)}"
        )
    return tuple(instances)


def validate_suite(suite: BenchmarkSuiteConfig) -> SuiteValidationReport:
    from cps_reason_bench.validation import validate_scenario_semantics

    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for instance in materialize_suite(suite):
        report = validate_scenario_semantics(instance.scenario)
        reports.append(
            {
                "entry_id": instance.suite_entry_id,
                "scenario": instance.scenario.name,
                **report.to_dict(),
            }
        )
        errors.extend(
            f"{instance.suite_entry_id}: {message}" for message in report.errors
        )
    return SuiteValidationReport(
        not errors, suite.identity, tuple(reports), tuple(errors)
    )


def _apply_suite_scenario_overrides(
    scenario: ScenarioConfig, suite: BenchmarkSuiteConfig
) -> ScenarioConfig:
    value = scenario.to_mapping()
    for key, override in (
        ("interaction", suite.interaction),
        ("context", suite.context),
    ):
        source = override.get("source", "scenario")
        if source not in {"scenario", "suite"}:
            raise ConfigError(f"Suite {key}.source must be 'scenario' or 'suite'")
        if source == "suite":
            merged = dict(value.get(key, {}))
            merged.update(
                {
                    item: copy.deepcopy(setting)
                    for item, setting in override.items()
                    if item != "source"
                }
            )
            value[key] = merged
    if suite.evaluation:
        merged_evaluation = dict(value.get("evaluation", {}))
        merged_evaluation.update(copy.deepcopy(suite.evaluation))
        value["evaluation"] = merged_evaluation
    return ScenarioConfig.from_mapping(value)


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()
