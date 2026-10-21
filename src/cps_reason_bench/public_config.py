"""Strict, network-free release configuration over the existing benchmark types."""

from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import yaml

from cps_reason_bench.config import ConfigError, load_scenario
from cps_reason_bench.models.config import ModelConfig, build_policy
from cps_reason_bench.suites import (
    BenchmarkSuiteConfig, SuiteProtocolConfig, SuiteScenarioEntry,
    SuiteScenarioInstance, load_suite, materialize_suite,
)
from cps_reason_bench.versions import content_hash


BUILTIN_SUITES = {
    "cps_reason_graph_v2": "graph_v2_complete.yaml",
    "cps_reason_graph_v2_pilot": "graph_v2_pilot.yaml",
    "cps_reason_graph_v2_local_smoke": "graph_v2_local_smoke.yaml",
    "cps_reason_graph_v2_cloud_qualification": "graph_v2_cloud_qualification.yaml",
}
BACKENDS = ("scripted", "random", "ollama", "openai-compatible", "openai_compatible")
DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class RunConfig:
    """Validated selection. Paths resolve relative to the input YAML, not cwd."""

    name: str
    description: str
    models: tuple[ModelConfig, ...]
    suite: BenchmarkSuiteConfig
    conditions: tuple[SuiteScenarioInstance, ...]
    repetitions: int
    simulator_seeds: tuple[int, ...]
    output_dir: Path
    max_episodes: int
    model_call_delay_seconds: float
    source_path: Path
    resolved: dict[str, Any]

    @property
    def episode_count(self) -> int:
        return len(self.models) * self.repetitions * sum(
            len(self.simulator_seeds or item.simulator_seeds) for item in self.conditions
        )

    @property
    def requires_confirmation(self) -> bool:
        return any(model.backend in {"openai-compatible", "openai_compatible"} for model in self.models)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent resolved configuration, never credential values."""
        return copy.deepcopy(self.resolved)


class _UniqueLoader(yaml.SafeLoader):
    """Reject accidental duplicate YAML keys instead of silently overriding them."""


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("Configuration mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_config_document(path: Path) -> dict[str, Any]:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"Cannot read configuration file: {path.name}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark else ""
        raise ConfigError(f"Invalid YAML in {path.name}{location}") from exc
    return _mapping(data, "configuration")


def _mapping(value: Any, label: str, allowed: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{label} must be a mapping with string keys")
    if allowed is not None and set(value) - allowed:
        raise ConfigError(f"Unknown fields in {label}: {sorted(set(value) - allowed)}")
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if type(value) not in {float, int} or not math.isfinite(value) or value < minimum:
        raise ConfigError(f"{label} must be a finite number >= {minimum}")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{label} must be true or false")
    return value


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{label} must be a list")
    result = [_text(item, label) for item in value]
    if len(set(result)) != len(result):
        raise ConfigError(f"{label} must not contain duplicates")
    return result


def _url(value: Any, label: str, *, local: bool) -> str:
    text = _text(value, label)
    try:
        parsed = urlsplit(text)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment)
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid or any(char.isspace() for char in text):
        raise ConfigError(f"{label} must be an HTTP(S) URL without credentials, query, or fragment")
    if not local and parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ConfigError(f"{label} must use HTTPS except for a loopback endpoint")
    return text.rstrip("/")


def _model(raw: Any, source: Path) -> ModelConfig:
    data = _mapping(raw, "model")
    if "path" in data:
        _mapping(data, "model reference", {"path"})
        source = (source.parent / _text(data["path"], "model.path")).resolve()
        data = read_config_document(source)
    data = _mapping(data, "model", {
        "model_config_schema_version", "name", "backend", "model", "base_url",
        "api_key_env", "timeout_seconds", "decoding", "backend_parameters",
        "response_normalization",
    })
    _integer(data.get("model_config_schema_version", 1), "model_config_schema_version", 1)
    data["name"] = _text(data.get("name"), "model.name")
    backend = _text(data.get("backend"), "model.backend")
    if backend not in BACKENDS:
        raise ConfigError(f"Unknown backend {backend!r}; available: {', '.join(BACKENDS)}")
    data["model"] = _text(data.get("model", backend), "model.model")
    data["timeout_seconds"] = _number(data.get("timeout_seconds", 120), "timeout_seconds", 0.001)
    cloud = backend in {"openai-compatible", "openai_compatible"}
    if cloud or backend == "ollama":
        data["base_url"] = _url(data.get("base_url", "http://localhost:11434" if not cloud else None), "base_url", local=not cloud)
    elif "base_url" in data or "api_key_env" in data:
        raise ConfigError("Baseline models do not accept base_url or api_key_env")
    if cloud:
        env = _text(data.get("api_key_env"), "api_key_env")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
            raise ConfigError("api_key_env must name an environment variable, not contain its value")
        data["api_key_env"] = env
    elif data.get("api_key_env") is not None:
        raise ConfigError("api_key_env is supported only by openai-compatible backends")
    decoding = _mapping(data.get("decoding", {}), "decoding", {"temperature", "top_p", "max_tokens", "seed", "extra"})
    decoding["temperature"] = _number(decoding.get("temperature", 0), "decoding.temperature")
    decoding["top_p"] = _number(decoding.get("top_p", 1), "decoding.top_p", 0.000001)
    if decoding["top_p"] > 1:
        raise ConfigError("decoding.top_p must be <= 1")
    decoding["max_tokens"] = _integer(decoding.get("max_tokens", 768), "decoding.max_tokens", 1)
    seed = decoding.setdefault("seed", None)
    if seed is not None and seed != "derived":
        _integer(seed, "decoding.seed")
    extra = _mapping(decoding.get("extra", {}), "decoding.extra", {"num_ctx", "num_thread"} if backend == "ollama" else set())
    for key, value in extra.items():
        _integer(value, f"decoding.extra.{key}", 1)
    decoding["extra"] = extra
    data["decoding"] = decoding
    params = _mapping(data.get("backend_parameters", {}), "backend_parameters", (
        {"think", "structured_output_mode"} if backend == "ollama" else
        {"provider", "structured_output_mode", "routing", "request_parameters", "transport_retry"} if cloud else set()
    ))
    if backend == "ollama":
        params["think"] = _boolean(params.get("think", False), "backend_parameters.think")
    if cloud or backend == "ollama":
        mode = params.setdefault("structured_output_mode", "prompt_only" if cloud else "json_schema")
        if not isinstance(mode, str) or mode not in {"prompt_only", "json_schema", "json_object"}:
            raise ConfigError("structured_output_mode must be prompt_only, json_schema, or json_object")
    if cloud:
        params["provider"] = _text(params.get("provider", "custom"), "backend_parameters.provider")
        if params["provider"] not in {"custom", "openai", "openrouter"}:
            raise ConfigError("provider must be custom, openai, or openrouter; use custom for other compatible services")
        routing = _mapping(params.get("routing", {}), "routing", {
            "mode", "provider_order", "only", "allow_fallbacks", "require_parameters", "quantizations",
            "data_collection", "zdr", "expected_provider", "expected_route", "expected_canonical_model",
        })
        for key in ("provider_order", "only", "quantizations"):
            if key in routing:
                _strings(routing[key], f"routing.{key}")
        for key in ("allow_fallbacks", "require_parameters", "zdr"):
            if key in routing:
                _boolean(routing[key], f"routing.{key}")
        for key in ("mode", "data_collection", "expected_provider", "expected_route", "expected_canonical_model"):
            if key in routing:
                _text(routing[key], f"routing.{key}")
        if params["provider"] != "openrouter" and set(routing) - {"mode"}:
            raise ConfigError("Routing controls other than direct mode require provider=openrouter")
        params["routing"] = routing
        request = _mapping(params.get("request_parameters", {}), "request_parameters", {
            "temperature", "top_p", "seed", "max_tokens_field", "reasoning_effort", "reasoning_exclude",
        })
        for key in ("temperature", "top_p", "seed", "reasoning_exclude"):
            if key in request:
                _boolean(request[key], f"request_parameters.{key}")
        for key in ("max_tokens_field", "reasoning_effort"):
            if key in request and request[key] is not None:
                _text(request[key], f"request_parameters.{key}")
        params["request_parameters"] = request
        retry = _mapping(params.get("transport_retry", {}), "transport_retry", {
            "max_attempts", "base_backoff_seconds", "max_backoff_seconds", "jitter_seconds",
        })
        retry["max_attempts"] = _integer(retry.get("max_attempts", 1), "transport_retry.max_attempts", 1)
        for key, default in (("base_backoff_seconds", 1), ("max_backoff_seconds", 30), ("jitter_seconds", 0.25)):
            retry[key] = _number(retry.get(key, default), f"transport_retry.{key}")
        params["transport_retry"] = retry
    data["backend_parameters"] = params
    normalization = _mapping(data.get("response_normalization", {}), "response_normalization", {"mode"})
    if not isinstance(normalization.get("mode", "none"), str):
        raise ConfigError("response_normalization.mode must be a string")
    data["response_normalization"] = normalization
    config = ModelConfig.from_mapping(data, source)
    # Constructors validate backend-specific combinations but perform no requests.
    build_policy(config, random_seed=0, requested_model_seed=None)
    return config


def condition_record(item: SuiteScenarioInstance) -> dict[str, Any]:
    scenario = item.scenario
    return {
        "id": item.suite_entry_id,
        "family": scenario.task_family,
        "profile": scenario.difficulty_level,
        "role": scenario.difficulty_profile.get("graph_v2", {}).get("semantic_instance"),
        "scenario_hash": content_hash(scenario.to_mapping()),
        "simulator_seeds": list(item.simulator_seeds),
        "generation_seed": item.generation_seed,
    }


def _custom_suite(paths: list[str], source: Path) -> BenchmarkSuiteConfig:
    entries = []
    scenarios = []
    for name in paths:
        path = (source.parent / name).resolve()
        # Use strict YAML syntax first; the existing schema handles physical semantics.
        read_config_document(path)
        scenario = load_scenario(path)
        scenarios.append(scenario)
        entries.append(SuiteScenarioEntry(scenario.name, path, None, (scenario.seed,)))
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise ConfigError("Custom scenarios must have distinct names")
    protocol = SuiteProtocolConfig.from_mapping({"prompt": {"id": "cps-reason-json-control", "version": "2.0.0"}})
    raw = {
        "suite_schema_version": 1, "name": "custom_scenarios", "version": "1.0.0",
        "scenarios": [{"id": item.name, "scenario_hash": content_hash(item.to_mapping())} for item in scenarios],
        "protocol": protocol.to_dict(),
    }
    return BenchmarkSuiteConfig(1, "custom_scenarios", "1.0.0", "User-authored complete scenarios",
        tuple(sorted({item.task_family for item in scenarios})), tuple(entries),
        {"source": "scenario"}, {"source": "scenario"}, protocol, {}, {}, {}, source, raw)


def load_config(path: str | Path, *, check_credentials: bool = False) -> RunConfig:
    """Load and materialize YAML without network access or output writes."""
    source = Path(path).resolve()
    data = read_config_document(source)
    _mapping(data, "configuration", {"config_version", "name", "description", "models", "benchmark",
        "repetitions", "simulator_seeds", "protocol", "output_dir", "runtime"})
    if _integer(data.get("config_version", 1), "config_version", 1) != 1:
        raise ConfigError("Only config_version 1 is supported")
    name = _text(data.get("name"), "name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ConfigError("name must use letters, digits, dots, underscores, or hyphens")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise ConfigError("description must be a string")
    if not isinstance(data.get("models"), list) or not data["models"]:
        raise ConfigError("models must be a non-empty list")
    models = tuple(_model(item, source) for item in data["models"])
    if len({item.name for item in models}) != len(models):
        raise ConfigError("Model configuration names must be unique")
    selection = _mapping(data.get("benchmark"), "benchmark", {"suite", "scenarios", "families", "roles", "profiles", "condition_ids", "limit"})
    if ("suite" in selection) == ("scenarios" in selection):
        raise ConfigError("benchmark requires exactly one of suite or scenarios")
    if "suite" in selection:
        suite_name = _text(selection["suite"], "benchmark.suite")
        if suite_name not in BUILTIN_SUITES:
            raise ConfigError(f"Unknown suite {suite_name!r}; available: {', '.join(BUILTIN_SUITES)}")
        suite = load_suite(DATA / "benchmark/suites" / BUILTIN_SUITES[suite_name])
    else:
        paths = _strings(selection["scenarios"], "benchmark.scenarios")
        if not paths:
            raise ConfigError("benchmark.scenarios cannot be empty")
        suite = _custom_suite(paths, source)
        selection["scenarios"] = [str((source.parent / path).resolve()) for path in paths]
    if "protocol" in data:
        protocol = _mapping(data["protocol"], "protocol", {"prompt", "schema", "output_validation_policy", "retry_policy"})
        merged = suite.protocol.to_dict()
        for key in ("prompt", "schema"):
            if key in protocol:
                part = _mapping(protocol[key], f"protocol.{key}", {"id", "version"})
                for field, value in part.items():
                    _text(value, f"protocol.{key}.{field}")
                merged[key].update(part)
        if "retry_policy" in protocol:
            retry = _mapping(protocol["retry_policy"], "protocol.retry_policy", {"name", "max_corrections"})
            _text(retry.get("name", "strict_zero_correction"), "retry_policy.name")
            _integer(retry.get("max_corrections", 0), "retry_policy.max_corrections")
            merged["retry_policy"].update(retry)
        if "output_validation_policy" in protocol:
            merged["output_validation_policy"] = protocol["output_validation_policy"]
        parsed = SuiteProtocolConfig.from_mapping(merged)
        if parsed != suite.protocol:
            raw = copy.deepcopy(suite.raw)
            raw["protocol"] = parsed.to_dict()
            suite = replace(suite, protocol=parsed, raw=raw)
    if suite.protocol.retry_policy.max_corrections and any(m.response_normalization["mode"] != "none" for m in models):
        raise ConfigError("Response normalization requires zero output corrections")
    instances = materialize_suite(suite)
    records = {item.suite_entry_id: condition_record(item) for item in instances}
    for field, key in (("families", "family"), ("roles", "role"), ("profiles", "profile"), ("condition_ids", "id")):
        values = _strings(selection.get(field, []), f"benchmark.{field}")
        available = {record[key] for record in records.values()} - {None}
        if set(values) - available:
            raise ConfigError(f"Unknown {field}: {sorted(set(values) - available)}; available: {sorted(available)}")
        if values:
            instances = tuple(item for item in instances if records[item.suite_entry_id][key] in values)
    if "limit" in selection:
        instances = instances[:_integer(selection["limit"], "benchmark.limit", 1)]
    if not instances:
        raise ConfigError("Benchmark filters produced an empty selection; filters intersect")
    repetitions = _integer(data.get("repetitions", 1), "repetitions", 1)
    seeds_raw = data.get("simulator_seeds", [])
    if not isinstance(seeds_raw, list):
        raise ConfigError("simulator_seeds must be a list of integers")
    seeds = tuple(_integer(seed, "simulator_seeds") for seed in seeds_raw)
    if len(set(seeds)) != len(seeds):
        raise ConfigError("simulator_seeds must not contain duplicates")
    runtime = _mapping(data.get("runtime", {}), "runtime", {"max_episodes", "model_call_delay_seconds"})
    maximum = _integer(runtime.get("max_episodes", 1000), "runtime.max_episodes", 1)
    delay = _number(runtime.get("model_call_delay_seconds", 0), "runtime.model_call_delay_seconds")
    output = (source.parent / _text(data.get("output_dir", f"runs/{name}"), "output_dir")).resolve()
    resolved_models = []
    for model in models:
        value = model.safe_dict()
        for field in ("backend_import", "capability_overrides", "retry_policy"):
            value.pop(field)
        value["response_normalization"] = model.response_normalization
        if value["api_key_env"] is None:
            value.pop("api_key_env")
        if value["base_url"] is None:
            value.pop("base_url")
        resolved_models.append(value)
    resolved = {"config_version": 1, "name": name, "description": description,
        "models": resolved_models, "benchmark": selection, "protocol": suite.protocol.to_dict(),
        "repetitions": repetitions, "simulator_seeds": list(seeds), "output_dir": str(output),
        "runtime": {"max_episodes": maximum, "model_call_delay_seconds": delay}}
    config = RunConfig(name, description, models, suite, instances, repetitions, seeds,
        output, maximum, delay, source, resolved)
    if config.episode_count > maximum:
        raise ConfigError(f"Planned {config.episode_count} episodes exceeds runtime.max_episodes={maximum}; explicitly raise the cap")
    if check_credentials:
        require_credentials(config)
    return config


def require_credentials(config: RunConfig) -> None:
    """Check environment names only; never return or serialize their values."""
    missing = [model.api_key_env for model in config.models if model.api_key_env and not os.environ.get(model.api_key_env, "").strip()]
    if missing:
        raise ConfigError(f"Missing credential environment variables: {', '.join(sorted(set(missing)))}; export them before running")


def instantiate_benchmark(config: RunConfig) -> tuple[SuiteScenarioInstance, ...]:
    """Return the selected materialized conditions, without running the simulator."""
    return config.conditions


def list_conditions(config: RunConfig) -> list[dict[str, Any]]:
    """Return condition identifiers, original seeds, role/profile, and content hashes."""
    return [condition_record(item) for item in config.conditions]
