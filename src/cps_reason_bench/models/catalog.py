"""One user-facing registry for local and OpenAI-compatible model configs."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from cps_reason_bench.config import ConfigError
from cps_reason_bench.models.config import ModelConfig
from cps_reason_bench.models.providers import provider_preset
from cps_reason_bench.models.security import is_secret_key
from cps_reason_bench.run_logging import atomic_write_text
from cps_reason_bench.versions import content_hash


MODEL_REGISTRY_SCHEMA_VERSION = 1
SUPPORTED_BACKENDS = {"ollama", "openai-compatible", "scripted", "random"}
PLACEHOLDER_MARKERS = ("REPLACE_WITH_", "example.invalid")
ALLOWED_ENTRY_FIELDS = {
    "response_normalization",
    "enabled",
    "label",
    "backend",
    "backend_import",
    "model",
    "provider",
    "base_url",
    "api_key_env",
    "timeout_seconds",
    "decoding",
    "backend_parameters",
    "retry_policy",
    "capability_overrides",
    "capabilities",
    "routing",
    "request_parameters",
    "structured_output_mode",
    "transport_retry",
    "pricing",
    "model_metadata",
    "headers",
}
ALLOWED_CAPABILITIES = {
    "chat_completions",
    "seed",
    "temperature",
    "json_object",
    "json_schema",
    "usage",
    "cost_metadata",
    "top_p",
    "max_output_tokens",
    "reasoning_control",
    "response_format",
}
ALLOWED_ROUTING_FIELDS = {
    "mode",
    "provider_order",
    "only",
    "allow_fallbacks",
    "require_parameters",
    "quantizations",
    "data_collection",
    "zdr",
    "expected_provider",
    "expected_route",
    "expected_canonical_model",
}
ALLOWED_REQUEST_PARAMETER_FIELDS = {
    "temperature",
    "top_p",
    "seed",
    "max_tokens_field",
    "reasoning_effort",
    "reasoning_exclude",
}
ALLOWED_TRANSPORT_FIELDS = {
    "max_attempts",
    "base_backoff_seconds",
    "max_backoff_seconds",
    "jitter_seconds",
}
ALLOWED_PRICING_FIELDS = {
    "input_per_million_usd",
    "output_per_million_usd",
    "cached_input_per_million_usd",
    "source",
    "retrieved_at_utc",
    "currency",
}


@dataclass(frozen=True)
class ModelCatalogEntry:
    config_id: str
    label: str
    enabled: bool
    provider: str
    scope: str
    config: ModelConfig
    declared_capabilities: dict[str, Any]
    pricing: dict[str, Any]
    model_metadata: dict[str, Any]

    @property
    def key_status(self) -> str:
        if self.scope == "local":
            return "not required"
        return "configured" if os.environ.get(self.config.api_key_env or "") else "missing"

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "enabled": self.enabled,
            "label": self.label,
            "provider": self.provider,
            "scope": self.scope,
            "model_configuration": self.config.identity,
        }


@dataclass(frozen=True)
class ModelCatalog:
    source_path: Path
    defaults: dict[str, Any]
    entries: dict[str, ModelCatalogEntry]
    raw: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.source_path),
            "content_hash": content_hash(self.safe_dict()),
        }

    def safe_dict(self) -> dict[str, Any]:
        return {
            "model_registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
            "defaults": copy.deepcopy(self.defaults),
            "models": {
                config_id: _entry_snapshot(entry)
                for config_id, entry in sorted(self.entries.items())
            },
        }

    def select(
        self,
        config_ids: Iterable[str] | None = None,
        *,
        enabled_default: bool = True,
        require_runtime_credentials: bool = True,
    ) -> tuple[ModelCatalogEntry, ...]:
        requested = tuple(dict.fromkeys(str(item).strip() for item in (config_ids or ()) if str(item).strip()))
        if not requested and enabled_default:
            requested = tuple(
                config_id for config_id, entry in self.entries.items() if entry.enabled
            )
        missing = sorted(set(requested) - set(self.entries))
        if missing:
            raise ConfigError(f"Unknown model configuration IDs: {missing}")
        if not requested:
            raise ConfigError("No model configurations were selected")
        selected = tuple(self.entries[config_id] for config_id in requested)
        for entry in selected:
            _validate_selected_entry(
                entry, require_runtime_credentials=require_runtime_credentials
            )
        return selected


def load_model_catalog(path: str | Path) -> ModelCatalog:
    source = Path(path).resolve()
    data = _load_unique_document(source)
    unknown_top = set(data) - {"model_registry_schema_version", "defaults", "models"}
    if unknown_top:
        raise ConfigError(f"Unknown model registry fields: {sorted(unknown_top)}")
    version = int(data.get("model_registry_schema_version", MODEL_REGISTRY_SCHEMA_VERSION))
    if version != MODEL_REGISTRY_SCHEMA_VERSION:
        raise ConfigError(f"Unsupported model_registry_schema_version: {version}")
    defaults = _mapping(data.get("defaults", {}), "defaults")
    models = _mapping(data.get("models"), "models")
    if not models:
        raise ConfigError("Model registry must define at least one model")
    entries: dict[str, ModelCatalogEntry] = {}
    for config_id, raw_entry in models.items():
        normalized_id = _config_id(config_id)
        if normalized_id in entries:
            raise ConfigError(f"Duplicate model configuration ID: {normalized_id}")
        entry_data = {**copy.deepcopy(defaults), **_mapping(raw_entry, f"models.{normalized_id}")}
        entries[normalized_id] = _parse_entry(normalized_id, entry_data, source)
    return ModelCatalog(source, defaults, entries, data)


def materialize_catalog_entries(
    catalog: ModelCatalog,
    entries: Iterable[ModelCatalogEntry],
    destination_root: str | Path = "/tmp/cps_reason_bench_model_registry",
) -> tuple[Path, ...]:
    selected = tuple(entries)
    selection_hash = content_hash(
        {
            "registry": catalog.identity["content_hash"],
            "entries": [entry.identity for entry in selected],
        }
    )
    destination = Path(destination_root) / selection_hash
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for entry in selected:
        path = destination / f"{entry.config_id}.yaml"
        text = yaml.safe_dump(entry.config.safe_dict(), sort_keys=False)
        if path.is_file() and path.read_text(encoding="utf-8") != text:
            raise ConfigError(f"Generated model snapshot collision at {path}")
        if not path.is_file():
            atomic_write_text(path, text)
        paths.append(path.resolve())
    return tuple(paths)


def write_catalog_snapshot(
    catalog: ModelCatalog,
    entries: Iterable[ModelCatalogEntry],
    destination: str | Path,
) -> Path:
    selected = tuple(entries)
    payload = {
        "model_registry_schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "defaults": {},
        "models": {
            entry.config_id: _entry_snapshot(entry) for entry in selected
        },
    }
    text = yaml.safe_dump(payload, sort_keys=False)
    path = Path(destination)
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        raise ConfigError(
            f"Existing model snapshot differs from selected model identity: {path}"
        )
    if not path.is_file():
        atomic_write_text(path, text)
    return path


def parse_model_ids(value: str | Iterable[str]) -> tuple[str, ...]:
    values = value.split(",") if isinstance(value, str) else value
    result = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not result:
        raise ConfigError("--models must name at least one model configuration ID")
    return result


def _parse_entry(config_id: str, data: dict[str, Any], source: Path) -> ModelCatalogEntry:
    _reject_unsafe_values(data, f"models.{config_id}")
    unknown = set(data) - ALLOWED_ENTRY_FIELDS
    if unknown:
        raise ConfigError(f"Unknown fields for model {config_id!r}: {sorted(unknown)}")
    backend = str(data.get("backend", "ollama")).strip().lower().replace("_", "-")
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"Model {config_id!r} has unsupported backend {backend!r}; "
            f"choose one of {sorted(SUPPORTED_BACKENDS)}"
        )
    model = str(data.get("model", "")).strip()
    if not model:
        raise ConfigError(f"Model {config_id!r} requires a non-empty model ID")
    enabled = bool(data.get("enabled", False))
    label = str(data.get("label", config_id)).strip() or config_id
    scope = "cloud" if backend == "openai-compatible" else "local"
    provider = str(
        data.get("provider", "ollama" if backend == "ollama" else "custom")
    ).strip().lower()
    preset = provider_preset(provider) if scope == "cloud" else None
    base_url = data.get("base_url") or (preset.base_url if preset else None)
    api_key_env = data.get("api_key_env") or (preset.api_key_env if preset else None)
    if scope == "cloud" and not base_url:
        raise ConfigError(f"Cloud model {config_id!r} requires base_url")
    if scope == "cloud" and not api_key_env:
        raise ConfigError(f"Cloud model {config_id!r} requires api_key_env")

    decoding = _mapping(data.get("decoding", {}), f"models.{config_id}.decoding")
    _validate_decoding(config_id, decoding)
    timeout = float(data.get("timeout_seconds", 120.0))
    if timeout <= 0:
        raise ConfigError(f"Model {config_id!r} timeout_seconds must be positive")
    capabilities = _mapping(
        data.get("capabilities", {}), f"models.{config_id}.capabilities"
    )
    unknown_capabilities = set(capabilities) - ALLOWED_CAPABILITIES
    if unknown_capabilities:
        raise ConfigError(
            f"Model {config_id!r} has unsupported capabilities: {sorted(unknown_capabilities)}"
        )
    for key, value in capabilities.items():
        if value not in {True, False, None, "unknown"}:
            raise ConfigError(
                f"Model {config_id!r} capability {key!r} must be true, false, null, or 'unknown'"
            )

    routing = _mapping(data.get("routing", {}), f"models.{config_id}.routing")
    _validate_routing(config_id, provider, routing)
    request_parameters = _mapping(
        data.get("request_parameters", {}),
        f"models.{config_id}.request_parameters",
    )
    _validate_request_parameters(config_id, request_parameters, capabilities)
    structured_mode = str(data.get("structured_output_mode", "prompt_only"))
    if structured_mode not in {"prompt_only", "json_object", "json_schema"}:
        raise ConfigError(
            f"Model {config_id!r} structured_output_mode must be prompt_only, json_object, or json_schema"
        )
    transport = _mapping(
        data.get("transport_retry", {}), f"models.{config_id}.transport_retry"
    )
    _validate_transport(config_id, transport)
    pricing = _mapping(data.get("pricing", {}), f"models.{config_id}.pricing")
    _validate_pricing(config_id, pricing)
    model_metadata = _mapping(
        data.get("model_metadata", {}), f"models.{config_id}.model_metadata"
    )
    headers = dict(preset.headers if preset else {})
    headers.update(_string_mapping(data.get("headers", {}), f"models.{config_id}.headers"))

    backend_parameters = _mapping(
        data.get("backend_parameters", {}), f"models.{config_id}.backend_parameters"
    )
    if backend == "ollama":
        backend_parameters.setdefault("think", False)
        backend_parameters["structured_output_mode"] = structured_mode
    elif backend == "openai-compatible":
        backend_parameters.update(
            {
                "provider": provider,
                "routing": routing,
                "request_parameters": request_parameters,
                "structured_output_mode": structured_mode,
                "transport_retry": transport,
                "headers": headers,
                "declared_capabilities": capabilities,
                "pricing": pricing,
                "model_metadata": model_metadata,
            }
        )

    model_mapping: dict[str, Any] = {
        "model_config_schema_version": 1,
        "name": config_id,
        "backend": backend,
        "model": model,
        "backend_import": data.get("backend_import"),
        "base_url": str(base_url) if base_url else None,
        "api_key_env": str(api_key_env) if api_key_env else None,
        "timeout_seconds": timeout,
        "decoding": decoding,
        "backend_parameters": backend_parameters,
        "retry_policy": _mapping(
            data.get("retry_policy", {}), f"models.{config_id}.retry_policy"
        ),
        "capability_overrides": _mapping(
            data.get("capability_overrides", {}),
            f"models.{config_id}.capability_overrides",
        ),
    }
    if "response_normalization" in data:
        model_mapping["response_normalization"] = data["response_normalization"]
    config = ModelConfig.from_mapping(model_mapping, source)
    return ModelCatalogEntry(
        config_id,
        label,
        enabled,
        provider,
        scope,
        config,
        capabilities,
        pricing,
        model_metadata,
    )


def _validate_selected_entry(
    entry: ModelCatalogEntry, *, require_runtime_credentials: bool
) -> None:
    text = json.dumps(entry.config.safe_dict(), sort_keys=True)
    if any(marker in text for marker in PLACEHOLDER_MARKERS):
        raise ConfigError(
            f"Selected model {entry.config_id!r} still contains a placeholder model or endpoint"
        )
    if (
        require_runtime_credentials
        and entry.scope == "cloud"
        and entry.key_status == "missing"
    ):
        raise ConfigError(
            f"{entry.config.api_key_env}: missing for selected model {entry.config_id}"
        )


def _validate_decoding(config_id: str, decoding: dict[str, Any]) -> None:
    allowed = {"temperature", "top_p", "max_tokens", "seed", "extra"}
    unknown = set(decoding) - allowed
    if unknown:
        raise ConfigError(
            f"Model {config_id!r} has unsupported decoding fields: {sorted(unknown)}"
        )
    if float(decoding.get("temperature", 0.0)) < 0:
        raise ConfigError(f"Model {config_id!r} temperature cannot be negative")
    top_p = float(decoding.get("top_p", 1.0))
    if not 0 < top_p <= 1:
        raise ConfigError(f"Model {config_id!r} top_p must be in (0, 1]")
    if int(decoding.get("max_tokens", 768)) <= 0:
        raise ConfigError(f"Model {config_id!r} max_tokens must be positive")
    seed = decoding.get("seed")
    if seed not in {None, "derived"} and not isinstance(seed, int):
        raise ConfigError(
            f"Model {config_id!r} decoding.seed must be an integer, null, or 'derived'"
        )
    extra = decoding.get("extra", {})
    if not isinstance(extra, Mapping):
        raise ConfigError(f"Model {config_id!r} decoding.extra must be a mapping")
    forbidden = {
        "plugins",
        "models",
        "route",
        "provider",
        "response_format",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "reasoning",
        "include_reasoning",
    } & set(extra)
    if forbidden:
        raise ConfigError(
            f"Model {config_id!r} decoding.extra cannot override benchmark routing/output fields: {sorted(forbidden)}"
        )


def _validate_routing(config_id: str, provider: str, routing: dict[str, Any]) -> None:
    unknown = set(routing) - ALLOWED_ROUTING_FIELDS
    if unknown:
        raise ConfigError(
            f"Model {config_id!r} has unsupported routing fields: {sorted(unknown)}"
        )
    default_mode = "dynamic" if provider == "openrouter" else "direct"
    mode = str(routing.get("mode", default_mode))
    if mode not in {"dynamic", "benchmark", "direct"}:
        raise ConfigError(f"Model {config_id!r} has invalid routing mode {mode!r}")
    if provider != "openrouter" and mode != "direct":
        raise ConfigError(
            f"Model {config_id!r} provider {provider!r} supports routing mode 'direct' only"
        )
    for key in ("provider_order", "only", "quantizations"):
        if key in routing and (
            not isinstance(routing[key], list)
            or any(not isinstance(item, str) or not item for item in routing[key])
        ):
            raise ConfigError(f"Model {config_id!r} routing.{key} must be a string list")
    if routing.get("data_collection") not in {None, "allow", "deny"}:
        raise ConfigError(
            f"Model {config_id!r} routing.data_collection must be 'allow' or 'deny'"
        )
    if routing.get("zdr") not in {None, True, False}:
        raise ConfigError(f"Model {config_id!r} routing.zdr must be boolean")
    for key in ("expected_provider", "expected_route", "expected_canonical_model"):
        if key in routing and (
            not isinstance(routing[key], str) or not routing[key].strip()
        ):
            raise ConfigError(f"Model {config_id!r} routing.{key} must be a string")
    if mode == "benchmark":
        if routing.get("allow_fallbacks", False) is not False:
            raise ConfigError(
                f"Model {config_id!r} benchmark routing requires allow_fallbacks: false"
            )
        if routing.get("require_parameters", True) is not True:
            raise ConfigError(
                f"Model {config_id!r} benchmark routing requires require_parameters: true"
            )


def _validate_request_parameters(
    config_id: str,
    parameters: dict[str, Any],
    capabilities: Mapping[str, Any],
) -> None:
    unknown = set(parameters) - ALLOWED_REQUEST_PARAMETER_FIELDS
    if unknown:
        raise ConfigError(
            f"Model {config_id!r} has unsupported request_parameters fields: {sorted(unknown)}"
        )
    for name in ("temperature", "top_p", "seed", "reasoning_exclude"):
        value = parameters.get(name)
        if value not in {None, True, False}:
            raise ConfigError(
                f"Model {config_id!r} request_parameters.{name} must be boolean"
            )
        if value is True and capabilities.get(name) is False:
            raise ConfigError(
                f"Model {config_id!r} requests unsupported parameter {name!r}"
            )
    reasoning_effort = parameters.get("reasoning_effort")
    if reasoning_effort not in {
        None,
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ConfigError(
            f"Model {config_id!r} request_parameters.reasoning_effort is invalid"
        )
    if reasoning_effort is not None and capabilities.get("reasoning_control") is False:
        raise ConfigError(
            f"Model {config_id!r} requests unsupported reasoning control"
        )
    field = parameters.get("max_tokens_field", "max_tokens")
    if field not in {"max_tokens", "max_completion_tokens"}:
        raise ConfigError(
            f"Model {config_id!r} request_parameters.max_tokens_field must be "
            "'max_tokens' or 'max_completion_tokens'"
        )


def _validate_transport(config_id: str, transport: dict[str, Any]) -> None:
    unknown = set(transport) - ALLOWED_TRANSPORT_FIELDS
    if unknown:
        raise ConfigError(
            f"Model {config_id!r} has unsupported transport_retry fields: {sorted(unknown)}"
        )
    attempts = int(transport.get("max_attempts", 3))
    if attempts < 1 or attempts > 10:
        raise ConfigError(f"Model {config_id!r} max_attempts must be between 1 and 10")
    for key, default in (
        ("base_backoff_seconds", 1.0),
        ("max_backoff_seconds", 30.0),
        ("jitter_seconds", 0.25),
    ):
        if float(transport.get(key, default)) < 0:
            raise ConfigError(f"Model {config_id!r} transport_retry.{key} cannot be negative")


def _validate_pricing(config_id: str, pricing: dict[str, Any]) -> None:
    unknown = set(pricing) - ALLOWED_PRICING_FIELDS
    if unknown:
        raise ConfigError(
            f"Model {config_id!r} has unsupported pricing fields: {sorted(unknown)}"
        )
    for key in (
        "input_per_million_usd",
        "output_per_million_usd",
        "cached_input_per_million_usd",
    ):
        value = pricing.get(key)
        if value is not None and float(value) < 0:
            raise ConfigError(f"Model {config_id!r} pricing.{key} cannot be negative")
    currency = pricing.get("currency", "USD")
    if currency != "USD":
        raise ConfigError(f"Model {config_id!r} pricing.currency currently supports USD only")


def _entry_snapshot(entry: ModelCatalogEntry) -> dict[str, Any]:
    safe = entry.config.safe_dict()
    backend_parameters = copy.deepcopy(safe.get("backend_parameters", {}))
    output: dict[str, Any] = {
        "enabled": entry.enabled,
        "label": entry.label,
        "backend": safe["backend"].replace("-", "_"),
        "model": safe["model"],
        "provider": entry.provider,
        "base_url": safe.get("base_url"),
        "api_key_env": safe.get("api_key_env"),
        "timeout_seconds": safe["timeout_seconds"],
        "decoding": safe["decoding"],
        "retry_policy": safe["retry_policy"],
        "capability_overrides": safe["capability_overrides"],
        "capabilities": entry.declared_capabilities,
        "pricing": entry.pricing,
        "model_metadata": entry.model_metadata,
    }
    if safe.get("backend_import"):
        output["backend_import"] = safe["backend_import"]
    if entry.config.backend == "ollama":
        output["structured_output_mode"] = backend_parameters.pop(
            "structured_output_mode", "json_schema"
        )
        output["backend_parameters"] = backend_parameters
    elif entry.config.backend == "openai-compatible":
        for key in (
            "routing",
            "request_parameters",
            "structured_output_mode",
            "transport_retry",
            "headers",
        ):
            output[key] = backend_parameters.pop(key, {})
        for key in ("declared_capabilities", "pricing", "model_metadata", "provider"):
            backend_parameters.pop(key, None)
        if backend_parameters:
            output["backend_parameters"] = backend_parameters
    if "response_normalization" in safe:
        output["response_normalization"] = safe["response_normalization"]
    return {key: value for key, value in output.items() if value not in (None, {}, [])}


def _load_unique_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Model registry does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(text, object_pairs_hook=_unique_pairs)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            loader = _UniqueKeyLoader(text)
            try:
                value = loader.get_single_data()
            finally:
                loader.dispose()
        else:
            raise ConfigError("Model registry must use .yaml, .yml, or .json")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load model registry {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("Model registry top level must be a mapping")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ConfigError(f"Duplicate configuration key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ConfigError(f"Duplicate configuration key: {key}")
        output[key] = value
    return output


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return {str(key): copy.deepcopy(nested) for key, nested in value.items()}


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    mapping = _mapping(value, name)
    if any(not isinstance(item, str) for item in mapping.values()):
        raise ConfigError(f"{name} values must be strings")
    return {key: str(item) for key, item in mapping.items()}


def _config_id(value: Any) -> str:
    config_id = str(value).strip()
    if not config_id or any(character.isspace() for character in config_id):
        raise ConfigError(f"Invalid model configuration ID: {value!r}")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in config_id):
        raise ConfigError(
            f"Model configuration ID {config_id!r} may contain letters, digits, '.', '_', and '-' only"
        )
    return config_id


def _reject_unsafe_values(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if is_secret_key(str(key)):
                raise ConfigError(
                    f"{path}.{key} is forbidden; store only an API-key environment variable name"
                )
            _reject_unsafe_values(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_unsafe_values(nested, f"{path}[{index}]")
