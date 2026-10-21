"""Reusable, secret-free model configuration files."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cps_reason_bench.config import ConfigError, load_document
from cps_reason_bench.models.base import DecodingParameters
from cps_reason_bench.models.registry import BackendBuildConfig, create_backend
from cps_reason_bench.policies import LLMPolicy, Policy, RandomPolicy, ScriptedPolicy
from cps_reason_bench.response_normalization import normalization_config
from cps_reason_bench.suites import RetryPolicyConfig
from cps_reason_bench.versions import (
    MODEL_CONFIG_SCHEMA_VERSION,
    PROMPT_TEMPLATE_ID,
    PROMPT_TEMPLATE_VERSION,
    content_hash,
)


SECRET_KEYS = {"api_key", "token", "password", "secret", "authorization"}


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    name: str
    backend: str
    model: str
    backend_import: str | None
    base_url: str | None
    api_key_env: str | None
    timeout_seconds: float
    decoding: dict[str, Any]
    backend_parameters: dict[str, Any]
    retry_policy: RetryPolicyConfig
    capability_overrides: dict[str, Any]
    source_path: Path | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "model": self.model,
            "content_hash": content_hash(self.safe_dict()),
        }

    def safe_dict(self) -> dict[str, Any]:
        result = {
            "model_config_schema_version": self.schema_version,
            "name": self.name,
            "backend": self.backend,
            "model": self.model,
            "backend_import": self.backend_import,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "decoding": copy.deepcopy(self.decoding),
            "backend_parameters": copy.deepcopy(self.backend_parameters),
            "retry_policy": self.retry_policy.to_dict(),
            "capability_overrides": copy.deepcopy(self.capability_overrides),
        }
        # Omit the strict default to preserve every existing model/episode identity.
        if self.response_normalization["mode"] != "none":
            result["response_normalization"] = self.response_normalization
        return result

    @property
    def response_normalization(self) -> dict[str, str]:
        return normalization_config(self.raw.get("response_normalization"))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], source_path: str | Path | None = None
    ) -> "ModelConfig":
        data = dict(value)
        _reject_secret_values(data)
        normalization = normalization_config(data.get("response_normalization"))
        if normalization["mode"] != "none" and RetryPolicyConfig.from_mapping(data.get("retry_policy")).max_corrections:
            raise ConfigError("Response normalization requires zero output corrections")
        version = int(data.get("model_config_schema_version", MODEL_CONFIG_SCHEMA_VERSION))
        if version != MODEL_CONFIG_SCHEMA_VERSION:
            raise ConfigError(f"Unsupported model_config_schema_version: {version}")
        name = _required(data.get("name"), "model configuration name")
        backend_value = data.get("backend", "scripted")
        if isinstance(backend_value, Mapping):
            backend_data = dict(backend_value)
            backend = _required(backend_data.get("name"), "backend.name")
            backend_import = backend_data.get("import_path")
        else:
            backend = _required(backend_value, "backend")
            backend_import = data.get("backend_import")
        model = str(data.get("model", backend))
        decoding = dict(data.get("decoding", {}))
        seed = decoding.get("seed")
        if seed not in {None, "derived"} and not isinstance(seed, int):
            raise ConfigError("decoding.seed must be an integer, null, or 'derived'")
        timeout = float(data.get("timeout_seconds", 120.0))
        if timeout <= 0:
            raise ConfigError("timeout_seconds must be positive")
        return cls(
            schema_version=version,
            name=name,
            backend=backend,
            model=model,
            backend_import=str(backend_import) if backend_import else None,
            base_url=str(data["base_url"]) if data.get("base_url") else None,
            api_key_env=str(data["api_key_env"]) if data.get("api_key_env") else None,
            timeout_seconds=timeout,
            decoding=decoding,
            backend_parameters=dict(data.get("backend_parameters", {})),
            retry_policy=RetryPolicyConfig.from_mapping(data.get("retry_policy")),
            capability_overrides=dict(data.get("capability_overrides", {})),
            source_path=Path(source_path).resolve() if source_path else None,
            raw=data,
        )


def load_model_config(path: str | Path) -> ModelConfig:
    source = Path(path).resolve()
    return ModelConfig.from_mapping(load_document(source), source)


def build_policy(
    config: ModelConfig,
    *,
    random_seed: int,
    requested_model_seed: int | None,
    retry_policy: RetryPolicyConfig | None = None,
    prompt_template_id: str = PROMPT_TEMPLATE_ID,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> Policy:
    if config.backend == "scripted":
        return ScriptedPolicy(configuration_name=config.name)
    if config.backend == "random":
        return RandomPolicy(random_seed, configuration_name=config.name)

    backend = create_backend(
        config.backend,
        BackendBuildConfig(
            model=config.model,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            timeout_seconds=config.timeout_seconds,
            parameters=config.backend_parameters,
        ),
        config.backend_import,
    )
    decoding = DecodingParameters(
        temperature=float(config.decoding.get("temperature", 0.0)),
        max_tokens=int(config.decoding.get("max_tokens", 768)),
        top_p=float(config.decoding.get("top_p", 1.0)),
        seed=requested_model_seed,
        extra=dict(config.decoding.get("extra", {})),
    )
    effective_retry = retry_policy or config.retry_policy
    return LLMPolicy(
        backend,
        decoding,
        effective_retry.max_corrections,
        retry_policy_name=effective_retry.name,
        prompt_template_id=prompt_template_id,
        prompt_template_version=prompt_template_version,
        model_configuration_name=config.name,
        capability_overrides=config.capability_overrides,
        response_normalization=config.response_normalization,
    )


def resolve_model_seed(config: ModelConfig, derived_seed: int) -> int | None:
    requested = config.decoding.get("seed")
    if requested == "derived":
        return derived_seed
    return int(requested) if requested is not None else None


def _required(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _reject_secret_values(value: Any, path: str = "model configuration") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            secret_key = (
                normalized_key in SECRET_KEYS
                or normalized_key == "access_token"
                or "password" in normalized_key
                or "secret" in normalized_key
            ) and not normalized_key.endswith("_env")
            if secret_key:
                raise ConfigError(
                    f"{path}.{key} is forbidden; store only an API-key environment variable name"
                )
            _reject_secret_values(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secret_values(nested, f"{path}[{index}]")
