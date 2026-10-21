"""Lightweight backend registry with import-path plugin loading."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from cps_reason_bench.config import ConfigError
from cps_reason_bench.models.base import ModelBackend
from cps_reason_bench.models.ollama import OllamaBackend
from cps_reason_bench.models.openai_compatible import OpenAICompatibleBackend


@dataclass(frozen=True)
class BackendBuildConfig:
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    parameters: dict[str, Any] = field(default_factory=dict)


BackendFactory = Callable[[BackendBuildConfig], ModelBackend]


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(
        self, name: str, factory: BackendFactory, *, replace: bool = False
    ) -> None:
        if not name or (name in self._factories and not replace):
            raise ValueError(f"Backend {name!r} is already registered or invalid")
        self._factories[name] = factory

    def create(
        self,
        name: str,
        config: BackendBuildConfig,
        import_path: str | None = None,
    ) -> ModelBackend:
        if import_path:
            factory = _load_factory(import_path)
            backend = factory(config)
            if not isinstance(backend, ModelBackend):
                raise TypeError(
                    f"Imported backend factory {import_path!r} did not return ModelBackend"
                )
            return backend
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ConfigError(
                f"Unknown backend {name!r}; registered backends: {sorted(self._factories)}"
            ) from exc
        backend = factory(config)
        if not isinstance(backend, ModelBackend):
            raise TypeError(f"Backend factory {name!r} did not return ModelBackend")
        return backend

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


BACKENDS = BackendRegistry()


def register_backend(
    name: str, factory: BackendFactory, *, replace: bool = False
) -> None:
    BACKENDS.register(name, factory, replace=replace)


def create_backend(
    name: str,
    config: BackendBuildConfig,
    import_path: str | None = None,
) -> ModelBackend:
    return BACKENDS.create(name, config, import_path)


def registered_backends() -> tuple[str, ...]:
    return BACKENDS.names()


def _load_factory(import_path: str) -> BackendFactory:
    if ":" not in import_path:
        raise ConfigError("backend_import must use 'module:object' syntax")
    module_name, object_name = import_path.split(":", 1)
    try:
        target = getattr(importlib.import_module(module_name), object_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"Could not load backend import path {import_path!r}: {exc}") from exc

    if isinstance(target, type) and issubclass(target, ModelBackend):
        def class_factory(config: BackendBuildConfig) -> ModelBackend:
            kwargs = dict(config.parameters)
            kwargs.setdefault("model", config.model)
            if config.base_url is not None:
                kwargs.setdefault("base_url", config.base_url)
            kwargs.setdefault("timeout_seconds", config.timeout_seconds)
            return target(**kwargs)

        return class_factory
    if callable(target):
        return target
    raise ConfigError(f"Imported backend object {import_path!r} is not callable")


def _ollama_factory(config: BackendBuildConfig) -> ModelBackend:
    parameters = dict(config.parameters)
    think = bool(parameters.pop("think", False))
    structured_output_mode = str(
        parameters.pop("structured_output_mode", "json_schema")
    )
    if parameters:
        raise ConfigError(f"Unsupported Ollama backend parameters: {sorted(parameters)}")
    return OllamaBackend(
        config.model,
        config.base_url or "http://localhost:11434",
        config.timeout_seconds,
        think,
        structured_output_mode,
    )


def _openai_compatible_factory(config: BackendBuildConfig) -> ModelBackend:
    if not config.base_url:
        raise ConfigError("openai-compatible backend requires base_url")
    parameters = dict(config.parameters)
    allowed = {
        "provider",
        "routing",
        "request_parameters",
        "structured_output_mode",
        "transport_retry",
        "headers",
        "declared_capabilities",
        "pricing",
        "model_metadata",
    }
    unknown = set(parameters) - allowed
    if unknown:
        raise ConfigError(
            f"Unsupported OpenAI-compatible backend parameters: {sorted(unknown)}"
        )
    return OpenAICompatibleBackend(
        config.model,
        config.base_url,
        api_key_env=config.api_key_env,
        timeout_seconds=config.timeout_seconds,
        provider=str(parameters.get("provider", "custom")),
        routing=parameters.get("routing"),
        request_parameters=parameters.get("request_parameters"),
        structured_output_mode=str(
            parameters.get("structured_output_mode", "prompt_only")
        ),
        transport_retry=parameters.get("transport_retry"),
        headers=parameters.get("headers"),
        declared_capabilities=parameters.get("declared_capabilities"),
        pricing=parameters.get("pricing"),
        model_metadata=parameters.get("model_metadata"),
    )


register_backend("ollama", _ollama_factory)
register_backend("openai-compatible", _openai_compatible_factory)
register_backend("openai_compatible", _openai_compatible_factory)
