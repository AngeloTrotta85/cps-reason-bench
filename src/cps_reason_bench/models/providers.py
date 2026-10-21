"""Configuration-only presets for OpenAI-compatible HTTP providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cps_reason_bench.config import ConfigError


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str | None
    api_key_env: str
    headers: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "headers": dict(self.headers),
            "description": self.description,
        }


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        description="OpenRouter's OpenAI-compatible multi-provider API.",
    ),
    "openai": ProviderPreset(
        "openai",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        description="Direct OpenAI API.",
    ),
    "together": ProviderPreset(
        "together",
        "https://api.together.ai/v1",
        "TOGETHER_API_KEY",
        description="Together AI's OpenAI-compatible API.",
    ),
    "groq": ProviderPreset(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        description="Groq's OpenAI-compatible API.",
    ),
    "custom": ProviderPreset(
        "custom",
        None,
        "CUSTOM_LLM_API_KEY",
        description="A user-supplied OpenAI-compatible endpoint.",
    ),
}


def provider_preset(name: str) -> ProviderPreset:
    normalized = str(name).strip().lower()
    try:
        return PROVIDER_PRESETS[normalized]
    except KeyError as exc:
        raise ConfigError(
            f"Unsupported provider preset {name!r}; choose one of "
            f"{sorted(PROVIDER_PRESETS)}"
        ) from exc


def list_provider_presets() -> tuple[ProviderPreset, ...]:
    return tuple(PROVIDER_PRESETS[name] for name in sorted(PROVIDER_PRESETS))
