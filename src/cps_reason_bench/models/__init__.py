"""Provider adapters for model inference."""

from cps_reason_bench.models.base import (
    BackendCapabilities,
    BackendError,
    ProviderError,
    DecodingParameters,
    ModelBackend,
    ModelRequest,
    ModelResponse,
)
from cps_reason_bench.models.ollama import OllamaBackend
from cps_reason_bench.models.openai_compatible import OpenAICompatibleBackend
from cps_reason_bench.models.registry import (
    BackendBuildConfig,
    create_backend,
    register_backend,
    registered_backends,
)

__all__ = [
    "BackendError",
    "ProviderError",
    "BackendCapabilities",
    "BackendBuildConfig",
    "DecodingParameters",
    "ModelBackend",
    "ModelRequest",
    "ModelResponse",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "create_backend",
    "register_backend",
    "registered_backends",
]
