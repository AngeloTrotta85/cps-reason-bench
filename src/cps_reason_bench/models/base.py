"""Provider-independent model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class BackendError(RuntimeError):
    """Raised when a model provider cannot return a usable response."""

    category = "backend_error"
    retryable = False
    billing_ambiguity = False
    request_state = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "retryable": self.retryable,
            "billing_ambiguity": self.billing_ambiguity,
            "request_state": self.request_state,
            "message": str(self),
        }


class ProviderError(BackendError):
    """Sanitized cloud-provider error with retry and billing semantics."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
        request_state: str = "unknown",
        billing_ambiguity: bool = False,
        metadata: dict[str, Any] | None = None,
        transport_attempts: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        self.request_state = request_state
        self.billing_ambiguity = billing_ambiguity
        self.metadata = dict(metadata or {})
        self.transport_attempts = tuple(transport_attempts)

    def with_attempts(self, attempts: list[dict[str, Any]]) -> "ProviderError":
        return ProviderError(
            str(self),
            category=self.category,
            retryable=self.retryable,
            http_status=self.http_status,
            retry_after_seconds=self.retry_after_seconds,
            request_state=self.request_state,
            billing_ambiguity=self.billing_ambiguity,
            metadata=self.metadata,
            transport_attempts=tuple(attempts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "http_status": self.http_status,
            "retry_after_seconds": self.retry_after_seconds,
            "metadata": self.metadata,
            "transport_attempts": list(self.transport_attempts),
        }


@dataclass(frozen=True)
class BackendCapabilities:
    chat_messages: bool | None = True
    structured_output: bool | None = None
    native_json_schema: bool | None = None
    model_seed: bool | None = None
    token_usage: bool | None = None
    latency: bool | None = True
    reasoning_controls: bool | None = None
    context_window_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_messages": self.chat_messages,
            "structured_output": self.structured_output,
            "native_json_schema": self.native_json_schema,
            "model_seed": self.model_seed,
            "token_usage": self.token_usage,
            "latency": self.latency,
            "reasoning_controls": self.reasoning_controls,
            "context_window_tokens": self.context_window_tokens,
        }


@dataclass(frozen=True)
class DecodingParameters:
    temperature: float = 0.0
    max_tokens: int = 768
    top_p: float = 1.0
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[dict[str, str], ...]
    response_schema: dict[str, Any] | None
    decoding: DecodingParameters


@dataclass(frozen=True)
class ModelResponse:
    content: str
    token_usage: dict[str, int | float | None] = field(default_factory=dict)
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelBackend(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one response without knowing anything about the simulator."""

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "capabilities": self.capabilities().to_dict(),
        }

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()
