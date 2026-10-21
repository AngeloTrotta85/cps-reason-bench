from __future__ import annotations

from cps_reason_bench.models.base import (
    BackendCapabilities,
    ModelBackend,
    ModelRequest,
    ModelResponse,
)
from cps_reason_bench.models.registry import BackendBuildConfig


class CustomBackend(ModelBackend):
    provider = "custom-test"

    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            '{"actions": [], "callback": null, "subscriptions": [], '
            '"context_requests": [], "reasoning": null}',
            {"total_tokens": 1},
            0.001,
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            chat_messages=True,
            structured_output=True,
            native_json_schema=False,
            model_seed=False,
            token_usage=True,
            latency=True,
            reasoning_controls=False,
        )


def create_backend(config: BackendBuildConfig) -> ModelBackend:
    return CustomBackend(config.model)

