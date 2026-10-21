"""Native Ollama adapter with JSON-schema output support."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from cps_reason_bench.models.base import (
    BackendError,
    BackendCapabilities,
    ModelBackend,
    ModelRequest,
    ModelResponse,
)


class OllamaBackend(ModelBackend):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        think: bool = False,
        structured_output_mode: str = "json_schema",
    ) -> None:
        if structured_output_mode not in {"prompt_only", "json_object", "json_schema"}:
            raise ValueError(
                "structured_output_mode must be prompt_only, json_object, or json_schema"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.structured_output_mode = structured_output_mode

    def generate(self, request: ModelRequest) -> ModelResponse:
        options: dict[str, Any] = {
            "temperature": request.decoding.temperature,
            "top_p": request.decoding.top_p,
            "num_predict": request.decoding.max_tokens,
            **request.decoding.extra,
        }
        if request.decoding.seed is not None:
            options["seed"] = request.decoding.seed
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(request.messages),
            "stream": False,
            "think": self.think,
            "options": options,
        }
        if self.structured_output_mode == "json_object":
            payload["format"] = "json"
        elif self.structured_output_mode == "json_schema":
            payload["format"] = request.response_schema or "json"
        started = time.perf_counter()
        response = _post_json(
            f"{self.base_url}/api/chat", payload, self.timeout_seconds, headers={}
        )
        latency = time.perf_counter() - started
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BackendError("Ollama response did not contain message.content") from exc
        usage = {
            "prompt_tokens": int(response.get("prompt_eval_count", 0)),
            "completion_tokens": int(response.get("eval_count", 0)),
            "total_tokens": int(response.get("prompt_eval_count", 0))
            + int(response.get("eval_count", 0)),
        }
        return ModelResponse(
            str(content),
            usage,
            latency,
            {
                "done_reason": response.get("done_reason"),
                "load_duration_ns": response.get("load_duration"),
                "eval_duration_ns": response.get("eval_duration"),
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "think": self.think,
            "structured_output_mode": self.structured_output_mode,
            "format_adapter": (
                "ollama_prompt_only_chat"
                if self.structured_output_mode == "prompt_only"
                else "ollama_native_chat_with_json_schema"
            ),
        }

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            chat_messages=True,
            structured_output=(
                None if self.structured_output_mode == "prompt_only" else True
            ),
            native_json_schema=(
                True if self.structured_output_mode == "json_schema" else None
            ),
            model_seed=True,
            token_usage=True,
            latency=True,
            reasoning_controls=True,
            context_window_tokens=None,
        )


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BackendError(f"Model backend returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"Could not reach model backend at {url}: {exc}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BackendError("Model backend returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise BackendError("Model backend response must be a JSON object")
    return value
