"""Generic OpenAI chat-completions adapter with benchmark-safe controls."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping

from cps_reason_bench.config import ConfigError
from cps_reason_bench.models.base import (
    BackendCapabilities,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    ProviderError,
)
from cps_reason_bench.models.security import (
    redact_secrets,
    safe_header_names,
    sanitize_url,
)


OUTPUT_BUDGET_EXHAUSTION_RATIO = 0.90
GOOGLE_MODEL_SEED_MODULUS = 2**31
GOOGLE_MODEL_SEED_ADAPTATION = "google_nonnegative_int32_modulo"


@dataclass(frozen=True)
class HTTPJSONResponse:
    payload: dict[str, Any]
    headers: dict[str, str]
    status: int


class OpenAICompatibleBackend(ModelBackend):
    """One backend for OpenRouter and direct OpenAI-compatible endpoints."""

    provider = "openai-compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float = 120.0,
        *,
        provider: str = "custom",
        routing: Mapping[str, Any] | None = None,
        request_parameters: Mapping[str, Any] | None = None,
        structured_output_mode: str = "prompt_only",
        transport_retry: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        declared_capabilities: Mapping[str, Any] | None = None,
        pricing: Mapping[str, Any] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: random.Random | None = None,
    ) -> None:
        if structured_output_mode not in {"prompt_only", "json_object", "json_schema"}:
            raise ConfigError(
                "structured_output_mode must be prompt_only, json_object, or json_schema"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.configured_provider = str(provider).lower()
        self.api_key_env = api_key_env
        self._api_key = api_key or (
            os.environ.get(api_key_env) if api_key_env else None
        )
        self.timeout_seconds = timeout_seconds
        self.routing = _normalized_routing(self.configured_provider, routing)
        self.request_parameters = _normalized_request_parameters(request_parameters)
        self.structured_output_mode = structured_output_mode
        self.transport_retry = _normalized_transport_retry(transport_retry)
        self._headers = {str(key): str(value) for key, value in (headers or {}).items()}
        self.declared_capabilities = dict(declared_capabilities or {})
        self.pricing = dict(pricing or {})
        self.model_metadata = dict(model_metadata or {})
        self._sleeper = sleeper
        self._random = random_source or random.Random()

    def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self._request_payload(request)
        headers = dict(self._headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        transport_attempts: list[dict[str, Any]] = []
        response: HTTPJSONResponse | None = None
        max_attempts = int(self.transport_retry["max_attempts"])
        for attempt in range(1, max_attempts + 1):
            try:
                response = _request_json(
                    endpoint,
                    payload,
                    self.timeout_seconds,
                    headers,
                    provider=self.configured_provider,
                )
                finish_error = self._provider_finish_error(
                    response,
                    request=request,
                    payload=payload,
                    endpoint=endpoint,
                    started_at=started_at,
                    latency_seconds=time.perf_counter() - started,
                )
                if finish_error is not None:
                    raise finish_error
                transport_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "success",
                        "http_status": response.status,
                        "delay_before_next_seconds": None,
                    }
                )
                break
            except ProviderError as exc:
                retry = bool(
                    exc.retryable
                    and not exc.billing_ambiguity
                    and attempt < max_attempts
                )
                delay = self._retry_delay(attempt, exc) if retry else None
                attempt_record = {
                    "attempt": attempt,
                    "status": "error",
                    "category": exc.category,
                    "http_status": exc.http_status,
                    "request_state": exc.request_state,
                    "billing_ambiguity": exc.billing_ambiguity,
                    "retry_after_seconds": exc.retry_after_seconds,
                    "delay_before_next_seconds": delay,
                }
                for name in ("request_id", "finish_reason", "normalized_usage"):
                    if exc.metadata.get(name) is not None:
                        attempt_record[name] = exc.metadata[name]
                transport_attempts.append(attempt_record)
                if not retry:
                    raise exc.with_attempts(transport_attempts) from exc
                self._sleeper(float(delay))
        if response is None:  # pragma: no cover - loop always returns or raises
            raise ProviderError(
                "Provider request did not complete",
                category="transient_provider_failure",
                retryable=True,
            )

        latency = time.perf_counter() - started
        body = response.payload
        usage = normalize_usage(body.get("usage"))
        returned_model = _optional_text(body.get("model"))
        actual_provider = _actual_provider(body, response.headers)
        requested_provider = _requested_provider(self.routing)
        requested_route = _requested_route(self.routing)
        actual_route = _actual_route(body, response.headers)
        seed_transport = _seed_transport_metadata(
            request.decoding.seed,
            payload.get("seed"),
            requested_provider,
        )
        accepted_models = {
            self.model,
            str(self.routing.get("expected_canonical_model") or self.model),
        }
        model_mismatch = bool(returned_model and returned_model not in accepted_models)
        route_mismatch = bool(
            requested_provider
            and actual_provider
            and actual_provider.lower() != requested_provider.lower()
        )
        route_mismatch = route_mismatch or bool(
            requested_route
            and actual_route
            and "/" in str(actual_route)
            and str(actual_route).casefold() != requested_route.casefold()
        )
        fallback_detected = bool(
            model_mismatch or route_mismatch or _provider_fallback_flag(body)
        )
        try:
            choice = body["choices"][0]
            message = choice["message"]
            content = _message_content(message.get("content"))
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "Compatible API response did not contain choices[0].message.content",
                category="invalid_provider_response",
                retryable=False,
                http_status=response.status,
                request_state="response_received",
                billing_ambiguity=True,
                metadata=_safe_response_error_metadata(
                    body, response.headers, self._api_key
                ),
                transport_attempts=tuple(transport_attempts),
            ) from exc
        finish_reason = _optional_text(choice.get("finish_reason"))
        reasoning_tokens = usage.get("reasoning_tokens")
        completion_tokens = usage.get("completion_tokens")
        observed_output_tokens = max(
            (
                float(value)
                for value in (reasoning_tokens, completion_tokens)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ),
            default=0.0,
        )
        output_budget_exhausted = bool(
            not content.strip()
            and finish_reason
            and finish_reason.lower() == "length"
            and observed_output_tokens
            >= request.decoding.max_tokens * OUTPUT_BUDGET_EXHAUSTION_RATIO
        )
        if not content.strip() and not output_budget_exhausted:
            error_metadata = _safe_response_error_metadata(
                body, response.headers, self._api_key
            )
            error_metadata.update(
                {
                    "configured_provider": self.configured_provider,
                    "requested_model": self.model,
                    "returned_model": returned_model,
                    "requested_provider": requested_provider,
                    "actual_provider": actual_provider,
                    "requested_route": requested_route,
                    "actual_route": actual_route,
                    "fallback_detected": fallback_detected,
                    "routing_mode": self.routing["mode"],
                    "routing_preferences": dict(self.routing),
                    "structured_output_mode": self.structured_output_mode,
                    "response_healing_requested": False,
                    "request_parameter_policy": dict(self.request_parameters),
                    "sent_request_parameters": _sent_request_parameters(payload),
                    "omitted_request_parameters": _omitted_request_parameters(payload),
                    **seed_transport,
                    "max_output_tokens_requested": request.decoding.max_tokens,
                    "latency_seconds": latency,
                    "request_started_at_utc": started_at,
                    "response_received_at_utc": datetime.now(timezone.utc).isoformat(),
                    "finish_reason": finish_reason,
                }
            )
            raise ProviderError(
                "Compatible API returned an empty model response",
                category="empty_response",
                retryable=False,
                http_status=response.status,
                request_state="response_received",
                billing_ambiguity=True,
                metadata=error_metadata,
                transport_attempts=tuple(transport_attempts),
            )
        if self.routing["mode"] == "benchmark" and fallback_detected:
            raise ProviderError(
                "Benchmark routing returned a different model or provider than requested",
                category="routing_mismatch",
                retryable=False,
                http_status=response.status,
                request_state="response_received",
                billing_ambiguity=True,
                metadata={
                    "requested_model": self.model,
                    "returned_model": returned_model,
                    "requested_provider": requested_provider,
                    "actual_provider": actual_provider,
                    "requested_route": requested_route,
                    "actual_route": actual_route,
                    "fallback_detected": fallback_detected,
                    "request_id": _request_id(body, response.headers),
                    "normalized_usage": usage,
                },
                transport_attempts=tuple(transport_attempts),
            )

        observed = {
            "chat_completions": True,
            "seed": (
                True if "seed" in payload and body.get("seed") is not None else None
            ),
            # Sending a parameter is not evidence that the provider applied it.
            "temperature": None,
            "top_p": None,
            "max_output_tokens": None,
            "reasoning_control": (
                True if isinstance(payload.get("reasoning"), Mapping) else None
            ),
            "response_format": None,
            "json_object": None,
            "json_schema": None,
            "usage": any(value is not None for value in usage.values()),
            "cost_metadata": usage["reported_cost_usd"] is not None,
        }
        metadata = {
            "provider": self.configured_provider,
            "api_compatibility": "openai_chat_completions",
            "base_url": sanitize_url(self.base_url),
            "endpoint": sanitize_url(endpoint),
            "requested_model": self.model,
            "returned_model": returned_model,
            "requested_provider": requested_provider,
            "actual_provider": actual_provider,
            "requested_route": requested_route,
            "actual_route": actual_route,
            "routing_mode": self.routing["mode"],
            "routing_preferences": dict(self.routing),
            "fallback_allowed": self.routing.get("allow_fallbacks"),
            "fallback_detected": fallback_detected,
            "quantization": _quantization(body),
            "request_id": _request_id(body, response.headers),
            "provider_generation_id": _optional_text(body.get("id")),
            "system_fingerprint": _optional_text(body.get("system_fingerprint")),
            "model_version": _optional_text(body.get("model_version")),
            "model_date": _optional_text(body.get("model_date")),
            "seed_acknowledged": (
                body.get("seed") is not None
                if self.request_parameters["seed"]
                else None
            ),
            "structured_output_mode": self.structured_output_mode,
            "requested_response_format": payload.get("response_format"),
            "response_healing_requested": False,
            "provider_side_response_repair": "not_requested_and_not_observable",
            "request_parameter_policy": dict(self.request_parameters),
            "sent_request_parameters": _sent_request_parameters(payload),
            **seed_transport,
            "provider_acknowledged_request_parameters": (
                ["seed"] if body.get("seed") is not None and "seed" in payload else []
            ),
            "reasoning_control": (
                dict(payload["reasoning"])
                if isinstance(payload.get("reasoning"), Mapping)
                else None
            ),
            "omitted_request_parameters": _omitted_request_parameters(payload),
            "declared_capabilities": dict(self.declared_capabilities),
            "observed_capabilities": observed,
            "capability_mismatches": _capability_mismatches(
                self.declared_capabilities, observed
            ),
            "normalized_usage": usage,
            "raw_provider_metadata": _safe_response_metadata(
                body, response.headers, self._api_key
            ),
            "transport_attempts": transport_attempts,
            "transport_retry_count": max(0, len(transport_attempts) - 1),
            "billing_ambiguity": any(
                bool(item.get("billing_ambiguity")) for item in transport_attempts
            ),
            "request_started_at_utc": started_at,
            "response_received_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": response.status,
            "finish_reason": finish_reason,
            "model_output_status": (
                "output_budget_exhausted"
                if output_budget_exhausted
                else "content_returned"
            ),
            "output_budget_exhausted": output_budget_exhausted,
        }
        return ModelResponse(content, usage, latency, metadata)

    def _provider_finish_error(
        self,
        response: HTTPJSONResponse,
        *,
        request: ModelRequest,
        payload: Mapping[str, Any],
        endpoint: str,
        started_at: str,
        latency_seconds: float,
    ) -> ProviderError | None:
        body = response.payload
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        finish_reason = (
            _optional_text(choice.get("finish_reason"))
            if isinstance(choice, Mapping)
            else None
        )
        provider_error, error_location = _embedded_provider_error(body)
        if provider_error is None and (
            not finish_reason or finish_reason.lower() != "error"
        ):
            return None

        usage = normalize_usage(body.get("usage"))
        explicitly_unbilled = _usage_is_explicit_zero(usage)
        category = _in_band_provider_error_category(provider_error)
        retryable_category = category in {
            "rate_limit",
            "timeout",
            "transient_provider_failure",
        }
        requested_provider = _requested_provider(self.routing)
        returned_model = _optional_text(body.get("model"))
        actual_provider = _actual_provider(body, response.headers)
        metadata = _safe_response_error_metadata(body, response.headers, self._api_key)
        metadata.update(
            {
                "provider": self.configured_provider,
                "endpoint": sanitize_url(endpoint),
                "configured_provider": self.configured_provider,
                "requested_model": self.model,
                "returned_model": returned_model,
                "requested_provider": requested_provider,
                "actual_provider": actual_provider,
                "actual_route": _actual_route(body, response.headers),
                "routing_mode": self.routing["mode"],
                "routing_preferences": dict(self.routing),
                "structured_output_mode": self.structured_output_mode,
                "response_healing_requested": False,
                "request_parameter_policy": dict(self.request_parameters),
                "sent_request_parameters": _sent_request_parameters(payload),
                "omitted_request_parameters": _omitted_request_parameters(payload),
                **_seed_transport_metadata(
                    request.decoding.seed,
                    payload.get("seed"),
                    requested_provider,
                ),
                "max_output_tokens_requested": request.decoding.max_tokens,
                "latency_seconds": latency_seconds,
                "request_started_at_utc": started_at,
                "response_received_at_utc": datetime.now(timezone.utc).isoformat(),
                "finish_reason": finish_reason,
                "provider_error_location": error_location,
                "normalized_usage": usage,
            }
        )
        return ProviderError(
            "Model provider returned an in-band generation error",
            category=category,
            retryable=retryable_category and explicitly_unbilled,
            http_status=response.status,
            retry_after_seconds=_provider_retry_delay(response.headers),
            request_state="response_received",
            billing_ambiguity=usage["reported_cost_usd"] is None,
            metadata=metadata,
        )

    def _request_payload(self, request: ModelRequest) -> dict[str, Any]:
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
        }
        supplied_forbidden = forbidden & set(request.decoding.extra)
        if supplied_forbidden:
            raise ConfigError(
                "Decoding extras cannot override routing or output controls: "
                f"{sorted(supplied_forbidden)}"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(request.messages),
            **request.decoding.extra,
        }
        if self.request_parameters["temperature"]:
            payload["temperature"] = request.decoding.temperature
        if self.request_parameters["top_p"]:
            payload["top_p"] = request.decoding.top_p
        payload[
            str(self.request_parameters["max_tokens_field"])
        ] = request.decoding.max_tokens
        if self.request_parameters["seed"] and request.decoding.seed is not None:
            payload["seed"] = _provider_compatible_model_seed(
                request.decoding.seed,
                _requested_provider(self.routing),
            )
        reasoning_effort = self.request_parameters.get("reasoning_effort")
        if reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": bool(self.request_parameters.get("reasoning_exclude", True)),
            }
        if self.structured_output_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.structured_output_mode == "json_schema":
            if request.response_schema is None:
                raise ConfigError("json_schema output mode requires a response schema")
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "cps_reason_decision",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        if (
            self.configured_provider == "openrouter"
            and self.routing["mode"] != "direct"
        ):
            provider: dict[str, Any] = {}
            if self.routing["mode"] == "benchmark":
                provider.update(
                    {
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                )
            else:
                for key in ("allow_fallbacks", "require_parameters"):
                    if self.routing.get(key) is not None:
                        provider[key] = bool(self.routing[key])
            for source, target in (
                ("provider_order", "order"),
                ("only", "only"),
                ("quantizations", "quantizations"),
            ):
                if self.routing.get(source):
                    provider[target] = list(self.routing[source])
            if self.routing.get("data_collection") is not None:
                provider["data_collection"] = self.routing["data_collection"]
            if self.routing.get("zdr") is not None:
                provider["zdr"] = bool(self.routing["zdr"])
            if provider:
                payload["provider"] = provider
        return payload

    def _retry_delay(self, attempt: int, error: ProviderError) -> float:
        if error.retry_after_seconds is not None:
            return min(
                float(self.transport_retry["max_backoff_seconds"]),
                max(0.0, float(error.retry_after_seconds)),
            )
        base = float(self.transport_retry["base_backoff_seconds"])
        maximum = float(self.transport_retry["max_backoff_seconds"])
        jitter = float(self.transport_retry["jitter_seconds"])
        return min(maximum, base * (2 ** max(0, attempt - 1))) + self._random.uniform(
            0.0, jitter
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "provider": self.configured_provider,
            "api_compatibility": "openai_chat_completions",
            "base_url": sanitize_url(self.base_url),
            "api_key_env": self.api_key_env,
            "api_key_status": "configured" if self._api_key else "missing",
            "timeout_seconds": self.timeout_seconds,
            "routing": dict(self.routing),
            "request_parameters": dict(self.request_parameters),
            "structured_output_mode": self.structured_output_mode,
            "transport_retry": dict(self.transport_retry),
            "custom_header_names": safe_header_names(self._headers),
            "declared_capabilities": dict(self.declared_capabilities),
            "pricing": dict(self.pricing),
            "model_metadata": dict(self.model_metadata),
            "format_adapter": "openai_compatible_chat_completions",
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
            model_seed=None,
            token_usage=None,
            latency=True,
            reasoning_controls=None,
            context_window_tokens=None,
        )


def normalize_usage(value: Any) -> dict[str, int | float | None]:
    usage = value if isinstance(value, Mapping) else {}
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, Mapping) else {}
    completion_details = (
        completion_details if isinstance(completion_details, Mapping) else {}
    )
    return {
        "prompt_tokens": _number_or_none(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        ),
        "completion_tokens": _number_or_none(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
        "reasoning_tokens": _number_or_none(
            completion_details.get("reasoning_tokens", usage.get("reasoning_tokens"))
        ),
        "cached_tokens": _number_or_none(
            prompt_details.get(
                "cached_tokens",
                usage.get("cached_tokens", usage.get("cache_read_input_tokens")),
            )
        ),
        "total_tokens": _number_or_none(usage.get("total_tokens")),
        "reported_cost_usd": _number_or_none(
            usage.get("cost", usage.get("reported_cost_usd"))
        ),
    }


def _usage_is_explicit_zero(usage: Mapping[str, Any]) -> bool:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "total_tokens",
        "reported_cost_usd",
    )
    return all(usage.get(name) == 0 for name in fields)


def _request_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: Mapping[str, str],
    *,
    provider: str,
) -> HTTPJSONResponse:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **dict(headers)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            response_headers = {
                str(key).lower(): str(value) for key, value in response.headers.items()
            }
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error_headers = {
            str(key).lower(): str(value) for key, value in exc.headers.items()
        }
        raise _http_provider_error(
            int(exc.code), detail, error_headers, provider
        ) from exc
    except TimeoutError as exc:
        raise ProviderError(
            "Model provider request timed out",
            category="timeout",
            retryable=True,
            request_state="sent_no_response",
            billing_ambiguity=True,
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ProviderError(
            "Could not reach model provider",
            category="connection_error",
            retryable=True,
            request_state="unknown",
            billing_ambiguity=True,
        ) from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            "Model provider returned invalid JSON",
            category="invalid_provider_response",
            retryable=False,
            http_status=status,
            request_state="response_received",
            billing_ambiguity=True,
            metadata={
                "provider": provider,
                "request_id": _header_value(
                    response_headers, "x-request-id", "request-id", "cf-ray"
                ),
            },
        ) from exc
    if not isinstance(value, dict):
        raise ProviderError(
            "Model provider response must be a JSON object",
            category="invalid_provider_response",
            retryable=False,
            http_status=status,
            request_state="response_received",
            billing_ambiguity=True,
            metadata={
                "provider": provider,
                "request_id": _header_value(
                    response_headers, "x-request-id", "request-id", "cf-ray"
                ),
            },
        )
    return HTTPJSONResponse(value, response_headers, status)


def _http_provider_error(
    status: int, body: str, headers: Mapping[str, str], provider: str
) -> ProviderError:
    categories = {
        400: (
            "permanent_configuration_error",
            False,
            "Provider rejected the request configuration",
        ),
        401: ("authentication_failure", False, "Provider authentication failed"),
        402: ("insufficient_credit", False, "Provider account has insufficient credit"),
        403: ("permission_failure", False, "Provider denied the request"),
        404: ("model_unavailable", False, "Requested provider model was not found"),
        408: ("timeout", True, "Provider request timed out"),
        409: (
            "transient_provider_failure",
            True,
            "Provider reported a transient conflict",
        ),
        429: ("rate_limit", True, "Provider rate limit was reached"),
    }
    if status in categories:
        category, retryable, message = categories[status]
    elif 500 <= status <= 599:
        category, retryable, message = (
            "transient_provider_failure",
            True,
            "Provider reported a transient server failure",
        )
    else:
        category, retryable, message = (
            "permanent_configuration_error",
            False,
            "Provider rejected the request",
        )
    metadata = {
        "provider": provider,
        "error": _safe_error_body(body),
        "request_id": _header_value(headers, "x-request-id", "request-id", "cf-ray"),
    }
    failed_usage = _safe_error_usage(body)
    if failed_usage is not None:
        metadata["normalized_usage"] = failed_usage
    return ProviderError(
        message,
        category=category,
        retryable=retryable,
        http_status=status,
        retry_after_seconds=_provider_retry_delay(headers),
        request_state="response_received",
        billing_ambiguity=False,
        metadata=metadata,
    )


def _normalized_routing(
    provider: str, value: Mapping[str, Any] | None
) -> dict[str, Any]:
    data = dict(value or {})
    mode = str(data.get("mode", "dynamic" if provider == "openrouter" else "direct"))
    if provider != "openrouter" and mode != "direct":
        raise ConfigError(f"Provider {provider!r} supports routing mode 'direct' only")
    if mode not in {"dynamic", "benchmark", "direct"}:
        raise ConfigError(f"Unsupported routing mode: {mode}")
    if mode == "benchmark":
        if data.get("allow_fallbacks", False) is not False:
            raise ConfigError("Benchmark routing requires allow_fallbacks=false")
        if data.get("require_parameters", True) is not True:
            raise ConfigError("Benchmark routing requires require_parameters=true")
    if data.get("data_collection") not in {None, "allow", "deny"}:
        raise ConfigError("routing.data_collection must be allow or deny")
    if data.get("zdr") not in {None, True, False}:
        raise ConfigError("routing.zdr must be boolean")
    return {
        "mode": mode,
        "provider_order": list(data.get("provider_order", [])),
        "only": list(data.get("only", [])),
        "allow_fallbacks": (
            False if mode == "benchmark" else data.get("allow_fallbacks")
        ),
        "require_parameters": (
            True if mode == "benchmark" else data.get("require_parameters")
        ),
        "quantizations": list(data.get("quantizations", [])),
        "data_collection": data.get("data_collection"),
        "zdr": data.get("zdr"),
        "expected_provider": data.get("expected_provider"),
        "expected_route": data.get("expected_route"),
        "expected_canonical_model": data.get("expected_canonical_model"),
    }


def _normalized_request_parameters(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data = dict(value or {})
    unknown = set(data) - {
        "temperature",
        "top_p",
        "seed",
        "max_tokens_field",
        "reasoning_effort",
        "reasoning_exclude",
    }
    if unknown:
        raise ConfigError(f"Unsupported request parameter controls: {sorted(unknown)}")
    for key in ("temperature", "top_p", "seed", "reasoning_exclude"):
        if data.get(key) not in {None, True, False}:
            raise ConfigError(f"request_parameters.{key} must be boolean")
    result = {
        "temperature": data.get("temperature", True),
        "top_p": data.get("top_p", True),
        "seed": data.get("seed", True),
        "max_tokens_field": str(data.get("max_tokens_field", "max_tokens")),
        "reasoning_effort": data.get("reasoning_effort"),
        "reasoning_exclude": data.get("reasoning_exclude", True),
    }
    if result["reasoning_effort"] not in {
        None,
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ConfigError("request_parameters.reasoning_effort is invalid")
    if result["max_tokens_field"] not in {
        "max_tokens",
        "max_completion_tokens",
    }:
        raise ConfigError(
            "request_parameters.max_tokens_field must be max_tokens or "
            "max_completion_tokens"
        )
    return result


def _normalized_transport_retry(value: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(value or {})
    result = {
        "max_attempts": int(data.get("max_attempts", 3)),
        "base_backoff_seconds": float(data.get("base_backoff_seconds", 1.0)),
        "max_backoff_seconds": float(data.get("max_backoff_seconds", 30.0)),
        "jitter_seconds": float(data.get("jitter_seconds", 0.25)),
    }
    if result["max_attempts"] < 1 or result["max_attempts"] > 10:
        raise ConfigError("transport_retry.max_attempts must be between 1 and 10")
    if any(result[key] < 0 for key in result if key != "max_attempts"):
        raise ConfigError("transport retry delays cannot be negative")
    return result


def _safe_error_body(body: str) -> dict[str, Any] | None:
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    error = value.get("error", value)
    if not isinstance(error, Mapping):
        return None
    allowed = {
        key: error.get(key)
        for key in ("type", "code", "param", "provider_name", "provider_code")
        if error.get(key) is not None
    }
    return redact_secrets(allowed)


def _safe_error_usage(body: str) -> dict[str, int | float | None] | None:
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    usage = normalize_usage(value.get("usage"))
    return usage if any(item is not None for item in usage.values()) else None


def _safe_response_metadata(
    body: Mapping[str, Any], headers: Mapping[str, str], api_key: str | None
) -> dict[str, Any]:
    allowed = {
        key: body.get(key)
        for key in (
            "id",
            "model",
            "provider",
            "created",
            "system_fingerprint",
            "model_version",
            "model_date",
            "seed",
            "usage",
            "openrouter_metadata",
        )
        if body.get(key) is not None
    }
    allowed["response_headers"] = {
        key: value
        for key, value in headers.items()
        if key
        in {
            "x-request-id",
            "request-id",
            "cf-ray",
            "x-openrouter-provider",
            "x-openrouter-generation-id",
            "x-ratelimit-limit-requests",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
        }
    }
    return redact_secrets(allowed, (api_key or "",))


def _safe_response_error_metadata(
    body: Mapping[str, Any], headers: Mapping[str, str], api_key: str | None
) -> dict[str, Any]:
    metadata = _safe_response_metadata(body, headers, api_key)
    metadata["request_id"] = _request_id(body, headers)
    metadata["response_shape"] = _response_shape(body)
    provider_error, error_location = _embedded_provider_error(body)
    if provider_error is not None:
        metadata["provider_error"] = _safe_provider_error(provider_error, api_key)
        metadata["provider_error_location"] = error_location
    usage = normalize_usage(body.get("usage"))
    if any(item is not None for item in usage.values()):
        metadata["normalized_usage"] = usage
    return metadata


def _embedded_provider_error(
    body: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    error = body.get("error")
    if isinstance(error, Mapping):
        return error, "top_level"
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        error = choices[0].get("error")
        if isinstance(error, Mapping):
            return error, "choice"
    return None, None


def _safe_provider_error(
    error: Mapping[str, Any], api_key: str | None
) -> dict[str, Any]:
    allowed = {
        key: error.get(key)
        for key in (
            "type",
            "code",
            "param",
            "provider_name",
            "provider_code",
            "error_type",
        )
        if error.get(key) is not None
    }
    nested = error.get("metadata")
    if isinstance(nested, Mapping):
        for key in ("error_type", "provider_code", "provider_name"):
            if key not in allowed and nested.get(key) is not None:
                allowed[key] = nested[key]
    return redact_secrets(allowed, (api_key or "",))


def _response_shape(body: Mapping[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = first_choice.get("message") if isinstance(first_choice, Mapping) else None
    return {
        "top_level_error_present": isinstance(body.get("error"), Mapping),
        "choices_type": type(choices).__name__,
        "choices_count": len(choices) if isinstance(choices, list) else None,
        "first_choice_type": type(first_choice).__name__,
        "message_type": type(message).__name__,
        "content_field_present": isinstance(message, Mapping) and "content" in message,
    }


def _in_band_provider_error_category(error: Mapping[str, Any] | None) -> str:
    if error is None:
        return "transient_provider_failure"
    nested = error.get("metadata")
    error_type = _optional_text(error.get("error_type"))
    if error_type is None and isinstance(nested, Mapping):
        error_type = _optional_text(nested.get("error_type"))
    error_type = (error_type or _optional_text(error.get("type")) or "").lower()
    code_value = error.get("code")
    try:
        code = int(code_value) if code_value is not None else None
    except (TypeError, ValueError):
        code = None

    if error_type == "rate_limit_exceeded" or code == 429:
        return "rate_limit"
    if error_type == "timeout" or code in {408, 504}:
        return "timeout"
    if error_type == "authentication" or code == 401:
        return "authentication_failure"
    if error_type == "payment_required" or code == 402:
        return "insufficient_credit"
    if error_type == "permission_denied" or code == 403:
        return "permission_failure"
    if error_type == "not_found" or code == 404:
        return "model_unavailable"
    if error_type in {
        "invalid_prompt",
        "invalid_request",
        "payload_too_large",
        "precondition_failed",
        "string_too_long",
        "unprocessable",
    } or (code is not None and 400 <= code < 500):
        return "permanent_configuration_error"
    return "transient_provider_failure"


def _actual_provider(body: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
    direct = _optional_text(body.get("provider")) or _header_value(
        headers, "x-openrouter-provider"
    )
    if direct:
        return direct
    metadata = body.get("openrouter_metadata")
    if isinstance(metadata, Mapping):
        return _optional_text(metadata.get("provider_name", metadata.get("provider")))
    return None


def _actual_route(body: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
    metadata = body.get("openrouter_metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("endpoint", metadata.get("route"))
        if value is not None:
            return str(value)
    return _actual_provider(body, headers)


def _quantization(body: Mapping[str, Any]) -> str | None:
    for source in (body, body.get("openrouter_metadata")):
        if isinstance(source, Mapping):
            value = source.get("quantization")
            if value is not None:
                return str(value)
    return None


def _provider_fallback_flag(body: Mapping[str, Any]) -> bool:
    metadata = body.get("openrouter_metadata")
    if not isinstance(metadata, Mapping):
        return False
    return bool(
        metadata.get("fallback_used")
        or metadata.get("fallback")
        or metadata.get("fallbacks")
    )


def _requested_provider(routing: Mapping[str, Any]) -> str | None:
    expected = routing.get("expected_provider")
    if expected:
        return str(expected)
    values = routing.get("provider_order") or routing.get("only") or []
    value = str(values[0]) if values else None
    return value.split("/", 1)[0] if value else None


def _requested_route(routing: Mapping[str, Any]) -> str | None:
    expected = routing.get("expected_route")
    if expected:
        return str(expected)
    values = routing.get("only") or routing.get("provider_order") or []
    return str(values[0]) if values else None


def _provider_compatible_model_seed(
    requested_seed: int, requested_provider: str | None
) -> int:
    if requested_provider and requested_provider.casefold() == "google":
        # Google's GenerationConfig seed is int32. Keep the benchmark's stable
        # logical seed and adapt only the value crossing this transport boundary.
        return requested_seed % GOOGLE_MODEL_SEED_MODULUS
    return requested_seed


def _seed_transport_metadata(
    requested_seed: int | None,
    sent_seed: Any,
    requested_provider: str | None,
) -> dict[str, Any]:
    if requested_seed is None or not isinstance(sent_seed, int):
        return {
            "requested_model_seed": requested_seed,
            "sent_model_seed": None,
            "model_seed_transport_adaptation": None,
        }
    adapted = sent_seed != requested_seed
    return {
        "requested_model_seed": requested_seed,
        "sent_model_seed": sent_seed,
        "model_seed_transport_adaptation": (
            GOOGLE_MODEL_SEED_ADAPTATION
            if adapted
            and requested_provider
            and requested_provider.casefold() == "google"
            else None
        ),
    }


def _request_id(body: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
    return _header_value(
        headers, "x-request-id", "request-id", "cf-ray"
    ) or _optional_text(body.get("request_id"))


def _header_value(headers: Mapping[str, str], *names: str) -> str | None:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = _header_value(headers, "retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            return max(
                0.0,
                (parsed - datetime.now(parsed.tzinfo or timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return None


def _provider_retry_delay(headers: Mapping[str, str]) -> float | None:
    explicit = _retry_after(headers)
    if explicit is not None:
        return explicit
    for name in (
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ):
        parsed = _duration_or_epoch_seconds(_header_value(headers, name))
        if parsed is not None:
            return parsed
    return None


def _duration_or_epoch_seconds(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value.strip().lower()
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    for suffix, multiplier in multipliers.items():
        if normalized.endswith(suffix):
            try:
                return max(0.0, float(normalized[: -len(suffix)]) * multiplier)
            except ValueError:
                return None
    try:
        numeric = float(normalized)
    except ValueError:
        return None
    if numeric > 1_000_000_000:
        return max(0.0, numeric - time.time())
    return max(0.0, numeric)


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _sent_request_parameters(payload: Mapping[str, Any]) -> list[str]:
    return sorted(
        key
        for key in (
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "reasoning",
            "include_reasoning",
        )
        if key in payload
    )


def _omitted_request_parameters(payload: Mapping[str, Any]) -> list[str]:
    return sorted(
        key
        for key in (
            "temperature",
            "top_p",
            "seed",
            "reasoning",
            "include_reasoning",
        )
        if key not in payload
    )


def _capability_mismatches(
    declared: Mapping[str, Any], observed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for name, declaration in declared.items():
        observation = observed.get(name)
        if (
            declaration in {True, False}
            and observation in {True, False}
            and declaration != observation
        ):
            output.append(
                {"capability": name, "declared": declaration, "observed": observation}
            )
    return output


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _optional_text(value: Any) -> str | None:
    return str(value) if value is not None and str(value) else None
