from __future__ import annotations

import json
import random
import unittest
from unittest.mock import patch

from cps_reason_bench.config import ConfigError
from cps_reason_bench.models import DecodingParameters, ModelRequest, ProviderError
from cps_reason_bench.models.openai_compatible import (
    HTTPJSONResponse,
    OpenAICompatibleBackend,
    _http_provider_error,
    normalize_usage,
)


MOCK_SECRET = "mock-cloud-secret-never-log"
REQUEST = ModelRequest(
    ({"role": "user", "content": "Return JSON."},),
    {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
    },
    DecodingParameters(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        seed=17,
    ),
)


def _success(
    *,
    model: str = "vendor/model",
    provider: str | None = "provider-a",
    metadata: dict[str, object] | None = None,
    echo_seed: bool = True,
) -> HTTPJSONResponse:
    body: dict[str, object] = {
        "id": "generation-1",
        "model": model,
        "choices": [
            {
                "message": {"content": '{"action":"noop"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
            "cost": 0.00012,
        },
        "system_fingerprint": "fp-test",
    }
    if echo_seed:
        body["seed"] = 17
    if provider is not None:
        body["provider"] = provider
    if metadata is not None:
        body["openrouter_metadata"] = metadata
    return HTTPJSONResponse(body, {"x-request-id": "request-1"}, 200)


def _finish_error_response(*, include_usage: bool = True) -> HTTPJSONResponse:
    response = _success(provider="Google", echo_seed=False)
    response.payload["choices"][0]["message"]["content"] = "{"  # type: ignore[index]
    response.payload["choices"][0]["finish_reason"] = "error"  # type: ignore[index]
    if include_usage:
        response.payload["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
            "cost": 0,
        }
    else:
        response.payload.pop("usage", None)
    return response


def _top_level_error_response(*, include_zero_usage: bool = False) -> HTTPJSONResponse:
    body: dict[str, object] = {
        "error": {
            "code": 502,
            "message": MOCK_SECRET,
            "metadata": {
                "error_type": "provider_unavailable",
                "provider_code": "upstream_invalid_response",
                "unsafe_detail": MOCK_SECRET,
            },
        }
    }
    if include_zero_usage:
        body["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
            "cost": 0,
        }
    return HTTPJSONResponse(body, {"cf-ray": "in-band-request-1"}, 200)


def _backend(**overrides: object) -> OpenAICompatibleBackend:
    values: dict[str, object] = {
        "model": "vendor/model",
        "base_url": "https://example.invalid/v1?unsafe=query",
        "api_key_env": "TEST_CLOUD_API_KEY",
        "provider": "openrouter",
        "routing": {
            "mode": "benchmark",
            "provider_order": ["provider-a"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "structured_output_mode": "prompt_only",
        "transport_retry": {
            "max_attempts": 1,
            "base_backoff_seconds": 0.5,
            "max_backoff_seconds": 5,
            "jitter_seconds": 0,
        },
        "random_source": random.Random(1),
    }
    values.update(overrides)
    return OpenAICompatibleBackend(**values)  # type: ignore[arg-type]


class OpenAICompatiblePayloadTests(unittest.TestCase):
    def test_prompt_only_is_default_and_benchmark_routing_is_explicit(self) -> None:
        payload = _backend()._request_payload(REQUEST)
        self.assertNotIn("response_format", payload)
        self.assertEqual("vendor/model", payload["model"])
        self.assertNotIn("models", payload)
        self.assertEqual(
            {
                "allow_fallbacks": False,
                "require_parameters": True,
                "order": ["provider-a"],
            },
            payload["provider"],
        )

    def test_dynamic_routing_options_are_transmitted_and_labeled(self) -> None:
        backend = _backend(
            routing={
                "mode": "dynamic",
                "provider_order": ["provider-a", "provider-b"],
                "allow_fallbacks": True,
                "require_parameters": False,
            }
        )
        payload = backend._request_payload(REQUEST)
        self.assertEqual(
            {
                "allow_fallbacks": True,
                "require_parameters": False,
                "order": ["provider-a", "provider-b"],
            },
            payload["provider"],
        )
        self.assertEqual("dynamic", backend.metadata()["routing"]["mode"])

    def test_privacy_routing_and_model_specific_parameters_are_explicit(self) -> None:
        backend = _backend(
            routing={
                "mode": "benchmark",
                "provider_order": ["Azure"],
                "only": ["Azure"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
                "zdr": True,
            },
            request_parameters={
                "temperature": False,
                "top_p": False,
                "seed": True,
                "max_tokens_field": "max_completion_tokens",
            },
        )
        payload = backend._request_payload(REQUEST)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(32, payload["max_completion_tokens"])
        self.assertEqual(17, payload["seed"])
        self.assertEqual("deny", payload["provider"]["data_collection"])
        self.assertTrue(payload["provider"]["zdr"])
        self.assertEqual(["Azure"], payload["provider"]["only"])

    def test_fixed_google_route_adapts_unsigned_seed_to_nonnegative_int32(self) -> None:
        requested_seed = 4_232_319_185
        request = ModelRequest(
            REQUEST.messages,
            REQUEST.response_schema,
            DecodingParameters(max_tokens=32, seed=requested_seed),
        )
        google = _backend(
            routing={
                "mode": "benchmark",
                "provider_order": ["Google"],
                "only": ["Google"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        )._request_payload(request)
        azure = _backend(
            routing={
                "mode": "benchmark",
                "provider_order": ["Azure"],
                "only": ["Azure"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        )._request_payload(request)

        self.assertEqual(2_084_835_537, google["seed"])
        self.assertEqual(requested_seed, azure["seed"])

    def test_unsupported_parameter_overrides_are_rejected(self) -> None:
        request = ModelRequest(
            REQUEST.messages,
            REQUEST.response_schema,
            DecodingParameters(
                temperature=0,
                top_p=1,
                max_tokens=32,
                seed=17,
                extra={"reasoning": {"enabled": True}},
            ),
        )
        with self.assertRaisesRegex(ConfigError, "cannot override"):
            _backend()._request_payload(request)

    def test_native_json_modes_are_opt_in(self) -> None:
        json_object = _backend(structured_output_mode="json_object")._request_payload(
            REQUEST
        )
        json_schema = _backend(structured_output_mode="json_schema")._request_payload(
            REQUEST
        )
        self.assertEqual({"type": "json_object"}, json_object["response_format"])
        self.assertEqual("json_schema", json_schema["response_format"]["type"])
        self.assertTrue(json_schema["response_format"]["json_schema"]["strict"])


class OpenAICompatibleResponseTests(unittest.TestCase):
    def test_environment_key_is_loaded_but_redacted_from_all_metadata(self) -> None:
        with patch.dict(
            "os.environ", {"TEST_CLOUD_API_KEY": MOCK_SECRET}, clear=True
        ), patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=_success(
                metadata={"authorization": MOCK_SECRET, "note": MOCK_SECRET}
            ),
        ) as request:
            backend = _backend()
            response = backend.generate(REQUEST)

        sent_headers = request.call_args.args[3]
        self.assertEqual(f"Bearer {MOCK_SECRET}", sent_headers["Authorization"])
        serialized = json.dumps(
            {"response": response.metadata, "backend": backend.metadata()},
            sort_keys=True,
        )
        self.assertNotIn(MOCK_SECRET, serialized)
        self.assertNotIn("?unsafe=query", serialized)
        self.assertEqual("request-1", response.metadata["request_id"])
        self.assertEqual("provider-a", response.metadata["actual_provider"])
        self.assertEqual(2, response.token_usage["reasoning_tokens"])
        self.assertEqual(3, response.token_usage["cached_tokens"])
        self.assertEqual(0.00012, response.token_usage["reported_cost_usd"])
        self.assertEqual(
            ["max_tokens", "seed", "temperature", "top_p"],
            response.metadata["sent_request_parameters"],
        )
        self.assertEqual(
            ["seed"], response.metadata["provider_acknowledged_request_parameters"]
        )
        self.assertIsNone(response.metadata["observed_capabilities"]["temperature"])
        self.assertFalse(response.metadata["response_healing_requested"])

    def test_missing_seed_echo_is_unknown_not_a_capability_mismatch(self) -> None:
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=_success(echo_seed=False),
        ):
            response = _backend(
                request_parameters={"seed": True},
                declared_capabilities={"seed": True},
            ).generate(REQUEST)

        self.assertIn("seed", response.metadata["sent_request_parameters"])
        self.assertIsNone(response.metadata["observed_capabilities"]["seed"])
        self.assertFalse(response.metadata["seed_acknowledged"])
        self.assertEqual([], response.metadata["capability_mismatches"])

    def test_google_seed_transport_adaptation_is_logged(self) -> None:
        requested_seed = 4_232_319_185
        request_value = ModelRequest(
            REQUEST.messages,
            REQUEST.response_schema,
            DecodingParameters(max_tokens=32, seed=requested_seed),
        )
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=_success(provider="Google", echo_seed=False),
        ) as transport:
            response = _backend(
                routing={
                    "mode": "benchmark",
                    "provider_order": ["Google"],
                    "only": ["Google"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            ).generate(request_value)

        self.assertEqual(2_084_835_537, transport.call_args.args[1]["seed"])
        self.assertEqual(requested_seed, response.metadata["requested_model_seed"])
        self.assertEqual(2_084_835_537, response.metadata["sent_model_seed"])
        self.assertEqual(
            "google_nonnegative_int32_modulo",
            response.metadata["model_seed_transport_adaptation"],
        )

    def test_empty_http_200_preserves_route_usage_and_probe_controls(self) -> None:
        empty = _success()
        empty.payload["choices"][0]["message"]["content"] = ""  # type: ignore[index]
        empty.payload["choices"][0]["finish_reason"] = "length"  # type: ignore[index]
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=empty,
        ):
            with self.assertRaises(ProviderError) as caught:
                _backend().generate(REQUEST)

        error = caught.exception
        self.assertEqual("empty_response", error.category)
        self.assertEqual("vendor/model", error.metadata["returned_model"])
        self.assertEqual("provider-a", error.metadata["actual_provider"])
        self.assertEqual("length", error.metadata["finish_reason"])
        self.assertEqual(32, error.metadata["max_output_tokens_requested"])
        self.assertEqual(
            0.00012,
            error.metadata["normalized_usage"]["reported_cost_usd"],
        )
        self.assertEqual(
            ["max_tokens", "seed", "temperature", "top_p"],
            error.metadata["sent_request_parameters"],
        )

    def test_reasoning_budget_exhaustion_is_returned_as_invalid_model_output(
        self,
    ) -> None:
        exhausted = _success()
        exhausted.payload["choices"][0]["message"]["content"] = ""  # type: ignore[index]
        exhausted.payload["choices"][0]["finish_reason"] = "length"  # type: ignore[index]
        exhausted.payload["usage"]["completion_tokens"] = 32  # type: ignore[index]
        exhausted.payload["usage"]["total_tokens"] = 43  # type: ignore[index]
        exhausted.payload["usage"]["completion_tokens_details"] = {  # type: ignore[index]
            "reasoning_tokens": 32
        }
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=exhausted,
        ):
            response = _backend().generate(REQUEST)

        self.assertEqual("", response.content)
        self.assertEqual(32, response.token_usage["reasoning_tokens"])
        self.assertTrue(response.metadata["output_budget_exhausted"])
        self.assertEqual(
            "output_budget_exhausted",
            response.metadata["model_output_status"],
        )
        self.assertEqual("length", response.metadata["finish_reason"])

    def test_output_budget_exhaustion_allows_provider_token_accounting_gap(
        self,
    ) -> None:
        exhausted = _success()
        exhausted.payload["choices"][0]["message"]["content"] = ""  # type: ignore[index]
        exhausted.payload["choices"][0]["finish_reason"] = "length"  # type: ignore[index]
        exhausted.payload["usage"]["completion_tokens"] = 30  # type: ignore[index]
        exhausted.payload["usage"]["total_tokens"] = 41  # type: ignore[index]
        exhausted.payload["usage"]["completion_tokens_details"] = {  # type: ignore[index]
            "reasoning_tokens": 29
        }
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=exhausted,
        ):
            response = _backend().generate(REQUEST)

        self.assertEqual("", response.content)
        self.assertEqual(29, response.token_usage["reasoning_tokens"])
        self.assertTrue(response.metadata["output_budget_exhausted"])
        self.assertEqual(
            "output_budget_exhausted",
            response.metadata["model_output_status"],
        )

    def test_missing_usage_remains_null(self) -> None:
        self.assertEqual(
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "cached_tokens": None,
                "total_tokens": None,
                "reported_cost_usd": None,
            },
            normalize_usage(None),
        )

    def test_dynamic_mode_records_returned_model_fallback(self) -> None:
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=_success(model="vendor/alternate", provider="provider-b"),
        ):
            response = _backend(
                routing={"mode": "dynamic", "allow_fallbacks": True}
            ).generate(REQUEST)
        self.assertTrue(response.metadata["fallback_detected"])
        self.assertEqual("vendor/alternate", response.metadata["returned_model"])
        self.assertEqual("provider-b", response.metadata["actual_provider"])

    def test_benchmark_mode_rejects_model_route_or_explicit_fallback(self) -> None:
        responses = (
            _success(model="vendor/alternate"),
            _success(provider="provider-b"),
            _success(metadata={"fallback_used": True}),
        )
        for response in responses:
            with self.subTest(response=response.payload), patch(
                "cps_reason_bench.models.openai_compatible._request_json",
                return_value=response,
            ):
                with self.assertRaisesRegex(
                    ProviderError, "different model or provider"
                ) as caught:
                    _backend().generate(REQUEST)
                self.assertEqual("routing_mismatch", caught.exception.category)
                self.assertTrue(caught.exception.billing_ambiguity)
                self.assertEqual(
                    0.00012,
                    caught.exception.metadata["normalized_usage"]["reported_cost_usd"],
                )


class ProviderFailureTests(unittest.TestCase):
    def test_permanent_http_failures_are_classified_and_never_retried(self) -> None:
        expected = {
            400: "permanent_configuration_error",
            401: "authentication_failure",
            402: "insufficient_credit",
            403: "permission_failure",
            404: "model_unavailable",
        }
        for status, category in expected.items():
            error = _http_provider_error(
                status,
                json.dumps(
                    {
                        "error": {
                            "type": "mock",
                            "message": MOCK_SECRET,
                            "authorization": MOCK_SECRET,
                        }
                    }
                ),
                {"x-request-id": f"request-{status}"},
                "openrouter",
            )
            with self.subTest(status=status), patch(
                "cps_reason_bench.models.openai_compatible._request_json",
                side_effect=error,
            ) as request:
                with self.assertRaises(ProviderError) as caught:
                    _backend(transport_retry={"max_attempts": 3}).generate(REQUEST)
                self.assertEqual(category, caught.exception.category)
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(1, request.call_count)
                self.assertNotIn(MOCK_SECRET, json.dumps(caught.exception.to_dict()))

    def test_rate_limit_honors_retry_after_and_records_transport_retry(self) -> None:
        delays: list[float] = []
        rate_limit = _http_provider_error(
            429,
            '{"error":{"type":"rate_limit"}}',
            {"Retry-After": "2", "x-request-id": "limited-1"},
            "openrouter",
        )
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[rate_limit, _success()],
        ) as request:
            response = _backend(
                transport_retry={
                    "max_attempts": 2,
                    "base_backoff_seconds": 0.5,
                    "max_backoff_seconds": 5,
                    "jitter_seconds": 0,
                },
                sleeper=delays.append,
            ).generate(REQUEST)
        self.assertEqual(2, request.call_count)
        self.assertEqual([2.0], delays)
        self.assertEqual(1, response.metadata["transport_retry_count"])
        self.assertFalse(rate_limit.billing_ambiguity)

    def test_provider_rate_limit_reset_header_is_understood(self) -> None:
        error = _http_provider_error(
            429,
            "{}",
            {"x-ratelimit-reset-requests": "250ms"},
            "openai",
        )
        self.assertAlmostEqual(0.25, error.retry_after_seconds or -1)

    def test_transient_failure_uses_bounded_exponential_retry(self) -> None:
        delays: list[float] = []
        transient = ProviderError(
            "temporary",
            category="transient_provider_failure",
            retryable=True,
            http_status=503,
            request_state="response_received",
            billing_ambiguity=False,
        )
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[transient, _success()],
        ):
            response = _backend(
                transport_retry={
                    "max_attempts": 2,
                    "base_backoff_seconds": 0.5,
                    "max_backoff_seconds": 1,
                    "jitter_seconds": 0,
                },
                sleeper=delays.append,
            ).generate(REQUEST)
        self.assertEqual([0.5], delays)
        self.assertEqual(2, len(response.metadata["transport_attempts"]))
        self.assertFalse(response.metadata["billing_ambiguity"])

    def test_nonbillable_finish_error_uses_bounded_transport_retry(self) -> None:
        delays: list[float] = []
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[_finish_error_response(), _success()],
        ) as request:
            response = _backend(
                transport_retry={
                    "max_attempts": 2,
                    "base_backoff_seconds": 0.5,
                    "max_backoff_seconds": 1,
                    "jitter_seconds": 0,
                },
                sleeper=delays.append,
            ).generate(REQUEST)

        self.assertEqual(2, request.call_count)
        self.assertEqual([0.5], delays)
        first = response.metadata["transport_attempts"][0]
        self.assertEqual("error", first["status"])
        self.assertEqual("transient_provider_failure", first["category"])
        self.assertEqual("error", first["finish_reason"])
        self.assertEqual(0, first["normalized_usage"]["reported_cost_usd"])
        self.assertFalse(first["billing_ambiguity"])

    def test_top_level_http_200_error_is_classified_and_safely_recorded(
        self,
    ) -> None:
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=_top_level_error_response(),
        ) as request:
            with self.assertRaises(ProviderError) as caught:
                _backend(transport_retry={"max_attempts": 3}).generate(REQUEST)

        error = caught.exception
        self.assertEqual(1, request.call_count)
        self.assertEqual("transient_provider_failure", error.category)
        self.assertFalse(error.retryable)
        self.assertTrue(error.billing_ambiguity)
        self.assertEqual("in-band-request-1", error.metadata["request_id"])
        self.assertEqual("top_level", error.metadata["provider_error_location"])
        self.assertEqual(
            {
                "code": 502,
                "error_type": "provider_unavailable",
                "provider_code": "upstream_invalid_response",
            },
            error.metadata["provider_error"],
        )
        self.assertTrue(error.metadata["response_shape"]["top_level_error_present"])
        self.assertNotIn(MOCK_SECRET, json.dumps(error.to_dict()))

    def test_explicitly_unbilled_http_200_error_uses_transport_retry(self) -> None:
        delays: list[float] = []
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[
                _top_level_error_response(include_zero_usage=True),
                _success(),
            ],
        ) as request:
            response = _backend(
                transport_retry={
                    "max_attempts": 2,
                    "base_backoff_seconds": 0.5,
                    "max_backoff_seconds": 1,
                    "jitter_seconds": 0,
                },
                sleeper=delays.append,
            ).generate(REQUEST)

        self.assertEqual(2, request.call_count)
        self.assertEqual([0.5], delays)
        first = response.metadata["transport_attempts"][0]
        self.assertEqual("transient_provider_failure", first["category"])
        self.assertEqual(0, first["normalized_usage"]["reported_cost_usd"])
        self.assertFalse(first["billing_ambiguity"])

    def test_finish_error_without_explicit_zero_cost_is_not_retried(self) -> None:
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            return_value=_finish_error_response(include_usage=False),
        ) as request:
            with self.assertRaises(ProviderError) as caught:
                _backend(transport_retry={"max_attempts": 3}).generate(REQUEST)

        error = caught.exception
        self.assertEqual(1, request.call_count)
        self.assertEqual("transient_provider_failure", error.category)
        self.assertFalse(error.retryable)
        self.assertTrue(error.billing_ambiguity)
        self.assertEqual("error", error.metadata["finish_reason"])

    def test_ambiguous_transport_failure_requires_explicit_retry(self) -> None:
        ambiguous = ProviderError(
            "response lost",
            category="timeout",
            retryable=True,
            request_state="sent_no_response",
            billing_ambiguity=True,
        )
        with patch(
            "cps_reason_bench.models.openai_compatible._request_json",
            side_effect=[ambiguous, _success()],
        ) as request:
            with self.assertRaises(ProviderError) as caught:
                _backend(transport_retry={"max_attempts": 3}).generate(REQUEST)
        self.assertEqual(1, request.call_count)
        self.assertTrue(caught.exception.billing_ambiguity)
        self.assertEqual(1, len(caught.exception.transport_attempts))

    def test_failed_response_usage_is_normalized_for_cost_retention(self) -> None:
        error = _http_provider_error(
            503,
            json.dumps(
                {
                    "error": {"type": "server_error"},
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 2,
                        "total_tokens": 22,
                        "cost": 0.0004,
                    },
                }
            ),
            {"x-request-id": "failed-cost-1"},
            "openrouter",
        )
        self.assertFalse(error.billing_ambiguity)
        self.assertEqual(
            0.0004,
            error.metadata["normalized_usage"]["reported_cost_usd"],
        )


if __name__ == "__main__":
    unittest.main()
