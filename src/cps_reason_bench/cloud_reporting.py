"""Provider and cost reports derived from sanitized model-interaction traces."""

from __future__ import annotations

import copy
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from cps_reason_bench.run_logging import atomic_write_json, atomic_write_text


PROVIDER_REPORT_FIELDS = (
    "model_configuration",
    "provider",
    "requested_model",
    "returned_model",
    "requested_provider",
    "actual_provider",
    "actual_route",
    "routing_mode",
    "fallback_allowed",
    "fallback_detected",
    "quantization",
    "structured_output_mode",
    "calls",
    "transport_attempts",
    "transport_retries",
    "errors",
    "error_categories",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "total_tokens",
    "reported_cost_usd",
    "locally_estimated_cost_usd",
    "conservative_cost_bound_usd",
    "cost_incomplete",
    "latency_seconds",
    "billing_ambiguities",
)


def enrich_episode_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_episode_record(record) for record in records]


def enrich_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(record)
    calls = provider_call_records(record)
    cloud_calls = [call for call in calls if call["scope"] == "cloud"]
    scope = "cloud" if cloud_calls else "local"
    selected = cloud_calls if cloud_calls else calls
    output["cloud"] = {
        "scope": scope,
        "provider": _single_or_none(call.get("provider") for call in selected),
        "routing_mode": _single_or_none(call.get("routing_mode") for call in selected),
        "structured_output_mode": _single_or_none(
            call.get("structured_output_mode") for call in selected
        ),
        "model_call_count": len(selected),
        "transport_attempt_count": sum(
            int(call.get("transport_attempts") or 1) for call in selected
        ),
        "transport_retry_count": sum(
            int(call.get("transport_retries") or 0) for call in selected
        ),
        "provider_error_count": sum(
            bool(call.get("error_category")) for call in selected
        ),
        "latency_seconds": _sum_known(call.get("latency_seconds") for call in selected),
        "prompt_tokens": _sum_known(call.get("prompt_tokens") for call in selected),
        "completion_tokens": _sum_known(
            call.get("completion_tokens") for call in selected
        ),
        "reasoning_tokens": _sum_known(
            call.get("reasoning_tokens") for call in selected
        ),
        "cached_tokens": _sum_known(call.get("cached_tokens") for call in selected),
        "total_tokens": _sum_known(call.get("total_tokens") for call in selected),
        "reported_cost_usd": _sum_known(
            call.get("reported_cost_usd") for call in cloud_calls
        ),
        "locally_estimated_cost_usd": _sum_known(
            call.get("locally_estimated_cost_usd") for call in cloud_calls
        ),
        "conservative_cost_bound_usd": _sum_known(
            call.get("conservative_cost_bound_usd") for call in cloud_calls
        ),
        "accounted_cost_usd": _accounted_cost(cloud_calls),
        "cost_incomplete": any(_call_cost_incomplete(call) for call in cloud_calls),
        "provider_cost_incomplete": any(
            _provider_cost_incomplete(call) for call in cloud_calls
        ),
        "currency": "USD" if cloud_calls else None,
        "billing_ambiguity": any(
            bool(call.get("billing_ambiguity")) for call in cloud_calls
        ),
    }
    return output


def provider_call_records(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = record.get("policy") if isinstance(record.get("policy"), Mapping) else {}
    metadata = (
        record.get("run_metadata")
        if isinstance(record.get("run_metadata"), Mapping)
        else {}
    )
    model_config = metadata.get("model_configuration")
    model_config = model_config if isinstance(model_config, Mapping) else {}
    parameters = model_config.get("backend_parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    pricing = parameters.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    backend = str(model_config.get("backend") or record.get("backend") or "unknown")
    configured_provider = str(
        parameters.get("provider") or policy.get("provider") or backend
    )
    scope = (
        "cloud" if backend in {"openai-compatible", "openai_compatible"} else "local"
    )
    events_path = Path(str(record.get("events_path") or ""))
    calls: list[dict[str, Any]] = []
    if not events_path.is_file():
        return calls
    interaction_index = -1
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return calls
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(event, dict)
            or event.get("record_type") != "model_interaction"
        ):
            continue
        interaction_index = int(event.get("interaction_index", interaction_index + 1))
        attempts = event.get("output_attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            backend_metadata = attempt.get("backend_metadata")
            backend_metadata = (
                backend_metadata if isinstance(backend_metadata, Mapping) else {}
            )
            backend_error = attempt.get("backend_error")
            backend_error = backend_error if isinstance(backend_error, Mapping) else {}
            error_metadata = backend_error.get("metadata")
            error_metadata = (
                error_metadata if isinstance(error_metadata, Mapping) else {}
            )
            normalized = backend_metadata.get("normalized_usage")
            if not isinstance(normalized, Mapping):
                normalized = error_metadata.get("normalized_usage")
            normalized = normalized if isinstance(normalized, Mapping) else {}
            attempt_usage = attempt.get("token_usage")
            attempt_usage = attempt_usage if isinstance(attempt_usage, Mapping) else {}
            usage = {
                key: normalized.get(key, attempt_usage.get(key))
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "cached_tokens",
                    "total_tokens",
                    "reported_cost_usd",
                )
            }
            estimated = (
                estimate_usage_cost(usage, pricing) if scope == "cloud" else None
            )
            billing_ambiguity = bool(
                backend_error.get("billing_ambiguity")
                or backend_metadata.get("billing_ambiguity")
            )
            conservative_bound = (
                _conservative_unknown_call_cost(
                    event.get("prompt"), model_config, pricing
                )
                if scope == "cloud" and billing_ambiguity and estimated is None
                else None
            )
            transport_attempts = backend_metadata.get("transport_attempts")
            if not isinstance(transport_attempts, list):
                transport_attempts = backend_error.get("transport_attempts")
            if not isinstance(transport_attempts, list):
                transport_attempts = []
            calls.append(
                {
                    "run_id": record.get("run_id"),
                    "episode_id": record.get("episode_id"),
                    "interaction_index": interaction_index,
                    "output_attempt": attempt.get("attempt"),
                    "model_configuration": record.get("model_configuration")
                    or model_config.get("name"),
                    "scope": scope,
                    "provider": backend_metadata.get("provider", configured_provider),
                    "requested_model": backend_metadata.get(
                        "requested_model",
                        model_config.get("model") or record.get("model"),
                    ),
                    "returned_model": backend_metadata.get("returned_model"),
                    "requested_provider": backend_metadata.get("requested_provider"),
                    "actual_provider": backend_metadata.get("actual_provider"),
                    "actual_route": backend_metadata.get("actual_route"),
                    "routing_mode": backend_metadata.get("routing_mode")
                    or (
                        parameters.get("routing", {}).get("mode")
                        if isinstance(parameters.get("routing"), Mapping)
                        else None
                    ),
                    "fallback_allowed": backend_metadata.get("fallback_allowed"),
                    "fallback_detected": backend_metadata.get("fallback_detected"),
                    "quantization": backend_metadata.get("quantization"),
                    "structured_output_mode": backend_metadata.get(
                        "structured_output_mode",
                        parameters.get("structured_output_mode"),
                    ),
                    "transport_attempts": len(transport_attempts) or 1,
                    "transport_retries": max(0, len(transport_attempts) - 1),
                    "error_category": backend_error.get("category"),
                    "request_state": backend_error.get("request_state"),
                    "billing_ambiguity": billing_ambiguity,
                    "latency_seconds": attempt.get("latency_seconds"),
                    **usage,
                    "locally_estimated_cost_usd": estimated,
                    "conservative_cost_bound_usd": conservative_bound,
                    "conservative_cost_bound_source": (
                        "logged_prompt_utf8_bound_and_configured_max_output"
                        if conservative_bound is not None
                        else None
                    ),
                    "currency": "USD" if scope == "cloud" else None,
                    "pricing_source": pricing.get("source"),
                    "pricing_source_timestamp": pricing.get("retrieved_at_utc"),
                    "request_id": backend_metadata.get("request_id")
                    or (error_metadata.get("request_id")),
                }
            )
    return calls


def estimate_usage_cost(
    usage: Mapping[str, Any], pricing: Mapping[str, Any]
) -> float | None:
    input_price = _number(pricing.get("input_per_million_usd"))
    output_price = _number(pricing.get("output_per_million_usd"))
    cached_price = _number(pricing.get("cached_input_per_million_usd"))
    prompt = _number(usage.get("prompt_tokens"))
    completion = _number(usage.get("completion_tokens"))
    cached = _number(usage.get("cached_tokens"))
    if (
        input_price is None
        or output_price is None
        or prompt is None
        or completion is None
    ):
        return None
    cached_tokens = min(prompt, cached or 0.0)
    uncached_tokens = max(0.0, prompt - cached_tokens)
    effective_cached_price = input_price if cached_price is None else cached_price
    cost = (
        uncached_tokens * input_price
        + cached_tokens * effective_cached_price
        + completion * output_price
    ) / 1_000_000.0
    return round(cost, 12)


def _conservative_unknown_call_cost(
    prompt: Any,
    model_config: Mapping[str, Any],
    pricing: Mapping[str, Any],
) -> float | None:
    decoding = model_config.get("decoding")
    decoding = decoding if isinstance(decoding, Mapping) else {}
    max_output_tokens = _number(decoding.get("max_tokens"))
    if prompt is None or max_output_tokens is None or max_output_tokens <= 0:
        return None
    try:
        serialized = json.dumps(
            prompt, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    # Byte-level tokenizers cannot exceed the serialized byte count except for
    # chat framing. Doubling it and adding fixed headroom keeps this a budget
    # upper bound rather than an estimate of likely usage.
    prompt_token_bound = len(serialized) * 2 + 4096
    return estimate_usage_cost(
        {
            "prompt_tokens": prompt_token_bound,
            "completion_tokens": max_output_tokens,
            "cached_tokens": 0,
        },
        pricing,
    )


def write_cloud_reports(
    records: Iterable[dict[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    enriched = enrich_episode_records(records)
    calls = [call for record in enriched for call in provider_call_records(record)]
    provider_rows = _aggregate_provider_calls(calls)
    cost_rows = _aggregate_costs(enriched)
    cloud_calls = [call for call in calls if call["scope"] == "cloud"]
    cost_payload = {
        "cost_report_schema_version": 1,
        "currency": "USD",
        "local_api_price_included": False,
        "cloud_call_count": len(cloud_calls),
        "provider_reported_cost_usd": _sum_known(
            call.get("reported_cost_usd") for call in cloud_calls
        ),
        "locally_estimated_cost_usd": _sum_known(
            call.get("locally_estimated_cost_usd") for call in cloud_calls
        ),
        "conservative_cost_bound_usd": _sum_known(
            call.get("conservative_cost_bound_usd") for call in cloud_calls
        ),
        "accounted_cost_usd": _accounted_cost(cloud_calls),
        "cost_incomplete": any(_call_cost_incomplete(call) for call in cloud_calls),
        "provider_cost_incomplete": any(
            _provider_cost_incomplete(call) for call in cloud_calls
        ),
        "billing_ambiguity_count": sum(
            bool(call.get("billing_ambiguity")) for call in cloud_calls
        ),
        "by_model": cost_rows,
        "assumptions": [
            "Provider-reported per-call cost is preferred when available.",
            "Local estimates use versioned/manual pricing in the selected model snapshot.",
            (
                "Billing-ambiguous calls without usage reserve a conservative bound "
                "from the logged prompt, configured output limit, and frozen pricing."
            ),
            (
                "Received non-ambiguous provider errors without usage are "
                "accounted as zero cost."
            ),
            "Local Ollama monetary cost is not estimated or compared with cloud API price.",
        ],
    }
    provider_path = destination / "provider_report.csv"
    cost_json_path = atomic_write_json(destination / "cost_report.json", cost_payload)
    cost_csv_path = destination / "cost_report.csv"
    _write_csv(provider_path, PROVIDER_REPORT_FIELDS, provider_rows)
    cost_fields = (
        "model_configuration",
        "provider",
        "episodes",
        "calls",
        "successful_episodes",
        "provider_reported_cost_usd",
        "locally_estimated_cost_usd",
        "conservative_cost_bound_usd",
        "accounted_cost_usd",
        "cost_per_episode_usd",
        "cost_per_successful_episode_usd",
        "cost_incomplete",
        "billing_ambiguities",
    )
    _write_csv(cost_csv_path, cost_fields, cost_rows)
    return {
        "provider_report_csv": provider_path,
        "cost_report_json": cost_json_path,
        "cost_report_csv": cost_csv_path,
    }


def _aggregate_provider_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    dimensions = PROVIDER_REPORT_FIELDS[:12]
    for call in calls:
        grouped[tuple(call.get(key) for key in dimensions)].append(call)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        rows.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "calls": len(values),
                "transport_attempts": sum(
                    int(item.get("transport_attempts") or 1) for item in values
                ),
                "transport_retries": sum(
                    int(item.get("transport_retries") or 0) for item in values
                ),
                "errors": sum(bool(item.get("error_category")) for item in values),
                "error_categories": ";".join(
                    sorted(
                        {
                            str(item["error_category"])
                            for item in values
                            if item.get("error_category")
                        }
                    )
                ),
                "prompt_tokens": _sum_known(
                    item.get("prompt_tokens") for item in values
                ),
                "completion_tokens": _sum_known(
                    item.get("completion_tokens") for item in values
                ),
                "reasoning_tokens": _sum_known(
                    item.get("reasoning_tokens") for item in values
                ),
                "cached_tokens": _sum_known(
                    item.get("cached_tokens") for item in values
                ),
                "total_tokens": _sum_known(item.get("total_tokens") for item in values),
                "reported_cost_usd": _sum_known(
                    item.get("reported_cost_usd") for item in values
                ),
                "locally_estimated_cost_usd": _sum_known(
                    item.get("locally_estimated_cost_usd") for item in values
                ),
                "conservative_cost_bound_usd": _sum_known(
                    item.get("conservative_cost_bound_usd") for item in values
                ),
                "cost_incomplete": any(
                    item.get("scope") == "cloud" and _call_cost_incomplete(item)
                    for item in values
                ),
                "latency_seconds": _sum_known(
                    item.get("latency_seconds") for item in values
                ),
                "billing_ambiguities": sum(
                    bool(item.get("billing_ambiguity")) for item in values
                ),
            }
        )
    return rows


def _aggregate_costs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cloud = record.get("cloud", {})
        if not isinstance(cloud, Mapping) or cloud.get("scope") != "cloud":
            continue
        grouped[
            (
                str(record.get("model_configuration") or "unknown"),
                str(cloud.get("provider") or "unknown"),
            )
        ].append(record)
    rows = []
    for (model, provider), values in sorted(grouped.items()):
        reported = _sum_known(item["cloud"].get("reported_cost_usd") for item in values)
        estimated = _sum_known(
            item["cloud"].get("locally_estimated_cost_usd") for item in values
        )
        conservative_bound = _sum_known(
            item["cloud"].get("conservative_cost_bound_usd") for item in values
        )
        accounted_values = [item["cloud"].get("accounted_cost_usd") for item in values]
        accounted = _sum_known(accounted_values)
        successes = sum(
            bool(item.get("metrics", {}).get("general", {}).get("task_success"))
            for item in values
        )
        rows.append(
            {
                "model_configuration": model,
                "provider": provider,
                "episodes": len(values),
                "calls": sum(
                    int(item["cloud"].get("model_call_count") or 0) for item in values
                ),
                "successful_episodes": successes,
                "provider_reported_cost_usd": reported,
                "locally_estimated_cost_usd": estimated,
                "conservative_cost_bound_usd": conservative_bound,
                "accounted_cost_usd": accounted,
                "cost_per_episode_usd": (
                    accounted / len(values)
                    if accounted is not None and values
                    else None
                ),
                "cost_per_successful_episode_usd": (
                    accounted / successes
                    if accounted is not None and successes
                    else None
                ),
                "cost_incomplete": any(
                    bool(item["cloud"].get("cost_incomplete")) for item in values
                ),
                "billing_ambiguities": sum(
                    bool(item["cloud"].get("billing_ambiguity")) for item in values
                ),
            }
        )
    return rows


def _accounted_cost(calls: Iterable[Mapping[str, Any]]) -> float | None:
    values = []
    for call in calls:
        value = _call_accounted_cost(call)
        if value is not None:
            values.append(value)
    return round(sum(values), 12) if values else None


def _call_accounted_cost(call: Mapping[str, Any]) -> float | None:
    for key in (
        "reported_cost_usd",
        "locally_estimated_cost_usd",
        "conservative_cost_bound_usd",
    ):
        value = _number(call.get(key))
        if value is not None:
            return float(value)
    if _known_nonbillable_call(call):
        return 0.0
    return None


def _call_cost_incomplete(call: Mapping[str, Any]) -> bool:
    return _call_accounted_cost(call) is None


def _provider_cost_incomplete(call: Mapping[str, Any]) -> bool:
    return _number(
        call.get("reported_cost_usd")
    ) is None and not _known_nonbillable_call(call)


def _known_nonbillable_call(call: Mapping[str, Any]) -> bool:
    has_usage = any(
        _number(call.get(key)) is not None
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "total_tokens",
        )
    )
    return (
        bool(call.get("error_category"))
        and not has_usage
        and (
            call.get("request_state") == "response_received"
            and call.get("billing_ambiguity") is False
        )
    )


def _sum_known(values: Iterable[Any]) -> float | int | None:
    normalized = [float(value) for value in values if _number(value) is not None]
    if not normalized:
        return None
    total = sum(normalized)
    return int(total) if total.is_integer() else round(total, 12)


def _single_or_none(values: Iterable[Any]) -> Any:
    normalized = {value for value in values if value is not None}
    return next(iter(normalized)) if len(normalized) == 1 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _write_csv(
    path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    atomic_write_text(path, stream.getvalue())
