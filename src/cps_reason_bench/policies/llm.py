"""Policy wrapper that turns a provider backend into benchmark decisions."""

from __future__ import annotations

import json
from typing import Any

from cps_reason_bench.models.base import (
    BackendError,
    DecodingParameters,
    ModelBackend,
    ModelRequest,
)
from cps_reason_bench.policies.base import Policy, PolicyRequest, PolicyResult
from cps_reason_bench.protocol import decision_response_schema, parse_decision
from cps_reason_bench.config import ConfigError
from cps_reason_bench.response_normalization import (
    normalization_config, normalization_identity, normalize_response,
)
from cps_reason_bench.versions import (
    PROMPT_TEMPLATE_ID,
    PROMPT_SYSTEM_TEXT,
    PROMPT_TEMPLATE_VERSION,
    content_hash,
    get_prompt_template,
)


SYSTEM_PROMPT = PROMPT_SYSTEM_TEXT


class LLMPolicy(Policy):
    policy_name = "llm"

    def __init__(
        self,
        backend: ModelBackend,
        decoding: DecodingParameters | None = None,
        max_output_retries: int = 0,
        *,
        retry_policy_name: str = "strict_zero_correction",
        prompt_template_id: str = PROMPT_TEMPLATE_ID,
        prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
        model_configuration_name: str | None = None,
        capability_overrides: dict[str, Any] | None = None,
        response_normalization: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self.decoding = decoding or DecodingParameters()
        self.max_output_retries = max(0, max_output_retries)
        self.retry_policy_name = retry_policy_name
        self.prompt_template_id = prompt_template_id
        self.prompt_template_version = prompt_template_version
        self.model_configuration_name = model_configuration_name
        self.capability_overrides = dict(capability_overrides or {})
        self.response_normalization = normalization_config(response_normalization)
        if self.response_normalization["mode"] != "none" and self.max_output_retries:
            raise ConfigError("Response normalization requires zero output corrections")
        self.system_prompt = get_prompt_template(
            self.prompt_template_id, self.prompt_template_version
        )

    def decide(self, request: PolicyRequest) -> PolicyResult:
        schema = decision_response_schema(request.constraints)
        messages = list(
            _build_messages(
                request,
                schema,
                self.system_prompt,
                self.prompt_template_version,
            )
        )
        original_prompt = tuple(messages)
        rendered_prompt_hash = content_hash(original_prompt)
        response_schema_hash = content_hash(schema)
        attempts: list[dict[str, Any]] = []
        total_usage: dict[str, int | float] = {}
        total_latency = 0.0
        last_raw: str | None = None
        last_parsed: dict[str, Any] | None = None
        last_errors: tuple[str, ...] = ()
        processing: dict[str, Any] = {}

        for attempt_number in range(1, self.max_output_retries + 2):
            try:
                response = self.backend.generate(
                    ModelRequest(tuple(messages), schema, self.decoding)
                )
            except BackendError as exc:
                if self.response_normalization["mode"] != "none":
                    rule = normalization_identity(self.response_normalization["mode"])
                    processing = {
                        "normalized_response": None, "normalization_applied": False,
                        "normalization_rule": self.response_normalization["mode"],
                        "normalization_version": rule["version"],
                        "normalization_rule_hash": rule["rule_hash"],
                        "normalization_success": False, "normalization_reason": "no_response_backend_error",
                        "raw_response_hash": None, "normalized_response_hash": None,
                    }
                error = f"Backend error: {exc}"
                classification = (
                    exc.to_dict()
                    if hasattr(exc, "to_dict")
                    else {"category": "backend_error", "message": str(exc)}
                )
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "raw_response": None,
                        **processing,
                        "parsed_response": None,
                        "validation_errors": [error],
                        "json_valid": False,
                        "schema_valid": False,
                        "decision_valid": False,
                        "validation_stage": "backend",
                        "backend_error": classification,
                    }
                )
                return PolicyResult(
                    decision=None,
                    prompt=original_prompt,
                    validation_errors=(error,),
                    attempts=tuple(attempts),
                    token_usage=total_usage,
                    latency_seconds=total_latency,
                    rendered_prompt_hash=rendered_prompt_hash,
                    response_schema_hash=response_schema_hash,
                    response_processing=processing,
                )

            last_raw = response.content
            total_latency += response.latency_seconds or 0.0
            _add_usage(total_usage, response.token_usage)
            content = response.content
            if self.response_normalization["mode"] != "none":
                normalized = normalize_response(content, self.response_normalization["mode"])
                processing = normalized.metadata()
                content = normalized.normalized_response
            parsed = parse_decision(content, request.constraints)
            last_parsed = parsed.parsed
            last_errors = parsed.errors
            attempts.append(
                {
                    "attempt": attempt_number,
                    "raw_response": response.content,
                    **processing,
                    "parsed_response": parsed.parsed,
                    "validation_errors": list(parsed.errors),
                    "validation_issues": [issue.to_dict() for issue in parsed.issues],
                    "json_valid": parsed.json_valid,
                    "schema_valid": parsed.schema_valid,
                    "decision_valid": parsed.decision_valid,
                    "token_usage": response.token_usage,
                    "latency_seconds": response.latency_seconds,
                    "backend_metadata": response.metadata,
                }
            )
            if parsed.valid:
                return PolicyResult(
                    decision=parsed.decision,
                    prompt=original_prompt,
                    raw_response=last_raw,
                    parsed_response=last_parsed,
                    attempts=tuple(attempts),
                    token_usage=total_usage,
                    latency_seconds=total_latency,
                    rendered_prompt_hash=rendered_prompt_hash,
                    response_schema_hash=response_schema_hash,
                    response_processing=processing,
                )

            if attempt_number <= self.max_output_retries:
                messages.extend(
                    [
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": (
                                "Your response failed protocol validation. Return a corrected JSON "
                                "object only. Errors: " + "; ".join(parsed.errors)
                            ),
                        },
                    ]
                )

        return PolicyResult(
            decision=None,
            prompt=original_prompt,
            raw_response=last_raw,
            parsed_response=last_parsed,
            validation_errors=last_errors,
            attempts=tuple(attempts),
            token_usage=total_usage,
            latency_seconds=total_latency,
            rendered_prompt_hash=rendered_prompt_hash,
            response_schema_hash=response_schema_hash,
            response_processing=processing,
        )

    def metadata(self) -> dict[str, Any]:
        backend_metadata = self.backend.metadata()
        capabilities = dict(backend_metadata.get("capabilities", {}))
        capabilities.update(self.capability_overrides)
        backend_metadata["capabilities"] = capabilities
        return {
            "policy": self.policy_name,
            **({"response_normalization": normalization_identity(self.response_normalization["mode"])}
               if self.response_normalization["mode"] != "none" else {}),
            **backend_metadata,
            "model_configuration": self.model_configuration_name,
            "decoding": self.decoding.to_dict(),
            "max_output_retries": self.max_output_retries,
            "retry_policy_name": self.retry_policy_name,
            "prompt_template": {
                "id": self.prompt_template_id,
                "version": self.prompt_template_version,
                "content_hash": content_hash(self.system_prompt),
            },
        }


def _build_messages(
    request: PolicyRequest,
    schema: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
) -> tuple[dict[str, str], ...]:
    constraints = request.constraints
    task = {
        "scenario": request.scenario.name,
        "family": request.scenario.task_family,
        "description": request.scenario.description,
        "instructions": request.scenario.task.get("instructions", "Maintain safety and achieve the goals."),
        "objectives": request.scenario.task.get("objectives", []),
        "safety_rules": request.scenario.task.get("safety_rules", []),
    }
    protocol = {
        "allowed_action_ids": list(constraints.allowed_action_ids),
        "maximum_actions_now": constraints.max_actions,
        "callbacks_enabled": constraints.allows_callbacks,
        "callback_step_range": (
            [constraints.min_callback_steps, constraints.max_callback_steps]
            if constraints.allows_callbacks
            else None
        ),
        "subscriptions_enabled": constraints.allows_subscriptions,
        "allowed_event_types": list(constraints.allowed_event_types),
        "threshold_filter_format": {
            "target": "node.variable for variable_threshold; resource id for resource_threshold",
            "operator": "gt|ge|lt|le|eq|ne",
            "threshold": "number",
        },
        "active_subscription_ids": list(constraints.active_subscription_ids),
        "allowed_context_requests": list(constraints.allowed_context_requests),
        "reasoning_enabled": constraints.reasoning_enabled,
    }
    if prompt_template_version != PROMPT_TEMPLATE_VERSION:
        protocol.update(
            {
                "action_parameter_schemas": constraints.action_parameter_schemas,
                "callback_semantics": (
                    "Request one future benchmark invocation; a new request replaces "
                    "the pending callback and cancel removes it."
                    if constraints.allows_callbacks
                    else "Callbacks are disabled."
                ),
                "subscription_semantics": (
                    "Add, update, or remove filters over permitted public simulator events."
                    if constraints.allows_subscriptions
                    else "Subscriptions are disabled."
                ),
                "context_request_semantics": (
                    "Request only listed public history items for later interactions."
                ),
            }
        )
    user = {
        "task": task,
        "interaction_protocol": protocol,
        "context": request.context,
        "response_schema": schema,
    }
    return (
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user, sort_keys=True)},
    )


def _add_usage(
    total: dict[str, int | float], update: dict[str, int | float | None]
) -> None:
    for key, value in update.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
