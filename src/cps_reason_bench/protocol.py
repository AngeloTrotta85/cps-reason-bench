"""Structured operational protocol between a policy and the benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


THRESHOLD_OPERATORS = {"gt", "ge", "lt", "le", "eq", "ne"}


@dataclass(frozen=True)
class ActionCommand:
    action_id: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "parameters": self.parameters}


@dataclass(frozen=True)
class CallbackRequest:
    after_steps: int | None = None
    after_time: float | None = None
    at_time: float | None = None
    cancel: bool = False

    def to_dict(self) -> dict[str, Any]:
        if self.cancel:
            return {"cancel": True}
        if self.after_steps is not None:
            return {"after_steps": self.after_steps}
        if self.after_time is not None:
            return {"after_time": self.after_time}
        return {"at_time": self.at_time}


@dataclass(frozen=True)
class SubscriptionCommand:
    operation: str
    subscription_id: str
    event_type: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operation": self.operation,
            "id": self.subscription_id,
        }
        if self.event_type is not None:
            result["event_type"] = self.event_type
            result["filters"] = self.filters
        return result


@dataclass(frozen=True)
class Decision:
    actions: tuple[ActionCommand, ...] = ()
    callback: CallbackRequest | None = None
    subscriptions: tuple[SubscriptionCommand, ...] = ()
    context_requests: tuple[str, ...] = ()
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "callback": self.callback.to_dict() if self.callback else None,
            "subscriptions": [item.to_dict() for item in self.subscriptions],
            "context_requests": list(self.context_requests),
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class DecisionConstraints:
    allowed_action_ids: tuple[str, ...]
    action_parameter_schemas: dict[str, dict[str, Any]]
    max_actions: int
    current_step: int
    current_time: float
    max_steps: int
    time_step: float
    allows_callbacks: bool
    min_callback_steps: int
    max_callback_steps: int
    allows_subscriptions: bool
    allowed_event_types: tuple[str, ...]
    active_subscription_ids: tuple[str, ...]
    allowed_context_requests: tuple[str, ...]
    reasoning_enabled: bool


@dataclass(frozen=True)
class ValidationIssue:
    stage: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "message": self.message}


@dataclass(frozen=True)
class ParseResult:
    decision: Decision | None
    parsed: dict[str, Any] | None
    errors: tuple[str, ...]
    issues: tuple[ValidationIssue, ...] = ()
    json_valid: bool = False
    schema_valid: bool = False
    decision_valid: bool = False

    @property
    def valid(self) -> bool:
        return self.decision is not None and not self.errors


def parse_decision(
    content: str | Mapping[str, Any], constraints: DecisionConstraints
) -> ParseResult:
    errors: list[str] = []
    if isinstance(content, str):
        try:
            parsed_value = json.loads(content)
        except json.JSONDecodeError as exc:
            message = f"Response is not valid JSON: {exc.msg}"
            return ParseResult(
                None,
                None,
                (message,),
                (ValidationIssue("json", message),),
                False,
                False,
                False,
            )
    elif isinstance(content, Mapping):
        parsed_value = dict(content)
    else:
        message = "Response must be a JSON object"
        return ParseResult(
            None,
            None,
            (message,),
            (ValidationIssue("schema", message),),
            False,
            False,
            False,
        )

    if not isinstance(parsed_value, dict):
        message = "Top-level response must be a JSON object"
        return ParseResult(
            None,
            None,
            (message,),
            (ValidationIssue("schema", message),),
            True,
            False,
            False,
        )

    allowed_keys = {"actions", "callback", "subscriptions", "context_requests", "reasoning"}
    unknown = sorted(set(parsed_value) - allowed_keys)
    if unknown:
        errors.append(f"Unknown top-level fields: {', '.join(unknown)}")
    required_keys = {"actions", "callback", "subscriptions", "context_requests"}
    missing = sorted(required_keys - set(parsed_value))
    if missing:
        errors.append(f"Missing required top-level fields: {', '.join(missing)}")

    actions = _parse_actions(parsed_value.get("actions", []), constraints, errors)
    callback = _parse_callback(parsed_value.get("callback"), constraints, errors)
    subscriptions = _parse_subscriptions(
        parsed_value.get("subscriptions", []), constraints, errors
    )
    context_requests = _parse_context_requests(
        parsed_value.get("context_requests", []), constraints, errors
    )
    reasoning = parsed_value.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        errors.append("reasoning must be a string or null")
        reasoning = None
    if reasoning and not constraints.reasoning_enabled:
        errors.append("reasoning is disabled for this run")

    decision = None
    if not errors:
        decision = Decision(
            actions=tuple(actions),
            callback=callback,
            subscriptions=tuple(subscriptions),
            context_requests=tuple(context_requests),
            reasoning=reasoning,
        )
    issues = tuple(ValidationIssue(_validation_stage(error), error) for error in errors)
    schema_valid = not any(issue.stage == "schema" for issue in issues)
    decision_valid = schema_valid and not any(
        issue.stage == "decision" for issue in issues
    ) and decision is not None
    return ParseResult(
        decision,
        parsed_value,
        tuple(errors),
        issues,
        True,
        schema_valid,
        decision_valid,
    )


def decision_response_schema(constraints: DecisionConstraints) -> dict[str, Any]:
    """Return the JSON schema supplied to structured-output-capable backends."""

    callback_schema: dict[str, Any]
    if constraints.allows_callbacks:
        callback_schema = {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "after_steps": {"type": "integer", "minimum": 1},
                        "after_time": {"type": "number", "exclusiveMinimum": 0},
                        "at_time": {"type": "number"},
                        "cancel": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            ]
        }
    else:
        callback_schema = {"type": "null"}

    action_variants = []
    for action_id in constraints.allowed_action_ids:
        parameter_schemas = constraints.action_parameter_schemas.get(action_id, {})
        schema_properties = {
            name: {
                key: value
                for key, value in parameter_schema.items()
                if key != "required"
            }
            for name, parameter_schema in parameter_schemas.items()
        }
        required_parameters = [
            name
            for name, parameter_schema in parameter_schemas.items()
            if parameter_schema.get("required", False)
        ]
        action_variants.append(
            {
                "type": "object",
                "properties": {
                    "action_id": {"const": action_id},
                    "parameters": {
                        "type": "object",
                        "properties": schema_properties,
                        "required": required_parameters,
                        "additionalProperties": False,
                    },
                },
                "required": ["action_id"],
                "additionalProperties": False,
            }
        )

    subscription_items: dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {"enum": ["add", "update", "remove"]},
            "id": {"type": "string"},
            "event_type": {"enum": list(constraints.allowed_event_types)},
            "filters": {"type": "object"},
        },
        "required": ["operation", "id"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "maxItems": constraints.max_actions,
                "items": {"oneOf": action_variants},
            },
            "callback": callback_schema,
            "subscriptions": {
                "type": "array",
                "maxItems": 8 if constraints.allows_subscriptions else 0,
                "items": subscription_items,
            },
            "context_requests": {
                "type": "array",
                "items": {"enum": list(constraints.allowed_context_requests)},
                "uniqueItems": True,
            },
            "reasoning": (
                {"type": ["string", "null"]}
                if constraints.reasoning_enabled
                else {"type": "null"}
            ),
        },
        "required": ["actions", "callback", "subscriptions", "context_requests"],
        "additionalProperties": False,
    }


def _parse_actions(
    value: Any, constraints: DecisionConstraints, errors: list[str]
) -> list[ActionCommand]:
    if not isinstance(value, list):
        errors.append("actions must be a list")
        return []
    if len(value) > constraints.max_actions:
        errors.append(f"At most {constraints.max_actions} actions are allowed per interaction")

    result: list[ActionCommand] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"actions[{index}] must be an object")
            continue
        unknown = set(item) - {"action_id", "parameters"}
        if unknown:
            errors.append(f"actions[{index}] has unknown fields: {', '.join(sorted(unknown))}")
        action_id = item.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            errors.append(f"actions[{index}].action_id must be a non-empty string")
            continue
        if action_id not in constraints.allowed_action_ids:
            errors.append(f"Unknown action_id {action_id!r}")
            continue
        parameters = item.get("parameters", {})
        if not isinstance(parameters, Mapping):
            errors.append(f"actions[{index}].parameters must be an object")
            continue
        parameter_data = dict(parameters)
        parameter_schemas = constraints.action_parameter_schemas.get(action_id, {})
        unknown_parameters = set(parameter_data) - set(parameter_schemas)
        if unknown_parameters:
            errors.append(
                f"actions[{index}] has unsupported parameters for {action_id!r}: "
                + ", ".join(sorted(unknown_parameters))
            )
            continue
        parameter_errors = _validate_action_parameters(
            parameter_data, parameter_schemas, f"actions[{index}].parameters"
        )
        errors.extend(parameter_errors)
        result.append(ActionCommand(action_id, parameter_data))
    return result


def _parse_callback(
    value: Any, constraints: DecisionConstraints, errors: list[str]
) -> CallbackRequest | None:
    if value is None:
        return None
    if not constraints.allows_callbacks:
        errors.append("callback requests are disabled by the interaction mode")
        return None
    if not isinstance(value, Mapping):
        errors.append("callback must be an object or null")
        return None

    callback = dict(value)
    unknown = set(callback) - {"after_steps", "after_time", "at_time", "cancel"}
    if unknown:
        errors.append(f"callback has unknown fields: {', '.join(sorted(unknown))}")
    if callback.get("cancel") is True:
        if len(callback) != 1:
            errors.append("callback.cancel cannot be combined with a callback time")
        return CallbackRequest(cancel=True)

    timing_fields = [key for key in ("after_steps", "after_time", "at_time") if key in callback]
    if len(timing_fields) != 1:
        errors.append("callback must specify exactly one of after_steps, after_time, or at_time")
        return None

    field_name = timing_fields[0]
    raw = callback[field_name]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        errors.append(f"callback.{field_name} must be numeric")
        return None

    if field_name == "after_steps":
        if not isinstance(raw, int):
            errors.append("callback.after_steps must be an integer")
            return None
        delta_steps = raw
        request = CallbackRequest(after_steps=raw)
    elif field_name == "after_time":
        if raw <= 0:
            errors.append("callback.after_time must be positive")
            return None
        delta_steps = math.ceil(float(raw) / constraints.time_step)
        request = CallbackRequest(after_time=float(raw))
    else:
        if float(raw) <= constraints.current_time:
            errors.append("callback.at_time must be in the future")
            return None
        delta_steps = math.ceil(
            (float(raw) - constraints.current_time) / constraints.time_step
        )
        request = CallbackRequest(at_time=float(raw))

    if not constraints.min_callback_steps <= delta_steps <= constraints.max_callback_steps:
        errors.append(
            "callback timing must be between "
            f"{constraints.min_callback_steps} and {constraints.max_callback_steps} steps ahead"
        )
    if constraints.current_step + delta_steps >= constraints.max_steps:
        errors.append("callback occurs at or after the terminal episode horizon")
    return request


def _parse_subscriptions(
    value: Any, constraints: DecisionConstraints, errors: list[str]
) -> list[SubscriptionCommand]:
    if not isinstance(value, list):
        errors.append("subscriptions must be a list")
        return []
    if value and not constraints.allows_subscriptions:
        errors.append("event subscriptions are disabled by the interaction mode")
        return []

    result: list[SubscriptionCommand] = []
    active_ids = set(constraints.active_subscription_ids)
    for index, item in enumerate(value):
        prefix = f"subscriptions[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        unknown = set(item) - {"operation", "id", "event_type", "filters"}
        if unknown:
            errors.append(f"{prefix} has unknown fields: {', '.join(sorted(unknown))}")
        operation = item.get("operation")
        subscription_id = item.get("id")
        if operation not in {"add", "update", "remove"}:
            errors.append(f"{prefix}.operation must be add, update, or remove")
            continue
        if not isinstance(subscription_id, str) or not subscription_id.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
            continue
        if operation == "remove":
            if subscription_id not in active_ids:
                errors.append(f"{prefix} cannot remove unknown subscription {subscription_id!r}")
                continue
            active_ids.remove(subscription_id)
            result.append(SubscriptionCommand(operation, subscription_id))
            continue

        if operation == "add" and subscription_id in active_ids:
            errors.append(f"{prefix} cannot add existing subscription {subscription_id!r}")
            continue
        if operation == "update" and subscription_id not in active_ids:
            errors.append(f"{prefix} cannot update unknown subscription {subscription_id!r}")
            continue

        event_type = item.get("event_type")
        if event_type not in constraints.allowed_event_types:
            errors.append(f"{prefix}.event_type is not allowed: {event_type!r}")
            continue
        filters = item.get("filters", {})
        if not isinstance(filters, Mapping):
            errors.append(f"{prefix}.filters must be an object")
            continue
        filter_data = dict(filters)
        _validate_event_filters(event_type, filter_data, prefix, errors)
        result.append(
            SubscriptionCommand(operation, subscription_id, str(event_type), filter_data)
        )
        active_ids.add(subscription_id)
    return result


def _validate_event_filters(
    event_type: str, filters: dict[str, Any], prefix: str, errors: list[str]
) -> None:
    if event_type in {"variable_threshold", "resource_threshold"}:
        if not isinstance(filters.get("target"), str) or not filters["target"]:
            errors.append(f"{prefix}.filters.target is required for threshold events")
        if filters.get("operator") not in THRESHOLD_OPERATORS:
            errors.append(
                f"{prefix}.filters.operator must be one of {sorted(THRESHOLD_OPERATORS)}"
            )
        threshold = filters.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            errors.append(f"{prefix}.filters.threshold must be numeric")


def _parse_context_requests(
    value: Any, constraints: DecisionConstraints, errors: list[str]
) -> list[str]:
    if not isinstance(value, list):
        errors.append("context_requests must be a list")
        return []
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"context_requests[{index}] must be a string")
        elif item not in constraints.allowed_context_requests:
            errors.append(f"Unsupported context request: {item!r}")
        elif item not in result:
            result.append(item)
    return result


def _validate_action_parameters(
    parameters: dict[str, Any], schemas: dict[str, Any], prefix: str
) -> list[str]:
    errors: list[str] = []
    for name, raw_schema in schemas.items():
        schema = raw_schema if isinstance(raw_schema, Mapping) else {}
        if schema.get("required", False) and name not in parameters:
            errors.append(f"{prefix}.{name} is required")
            continue
        if name not in parameters:
            continue
        value = parameters[name]
        expected_type = schema.get("type")
        valid_type = {
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
        }.get(expected_type, True)
        if not valid_type:
            errors.append(f"{prefix}.{name} must have type {expected_type}")
            continue
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{prefix}.{name} must be one of {schema['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{prefix}.{name} is below its minimum")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{prefix}.{name} is above its maximum")
    return errors


def _validation_stage(message: str) -> str:
    """Classify static wire/schema failures separately from contextual decisions."""

    decision_markers = (
        "Unknown action_id",
        "unsupported parameters",
        " is required",
        "below its minimum",
        "above its maximum",
        "callback requests are disabled",
        "callback timing must",
        "callback occurs at or after",
        "callback.at_time must be in the future",
        "event subscriptions are disabled",
        "cannot remove unknown subscription",
        "cannot add existing subscription",
        "cannot update unknown subscription",
        "event_type is not allowed",
        "Unsupported context request",
        "reasoning is disabled",
    )
    return "decision" if any(marker in message for marker in decision_markers) else "schema"
