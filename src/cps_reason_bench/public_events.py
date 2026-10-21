"""Allowlisted projections from internal simulator events to model-visible events."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from cps_reason_bench.types import SimulatorEvent


# Unknown fields are private by default. Adding data to an internal event therefore
# cannot expand the model-visible interface without an explicit change here.
PUBLIC_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "action_applied": ("action_id",),
    "action_failed": ("action_id", "reason"),
    "action_scheduled": ("action_id", "due_step"),
    "actuator_failed": ("action_id",),
    "communication_sent": ("recipient",),
    "delayed_action_effect": ("effect_observed",),
    "edge_changed": ("edge_id", "enabled"),
    "fault_activated": ("fault_id", "kind", "target"),
    "fault_detected": ("target",),
    "fault_repaired": ("target",),
    "inspection_completed": (
        "target",
        "component_type",
        "status",
        "value",
        "faults",
    ),
    "invalid_action": ("action_id",),
    "resource_changed": ("target", "old", "new"),
    "safety_violation": ("constraint_id",),
    "sensor_unavailable": ("sensor_id",),
    "task_condition": ("goal_id",),
    "variable_changed": ("target", "old", "new"),
}


# This catalog documents fields intentionally retained only in internal traces.
INTERNAL_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "action_applied": ("operation", "details"),
    "action_failed": ("message", "delayed_effect"),
    "action_scheduled": ("requested_step", "requested_time", "command"),
    "actuator_failed": ("message", "delayed_effect"),
    "communication_sent": ("message",),
    "delayed_action_effect": (
        "action_id",
        "command",
        "requested_step",
        "requested_time",
        "completed_step",
        "completed_time",
        "details",
    ),
    "edge_changed": ("old", "new", "source"),
    "fault_activated": ("value",),
    "fault_detected": ("faults",),
    "fault_repaired": ("fault_ids",),
    "invalid_action": ("message",),
    "resource_changed": (),
    "safety_violation": ("constraint",),
    "sensor_unavailable": (),
    "task_condition": (),
    "variable_changed": ("source",),
}


ACTION_OUTCOME_EVENT_TYPES = frozenset(
    {
        "action_applied",
        "action_failed",
        "action_scheduled",
        "actuator_failed",
        "communication_sent",
        "delayed_action_effect",
        "fault_detected",
        "fault_repaired",
        "inspection_completed",
        "invalid_action",
    }
)


def project_public_event_data(event: SimulatorEvent) -> dict[str, Any]:
    """Return only fields explicitly allowed for this public event type."""

    if not event.publicly_observable:
        return {}
    allowed = PUBLIC_EVENT_FIELDS.get(event.event_type, ())
    return {
        key: copy.deepcopy(event.public_data[key])
        for key in allowed
        if key in event.public_data
    }


def public_event_record(event: SimulatorEvent) -> dict[str, Any] | None:
    """Build the complete public event record, including its public timestamp."""

    if not event.publicly_observable:
        return None
    return {
        "event_type": event.event_type,
        "step": event.step,
        "time": event.time,
        "details": project_public_event_data(event),
    }


def event_projection_catalog() -> list[dict[str, Any]]:
    """Return a serializable description used by release-audit reports."""

    event_types = sorted(set(PUBLIC_EVENT_FIELDS) | set(INTERNAL_EVENT_FIELDS))
    return [
        {
            "public_event_type": event_type,
            "public_timestamp_fields": ["step", "time"],
            "permitted_public_fields": list(PUBLIC_EVENT_FIELDS.get(event_type, ())),
            "internal_only_fields": list(INTERNAL_EVENT_FIELDS.get(event_type, ())),
            "unknown_fields_default": "internal_only",
        }
        for event_type in event_types
    ]


def project_mapping(
    event_type: str, public_data: Mapping[str, Any]
) -> dict[str, Any]:
    """Project a payload without constructing an event; useful for audit tests."""

    allowed = PUBLIC_EVENT_FIELDS.get(event_type, ())
    return {
        key: copy.deepcopy(public_data[key])
        for key in allowed
        if key in public_data
    }
