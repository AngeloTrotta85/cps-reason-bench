"""Small predefined event-subscription system managed by the harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from cps_reason_bench.protocol import SubscriptionCommand
from cps_reason_bench.public_events import project_public_event_data
from cps_reason_bench.types import EventTrigger, SimulatorEvent


@dataclass(frozen=True)
class Subscription:
    subscription_id: str
    event_type: str
    filters: dict[str, Any]
    created_step: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.subscription_id,
            "event_type": self.event_type,
            "filters": self.filters,
            "created_step": self.created_step,
        }


class SubscriptionManager:
    def __init__(self, maximum: int = 8) -> None:
        self.maximum = maximum
        self.active: dict[str, Subscription] = {}
        self.useful_trigger_count = 0

    def validate(self, commands: Iterable[SubscriptionCommand]) -> list[str]:
        ids = set(self.active)
        errors: list[str] = []
        for command in commands:
            if command.operation == "add":
                if command.subscription_id in ids:
                    errors.append(f"Subscription {command.subscription_id!r} already exists")
                else:
                    ids.add(command.subscription_id)
            elif command.operation == "update":
                if command.subscription_id not in ids:
                    errors.append(f"Subscription {command.subscription_id!r} does not exist")
            elif command.operation == "remove":
                if command.subscription_id not in ids:
                    errors.append(f"Subscription {command.subscription_id!r} does not exist")
                else:
                    ids.remove(command.subscription_id)
        if len(ids) > self.maximum:
            errors.append(f"At most {self.maximum} active subscriptions are allowed")
        return errors

    def apply(self, commands: Iterable[SubscriptionCommand], step: int) -> None:
        commands = list(commands)
        errors = self.validate(commands)
        if errors:
            raise ValueError("; ".join(errors))
        for command in commands:
            if command.operation == "remove":
                del self.active[command.subscription_id]
                continue
            assert command.event_type is not None
            created_step = (
                self.active[command.subscription_id].created_step
                if command.operation == "update"
                else step
            )
            self.active[command.subscription_id] = Subscription(
                command.subscription_id,
                command.event_type,
                command.filters,
                created_step,
            )

    def match(
        self,
        events: Iterable[SimulatorEvent],
        include_event_values: bool = False,
    ) -> list[EventTrigger]:
        triggers: list[EventTrigger] = []
        for event in events:
            if not event.publicly_observable:
                continue
            for subscription in self.active.values():
                if not _matches(subscription, event):
                    continue
                details = {
                    "configured_event_type": subscription.event_type,
                    "filters": subscription.filters,
                    **project_public_event_data(event),
                }
                if include_event_values:
                    details["event_data"] = project_public_event_data(event)
                triggers.append(
                    EventTrigger(
                        subscription.subscription_id,
                        subscription.event_type,
                        event.step,
                        event.time,
                        details,
                    )
                )
        self.useful_trigger_count += len(triggers)
        return triggers

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.active.values()]


def _matches(subscription: Subscription, event: SimulatorEvent) -> bool:
    event_type = subscription.event_type
    filters = subscription.filters
    if event_type == "variable_threshold":
        if event.event_type != "variable_changed":
            return False
        return _threshold_match(filters, event.data)
    if event_type == "resource_threshold":
        if event.event_type != "resource_changed":
            return False
        return _threshold_match(filters, event.data)

    if event.event_type != event_type:
        return False
    for key, expected in filters.items():
        if key == "target":
            actual = next(
                (
                    event.data[candidate]
                    for candidate in ("target", "sensor_id", "action_id", "edge_id")
                    if candidate in event.data
                ),
                None,
            )
        else:
            actual = event.data.get(key)
        if actual != expected:
            return False
    return True


def _threshold_match(filters: dict[str, Any], data: dict[str, Any]) -> bool:
    if data.get("target") != filters.get("target"):
        return False
    operator = str(filters.get("operator"))
    threshold = filters.get("threshold")
    old_matches = _compare(data.get("old"), operator, threshold)
    new_matches = _compare(data.get("new"), operator, threshold)
    return new_matches and (not filters.get("crossing_only", True) or not old_matches)


def _compare(value: Any, operator: str, threshold: Any) -> bool:
    try:
        if operator == "gt":
            return value > threshold
        if operator == "ge":
            return value >= threshold
        if operator == "lt":
            return value < threshold
        if operator == "le":
            return value <= threshold
        if operator == "eq":
            return value == threshold
        if operator == "ne":
            return value != threshold
    except TypeError:
        return False
    return False
