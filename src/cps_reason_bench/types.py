"""Shared data types used at module boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulatorEvent:
    """An internal simulator event with a separately controlled public payload."""

    event_type: str
    step: int
    time: float
    data: dict[str, Any] = field(default_factory=dict)
    public_data: dict[str, Any] = field(default_factory=dict)
    publicly_observable: bool = True

    def to_log_dict(self) -> dict[str, Any]:
        result = {
            "event_type": self.event_type,
            "step": self.step,
            "time": self.time,
            "data": self.data,
            "public_data": self.public_data,
        }
        if not self.publicly_observable:
            result["publicly_observable"] = False
        return result


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    accepted: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    reason_code: str | None = None
    public_message: str | None = None
    public_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "accepted": self.accepted,
            "message": self.message,
            "details": self.details,
            "reason_code": self.reason_code,
            "public": self.to_public_dict(),
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "accepted": self.accepted,
            "message": self.public_message or self.message,
            "details": (
                self.details if self.public_details is None else self.public_details
            ),
            "feedback_available": True,
        }


@dataclass(frozen=True)
class EventTrigger:
    subscription_id: str
    event_type: str
    step: int
    time: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "event_type": self.event_type,
            "step": self.step,
            "time": self.time,
            "details": self.details,
        }
