"""Harness-owned callback scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cps_reason_bench.protocol import CallbackRequest


@dataclass(frozen=True)
class ScheduledCallback:
    requested_at_step: int
    due_step: int
    due_time: float
    request: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_at_step": self.requested_at_step,
            "due_step": self.due_step,
            "due_time": self.due_time,
            "request": self.request,
        }


class CallbackScheduler:
    def __init__(self) -> None:
        self.pending: ScheduledCallback | None = None
        self.history: list[ScheduledCallback] = []
        self.triggered_history: list[ScheduledCallback] = []
        self.cancel_count = 0
        self.replacement_count = 0

    def apply(
        self,
        request: CallbackRequest | None,
        current_step: int,
        current_time: float,
        time_step: float,
    ) -> ScheduledCallback | None:
        if request is None:
            return self.pending
        if request.cancel:
            if self.pending is not None:
                self.cancel_count += 1
            self.pending = None
            return None
        if request.after_steps is not None:
            due_step = current_step + request.after_steps
        elif request.after_time is not None:
            due_step = current_step + math.ceil(request.after_time / time_step)
        else:
            assert request.at_time is not None
            due_step = current_step + math.ceil((request.at_time - current_time) / time_step)
        scheduled = ScheduledCallback(
            requested_at_step=current_step,
            due_step=due_step,
            due_time=round(due_step * time_step, 10),
            request=request.to_dict(),
        )
        if self.pending is not None:
            self.replacement_count += 1
        self.pending = scheduled
        self.history.append(scheduled)
        return scheduled

    def consume_if_due(self, step: int) -> ScheduledCallback | None:
        if self.pending is None or self.pending.due_step > step:
            return None
        callback = self.pending
        self.pending = None
        self.triggered_history.append(callback)
        return callback
