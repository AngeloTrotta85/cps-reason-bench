"""Explicit, benchmark-maintained context and memory modes."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from cps_reason_bench.config import ContextConfig
from cps_reason_bench.types import EventTrigger


class ContextManager:
    def __init__(self, config: ContextConfig) -> None:
        self.config = config
        self.interactions: list[dict[str, Any]] = []
        self.requested_items: list[str] = []
        self._public_event_cursor = 0

    def add_requests(self, requests: Iterable[str]) -> None:
        for item in requests:
            if item not in self.requested_items:
                self.requested_items.append(item)
        self.requested_items = self.requested_items[-self.config.max_context_requests :]

    def build(
        self,
        observation: dict[str, Any],
        triggers: list[EventTrigger],
        active_subscriptions: list[dict[str, Any]],
        public_action_history: list[dict[str, Any]],
        public_event_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        visible_observation = copy.deepcopy(observation)
        if not self.config.include_action_outcomes:
            visible_observation.pop("last_action_outcomes", None)
        context: dict[str, Any] = {
            "observation": visible_observation,
            "invocation_triggers": [trigger.to_dict() for trigger in triggers],
        }
        if self.config.include_public_events:
            context["public_events_since_last_interaction"] = copy.deepcopy(
                public_event_history[self._public_event_cursor :]
            )
        self._public_event_cursor = len(public_event_history)
        if self.config.mode == "previous" and self.interactions:
            context["previous_interaction"] = copy.deepcopy(self.interactions[-1])
        elif self.config.mode == "history":
            context["interaction_history"] = copy.deepcopy(
                self.interactions[-self.config.history_limit :]
            )
        elif self.config.mode == "summary":
            context["structured_summary"] = self._summary()

        requested: dict[str, Any] = {}
        for item in self.requested_items:
            if item == "active_subscriptions":
                requested[item] = copy.deepcopy(active_subscriptions)
            elif item == "action_history":
                requested[item] = self._visible_action_history(
                    public_action_history[-self.config.history_limit :]
                )
            elif item == "event_history":
                requested[item] = copy.deepcopy(
                    public_event_history[-self.config.history_limit :]
                )
            elif item == "previous_decision" and self.interactions:
                requested[item] = copy.deepcopy(self.interactions[-1].get("decision"))
            elif item.startswith("sensor:"):
                sensor_id = item.split(":", 1)[1]
                if sensor_id in observation.get("sensors", {}):
                    requested[item] = copy.deepcopy(observation["sensors"][sensor_id])
            elif item.startswith("resource:"):
                resource_id = item.split(":", 1)[1]
                if resource_id in observation.get("resources", {}):
                    requested[item] = copy.deepcopy(observation["resources"][resource_id])
        if requested:
            context["requested_context"] = requested
        return context

    def record(
        self,
        step: int,
        observation: dict[str, Any],
        decision: dict[str, Any] | None,
        outcomes: list[dict[str, Any]],
    ) -> None:
        visible_observation = copy.deepcopy(observation)
        if not self.config.include_action_outcomes:
            visible_observation.pop("last_action_outcomes", None)
        interaction = {
            "step": step,
            "observation": visible_observation,
            "decision": copy.deepcopy(decision),
        }
        if self.config.include_action_outcomes:
            interaction["action_outcomes"] = copy.deepcopy(outcomes)
        self.interactions.append(interaction)

    def _summary(self) -> dict[str, Any]:
        accepted = 0
        rejected = 0
        actions: list[str] = []
        for interaction in self.interactions:
            if self.config.include_action_outcomes:
                for outcome in interaction.get("action_outcomes", []):
                    actions.append(str(outcome.get("action_id")))
                    if outcome.get("accepted"):
                        accepted += 1
                    else:
                        rejected += 1
            else:
                decision = interaction.get("decision")
                if isinstance(decision, dict):
                    actions.extend(
                        str(command.get("action_id"))
                        for command in decision.get("actions", [])
                        if isinstance(command, dict) and command.get("action_id")
                    )
        last = self.interactions[-1] if self.interactions else None
        summary = {
            "interaction_count": len(self.interactions),
            "recent_actions": actions[-self.config.history_limit :],
            "last_interaction_step": last.get("step") if last else None,
        }
        if self.config.include_action_outcomes:
            summary.update(
                {
                    "accepted_action_count": accepted,
                    "rejected_action_count": rejected,
                }
            )
        return summary

    def _visible_action_history(
        self, history: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        records = copy.deepcopy(list(history))
        if not self.config.include_action_outcomes:
            for record in records:
                record.pop("outcome", None)
        return records
