"""PICC v1: stateless, public-evidence heuristics, not a reference controller.

This module deliberately has no runner, scenario, evaluator, filesystem or model
imports. All memory comes from the supplied bounded public context. Ambiguous
polarity/magnitude choices are deferred, never broken by action-ID ordering.
"""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass


CONTROLLER_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class PublicInput:
    """Detached JSON, with no live references back to the runner's objects."""

    serialized: str

    def to_dict(self) -> dict:
        return json.loads(self.serialized)


class PublicInformationCalibrationController:
    """Choose at most one command per invocation, solely from public cues."""

    __slots__ = ()

    def decide(self, public: PublicInput) -> tuple[dict, dict]:
        if type(public) is not PublicInput:
            raise TypeError("PICC accepts only a sanitized PublicInput")
        view = public.to_dict()
        ctx = view["context"]
        obs = ctx["observation"]
        protocol = view["interaction_protocol"]
        family = view["task"]["family"]
        actions = obs.get("available_actions", [])
        evidence = []

        def cite(path, value):
            evidence.append({"path": path, "value": value})

        cite(["task", "family"], family)
        cite(["task", "instructions"], view["task"]["instructions"])
        catalogue = {a["id"]: a for a in actions}
        # Only public outcomes count as acceptance; a submitted decision is not proof.
        records = []
        for path, interaction in self._interactions(ctx):
            for index, outcome in enumerate(interaction.get("action_outcomes", [])):
                records.append((interaction.get("step", -1), outcome, path + ["action_outcomes", index]))
        for index, item in enumerate(ctx.get("requested_context", {}).get("action_history", [])):
            if "outcome" in item:
                records.append((item.get("step", -1), item["outcome"],
                                ["context", "requested_context", "action_history", index, "outcome"]))
        for index, outcome in enumerate(obs.get("last_action_outcomes", [])):
            records.append((obs.get("step", 0) - 1, outcome,
                            ["context", "observation", "last_action_outcomes", index]))
        records.sort(key=lambda r: r[0])

        def accepted(action_id):
            matches = [r for r in records if r[1].get("action_id") == action_id and r[1].get("accepted") is True]
            if matches:
                cite(matches[-1][2], matches[-1][1])
            return matches

        # Current trustworthy sensor readings take precedence over retained inspections.
        facts = {}
        for path, interaction in self._interactions(ctx):
            self._inspection_facts(interaction.get("observation", {}), path + ["observation"], facts)
        self._inspection_facts(obs, ["context", "observation"], facts)
        for sensor_id, sensor in obs.get("sensors", {}).items():
            if sensor.get("status") == "ok" and sensor.get("age_steps", 0) == 0:
                facts[sensor["target"]] = (sensor.get("value"), ["context", "observation", "sensors", sensor_id])
        for edge in obs.get("topology", {}).get("edges", []):
            if "enabled" in edge:
                index = obs["topology"]["edges"].index(edge)
                facts["edge:" + edge["id"]] = (edge["enabled"], ["context", "observation", "topology", "edges", index])

        def fact(target):
            if target in facts:
                value, path = facts[target]
                node = view
                for key in path:
                    node = node[key]
                cite(path, node)
                return value
            return None

        def description_matches(prefix):
            return [a for a in actions if a.get("description", "").lower().startswith(prefix)]

        def emit(rule, action=None, callback=False):
            decision = {"actions": [], "callback": None, "subscriptions": [], "context_requests": []}
            if action is not None and protocol["maximum_actions_now"] > 0:
                cite(["context", "observation", "available_actions", actions.index(action)], action)
                decision["actions"] = [{"action_id": action["id"], "parameters": {}}]
            if callback and protocol["callbacks_enabled"]:
                cite(["interaction_protocol", "callback_step_range"], protocol["callback_step_range"])
                # Poll at the permitted minimum; never assume how a hidden clock advances.
                decision["callback"] = {"after_steps": protocol["callback_step_range"][0]}
            decision["context_requests"] = [x for x in ("action_history", "event_history")
                                             if x in protocol["allowed_context_requests"]]
            return decision, {"version": CONTROLLER_VERSION, "rule": rule, "evidence": evidence,
                              "selection": "public_cue" if action else "defer_or_observe",
                              "unbounded_private_memory": False}

        # A met public goal justifies passive continuation, not a private termination test.
        if obs.get("goal_status") and all(g.get("met") is True for g in obs["goal_status"]):
            cite(["context", "observation", "goal_status"], obs["goal_status"])
            return emit("public_goal_met")

        if family == "conditional_safety":
            safeguards = [key for key in facts if key.endswith(".safeguard")]
            protection = fact(safeguards[0]) if len(safeguards) == 1 else None
            if protection is not True:
                commands = description_matches("inspect") if protection is None else description_matches("enable")
                commands = [a for a in commands if "safeguard" in a.get("description", "").lower()]
                if len(commands) == 1 and not accepted(commands[0]["id"]):
                    return emit("inspect_unknown_safeguard" if protection is None else "enable_absent_safeguard", commands[0])
                return emit("safeguard_not_publicly_established")

        if family == "temporal_tracking":
            for action in description_matches("attribute the downstream response to "):
                source = action["description"].removeprefix("Attribute the downstream response to ").rstrip(".")
                if source in catalogue and accepted(source) and not accepted(action["id"]):
                    if self._ready(action, fact):
                        return emit("attribute_publicly_ready_source", action)

        edges = obs.get("topology", {}).get("edges", [])
        if family == "interference":
            inspections = description_matches("inspect the relevant graph or sensor evidence")
            if any("edge:" + e["id"] not in facts for e in edges) and len(inspections) == 1:
                if not accepted(inspections[0]["id"]):
                    cite(["context", "observation", "topology"], obs.get("topology", {}))
                    return emit("inspect_unknown_relation", inspections[0])

        disabled = [e for e in edges if fact("edge:" + e["id"]) is False]
        restorers = description_matches("restore")
        if len(disabled) == 1 and len(restorers) == 1:
            # A sole transfer edge or a uniquely named route is an operational cue.
            edge = disabled[0]
            description = restorers[0]["description"].lower()
            route = re.search(r"route ([a-z])\b", description)
            matches = len(edges) == 1 or (route and edge["source"] == "source_" + route[1])
            if matches and not accepted(restorers[0]["id"]):
                return emit("restore_observed_disabled_path", restorers[0])

        if family == "adaptation":
            repairs = description_matches("repair the failed")
            events = list(ctx.get("public_events_since_last_interaction", []))
            events += list(ctx.get("requested_context", {}).get("event_history", []))
            failed = [r for r in records if r[1].get("accepted") is False and
                      any(word in r[1].get("message", "").lower() for word in ("actuator unavailable", "actuator failed"))]
            failure_events = [e for e in events if e.get("event_type") in {"actuator_failed", "fault_detected"}]
            if len(repairs) == 1 and (disabled or failed or failure_events) and not accepted(repairs[0]["id"]):
                # Record actual public inputs, never the private fault schedule.
                cite(["context"], ctx)
                return emit("repair_after_public_failure", repairs[0])

        candidates = [a for a in actions if a.get("description", "").lower().startswith(("energize", "apply"))
                      and "opposite" not in a["description"].lower() and "opposing" not in a["description"].lower()]
        useful = [a for a in candidates if "useful local command" in a["description"].lower()]
        choice = useful[0] if len(useful) == 1 else None
        if choice is None and len(candidates) == 1:
            choice = candidates[0]
        if choice is None and candidates:
            # Direction follows public source/target ordering, not hidden coefficients.
            reachable = []
            for action in candidates:
                match = re.search(r"(?:source|route) ([a-z])\b", action["description"].lower())
                if match and self._path("source_" + match[1], "sink", edges, facts):
                    reachable.append(action)
            if len(reachable) == 1:
                cite(["context", "observation", "topology"], obs.get("topology", {}))
                choice = reachable[0]
        if choice is None:
            return emit("unresolved_action_choice", callback=family == "prospective_reasoning")

        previous = accepted(choice["id"])
        if previous:
            # Re-intervene only after a publicly accepted repair later than the pulse.
            repaired = [r for r in records if r[1].get("accepted") is True and
                        r[1].get("action_id") in catalogue and catalogue[r[1]["action_id"]].get("description", "").lower().startswith(("repair", "restore"))]
            if family != "adaptation" or not repaired or repaired[-1][0] <= previous[-1][0]:
                return emit("accepted_effect_pending", callback=family == "prospective_reasoning")
            cite(repaired[-1][2], repaired[-1][1])

        if family == "planning":
            preparations = [a for a in actions if a.get("description") == f"Prepare the component used by {choice['id']}."]
            if preparations:
                preparation = preparations[0]
                if not accepted(preparation["id"]):
                    return emit("prepare_public_prerequisite", preparation)
                # Accepted named preparation is an explicit public operational cue,
                # not access to the hidden armed variable or its private effect.
                return emit("activate_after_accepted_preparation", choice)
        if not self._ready(choice, fact):
            return emit("public_readiness_not_established", callback=family == "prospective_reasoning")
        return emit("activate_publicly_supported_action", choice)

    @staticmethod
    def _ready(action, fact):
        comparisons = {"eq": operator.eq, "ne": operator.ne, "ge": operator.ge,
                       "gt": operator.gt, "le": operator.le, "lt": operator.lt}
        for condition in action.get("preconditions", []):
            value = fact(condition["target"])
            compare = comparisons.get(condition.get("operator", "eq"))
            if value is None or compare is None:
                return False
            try:
                if not compare(value, condition["value"]):
                    return False
            except TypeError:
                return False
        return True

    @staticmethod
    def _interactions(ctx):
        items = [(["context", "interaction_history", i], value)
                 for i, value in enumerate(ctx.get("interaction_history", []))]
        if ctx.get("previous_interaction"):
            items.append((["context", "previous_interaction"], ctx["previous_interaction"]))
        return items

    @staticmethod
    def _inspection_facts(obs, path, facts):
        for index, item in enumerate(obs.get("inspection_results", [])):
            key = item.get("target")
            if item.get("component_type") == "edge" and "enabled" in item.get("status", {}):
                facts["edge:" + key] = (item["status"]["enabled"], path + ["inspection_results", index])
            elif "value" in item:
                facts[key] = (item["value"], path + ["inspection_results", index])

    @staticmethod
    def _path(source, target, edges, facts):
        seen, pending = set(), [source]
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(e["target"] for e in edges if e["source"] == node and
                               facts.get("edge:" + e["id"], (None,))[0] is not False)
        return False
