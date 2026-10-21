"""Trusted adapter: remove identity/private fields before calling pure PICC.

The standard PolicyRequest contains ScenarioConfig for historical baselines.
Only this adapter sees that object; it never forwards or retains it. The core
controller has no scenario, evaluator, simulator or unbounded memory handle.
"""

from __future__ import annotations

import json

from cps_reason_bench.policies.base import Policy, PolicyRequest, PolicyResult
from cps_reason_bench.policies.public_information import (
    CONTROLLER_VERSION, PublicInformationCalibrationController, PublicInput,
)
from cps_reason_bench.protocol import decision_response_schema, parse_decision
from cps_reason_bench.versions import content_hash


PROHIBITED = frozenset({
    "scenario", "scenario_id", "scenario_name", "scenario_configuration", "condition_id", "condition_name",
    "profile", "profile_id", "profile_label", "role", "role_id", "position", "entry", "instance", "seed",
    "difficulty", "difficulty_level", "difficulty_profile", "graph_role", "instance_index",
    "generation", "generation_id", "generation_seed", "evaluation", "reference", "reference_policy",
    "reference_actions", "reference_trajectory", "expected_exact_action_sequence",
    "private_state", "true_state", "full_state", "faults", "fault_schedule", "effects",
    "coefficient", "coupling", "goal_threshold", "safety_constraints", "metadata",
})


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in PROHIBITED and
                not k.startswith(("reference_", "private_", "evaluator_", "generation_"))}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("Public input must contain JSON values, not live objects")


def _select(value, keys):
    return {k: _clean(value[k]) for k in keys if k in value}


def _observation(obs):
    result = _select(obs, ("step", "time", "resources", "goal_status", "inspection_results", "last_action_outcomes"))
    result["topology"] = {
        "nodes": [_select(n, ("id", "label")) for n in obs.get("topology", {}).get("nodes", [])],
        "edges": [_select(e, ("id", "source", "target", "enabled")) for e in obs.get("topology", {}).get("edges", [])],
    }
    result["sensors"] = {key: _select(s, ("target", "status", "value", "age_steps"))
                         for key, s in obs.get("sensors", {}).items()}
    result["available_actions"] = [_select(a, ("id", "description", "preconditions"))
                                   for a in obs.get("available_actions", [])]
    return result


def sanitize_request(request: PolicyRequest) -> PublicInput:
    ctx = request.context
    context = _select(ctx, ("invocation_triggers", "public_events_since_last_interaction", "structured_summary"))
    if "requested_context" in ctx:
        context["requested_context"] = _select(ctx["requested_context"],
            ("action_history", "event_history", "active_subscriptions", "previous_decision"))
    context["observation"] = _observation(ctx["observation"])
    for key in ("previous_interaction", "interaction_history"):
        if key not in ctx:
            continue
        items = [ctx[key]] if key == "previous_interaction" else ctx[key]
        projected = []
        for item in items:
            clean = _select(item, ("step", "decision", "action_outcomes"))
            clean["observation"] = _observation(item.get("observation", {}))
            projected.append(clean)
        context[key] = projected[0] if key == "previous_interaction" else projected
    c = request.constraints
    # These are the same public fields exposed by LLMPolicy's protocol. Do not
    # forward DecisionConstraints: it also carries private horizon/time-step data.
    protocol = {
        "maximum_actions_now": c.max_actions,
        "callbacks_enabled": c.allows_callbacks,
        "callback_step_range": [c.min_callback_steps, c.max_callback_steps] if c.allows_callbacks else None,
        "subscriptions_enabled": c.allows_subscriptions,
        "allowed_event_types": list(c.allowed_event_types),
        "allowed_context_requests": list(c.allowed_context_requests),
        "allowed_action_ids": list(c.allowed_action_ids),
    }
    task = {"family": request.scenario.task_family,
            "instructions": request.scenario.task.get("instructions", "Maintain safety and achieve the goals."),
            "objectives": request.scenario.task.get("objectives", []),
            "safety_rules": request.scenario.task.get("safety_rules", [])}
    payload = _clean({"task": task, "context": context, "interaction_protocol": protocol})
    return PublicInput(json.dumps(payload, sort_keys=True, allow_nan=False))


class PublicInformationCalibrationPolicy(Policy):
    policy_name = "public_information_calibration"

    def decide(self, request: PolicyRequest) -> PolicyResult:
        public = sanitize_request(request)
        decision, rationale = PublicInformationCalibrationController().decide(public)
        raw = json.dumps(decision, sort_keys=True)
        parsed = parse_decision(raw, request.constraints)
        prompt = ({"role": "user", "content": public.serialized},)
        return PolicyResult(
            decision=parsed.decision, prompt=prompt, raw_response=raw,
            parsed_response=parsed.parsed, validation_errors=parsed.errors,
            attempts=({"attempt": 1, "json_valid": parsed.json_valid,
                       "schema_valid": parsed.schema_valid, "decision_valid": parsed.decision_valid,
                       "validation_errors": list(parsed.errors)},),
            rendered_prompt_hash=content_hash(prompt),
            response_schema_hash=content_hash(decision_response_schema(request.constraints)),
            response_processing={"picc_public_input": public.to_dict(), "picc_rationale": rationale},
        )

    def metadata(self) -> dict:
        return {"policy": self.policy_name, "model": "PICC", "controller_version": CONTROLLER_VERSION,
                "deterministic": True, "memory": "runner_supplied_public_context_only",
                "outside_frozen_9600": True}
