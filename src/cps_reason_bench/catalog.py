"""Public and answer-bearing diagnostic Markdown catalogs for benchmark suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cps_reason_bench.suites import BenchmarkSuiteConfig, materialize_suite
from cps_reason_bench.validation import validate_scenario_semantics


EXPECTED_CHALLENGES = {
    "prospective_reasoning": (
        "Arrange timely future monitoring or intervention without unnecessary polling."
    ),
    "planning": (
        "Respect prerequisite structure and resource limits across a multi-step repair."
    ),
    "interference": (
        "Reconcile conflicting physical evidence before selecting a consequential action."
    ),
    "conditional_safety": (
        "Gather relevant evidence and enforce one or more conditional physical rules."
    ),
    "adaptation": (
        "Revise an operational action-effect model after observable plant behavior changes."
    ),
    "temporal_tracking": (
        "Associate a delayed physical response with explicit prior observations and actions."
    ),
}


def build_catalog_records(
    suite: BenchmarkSuiteConfig,
    *,
    diagnostic: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for instance in materialize_suite(suite):
        scenario = instance.scenario
        validation = validate_scenario_semantics(scenario)
        profile = dict(scenario.difficulty_profile)
        reasoning = dict(profile.pop("reasoning_profile", {}))
        profile.pop("family_parameters", None)
        reference = validation.reference
        record: dict[str, Any] = {
            "scenario_identifier": scenario.name,
            "suite_name": suite.name,
            "suite_version": suite.version,
            "suite_entry_id": instance.suite_entry_id,
            "task_family": scenario.task_family,
            "controlled_difficulty_level": scenario.difficulty_level,
            "structural_difficulty_profile": profile,
            "reasoning_profile": reasoning,
            "simulator_seeds": list(instance.simulator_seeds),
            "generation_seed": instance.generation_seed,
            "generator_version": scenario.generation.get("generator_version"),
            "interaction_mode": scenario.interaction.mode,
            "context_mode": scenario.context.mode,
            "history_limit": scenario.context.history_limit,
            "number_of_nodes": len(scenario.graph["nodes"]),
            "number_of_variables": sum(
                len(node["variables"])
                for node in scenario.graph["nodes"].values()
            ),
            "number_of_sensors": len(scenario.sensors),
            "number_of_actions": len(scenario.actions),
            "distractor_action_count": sum(
                action.get("role") == "distractor" for action in scenario.actions
            ),
            "number_of_constraints": len(scenario.constraints),
            "number_of_goals": len(scenario.goals),
            "number_of_faults_or_regime_changes": len(scenario.faults),
            "expected_reasoning_challenge": EXPECTED_CHALLENGES.get(
                scenario.task_family,
                "Use public CPS evidence to select valid and safe actions.",
            ),
            "reference_type": reference.policy_kind if reference else None,
            "reference_valid": bool(
                reference and reference.script_valid and reference.task_success
            ),
            "solver_status": reference.solver.status if reference else None,
            "family_invariant_status": validation.family_invariants.status,
            "semantic_valid": validation.valid,
            "validation_warning_count": len(validation.warnings),
        }
        if diagnostic and reference:
            record["diagnostic_answers"] = {
                "reference_action_sequence": list(reference.valid_action_sequence),
                "required_event_subscriptions": list(
                    reference.required_event_subscriptions
                ),
                "required_inspections": list(reference.required_inspections),
                "solver_solution_sequences": [
                    list(sequence) for sequence in reference.solver.solution_sequences
                ],
            }
        records.append(record)
    return records


def render_catalog_markdown(
    suite: BenchmarkSuiteConfig,
    *,
    diagnostic: bool = False,
) -> str:
    records = build_catalog_records(suite, diagnostic=diagnostic)
    title = f"CPS Reason Bench Catalog: {suite.name}@{suite.version}"
    lines = [f"# {title}", ""]
    if diagnostic:
        lines.extend(
            [
                "> **DIAGNOSTIC ANSWER CATALOG.** This document contains reference "
                "actions and benchmark answers. Do not provide it to evaluated models.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> Public inspection catalog. Reference action sequences, hidden state, "
                "fault schedules, and solver solutions are intentionally omitted.",
                "",
            ]
        )
    lines.extend(
        [
            "Controlled levels describe intended structural and reasoning progression. "
            "They are not claims of empirical model difficulty.",
            "",
            "| Scenario | Family | Level | Simulator seeds | Interaction | Context | "
            "Nodes | Variables | Sensors | Actions | Distractors | Constraints | Goals | "
            "Faults | Reference | Solver | Invariants |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| {scenario_identifier} | {task_family} | {controlled_difficulty_level} | "
            "{simulator_seeds} | {interaction_mode} | {context_mode} | "
            "{number_of_nodes} | {number_of_variables} | {number_of_sensors} | "
            "{number_of_actions} | {distractor_action_count} | "
            "{number_of_constraints} | {number_of_goals} | "
            "{number_of_faults_or_regime_changes} | {reference_type} "
            "({reference_valid}) | {solver_status} | {family_invariant_status} |".format(
                **record
            )
        )
    lines.append("")
    for record in records:
        lines.extend(
            [
                f"## {record['scenario_identifier']}",
                "",
                record["expected_reasoning_challenge"],
                "",
                f"- Suite entry: `{record['suite_entry_id']}`",
                f"- Suite: `{record['suite_name']}@{record['suite_version']}`",
                f"- Generator version: `{record['generator_version']}`",
                f"- Generation seed: `{record['generation_seed']}`",
                f"- Simulator seeds: `{json.dumps(record['simulator_seeds'])}`",
                f"- Semantic validation: `{record['semantic_valid']}`",
                f"- Family invariants: `{record['family_invariant_status']}`",
                "",
                "### Structural Difficulty Profile",
                "",
                "```json",
                json.dumps(
                    record["structural_difficulty_profile"], indent=2, sort_keys=True
                ),
                "```",
                "",
                "### Intended Reasoning Profile",
                "",
                "```json",
                json.dumps(record["reasoning_profile"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
        if diagnostic:
            lines.extend(
                [
                    "### Diagnostic Answers",
                    "",
                    "```json",
                    json.dumps(record["diagnostic_answers"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_catalog(
    suite: BenchmarkSuiteConfig,
    output_path: str | Path,
    *,
    diagnostic: bool = False,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_catalog_markdown(suite, diagnostic=diagnostic), encoding="utf-8"
    )
    return destination
