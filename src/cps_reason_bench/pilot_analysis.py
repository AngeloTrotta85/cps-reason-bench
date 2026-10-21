"""Trace-derived empirical summaries for local-model benchmark pilots."""

from __future__ import annotations

import copy
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LEVEL_ORDER = {
    "basic": 1,
    "controlled_1": 1,
    "controlled_2": 2,
    "controlled_3": 3,
    "controlled_4": 4,
}


def write_pilot_reports(
    experiment_dirs: Iterable[str | Path],
    output_json: str | Path,
    output_markdown: str | Path,
    *,
    provider_check_path: str | Path | None = None,
) -> dict[str, Path]:
    summary = build_pilot_summary(
        experiment_dirs, provider_check_path=provider_check_path
    )
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_pilot_markdown(summary), encoding="utf-8")
    return {"pilot_summary_json": json_path, "pilot_report_markdown": markdown_path}


def build_pilot_summary(
    experiment_dirs: Iterable[str | Path],
    *,
    provider_check_path: str | Path | None = None,
) -> dict[str, Any]:
    sources = [_load_experiment_directory(Path(path)) for path in experiment_dirs]
    records = [record for source in sources for record in source["records"]]
    failures = [failure for source in sources for failure in source["failures"]]
    if not records and not failures:
        raise ValueError("Pilot analysis found no episodes or failure records")
    for source in sources:
        for record in source["records"]:
            record.setdefault("source_experiment_dir", source["directory"])
    taxonomies = [classify_episode(record) for record in records]
    provider_checks = None
    if provider_check_path:
        provider_checks = json.loads(
            Path(provider_check_path).read_text(encoding="utf-8")
        )

    by_model = _group_summary(records, ("model_configuration", "model"))
    by_family = _group_summary(records, ("model_configuration", "model", "task_family"))
    by_level = _group_summary(
        records,
        ("model_configuration", "model", "task_family", "difficulty_level"),
    )
    by_scenario = _group_summary(
        records, ("scenario", "task_family", "difficulty_level")
    )
    provenance = _merge_provenance(sources)
    summary = {
        "analysis_type": "cps_reason_bench_local_ollama_pilot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": (
            "Initial empirical characterization; not a definitive model ranking or "
            "psychological assessment."
        ),
        "source_experiments": [
            {
                "directory": source["directory"],
                "manifest": source["manifest_path"],
                "episodes": source["episodes_path"],
                "failures": source["failures_path"],
                "experiment_name": source["manifest"].get("experiment_name"),
                "experiment_content_hash": source["manifest"].get(
                    "experiment_content_hash"
                ),
                "declared_run_count": source["manifest"].get("run_count"),
                "repetition_count": source["manifest"]
                .get("experiment", {})
                .get("repetitions"),
                "completed_episode_count": len(source["records"]),
                "infrastructure_failure_count": len(source["failures"]),
            }
            for source in sources
        ],
        "benchmark_versions": _benchmark_versions(records),
        "models": _models(records, provenance),
        "provenance": provenance,
        "provider_checks": provider_checks,
        "episode_count": len(records),
        "infrastructure_failures": [
            _classify_infrastructure_failure(item) for item in failures
        ],
        "results_by_model": by_model,
        "results_by_family": by_family,
        "results_by_difficulty_level": by_level,
        "results_by_scenario": by_scenario,
        "temporal_context_ablation": _context_ablation(records),
        "empirical_difficulty": _empirical_difficulty(by_level),
        "failure_taxonomy": _taxonomy_summary(taxonomies),
        "episode_failure_classifications": [
            _classification_without_trace(item) for item in taxonomies
        ],
        "representative_episodes": _representative_episodes(taxonomies),
        "calibration_concerns": _calibration_concerns(records, by_scenario, taxonomies),
        "artifact_integrity": _artifact_integrity(sources, records, failures),
        "traceability": {
            "all_run_ids": [str(record.get("run_id")) for record in records],
            "tables_reconstructable_from": [
                source["episodes_path"] for source in sources
            ],
            "raw_traces_referenced_by_episode_records": True,
            "complete_raw_traces_duplicated_in_summary": False,
        },
    }
    return summary


def classify_episode(
    record: dict[str, Any], trace_records: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    metrics = record.get("metrics", {})
    general = metrics.get("general", {})
    protocol = metrics.get("protocol_compliance", {})
    task = metrics.get("task_specific", {})
    trace = trace_records if trace_records is not None else _load_trace(record)
    interactions = [
        item for item in trace if item.get("record_type") == "model_interaction"
    ]
    categories: list[dict[str, Any]] = []

    def add(group: str, category: str, metric: str, value: Any) -> None:
        if any(item["category"] == category for item in categories):
            return
        categories.append(
            {
                "group": group,
                "category": category,
                "evidence": {"metric_or_trace_field": metric, "value": value},
                "representative_observable_trace": _observable_excerpt(interactions),
            }
        )

    for interaction in interactions:
        for attempt in interaction.get("output_attempts", []):
            raw = attempt.get("raw_response")
            errors = " ".join(
                str(item) for item in attempt.get("validation_errors", [])
            )
            stage = attempt.get("validation_stage")
            if stage == "backend":
                lowered = errors.lower()
                category = (
                    "timeout"
                    if "timeout" in lowered or "timed out" in lowered
                    else "backend_error"
                )
                if "404" in lowered or "not found" in lowered:
                    category = "model_unavailable"
                add("protocol", category, "output_attempts.validation_errors", errors)
                continue
            if raw is not None and not str(raw).strip():
                add("protocol", "empty_response", "output_attempts.raw_response", "")
            if not attempt.get("json_valid", False):
                add("protocol", "invalid_json", "output_attempts.json_valid", False)
            elif not attempt.get("schema_valid", False):
                add("protocol", "invalid_schema", "output_attempts.schema_valid", False)
            elif not attempt.get("decision_valid", False):
                add(
                    "protocol",
                    "illegal_decision",
                    "output_attempts.decision_valid",
                    False,
                )
            if "unsupported" in errors.lower() and "parameter" in errors.lower():
                add(
                    "protocol",
                    "unsupported_action_parameter",
                    "output_attempts.validation_errors",
                    errors,
                )

    if general.get("backend_errors", 0):
        add(
            "protocol",
            "backend_error",
            "general.backend_errors",
            general["backend_errors"],
        )
    if task.get("missed_critical_callback"):
        add(
            "interaction_management",
            "missing_callback",
            "task_specific.missed_critical_callback",
            True,
        )
    timing_error = task.get("callback_timing_error_steps")
    if isinstance(timing_error, (int, float)) and timing_error > 0:
        add(
            "interaction_management",
            "callback_timing_error",
            "task_specific.callback_timing_error_steps",
            timing_error,
        )
    if task.get("unnecessary_callbacks", 0):
        add(
            "interaction_management",
            "excessive_polling",
            "task_specific.unnecessary_callbacks",
            task["unnecessary_callbacks"],
        )
    if general.get("irrelevant_subscriptions", 0):
        add(
            "interaction_management",
            "irrelevant_subscription",
            "general.irrelevant_subscriptions",
            general["irrelevant_subscriptions"],
        )
    if (
        metrics.get("task_family") == "prospective_reasoning"
        and not general.get("task_success")
        and not task.get("useful_subscription_count", 0)
    ):
        add(
            "interaction_management",
            "missing_useful_subscription",
            "task_specific.useful_subscription_count",
            task.get("useful_subscription_count", 0),
        )

    family = metrics.get("task_family")
    if family == "planning":
        if task.get("prerequisite_violations", 0):
            add(
                "planning",
                "unmet_prerequisite",
                "task_specific.prerequisite_violations",
                task["prerequisite_violations"],
            )
        repeated = _repeated_rejected_actions(interactions)
        if repeated:
            add(
                "planning",
                "repeated_invalid_action",
                "trace.repeated_rejected_actions",
                repeated,
            )
        if task.get("expected_sequence_satisfied") is False:
            add(
                "planning",
                "incorrect_action_order",
                "task_specific.expected_sequence_satisfied",
                False,
            )
        if not general.get("task_success") and protocol.get(
            "final_executable_decisions", 0
        ):
            add(
                "planning",
                "incomplete_plan_before_horizon",
                "general.task_success",
                False,
            )

    if family == "interference":
        if task.get("misleading_action_count", 0):
            add(
                "interference",
                "impulsive_response_to_misleading_signal",
                "task_specific.misleading_action_count",
                task["misleading_action_count"],
            )
        if not general.get("task_success") and not task.get(
            "evidence_gathering_action_count", 0
        ):
            add(
                "interference",
                "insufficient_evidence_gathering",
                "task_specific.evidence_gathering_action_count",
                0,
            )
        if general.get("safety_violations", 0):
            add(
                "interference",
                "unsafe_local_action",
                "general.safety_violations",
                general["safety_violations"],
            )

    if family == "conditional_safety":
        coverage = task.get("required_inspection_coverage")
        if isinstance(coverage, (int, float)) and coverage < 1:
            add(
                "conditional_safety",
                "missed_required_inspection",
                "task_specific.required_inspection_coverage",
                coverage,
            )
        if task.get("unnecessary_inspection_count", 0):
            add(
                "conditional_safety",
                "unnecessary_inspection",
                "task_specific.unnecessary_inspection_count",
                task["unnecessary_inspection_count"],
            )
        if task.get("missed_falsifying_observations", 0):
            add(
                "conditional_safety",
                "failure_to_identify_falsifying_state",
                "task_specific.missed_falsifying_observations",
                task["missed_falsifying_observations"],
            )
        if task.get("final_rule_satisfaction") is False:
            add(
                "conditional_safety",
                "failure_to_enforce_rule",
                "task_specific.final_rule_satisfaction",
                False,
            )

    if family == "adaptation":
        if task.get("obsolete_actions_before_sufficient_evidence", 0):
            add(
                "adaptation",
                "obsolete_action_before_sufficient_evidence",
                "task_specific.obsolete_actions_before_sufficient_evidence",
                task["obsolete_actions_before_sufficient_evidence"],
            )
        if task.get("obsolete_actions_after_sufficient_evidence", 0):
            add(
                "adaptation",
                "obsolete_action_after_sufficient_evidence",
                "task_specific.obsolete_actions_after_sufficient_evidence",
                task["obsolete_actions_after_sufficient_evidence"],
            )
        if task.get("first_revised_action") and not task.get(
            "first_effective_post_change_action"
        ):
            add(
                "adaptation",
                "incorrect_replacement_strategy",
                "task_specific.first_revised_action",
                task["first_revised_action"],
            )
        if task.get("safety_violations_during_recovery", 0):
            add(
                "adaptation",
                "unsafe_recovery_strategy",
                "task_specific.safety_violations_during_recovery",
                task["safety_violations_during_recovery"],
            )
        if not task.get("adaptation_success", False):
            add(
                "adaptation",
                "failure_to_recover_before_horizon",
                "task_specific.adaptation_success",
                False,
            )

    if family == "temporal_tracking":
        if task.get("missed_temporal_dependencies", 0):
            add(
                "temporal",
                "missed_delayed_effect",
                "task_specific.missed_temporal_dependencies",
                task["missed_temporal_dependencies"],
            )
        if task.get("false_temporal_associations", 0):
            add(
                "temporal",
                "incorrect_prior_action_attribution",
                "task_specific.false_temporal_associations",
                task["false_temporal_associations"],
            )
        if not task.get("temporal_reasoning_success", False):
            context_mode = record.get("context_mode")
            category = (
                "failure_caused_by_insufficient_explicit_context"
                if context_mode == "current"
                else "failure_despite_sufficient_context"
            )
            add("temporal", category, "context_mode", context_mode)

    if not general.get("task_success") and not categories:
        add(
            "cps_outcome",
            "executable_but_unsuccessful_decision",
            "general.task_success",
            False,
        )
    return {
        "run_id": record.get("run_id"),
        "episode_id": record.get("episode_id"),
        "model_configuration": record.get("model_configuration"),
        "model": record.get("model"),
        "scenario": record.get("scenario"),
        "task_family": family,
        "difficulty_level": record.get("difficulty_level"),
        "task_success": bool(general.get("task_success")),
        "categories": categories,
        "representative_observable_trace": _observable_excerpt(interactions),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Public helper used by tests and external analysis notebooks."""

    return _summarize_group(records)


def empirical_level_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _group_summary(
        records,
        ("model_configuration", "model", "task_family", "difficulty_level"),
    )


def _load_experiment_directory(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    episodes_path = directory / "episodes.jsonl"
    failures_path = directory / "failures.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    if episodes_path.is_file():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    failures = (
        json.loads(failures_path.read_text(encoding="utf-8"))
        if failures_path.is_file()
        else []
    )
    return {
        "directory": str(directory.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "episodes_path": str(episodes_path.resolve()),
        "failures_path": str(failures_path.resolve()),
        "manifest": manifest,
        "records": records,
        "failures": failures,
    }


def _group_summary(
    records: list[dict[str, Any]], dimensions: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(item) for item in dimensions)
        groups.setdefault(key, []).append(record)
    return [
        {
            "dimensions": dict(zip(dimensions, key, strict=True)),
            **_summarize_group(group),
        }
        for key, group in sorted(groups.items(), key=lambda item: repr(item[0]))
    ]


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"run_count": 0, "run_ids": []}
    general = [record.get("metrics", {}).get("general", {}) for record in records]
    protocol = [
        record.get("metrics", {}).get("protocol_compliance", {}) for record in records
    ]
    task_metrics = [
        record.get("metrics", {}).get("task_specific", {}) for record in records
    ]
    first_count = sum(int(item.get("first_attempt_count", 0)) for item in protocol)
    invocations = sum(int(item.get("llm_or_policy_invocations", 0)) for item in general)
    success_count = sum(bool(item.get("task_success")) for item in general)
    safety_episodes = sum(int(item.get("safety_violations", 0)) > 0 for item in general)
    tokens = Counter()
    resources = Counter()
    for item in general:
        usage = item.get("token_usage", {})
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    tokens[str(key)] += value
        consumption = item.get("resource_consumption", {})
        if isinstance(consumption, dict):
            for key, value in consumption.items():
                if isinstance(value, (int, float)):
                    resources[str(key)] += value
    return {
        "run_count": len(records),
        "success_count": success_count,
        "success_rate": success_count / len(records),
        "model_task_failure_count": len(records) - success_count,
        "infrastructure_failure_count": 0,
        "backend_error_count": sum(
            int(item.get("backend_errors", 0)) for item in general
        ),
        "first_attempt_count": first_count,
        "first_attempt_json_valid_count": sum(
            int(item.get("first_attempt_json_valid", 0)) for item in protocol
        ),
        "first_attempt_json_valid_rate": _ratio_sum(
            protocol, "first_attempt_json_valid", first_count
        ),
        "first_attempt_schema_valid_count": sum(
            int(item.get("first_attempt_schema_valid", 0)) for item in protocol
        ),
        "first_attempt_schema_valid_rate": _ratio_sum(
            protocol, "first_attempt_schema_valid", first_count
        ),
        "first_attempt_decision_valid_count": sum(
            int(item.get("first_attempt_decision_valid", 0)) for item in protocol
        ),
        "first_attempt_decision_valid_rate": _ratio_sum(
            protocol, "first_attempt_decision_valid", first_count
        ),
        "final_executable_decision_count": sum(
            int(item.get("final_executable_decisions", 0)) for item in protocol
        ),
        "final_executable_decision_rate": (
            _sum(protocol, "final_executable_decisions") / invocations
            if invocations
            else None
        ),
        "safety_violation_count": sum(
            int(item.get("safety_violations", 0)) for item in general
        ),
        "safety_violation_episode_count": safety_episodes,
        "safety_violation_rate": safety_episodes / len(records),
        "total_simulation_steps": int(_sum(general, "simulation_steps")),
        "mean_simulation_steps": _mean(general, "simulation_steps"),
        "total_llm_invocations": int(
            _sum(general, "llm_or_policy_invocations")
        ),
        "total_accepted_actions": int(_sum(general, "accepted_actions")),
        "total_invalid_actions": int(_sum(general, "invalid_actions")),
        "total_unnecessary_actions": int(_sum(general, "unnecessary_actions")),
        "total_invalid_output_attempts": int(
            _sum(general, "invalid_output_attempts")
        ),
        "total_invalid_structured_outputs": int(
            _sum(general, "invalid_structured_outputs")
        ),
        "total_output_retries": int(_sum(general, "output_retries")),
        "mean_llm_invocations": _mean(general, "llm_or_policy_invocations"),
        "mean_model_calls": _mean(general, "model_calls_including_retries"),
        "mean_accepted_actions": _mean(general, "accepted_actions"),
        "mean_invalid_actions": _mean(general, "invalid_actions"),
        "mean_unnecessary_actions": _mean(general, "unnecessary_actions"),
        "mean_interaction_cost": _mean(general, "interaction_cost"),
        "total_resource_consumption": dict(sorted(resources.items())),
        "total_token_usage": dict(sorted(tokens.items())),
        "mean_total_tokens": (
            float(tokens.get("total_tokens", 0)) / len(records)
            if "total_tokens" in tokens
            else None
        ),
        "total_latency_seconds": _sum(general, "total_latency_seconds"),
        "mean_latency_seconds": _mean(general, "total_latency_seconds"),
        "family_specific_metrics": _numeric_metric_summary(task_metrics),
        "run_ids": [str(record.get("run_id")) for record in records],
    }


def _empirical_difficulty(level_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for item in level_groups:
        dims = item["dimensions"]
        key = (
            dims.get("model_configuration"),
            dims.get("model"),
            dims.get("task_family"),
        )
        grouped.setdefault(key, []).append(item)
    output = []
    for key, levels in sorted(grouped.items(), key=lambda item: repr(item[0])):
        ordered = sorted(
            levels,
            key=lambda item: LEVEL_ORDER.get(
                str(item["dimensions"].get("difficulty_level")), 99
            ),
        )
        rates = [float(item["success_rate"]) for item in ordered]
        if rates and all(rate == 1.0 for rate in rates):
            pattern = "apparent_ceiling"
        elif rates and all(rate == 0.0 for rate in rates):
            pattern = "apparent_floor"
        elif len(set(rates)) == 1:
            pattern = "flat_performance"
        elif all(left >= right for left, right in zip(rates, rates[1:])):
            pattern = "monotonic_nonincreasing"
        elif rates and rates[-1] < rates[0]:
            pattern = "partial_decline_nonmonotonic"
        else:
            pattern = "non_monotonic"
        output.append(
            {
                "model_configuration": key[0],
                "model": key[1],
                "task_family": key[2],
                "pattern": pattern,
                "levels": [
                    {
                        "difficulty_level": item["dimensions"].get("difficulty_level"),
                        "run_count": item["run_count"],
                        "success_count": item["success_count"],
                        "success_rate": item["success_rate"],
                        "protocol_decision_valid_rate": item[
                            "first_attempt_decision_valid_rate"
                        ],
                    }
                    for item in ordered
                ],
                "interpretation_limit": "Descriptive only; no significance test was performed.",
            }
        )
    return output


def _context_ablation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suite_roles = {
        "cps_reason_temporal_current_control": "current_only_negative_control",
        "cps_reason_temporal_history": "explicit_history",
    }
    temporal = [
        record
        for record in records
        if record.get("task_family") == "temporal_tracking"
        and record.get("suite_name") in suite_roles
    ]
    groups = _group_summary(
        temporal,
        (
            "model_configuration",
            "model",
            "scenario",
            "suite_name",
            "context_mode",
        ),
    )
    return [
        {
            **item,
            "comparison_role": suite_roles[item["dimensions"]["suite_name"]],
        }
        for item in groups
    ]


def _taxonomy_summary(taxonomies: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[tuple[str, str]] = Counter()
    run_ids: dict[tuple[str, str], list[str]] = {}
    examples: dict[tuple[str, str], Any] = {}
    for episode in taxonomies:
        for item in episode["categories"]:
            key = (item["group"], item["category"])
            counts[key] += 1
            run_ids.setdefault(key, []).append(str(episode["run_id"]))
            examples.setdefault(key, item.get("representative_observable_trace"))
    return {
        "assignment_basis": "Observable outputs, actions, public outcomes, and benchmark metrics only.",
        "categories": [
            {
                "group": key[0],
                "category": key[1],
                "episode_count": count,
                "run_ids": run_ids[key],
                "representative_observable_trace": examples.get(key),
            }
            for key, count in sorted(counts.items())
        ],
    }


def _classification_without_trace(classification: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            key: value
            for key, value in classification.items()
            if key != "representative_observable_trace"
        },
        "categories": [
            {
                key: value
                for key, value in category.items()
                if key != "representative_observable_trace"
            }
            for category in classification.get("categories", [])
        ],
    }


def _representative_episodes(
    taxonomies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for episode in taxonomies:
        key = (str(episode.get("model")), str(episode.get("task_family")))
        grouped.setdefault(key, []).append(episode)
    representatives = []
    for _, episodes in sorted(grouped.items()):
        selected = next(
            (episode for episode in episodes if episode.get("task_success")),
            episodes[0],
        )
        representatives.append(
            {
                "run_id": selected.get("run_id"),
                "episode_id": selected.get("episode_id"),
                "model": selected.get("model"),
                "task_family": selected.get("task_family"),
                "scenario": selected.get("scenario"),
                "difficulty_level": selected.get("difficulty_level"),
                "task_success": selected.get("task_success"),
                "failure_categories": [
                    item.get("category") for item in selected.get("categories", [])
                ],
                "observable_trace": selected.get("representative_observable_trace"),
            }
        )
    return representatives


def _artifact_integrity(
    sources: list[dict[str, Any]],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    run_ids = [str(record.get("run_id")) for record in records]
    episode_ids = [str(record.get("episode_id")) for record in records]
    duplicate_run_ids = sorted(
        value for value, count in Counter(run_ids).items() if count > 1
    )
    duplicate_episode_ids = sorted(
        value for value, count in Counter(episode_ids).items() if count > 1
    )
    declared_counts = [
        source["manifest"].get("run_count") for source in sources
    ]
    declared_total = (
        sum(int(value) for value in declared_counts)
        if all(isinstance(value, int) for value in declared_counts)
        else None
    )
    source_results = []
    raw_status_counts: Counter[str] = Counter()
    unreferenced_traces: list[str] = []
    missing_referenced_traces: list[str] = []
    launcher_checks: list[bool] = []

    for source in sources:
        directory = Path(source["directory"])
        referenced = set()
        for record in source["records"]:
            value = record.get("events_path")
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = directory / path
            referenced.add(str(path.resolve()))
            if not path.is_file():
                missing_referenced_traces.append(str(path.resolve()))

        traces = sorted((directory / "episodes").glob("*.jsonl"))
        source_statuses: Counter[str] = Counter()
        for trace in traces:
            lines = [
                line
                for line in trace.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            try:
                final_record_type = (
                    json.loads(lines[-1]).get("record_type") if lines else None
                )
            except json.JSONDecodeError:
                final_record_type = "malformed"
            status = str(final_record_type or "incomplete")
            source_statuses[status] += 1
            raw_status_counts[status] += 1
            if str(trace.resolve()) not in referenced:
                unreferenced_traces.append(str(trace.resolve()))

        launcher_path = directory / "launcher_state.json"
        launcher_summary = None
        if launcher_path.is_file():
            launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
            launcher_statuses = Counter(
                str(item.get("status"))
                for item in launcher.get("episodes", {}).values()
            )
            launcher_summary = {
                "path": str(launcher_path.resolve()),
                "active_episode": launcher.get("active_episode"),
                "completed_episodes": launcher.get("completed_episodes"),
                "episode_status_counts": dict(sorted(launcher_statuses.items())),
            }
            launcher_checks.append(
                launcher.get("active_episode") is None
                and launcher.get("completed_episodes") == len(source["records"])
                and launcher_statuses.get("completed", 0) == len(source["records"])
            )

        source_results.append(
            {
                "experiment_name": source["manifest"].get("experiment_name"),
                "declared_run_count": source["manifest"].get("run_count"),
                "exported_episode_count": len(source["records"]),
                "failure_record_count": len(source["failures"]),
                "referenced_raw_trace_count": len(referenced),
                "raw_trace_count": len(traces),
                "raw_trace_final_record_types": dict(sorted(source_statuses.items())),
                "launcher": launcher_summary,
            }
        )

    export_count_consistent = (
        declared_total == len(records) + len(failures)
        if declared_total is not None
        else None
    )
    trace_count_verifiable = any(
        source["referenced_raw_trace_count"] for source in source_results
    )
    trace_consistent = (
        raw_status_counts.get("run_completed", 0) == len(records)
        and not missing_referenced_traces
        if trace_count_verifiable
        else None
    )
    launcher_consistent = all(launcher_checks) if launcher_checks else None
    consistency_checks = [
        value
        for value in (
            export_count_consistent,
            trace_consistent,
            launcher_consistent,
        )
        if value is not None
    ]
    return {
        "declared_run_count": declared_total,
        "exported_episode_count": len(records),
        "infrastructure_failure_record_count": len(failures),
        "model_task_failure_count": sum(
            not record.get("metrics", {}).get("general", {}).get("task_success")
            for record in records
        ),
        "unique_run_id_count": len(set(run_ids)),
        "unique_episode_id_count": len(set(episode_ids)),
        "duplicate_run_ids": duplicate_run_ids,
        "duplicate_episode_ids": duplicate_episode_ids,
        "raw_trace_count": sum(raw_status_counts.values()),
        "raw_trace_final_record_types": dict(sorted(raw_status_counts.items())),
        "unreferenced_raw_traces": unreferenced_traces,
        "missing_referenced_raw_traces": missing_referenced_traces,
        "export_count_consistent": export_count_consistent,
        "trace_count_consistent": trace_consistent,
        "launcher_state_consistent": launcher_consistent,
        "stored_outputs_consistent": (
            not duplicate_run_ids
            and not duplicate_episode_ids
            and all(consistency_checks)
        ),
        "sources": source_results,
    }


def _calibration_concerns(
    records: list[dict[str, Any]],
    scenario_groups: list[dict[str, Any]],
    taxonomies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_run = {item["run_id"]: item for item in taxonomies}
    concerns = []
    for group in scenario_groups:
        dims = group["dimensions"]
        scenario = dims.get("scenario")
        scenario_records = [
            record for record in records if record.get("scenario") == scenario
        ]
        rate = float(group["success_rate"])
        protocol_rate = group.get("first_attempt_decision_valid_rate")
        concern = None
        explanation = None
        recommendation = None
        confidence = "low"
        if len(scenario_records) >= 4 and rate >= 0.9:
            concern = "apparent_ceiling"
            explanation = (
                "Nearly all observed runs succeeded across the tested local models."
            )
            recommendation = "Retain for now; collect broader models or seeds before revising a future suite."
            confidence = "moderate"
        elif (
            len(scenario_records) >= 4
            and rate <= 0.1
            and protocol_rate is not None
            and protocol_rate >= 0.8
        ):
            concern = "apparent_floor_with_valid_protocol"
            explanation = "Models usually produced protocol-valid decisions but almost never completed the task."
            recommendation = "Inspect public information and representative traces; collect more repetitions before any future-version change."
            confidence = "moderate"
        if not concern:
            continue
        affected = [str(record.get("run_id")) for record in scenario_records]
        representative = next(
            (
                category.get("representative_observable_trace")
                for run_id in affected
                for category in by_run.get(run_id, {}).get("categories", [])
                if category.get("representative_observable_trace")
            ),
            None,
        )
        concerns.append(
            {
                "scenario_id": scenario,
                "concern": concern,
                "observed_evidence": {
                    "run_count": group["run_count"],
                    "success_count": group["success_count"],
                    "success_rate": rate,
                    "first_attempt_decision_valid_rate": protocol_rate,
                },
                "affected_models": sorted(
                    {str(record.get("model")) for record in scenario_records}
                ),
                "affected_repetitions": sorted(
                    {record.get("repetition_index") for record in scenario_records}
                ),
                "run_ids": affected,
                "relevant_metrics": {
                    "safety_violation_rate": group["safety_violation_rate"],
                    "mean_invalid_actions": group["mean_invalid_actions"],
                },
                "representative_observable_trace": representative,
                "possible_explanation": explanation,
                "confidence": confidence,
                "recommended_future_action": recommendation,
                "frozen_content_action": "No core_v2@2.0.0 content was modified.",
            }
        )
    return concerns


def render_pilot_markdown(summary: dict[str, Any]) -> str:
    versions = summary.get("benchmark_versions", {})
    models = summary.get("models", [])
    model_results = summary.get("results_by_model", [])
    integrity = summary.get("artifact_integrity", {})
    unreferenced_trace_count = len(integrity.get("unreferenced_raw_traces", []))
    trace_inventory_note = (
        "A non-completed unreferenced trace is retained as historical interruption "
        "evidence and is not counted as a completed episode."
        if unreferenced_trace_count
        else "No unreferenced raw trace was found."
    )
    lines = [
        "# Local Ollama Core v2 Pilot Results",
        "",
        "> Initial empirical characterization only. This is not a definitive model ranking, "
        "a statistical validation, or a psychological assessment.",
        "",
        "## Completion and Artifact Integrity",
        "",
        f"Stored-output consistency check: `{integrity.get('stored_outputs_consistent')}`. "
        f"Declared runs: {integrity.get('declared_run_count')}; exported episodes: "
        f"{integrity.get('exported_episode_count')}; unique run IDs: "
        f"{integrity.get('unique_run_id_count')}; infrastructure failure records: "
        f"{integrity.get('infrastructure_failure_record_count')}; model-task failures: "
        f"{integrity.get('model_task_failure_count')}.",
        "",
        f"Raw trace final-record inventory: `{_display(integrity.get('raw_trace_final_record_types'))}`. "
        f"Unreferenced preserved traces: {unreferenced_trace_count}; "
        f"missing referenced traces: {len(integrity.get('missing_referenced_raw_traces', []))}. "
        f"{trace_inventory_note}",
        "",
        "## 1. Experiment Objective",
        "",
        "Evaluate protocol compliance and observable CPS decision behavior for locally installed "
        "Ollama models across the frozen six-family benchmark.",
        "",
        "## 2. Benchmark Version",
        "",
        f"Package versions observed: `{_display(versions.get('package_versions'))}`.",
        "",
        "## 3. Suite Version",
        "",
        f"Suites observed: `{_display(versions.get('suites'))}`.",
        "",
        "## 4. Prompt Version",
        "",
        f"Prompts observed: `{_display(versions.get('prompts'))}`.",
        "",
        "## 5. Protocol Version",
        "",
        f"Protocols observed: `{_display(versions.get('protocols'))}`.",
        "",
        "## 6. Exact Models",
        "",
    ]
    lines.extend(_model_table(models))
    lines.extend(["", "## 7. Model Digests", ""])
    for model in models:
        lines.append(f"- `{model.get('model')}`: `{model.get('digest')}`")
    lines.extend(["", "## 8. Quantization", ""])
    for model in models:
        lines.append(
            f"- `{model.get('model')}`: `{model.get('quantization_level')}`"
        )
    provenance = summary.get("provenance", {})
    runtime = provenance.get("runtime", {})
    ollama = provenance.get("ollama", {})
    lines.extend(["", "## 9. Ollama Version", ""])
    lines.append(
        f"Ollama running at provenance capture: `{ollama.get('running')}`; "
        f"observed version(s): `{_display(ollama.get('observed_versions') or [ollama.get('version')])}`."
    )
    lines.extend(["", "## 10. Python Version", ""])
    lines.append(f"`{_display(runtime.get('python'))}`")
    lines.extend(["", "## 11. Hardware", ""])
    lines.extend(_hardware_lines(provenance))
    lines.extend(["", "## 12. Decoding Settings", ""])
    for model in models:
        lines.append(
            f"- `{model.get('model_configuration')}`: `{_display(model.get('decoding'))}`"
        )
    lines.extend(["", "## 13. Model Seed Behavior", ""])
    for model in models:
        lines.append(
            f"- `{model.get('model')}`: `{_display(model.get('model_seed_behavior'))}`"
        )
    lines.append(
        "Requested seeds are recorded, but deterministic model generation is not claimed."
    )
    lines.extend(["", "## 14. Repetition Count", ""])
    repetition_counts = sorted(
        {
            source.get("repetition_count")
            for source in summary.get("source_experiments", [])
            if source.get("repetition_count") is not None
        }
    )
    lines.append(
        f"Configured repetition count(s): `{_display(repetition_counts)}`. Raw counts are retained in every table."
    )
    lines.extend(["", "## 15. Experiment Size", ""])
    for source in summary.get("source_experiments", []):
        lines.append(
            f"- `{source.get('experiment_name')}`: {source.get('declared_run_count')} declared runs; `{source.get('directory')}`"
        )
    lines.append(f"Total declared experiment size: {integrity.get('declared_run_count')} runs.")
    lines.extend(["", "## 16. Completed Runs", ""])
    for source in summary.get("source_experiments", []):
        lines.append(
            f"- `{source.get('experiment_name')}`: {source.get('completed_episode_count')}/{source.get('declared_run_count')} completed."
        )
    lines.append(
        f"Total completed: {summary.get('episode_count')}; pending: "
        f"{(integrity.get('declared_run_count') or 0) - (integrity.get('exported_episode_count') or 0) - (integrity.get('infrastructure_failure_record_count') or 0)}."
    )
    lines.extend(["", "## 17. Infrastructure Failures", ""])
    failures = summary.get("infrastructure_failures", [])
    lines.append(
        "None recorded."
        if not failures
        else f"Recorded failures: `{_display(failures)}`"
    )
    lines.extend(["", "## 18. Protocol-Compliance Results", ""])
    lines.extend(_result_table(model_results, protocol=True))
    lines.append(_protocol_failure_statement(summary))
    lines.extend(["", "## 19. Task Results", ""])
    lines.extend(_task_table(model_results))
    lines.append("\nNo combined intelligence score is calculated.")
    lines.extend(["", "## 20. Safety Results", ""])
    lines.extend(_safety_table(model_results))
    lines.append(
        "Safety is reported independently and is not compensated by task success or efficiency."
    )
    lines.extend(["", "## 21. Results by Family", ""])
    lines.extend(_result_table(summary.get("results_by_family", [])))
    lines.extend(["", "## 22. Results by Difficulty", ""])
    lines.extend(_result_table(summary.get("results_by_difficulty_level", [])))
    lines.extend(["", "## 23. Prospective Reasoning", ""])
    lines.extend(
        _family_metric_table(
            summary,
            "prospective_reasoning",
            (
                "callback_count",
                "triggered_callback_count",
                "callback_in_target_window",
                "missed_critical_callback",
                "useful_subscription_count",
                "missed_relevant_events",
                "reaction_delay_steps",
            ),
        )
    )
    lines.extend(["", "## 24. Multi-Step Planning", ""])
    lines.extend(
        _family_metric_table(
            summary,
            "planning",
            (
                "planning_success",
                "prerequisite_violations",
                "repeated_rejected_actions_with_clear_feedback",
                "required_action_coverage",
                "excess_actions_vs_reference",
            ),
            include_resources=True,
        )
    )
    lines.extend(["", "## 25. Interference", ""])
    lines.extend(
        _family_metric_table(
            summary,
            "interference",
            (
                "misleading_action_count",
                "evidence_gathering_action_count",
                "unnecessary_emergency_action_count",
                "interference_safety_outcome",
            ),
        )
    )
    lines.extend(["", "## 26. Conditional Safety", ""])
    lines.extend(
        _family_metric_table(
            summary,
            "conditional_safety",
            (
                "required_inspection_coverage",
                "missed_falsifying_observations",
                "unnecessary_inspection_count",
                "final_rule_satisfaction",
            ),
            include_resources=True,
        )
    )
    lines.extend(["", "## 27. Adaptation", ""])
    lines.extend(
        _family_metric_table(
            summary,
            "adaptation",
            (
                "adaptation_success",
                "first_revised_action_step",
                "obsolete_actions_before_sufficient_evidence",
                "obsolete_actions_after_sufficient_evidence",
                "safety_violations_during_recovery",
                "adaptation_recovery_demonstrated",
            ),
            include_resources=True,
        )
    )
    lines.extend(["", "## 28. Temporal Tracking", ""])
    lines.extend(
        _family_metric_table(
            summary,
            "temporal_tracking",
            (
                "temporal_reasoning_success",
                "delayed_effect_detection_accuracy",
                "delayed_effect_attribution_accuracy",
                "correct_previous_action_association",
                "false_temporal_associations",
                "missed_temporal_dependencies",
            ),
        )
    )
    lines.append(
        "These are psychologically inspired CPS probes, not measurements of human working memory."
    )
    lines.extend(["", "## 29. Context Ablation", ""])
    context_ablation = summary.get("temporal_context_ablation", [])
    if context_ablation:
        lines.extend(_result_table(context_ablation))
        lines.append(
            "\nThis is an information-availability comparison, not a human working-memory measurement."
        )
    else:
        lines.append(
            "The dedicated paired temporal current-only/history experiment was not "
            "included or has no completed records; no context-ablation result is reported."
        )
    lines.extend(["", "## 30. Latency", ""])
    lines.extend(_latency_table(model_results))
    lines.extend(["", "## 31. Token Usage", ""])
    lines.extend(_token_table(model_results))
    lines.extend(["", "## 32. Observable Failure Taxonomy", ""])
    lines.extend(_taxonomy_table(summary))
    lines.append(
        "Categories describe observable outputs, actions, outcomes, and metrics only; they do not infer hidden reasoning. Full category run-ID lists are in the machine-readable summary."
    )
    lines.extend(["", "## 33. Empirical Difficulty", ""])
    for item in summary.get("empirical_difficulty", []):
        counts = ", ".join(
            f"{level['difficulty_level']}: {level['success_count']}/{level['run_count']}"
            for level in item["levels"]
        )
        lines.append(
            f"- `{item['model']}` / `{item['task_family']}`: **{item['pattern']}** ({counts}). Descriptive only."
        )
    lines.append(
        "Intended controlled levels are not assumed to be empirically monotonic, and no significance test was performed."
    )
    lines.extend(["", "## 34. Benchmark-Calibration Concerns", ""])
    lines.extend(_calibration_lines(summary))
    lines.extend(["", "## Representative Observable Traces", ""])
    lines.extend(_representative_trace_lines(summary))
    lines.extend(["", "## 35. Limitations", ""])
    lines.extend(_limitations(summary))
    lines.extend(["", "## 36. Future Work", ""])
    lines.extend(
        [
            "- Add a practical larger same-family model only after it is deliberately installed.",
            "- Collect more repetitions and additional simulator seeds before inferential claims.",
            "- Review flagged traces without changing frozen `core_v2@2.0.0`.",
            "- Run the already-defined paired temporal context ablation as a separate experiment when scientifically prioritized.",
            "- Record any benchmark-content revision only in a future version such as `core_v3`.",
        ]
    )
    lines.extend(["", "## Supporting Artifacts", ""])
    for source in summary.get("source_experiments", []):
        lines.append(f"- `{source['episodes']}` and `{source['manifest']}`")
    return "\n".join(lines).rstrip() + "\n"


def _benchmark_versions(records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [record.get("run_metadata", {}) for record in records]
    versions = [item.get("benchmark_versions", {}) for item in metadata]
    return {
        "package_versions": sorted(
            {
                item.get("package_version")
                for item in versions
                if item.get("package_version")
            }
        ),
        "suites": sorted(
            {
                f"{item.get('suite_name')}@{item.get('suite_version')}"
                for item in metadata
                if item.get("suite_name")
            }
        ),
        "prompts": sorted(
            {
                f"{item.get('prompt_template_id')}@{item.get('prompt_template_version')}:{item.get('prompt_template_hash')}"
                for item in versions
                if item.get("prompt_template_id")
            }
        ),
        "protocols": sorted(
            {
                f"{item.get('protocol_schema_id')}@{item.get('protocol_schema_version')}"
                for item in versions
                if item.get("protocol_schema_id")
            }
        ),
    }


def _merge_provenance(sources: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        source["manifest"].get("provenance")
        for source in sources
        if source["manifest"].get("provenance")
    ]
    if not values:
        return {"status": "unavailable"}
    first = copy.deepcopy(values[0])
    ollama_values = [
        item.get("ollama", {})
        for item in values
        if isinstance(item.get("ollama"), dict)
    ]
    if ollama_values:
        merged_ollama = copy.deepcopy(ollama_values[0])
        for field in ("installed_models", "selected_models"):
            by_identifier: dict[str, dict[str, Any]] = {}
            for ollama in ollama_values:
                records = ollama.get(field, [])
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict) or not record.get("identifier"):
                        continue
                    identifier = str(record["identifier"])
                    current = by_identifier.setdefault(identifier, {})
                    current.update(
                        {
                            key: copy.deepcopy(value)
                            for key, value in record.items()
                            if value is not None
                        }
                    )
            merged_ollama[field] = [by_identifier[key] for key in sorted(by_identifier)]
        merged_ollama["errors"] = sorted(
            {
                str(error)
                for ollama in ollama_values
                for error in ollama.get("errors", [])
            }
        )
        merged_ollama["running"] = all(
            bool(ollama.get("running")) for ollama in ollama_values
        )
        merged_ollama["observed_versions"] = sorted(
            {
                str(ollama["version"])
                for ollama in ollama_values
                if ollama.get("version")
            }
        )
        first["ollama"] = merged_ollama
    first["all_provenance_ids"] = sorted(
        {item.get("provenance_id") for item in values if item.get("provenance_id")}
    )
    return first


def _models(
    records: list[dict[str, Any]], provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    keys = sorted(
        {
            (record.get("model_configuration"), record.get("model"))
            for record in records
        },
        key=repr,
    )
    selected = (
        provenance.get("ollama", {}).get("selected_models", [])
        if isinstance(provenance.get("ollama"), dict)
        else []
    )
    for configuration, model in keys:
        sample = next(
            record
            for record in records
            if record.get("model_configuration") == configuration
            and record.get("model") == model
        )
        metadata = next(
            (item for item in selected if item.get("identifier") == model), {}
        )
        model_config = sample.get("run_metadata", {}).get("model_configuration", {})
        model_records = [
            record
            for record in records
            if record.get("model_configuration") == configuration
            and record.get("model") == model
        ]
        requested_seeds = {
            record.get("run_metadata", {}).get("seeds", {}).get("requested_model_seed")
            for record in model_records
            if record.get("run_metadata", {})
            .get("seeds", {})
            .get("requested_model_seed")
            is not None
        }
        seed_support = {
            record.get("run_metadata", {})
            .get("seeds", {})
            .get("backend_reports_model_seed_support")
            for record in model_records
        }
        output.append(
            {
                "model_configuration": configuration,
                "model": model,
                "backend": sample.get("backend"),
                "decoding": model_config.get("decoding")
                or sample.get("policy", {}).get("decoding"),
                "model_seed_behavior": {
                    "configured_seed": model_config.get("decoding", {}).get("seed"),
                    "requested_seed_count": len(requested_seeds),
                    "backend_reports_model_seed_support": (
                        next(iter(seed_support)) if len(seed_support) == 1 else None
                    ),
                    "determinism_claim": False,
                },
                **{
                    key: metadata.get(key)
                    for key in (
                        "digest",
                        "family",
                        "families",
                        "parameter_size",
                        "quantization_level",
                        "context_length",
                    )
                },
            }
        )
    return output


def _classify_infrastructure_failure(failure: dict[str, Any]) -> dict[str, Any]:
    message = str(failure.get("message", ""))
    lowered = message.lower()
    category = (
        "timeout"
        if "timeout" in lowered or "timed out" in lowered
        else "backend_or_infrastructure_error"
    )
    if "not found" in lowered or "404" in lowered:
        category = "model_unavailable"
    return {**failure, "category": category}


def _load_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = record.get("events_path")
    if not path:
        return []
    source = Path(str(path))
    if not source.is_file():
        return []
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(json.loads(line))
    return output


def _observable_excerpt(interactions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not interactions:
        return None
    selected_index = next(
        (
            index
            for index, entry in enumerate(interactions)
            if entry.get("validation_errors") or entry.get("public_action_outcomes")
        ),
        0,
    )
    item = interactions[selected_index]
    raw = item.get("raw_response")
    return {
        "interaction_index": item.get("interaction_index"),
        "step": item.get("step"),
        "invocation_triggers": item.get("invocation_triggers"),
        "visible_observation": item.get("visible_observation"),
        "parsed_response": item.get("parsed_response"),
        "raw_response_excerpt": str(raw)[:300] if raw is not None else None,
        "validation_errors": item.get("validation_errors"),
        "public_action_outcomes": item.get("public_action_outcomes"),
        "next_visible_observation": (
            interactions[selected_index + 1].get("visible_observation")
            if selected_index + 1 < len(interactions)
            else None
        ),
    }


def _repeated_rejected_actions(interactions: list[dict[str, Any]]) -> list[str]:
    rejected = []
    for interaction in interactions:
        for outcome in interaction.get("public_action_outcomes", []):
            if not outcome.get("accepted") and outcome.get("action_id"):
                rejected.append(str(outcome["action_id"]))
    counts = Counter(rejected)
    return sorted(action for action, count in counts.items() if count > 1)


def _numeric_metric_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for item in values
            for key, value in item.items()
            if isinstance(value, (int, float, bool))
        }
    )
    output = {}
    for key in keys:
        numbers = [
            float(item[key])
            for item in values
            if isinstance(item.get(key), (int, float, bool))
        ]
        output[key] = {
            "count": len(numbers),
            "mean": statistics.fmean(numbers),
            "minimum": min(numbers),
            "maximum": max(numbers),
        }
    return output


def _sum(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item.get(key, 0) or 0) for item in items)


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(item[key])
        for item in items
        if isinstance(item.get(key), (int, float, bool))
    ]
    return statistics.fmean(values) if values else None


def _ratio_sum(items: list[dict[str, Any]], key: str, denominator: int) -> float | None:
    return _sum(items, key) / denominator if denominator else None


def _display(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _model_table(models: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Configuration | Model | Digest | Family | Parameters | Quantization | Context |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for item in models:
        lines.append(
            f"| `{item.get('model_configuration')}` | `{item.get('model')}` | `{item.get('digest')}` | `{item.get('family')}` | `{item.get('parameter_size')}` | `{item.get('quantization_level')}` | {item.get('context_length')} |"
        )
    return lines


def _hardware_lines(provenance: dict[str, Any]) -> list[str]:
    runtime = provenance.get("runtime", {})
    return [
        f"- OS/kernel/architecture: `{runtime.get('operating_system')}` / `{runtime.get('kernel_version')}` / `{runtime.get('machine_architecture')}`",
        f"- CPU: `{runtime.get('cpu_model')}`, logical CPUs `{runtime.get('logical_cpu_count')}`",
        f"- RAM bytes: `{runtime.get('total_system_ram_bytes')}`",
        f"- GPUs: `{_display(runtime.get('gpu_models'))}`",
        f"- Source revision: `{_display(runtime.get('source_revision'))}`",
    ]


def _result_table(groups: list[dict[str, Any]], protocol: bool = False) -> list[str]:
    if protocol:
        lines = [
            "| Group | Runs | Attempts | JSON valid | Schema valid | Decision valid | Executable | Backend errors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in groups:
            lines.append(
                f"| `{_display(item.get('dimensions'))}` | {item.get('run_count')} | {item.get('first_attempt_count')} | {_count_rate(item, 'first_attempt_json_valid')} | {_count_rate(item, 'first_attempt_schema_valid')} | {_count_rate(item, 'first_attempt_decision_valid')} | {_count_rate(item, 'final_executable_decision')} | {item.get('backend_error_count')} |"
            )
        return lines
    lines = [
        "| Group | Runs | Success | Rate | Safety episodes | Safety rate | Invalid actions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in groups:
        lines.append(
            f"| `{_display(item.get('dimensions'))}` | {item.get('run_count')} | {item.get('success_count')} | {_fmt(item.get('success_rate'))} | {item.get('safety_violation_episode_count')} | {_fmt(item.get('safety_violation_rate'))} | {_fmt(item.get('mean_invalid_actions'))} |"
        )
    return lines


def _task_table(groups: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Runs | Success | Model-task failures | Steps | LLM interactions | Accepted | Invalid | Unnecessary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in groups:
        lines.append(
            f"| `{item.get('dimensions', {}).get('model')}` | {item.get('run_count')} | {item.get('success_count')} | {item.get('model_task_failure_count')} | {item.get('total_simulation_steps')} | {item.get('total_llm_invocations')} | {item.get('total_accepted_actions')} | {item.get('total_invalid_actions')} | {item.get('total_unnecessary_actions')} |"
        )
    return lines


def _safety_table(groups: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Runs | Episodes with violation | Episode rate | Total violations |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in groups:
        lines.append(
            f"| `{item.get('dimensions', {}).get('model')}` | {item.get('run_count')} | {item.get('safety_violation_episode_count')} | {_fmt(item.get('safety_violation_rate'))} | {item.get('safety_violation_count')} |"
        )
    return lines


def _family_metric_table(
    summary: dict[str, Any],
    family: str,
    metric_names: tuple[str, ...],
    *,
    include_resources: bool = False,
) -> list[str]:
    values = [
        item
        for item in summary.get("results_by_family", [])
        if item.get("dimensions", {}).get("task_family") == family
    ]
    if not values:
        return ["No completed episodes were available."]
    headers = ["Model", "Success", *metric_names]
    if include_resources:
        headers.append("resource totals")
    lines = [
        "Values below are per-episode means unless identified as totals.",
        "",
        "| " + " | ".join(f"`{header}`" for header in headers) + " |",
        "| " + " | ".join("---:" if index else "---" for index in range(len(headers))) + " |",
    ]
    for item in values:
        cells = [
            f"`{item.get('dimensions', {}).get('model')}`",
            f"{item.get('success_count')}/{item.get('run_count')}",
        ]
        metrics = item.get("family_specific_metrics", {})
        cells.extend(
            _fmt(metrics.get(name, {}).get("mean")) for name in metric_names
        )
        if include_resources:
            cells.append(f"`{_display(item.get('total_resource_consumption', {}))}`")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _latency_table(groups: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Runs | Mean model calls | Mean episode latency (s) | Total latency (s) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in groups:
        lines.append(
            f"| `{item.get('dimensions', {}).get('model')}` | {item.get('run_count')} | {_fmt(item.get('mean_model_calls'))} | {_fmt(item.get('mean_latency_seconds'))} | {_fmt(item.get('total_latency_seconds'))} |"
        )
    return lines


def _token_table(groups: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Runs | Prompt tokens | Completion tokens | Total tokens | Mean total per episode |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in groups:
        usage = item.get("total_token_usage", {})
        lines.append(
            f"| `{item.get('dimensions', {}).get('model')}` | {item.get('run_count')} | {usage.get('prompt_tokens', 'NA')} | {usage.get('completion_tokens', 'NA')} | {usage.get('total_tokens', 'NA')} | {_fmt(item.get('mean_total_tokens'))} |"
        )
    return lines


def _protocol_failure_statement(summary: dict[str, Any]) -> str:
    protocol_categories = {
        item.get("category"): item.get("episode_count")
        for item in summary.get("failure_taxonomy", {}).get("categories", [])
        if item.get("group") == "protocol"
    }
    if not protocol_categories:
        return (
            "No empty responses, invalid JSON, schema-invalid outputs, illegal "
            "decisions, timeouts, or backend failures were observed."
        )
    return f"Observed protocol failure categories: `{_display(protocol_categories)}`."


def _taxonomy_table(summary: dict[str, Any]) -> list[str]:
    values = summary.get("failure_taxonomy", {}).get("categories", [])
    if not values:
        return ["No failure categories were assigned."]
    lines = [
        "| Group | Category | Episodes | Representative run |",
        "| --- | --- | ---: | --- |",
    ]
    for item in values:
        run_ids = item.get("run_ids", [])
        representative = run_ids[0] if run_ids else "unavailable"
        lines.append(
            f"| `{item.get('group')}` | `{item.get('category')}` | {item.get('episode_count')} | `{representative}` |"
        )
    return lines


def _calibration_lines(summary: dict[str, Any]) -> list[str]:
    concerns = summary.get("calibration_concerns", [])
    if not concerns:
        return [
            "No automatic floor/ceiling calibration threshold was reached. This does not prove content validity."
        ]
    lines = [
        "These are evidence-triggered review items, not automatic changes to the frozen suite."
    ]
    for item in concerns:
        run_ids = item.get("run_ids", [])
        lines.extend(
            [
                "",
                f"### `{item.get('scenario_id')}`: {item.get('concern')}",
                "",
                f"- Models: `{_display(item.get('affected_models'))}`; repetitions: `{_display(item.get('affected_repetitions'))}`; representative run: `{run_ids[0] if run_ids else 'unavailable'}`.",
                f"- Evidence: `{_display(item.get('observed_evidence'))}`; relevant metrics: `{_display(item.get('relevant_metrics'))}`.",
                f"- Observable excerpt: `{_display(_trace_digest(item.get('representative_observable_trace')))}`.",
                f"- Assessment: confidence `{item.get('confidence')}`. {item.get('possible_explanation')}",
                f"- Recommendation: {item.get('recommended_future_action')} {item.get('frozen_content_action')}",
            ]
        )
    return lines


def _representative_trace_lines(summary: dict[str, Any]) -> list[str]:
    episodes = summary.get("representative_episodes", [])
    if not episodes:
        return ["No representative episode excerpts were available."]
    lines = [
        "One stored, observable excerpt is selected per model and family. Successful episodes are preferred where available."
    ]
    for episode in episodes:
        outcome = "success" if episode.get("task_success") else "model-task failure"
        lines.extend(
            [
                "",
                f"### `{episode.get('model')}` / `{episode.get('task_family')}`",
                "",
                f"- Run: `{episode.get('run_id')}`; scenario `{episode.get('scenario')}`; level `{episode.get('difficulty_level')}`; outcome **{outcome}**.",
                f"- Assigned categories: `{_display(episode.get('failure_categories'))}`.",
                f"- Public observation, decision, outcome, and next visible state: `{_display(_trace_digest(episode.get('observable_trace')))}`.",
            ]
        )
    protocol_invalid = any(
        item.get("group") == "protocol"
        for item in summary.get("failure_taxonomy", {}).get("categories", [])
    )
    if not protocol_invalid:
        lines.extend(
            [
                "",
                "No protocol-invalid representative exists because every recorded decision attempt was protocol-valid.",
            ]
        )
    return lines


def _trace_digest(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if not trace:
        return None
    decision = trace.get("parsed_response") or {}
    return {
        "step": trace.get("step"),
        "public_observation": _observation_digest(trace.get("visible_observation")),
        "decision": {
            "actions": decision.get("actions"),
            "callback": decision.get("callback"),
            "subscriptions": decision.get("subscriptions"),
        },
        "validation_errors": trace.get("validation_errors"),
        "public_action_outcomes": trace.get("public_action_outcomes"),
        "next_visible_state": _observation_digest(
            trace.get("next_visible_observation")
        ),
    }


def _observation_digest(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not observation:
        return None
    sensors = {}
    for name, sensor in observation.get("sensors", {}).items():
        if not isinstance(sensor, dict):
            sensors[name] = sensor
            continue
        sensors[name] = {
            key: sensor.get(key)
            for key in ("value", "status", "age_steps")
            if key in sensor
        }
    return {
        "step": observation.get("step"),
        "time": observation.get("time"),
        "sensors": sensors,
        "resources": observation.get("resources"),
    }


def _count_rate(item: dict[str, Any], stem: str) -> str:
    return f"{item.get(stem + '_count')} ({_fmt(item.get(stem + '_rate'))})"


def _limitations(summary: dict[str, Any]) -> list[str]:
    models = summary.get("models", [])
    runtime = summary.get("provenance", {}).get("runtime", {})
    lines = [
        f"- Only {len(models)} locally installed model configuration(s) were evaluated.",
        "- Repetition counts are small; no statistical significance is claimed.",
        "- Requested seeds do not prove deterministic model generation.",
        "- Latency is machine-specific and includes model loading where reported by the backend.",
        "- Failure categories use observable behavior and do not infer hidden reasoning.",
    ]
    if not runtime.get("source_revision", {}).get("git_commit"):
        lines.append("- A Git source revision was unavailable in this workspace.")
    if not any(
        item.get("memory_total_bytes") for item in runtime.get("gpu_models", [])
    ):
        lines.append("- Dedicated GPU memory was unavailable or not detectable.")
    return lines


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4g}"
