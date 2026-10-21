"""Research-oriented summaries that keep task-family metrics separate."""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cps_reason_bench.run_logging import atomic_write_text


FAMILY_REPORT_METRICS: dict[str, tuple[str, ...]] = {
    "prospective_reasoning": (
        "general.task_success",
        "task_specific.callback_timing_error_steps",
        "task_specific.callback_count",
        "general.interaction_cost",
        "general.useful_subscription_triggers",
        "task_specific.redundant_subscriptions",
        "task_specific.missed_relevant_events",
        "task_specific.reaction_delay_steps",
    ),
    "planning": (
        "general.task_success",
        "task_specific.planning_action_attempts",
        "general.accepted_actions",
        "task_specific.excess_actions_vs_reference",
        "task_specific.prerequisite_violations",
        "general.resource_consumption",
        "general.safety_violations",
    ),
    "interference": (
        "general.task_success",
        "task_specific.misleading_action_rate",
        "task_specific.unnecessary_emergency_action_rate",
        "task_specific.evidence_gathering_action_count",
        "task_specific.interference_safety_outcome",
    ),
    "conditional_safety": (
        "general.task_success",
        "task_specific.required_inspection_coverage",
        "task_specific.missed_falsifying_observations",
        "task_specific.unnecessary_inspection_count",
        "task_specific.final_rule_satisfaction",
        "general.resource_consumption",
        "general.safety_violations",
    ),
    "adaptation": (
        "general.task_success",
        "task_specific.adaptation_success",
        "task_specific.obsolete_actions_after_change",
        "task_specific.obsolete_actions_before_sufficient_evidence",
        "task_specific.obsolete_actions_after_sufficient_evidence",
        "task_specific.adaptation_delay_steps",
        "task_specific.first_effective_post_change_action",
        "task_specific.recovery_time_steps",
        "task_specific.recovery_resource_cost",
        "task_specific.safety_violations_during_recovery",
    ),
    "temporal_tracking": (
        "general.task_success",
        "task_specific.temporal_reasoning_success",
        "task_specific.temporal_comparison_accuracy",
        "task_specific.delayed_effect_detection_accuracy",
        "task_specific.delayed_effect_attribution_accuracy",
        "task_specific.correct_previous_action_association",
        "task_specific.false_temporal_associations",
        "task_specific.missed_temporal_dependencies",
    ),
}

FAMILY_REQUIRED_METRICS: dict[str, tuple[str, ...]] = {
    family: tuple(
        metric
        for metric in metrics
        if metric
        not in {
            "task_specific.callback_timing_error_steps",
            "general.resource_consumption",
            "task_specific.first_effective_post_change_action",
            "task_specific.recovery_resource_cost",
        }
    )
    for family, metrics in FAMILY_REPORT_METRICS.items()
}

REPORT_GROUP_BY = (
    "task_family",
    "model_configuration",
    "backend",
    "model",
    "difficulty_level",
)


def load_episode_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Episode JSONL line {line_number} is not an object")
        records.append(value)
    return records


def build_family_report(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(item) for item in REPORT_GROUP_BY)
        groups.setdefault(key, []).append(record)

    output_groups: list[dict[str, Any]] = []
    for key, grouped in sorted(groups.items(), key=lambda item: repr(item[0])):
        dimensions = dict(zip(REPORT_GROUP_BY, key, strict=True))
        family = str(dimensions["task_family"])
        paths = FAMILY_REPORT_METRICS.get(family, ("general.task_success",))
        numeric: dict[str, dict[str, float | int]] = {}
        categorical: dict[str, dict[str, Any]] = {}
        for path in paths:
            values = [
                value
                for record in grouped
                if (value := metric_value(record.get("metrics", {}), path)) is not None
            ]
            scalar_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float, bool))
            ]
            if scalar_values and len(scalar_values) == len(values):
                numeric[path] = _statistics(scalar_values)
            elif values:
                normalized = [
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else str(value)
                    for value in values
                ]
                counts = Counter(normalized)
                categorical[path] = {
                    "count": len(normalized),
                    "value_counts": dict(sorted(counts.items())),
                }
        output_groups.append(
            {
                "dimensions": dimensions,
                "run_count": len(grouped),
                "run_ids": [str(record["run_id"]) for record in grouped],
                "metrics": numeric,
                "categorical_metrics": categorical,
            }
        )
    return {
        "group_by": list(REPORT_GROUP_BY),
        "global_score": None,
        "global_score_reason": "Task-family metrics remain decomposed.",
        "groups": output_groups,
    }


def write_family_reports(
    records: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = build_family_report(records)
    json_path = destination / "family_report.json"
    csv_path = destination / "family_report.csv"
    markdown_path = destination / "family_report.md"
    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    try:
        _write_csv(report, temporary_csv)
        os.replace(temporary_csv, csv_path)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()
    atomic_write_text(markdown_path, _render_markdown(report))
    return {
        "family_report_json": json_path,
        "family_report_csv": csv_path,
        "family_report_markdown": markdown_path,
    }


def metric_value(metrics: dict[str, Any], path: str) -> Any:
    current: Any = metrics
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def metric_is_active(metrics: dict[str, Any], path: str) -> bool:
    return metric_value(metrics, path) is not None


def _statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        *REPORT_GROUP_BY,
        "run_count",
        "metric",
        "kind",
        "count",
        "mean",
        "median",
        "stddev",
        "minimum",
        "maximum",
        "category",
        "category_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in report["groups"]:
            base = {**group["dimensions"], "run_count": group["run_count"]}
            for metric, values in group["metrics"].items():
                writer.writerow({**base, "metric": metric, "kind": "numeric", **values})
            for metric, values in group["categorical_metrics"].items():
                for category, count in values["value_counts"].items():
                    writer.writerow(
                        {
                            **base,
                            "metric": metric,
                            "kind": "categorical",
                            "count": values["count"],
                            "category": category,
                            "category_count": count,
                        }
                    )


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CPS Reason Bench Family Report",
        "",
        "> Metrics are reported by task family. No global intelligence score is defined.",
        "",
    ]
    for group in report["groups"]:
        dimensions = group["dimensions"]
        lines.extend(
            [
                f"## {dimensions['task_family']} / {dimensions['model_configuration']} / {dimensions['difficulty_level']}",
                "",
                f"Runs: {group['run_count']}",
                "",
                "| Metric | Count | Mean | Median | Min | Max |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric, values in group["metrics"].items():
            lines.append(
                f"| `{metric}` | {values['count']} | {_format(values['mean'])} | "
                f"{_format(values['median'])} | {_format(values['minimum'])} | "
                f"{_format(values['maximum'])} |"
            )
        if not group["metrics"]:
            lines.append("| No numeric metrics available | 0 |  |  |  |  |")
        if group["categorical_metrics"]:
            lines.extend(["", "Categorical metrics:", ""])
            for metric, values in group["categorical_metrics"].items():
                rendered = ", ".join(
                    f"`{value}`: {count}"
                    for value, count in values["value_counts"].items()
                )
                lines.append(f"- `{metric}`: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format(value: float | int) -> str:
    return f"{float(value):.6g}"
