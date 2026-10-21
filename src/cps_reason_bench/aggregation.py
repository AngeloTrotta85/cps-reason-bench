"""Episode-level export and transparent statistical aggregation."""

from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from cps_reason_bench.run_logging import atomic_write_text


DEFAULT_GROUP_BY = (
    "suite_name",
    "suite_version",
    "model_configuration",
    "backend",
    "model",
    "task_family",
    "difficulty_level",
    "scenario",
    "simulator_seed",
    "interaction_mode",
    "context_mode",
)


def episode_record(summary: dict[str, Any], summary_path: str | Path) -> dict[str, Any]:
    metrics = summary["metrics"]
    metadata = summary.get("run_metadata", {})
    policy = summary.get("policy", {})
    scenario_config = summary.get("scenario_configuration", {})
    model_configuration = metadata.get("model_configuration") or {}
    if isinstance(model_configuration, dict):
        model_configuration_name = model_configuration.get("name")
    else:
        model_configuration_name = str(model_configuration)
    analysis = metadata.get("analysis", {})
    dimensions = {
        "run_id": summary["run_id"],
        "episode_id": metadata.get("episode_id"),
        "experiment_name": metadata.get("experiment_name"),
        "suite_name": metadata.get("suite_name"),
        "suite_version": metadata.get("suite_version"),
        "suite_entry_id": metadata.get("suite_entry_id"),
        "model_configuration": model_configuration_name
        or policy.get("model_configuration"),
        "backend": policy.get("provider"),
        "model": policy.get("model"),
        "task_family": metrics.get("task_family"),
        "difficulty": metrics.get("difficulty"),
        "difficulty_level": metrics.get("difficulty_level"),
        "scenario": metrics.get("scenario"),
        "scenario_generation_seed": metadata.get("seeds", {}).get(
            "scenario_generation_seed"
        ),
        "simulator_seed": metadata.get("seeds", {}).get("simulator_seed"),
        "random_baseline_seed": metadata.get("seeds", {}).get("random_baseline_seed"),
        "requested_model_seed": metadata.get("seeds", {}).get("requested_model_seed"),
        "repetition_index": metadata.get("seeds", {}).get("repetition_index"),
        "semantic_scenario_hash": analysis.get("semantic_scenario_hash"),
        "physical_scenario_hash": analysis.get("physical_scenario_hash"),
        "physical_plant_hash": analysis.get("physical_plant_hash"),
        "physical_instance_id": analysis.get("physical_instance_id"),
        "paired_condition_id": analysis.get("paired_condition_id"),
        "analysis_unit_id": analysis.get("analysis_unit_id"),
        "replicate_id": analysis.get("replicate_id"),
        "interaction_mode": scenario_config.get("interaction", {}).get("mode"),
        "context_mode": scenario_config.get("context", {}).get("mode"),
        "summary_path": str(summary_path),
        "events_path": summary.get("events_file"),
    }
    return {
        **dimensions,
        "metrics": metrics,
        "run_metadata": metadata,
        "policy": policy,
    }


def write_experiment_exports(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    group_by: Iterable[str] = DEFAULT_GROUP_BY,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    episodes_jsonl = destination / "episodes.jsonl"
    episodes_csv = destination / "episodes.csv"
    aggregate_json = destination / "aggregates.json"
    aggregate_csv = destination / "aggregates.csv"

    atomic_write_text(
        episodes_jsonl,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )
    flat_rows = [flatten_episode_record(record) for record in records]
    _atomic_csv_write(episodes_csv, lambda path: _write_wide_csv(flat_rows, path))
    aggregates = aggregate_episode_records(records, tuple(group_by))
    atomic_write_text(
        aggregate_json,
        json.dumps(
            {
                "group_by": list(group_by),
                "standard_deviation": "population (ddof=0)",
                "groups": aggregates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_csv_write(
        aggregate_csv,
        lambda path: _write_aggregate_csv(aggregates, tuple(group_by), path),
    )
    return {
        "episodes_jsonl": episodes_jsonl,
        "episodes_csv": episodes_csv,
        "aggregates_json": aggregate_json,
        "aggregates_csv": aggregate_csv,
    }


def flatten_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key not in {"metrics", "run_metadata", "policy"}
    }
    row.update(_flatten(record.get("metrics", {})))
    return {
        key: value
        for key, value in row.items()
        if "general.final_system_state" not in key
    }


def aggregate_episode_records(
    records: list[dict[str, Any]], group_by: tuple[str, ...] = DEFAULT_GROUP_BY
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(dimension) for dimension in group_by)
        groups.setdefault(key, []).append(record)

    results: list[dict[str, Any]] = []
    for key, group_records in sorted(groups.items(), key=lambda item: repr(item[0])):
        rows = [flatten_episode_record(record) for record in group_records]
        numeric_keys = sorted(
            {
                column
                for row in rows
                for column, value in row.items()
                if column.startswith(
                    ("general.", "task_specific.", "protocol_compliance.")
                )
                and _numeric(value) is not None
                and "final_system_state" not in column
            }
        )
        metrics: dict[str, dict[str, float | int]] = {}
        for metric in numeric_keys:
            values = [
                numeric
                for row in rows
                if (numeric := _numeric(row.get(metric))) is not None
            ]
            if values:
                metrics[metric] = _statistics(values)
        successes = [
            bool(record.get("metrics", {}).get("general", {}).get("task_success"))
            for record in group_records
        ]
        results.append(
            {
                "dimensions": dict(zip(group_by, key, strict=True)),
                "run_count": len(group_records),
                "success_rate": sum(successes) / len(successes),
                "run_ids": [record["run_id"] for record in group_records],
                "metrics": metrics,
            }
        )
    return results


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(nested, child))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        flattened[prefix] = value
    else:
        flattened[prefix] = json.dumps(value, sort_keys=True)
    return flattened


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _write_wide_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_aggregate_csv(
    aggregates: list[dict[str, Any]], group_by: tuple[str, ...], path: Path
) -> None:
    fields = [
        *group_by,
        "run_count",
        "success_rate",
        "metric",
        "count",
        "mean",
        "median",
        "stddev",
        "minimum",
        "maximum",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in aggregates:
            base = {
                **group["dimensions"],
                "run_count": group["run_count"],
                "success_rate": group["success_rate"],
            }
            for metric, values in group["metrics"].items():
                writer.writerow({**base, "metric": metric, **values})


def _atomic_csv_write(path: Path, writer: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
