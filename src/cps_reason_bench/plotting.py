"""Reproducible scientific plots derived only from public episode exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from cps_reason_bench.family_reports import load_episode_records
from cps_reason_bench.run_logging import atomic_write_json, atomic_write_text
from cps_reason_bench.versions import PACKAGE_VERSION


FAMILY_ORDER = (
    "prospective_reasoning",
    "planning",
    "interference",
    "conditional_safety",
    "adaptation",
    "temporal_tracking",
)
LEVEL_ORDER = (
    "sanity_1",
    "sanity_2",
    "sanity_3",
    "sanity_4",
    "entry_1",
    "entry_2",
    "entry_3",
    "bridge",
    "basic",
    "controlled_1",
    "controlled_2",
    "controlled_3",
    "controlled_4",
)
COLORS = (
    "#2f6b9a",
    "#d17a22",
    "#3f7f58",
    "#b54a4a",
    "#74569b",
    "#65717c",
)
WILSON_95 = "Wilson score interval, 95% confidence (z=1.959963984540054)"
RUNTIME_PROVENANCE_KEYS = (
    "collection_scope",
    "cpu_model",
    "gpu_models",
    "kernel_version",
    "logical_cpu_count",
    "machine_architecture",
    "operating_system",
    "python",
    "source_revision",
    "total_system_ram_bytes",
)

FAMILY_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    "prospective_reasoning": (
        ("callback_timing_error_steps", "Callback timing error"),
        ("missed_relevant_events", "Missed events"),
        ("useful_subscription_triggers", "Useful subscriptions"),
    ),
    "planning": (
        ("prerequisite_violations", "Prerequisite violations"),
        ("excess_actions_vs_reference", "Excess actions"),
    ),
    "interference": (
        ("misleading_action_rate", "Misleading-action rate"),
        ("evidence_gathering_action_count", "Evidence actions"),
    ),
    "conditional_safety": (
        ("required_inspection_coverage", "Inspection coverage"),
        ("unnecessary_inspection_count", "Unnecessary inspections"),
    ),
    "adaptation": (
        ("adaptation_success", "Adaptation success"),
        ("adaptation_delay_steps", "Adaptation delay"),
        (
            "obsolete_actions_after_sufficient_evidence",
            "Obsolete actions after evidence",
        ),
    ),
    "temporal_tracking": (
        ("delayed_effect_detection_accuracy", "Effect detection"),
        ("delayed_effect_attribution_accuracy", "Attribution accuracy"),
        ("false_temporal_associations", "False associations"),
    ),
}


class PlottingUnavailable(RuntimeError):
    """Raised when optional Matplotlib support is not installed."""


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binary proportion."""

    if total <= 0:
        raise ValueError("Wilson intervals require a positive sample count")
    if successes < 0 or successes > total:
        raise ValueError("Wilson successes must be between zero and total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def build_plot_data(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build JSON-serializable plot aggregates without retaining simulator state."""

    rows = [_public_episode_row(record) for record in records]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError("No valid episode records were provided")

    foundation_rows = [row for row in rows if row["suite_layer"] == "foundation"]
    protocol_rows = [row for row in rows if row["suite_layer"] == "protocol_sanity"]
    family_rows = [row for row in rows if row["family"] in FAMILY_ORDER]

    return {
        "record_count": len(rows),
        "run_ids": sorted({str(row["run_id"]) for row in rows}),
        "models": sorted({str(row["model"]) for row in rows}),
        "model_configurations": sorted(
            {str(row["model_configuration"]) for row in rows}
        ),
        "families": _ordered_values(
            {str(row["family"]) for row in family_rows}, FAMILY_ORDER
        ),
        "levels": _ordered_values({str(row["level"]) for row in rows}, LEVEL_ORDER),
        "suite_versions": sorted(
            {f"{row['suite_name']}@{row['suite_version']}" for row in rows}
        ),
        "episodes": rows,
        "success_by_model_family": _proportion_groups(
            family_rows, ("model", "family"), "success"
        ),
        "success_by_model_level_family": _proportion_groups(
            family_rows, ("model", "family", "level"), "success"
        ),
        "safety_by_model_family": _proportion_groups(
            family_rows, ("model", "family"), "safety_event"
        ),
        "protocol_by_model": _protocol_groups(rows),
        "protocol_sanity_by_model": _protocol_groups(protocol_rows),
        "efficiency_by_model": _mean_groups(
            rows,
            ("model",),
            (
                "accepted_actions",
                "invalid_actions",
                "unnecessary_actions",
                "interactions",
                "excess_actions_vs_reference",
            ),
        ),
        "family_metrics": {
            family: _mean_groups(
                [row for row in rows if row["family"] == family],
                ("model",),
                tuple(metric for metric, _label in FAMILY_METRICS[family]),
            )
            for family in FAMILY_ORDER
        },
        "latency_by_model": _mean_groups(rows, ("model",), ("total_latency_seconds",)),
        "latency_by_family": _mean_groups(
            family_rows, ("family_series",), ("total_latency_seconds",)
        ),
        "tokens_by_model": _mean_groups(rows, ("model",), ("total_tokens",)),
        "cloud_tokens_by_model": _mean_groups(
            [row for row in rows if row.get("cloud_scope") == "cloud"],
            ("model",),
            ("prompt_tokens", "completion_tokens"),
        ),
        "cloud_cost_by_model": _cloud_cost_groups(rows),
        "foundation_success_by_model": _proportion_groups(
            foundation_rows, ("model",), "success"
        ),
        "foundation_success_by_model_family": _proportion_groups(
            foundation_rows, ("model", "family"), "success"
        ),
        "foundation_success_by_model_level": _proportion_groups(
            foundation_rows, ("model", "level"), "success"
        ),
        "foundation_safety_by_model": _proportion_groups(
            foundation_rows, ("model",), "safety_event"
        ),
        "suite_layer_by_model": _suite_layer_groups(rows),
    }


def write_result_plots(
    episode_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    matplotlib_loader: Callable[[], tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Write PNG/SVG figures, a manifest, and a foundation Markdown report."""

    if not episode_paths:
        raise ValueError("At least one episodes.jsonl path is required")
    sources = [Path(path).resolve() for path in episode_paths]
    records: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    for source in sources:
        loaded = load_episode_records(source)
        records.extend(loaded)
        source_metadata.append(
            {
                "path": str(source),
                "sha256": _file_sha256(source),
                "record_count": len(loaded),
                **_source_provenance(source),
            }
        )
    data = build_plot_data(records)
    matplotlib, pyplot = _resolve_matplotlib(matplotlib_loader)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib(matplotlib)

    manifest: dict[str, Any] = {
        "plot_manifest_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_package_version": PACKAGE_VERSION,
        "suite_versions": data["suite_versions"],
        "source_episode_files": source_metadata,
        "source_run_ids": data["run_ids"],
        "record_count": data["record_count"],
        "plotting_software": {
            "name": "matplotlib",
            "version": str(matplotlib.__version__),
        },
        "uncertainty_methods": {
            "binary_proportions": WILSON_95,
            "means": "No interval; arithmetic mean with sample count shown",
        },
        "plots": [],
        "skipped_plots": [],
    }
    writer = _PlotWriter(destination, pyplot, manifest)

    _plot_success_by_model_family(data, writer)
    for family in FAMILY_ORDER:
        _plot_success_by_level(data, family, writer)
    _plot_model_family_heatmap(data, writer)
    _plot_protocol_compliance(data, writer)
    _plot_safety(data, writer)
    _plot_efficiency(data, writer)
    for family in FAMILY_ORDER:
        _plot_family_metrics(data, family, writer)
    _plot_latency_and_tokens(data, writer)
    _plot_cloud_operations(data, writer)
    _plot_suite_layer_comparison(data, writer)

    report_path: Path | None = None
    if any(
        row["suite_layer"] in {"protocol_sanity", "foundation"}
        for row in data["episodes"]
    ):
        report_path = destination / "foundation_report.md"
        atomic_write_text(
            report_path,
            _render_foundation_report(data, manifest["plots"]),
        )
        manifest["foundation_report"] = str(report_path)

    manifest_path = destination / "plot_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "plots": tuple(manifest["plots"]),
        "foundation_report": report_path,
    }


def _public_episode_row(record: dict[str, Any]) -> dict[str, Any] | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    general = metrics.get("general", {})
    compliance = metrics.get("protocol_compliance", {})
    task = metrics.get("task_specific", {})
    policy = record.get("policy", {})
    if not isinstance(general, dict) or not isinstance(compliance, dict):
        return None
    family = str(record.get("task_family") or metrics.get("task_family") or "unknown")
    suite_name = str(record.get("suite_name") or "unknown_suite")
    suite_version = str(record.get("suite_version") or "unknown")
    model_configuration = str(
        record.get("model_configuration")
        or record.get("model")
        or record.get("backend")
        or "unknown_model"
    )
    layer = _suite_layer(suite_name, family)
    model = f"{model_configuration} | {suite_name}@{suite_version}"
    row: dict[str, Any] = {
        "run_id": str(record.get("run_id") or record.get("episode_id") or "unknown"),
        "model": model,
        "model_configuration": model_configuration,
        "backend": record.get("backend"),
        "model_identifier": record.get("model"),
        "family": family,
        "level": str(
            record.get("difficulty_level")
            or metrics.get("difficulty_level")
            or "unknown"
        ),
        "suite_name": suite_name,
        "suite_version": suite_version,
        "suite_layer": layer,
        "suite_label": f"{layer}: {suite_name}@{suite_version}",
        "family_series": f"{family} | {suite_name}@{suite_version}",
        "success": bool(general.get("task_success", False)),
        "safety_event": _numeric(general.get("safety_violations"), 0.0) > 0.0,
        "safety_violations": _numeric(general.get("safety_violations"), 0.0),
        "accepted_actions": _optional_numeric(general.get("accepted_actions")),
        "invalid_actions": _optional_numeric(general.get("invalid_actions")),
        "unnecessary_actions": _optional_numeric(general.get("unnecessary_actions")),
        "interactions": _optional_numeric(
            general.get("llm_or_policy_invocations", general.get("interaction_cost"))
        ),
        "total_latency_seconds": (
            _optional_numeric(general.get("total_latency_seconds"))
            if _capability_available(policy, "latency")
            else None
        ),
        "total_tokens": _total_tokens(general.get("token_usage")),
        "cloud_scope": (
            record.get("cloud", {}).get("scope")
            if isinstance(record.get("cloud"), dict)
            else None
        ),
        "prompt_tokens": (
            _optional_numeric(record.get("cloud", {}).get("prompt_tokens"))
            if isinstance(record.get("cloud"), dict)
            else None
        ),
        "completion_tokens": (
            _optional_numeric(record.get("cloud", {}).get("completion_tokens"))
            if isinstance(record.get("cloud"), dict)
            else None
        ),
        "accounted_cost_usd": (
            _optional_numeric(record.get("cloud", {}).get("accounted_cost_usd"))
            if isinstance(record.get("cloud"), dict)
            else None
        ),
        "first_attempt_count": int(_numeric(compliance.get("first_attempt_count"), 0)),
        "first_attempt_json_valid": int(
            _numeric(compliance.get("first_attempt_json_valid"), 0)
        ),
        "first_attempt_schema_valid": int(
            _numeric(compliance.get("first_attempt_schema_valid"), 0)
        ),
        "first_attempt_decision_valid": int(
            _numeric(compliance.get("first_attempt_decision_valid"), 0)
        ),
        "final_executable_decisions": int(
            _numeric(compliance.get("final_executable_decisions"), 0)
        ),
    }
    for metric, _label in {
        item for values in FAMILY_METRICS.values() for item in values
    }:
        value = None
        if metric == "useful_subscription_triggers":
            value = general.get(metric)
        elif isinstance(task, dict):
            value = task.get(metric)
        row[metric] = _optional_numeric(value)
    row["excess_actions_vs_reference"] = _optional_numeric(
        task.get("excess_actions_vs_reference") if isinstance(task, dict) else None
    )
    return row


def _suite_layer(suite_name: str, family: str) -> str:
    lowered = suite_name.lower()
    if family == "protocol_sanity" or "protocol_sanity" in lowered:
        return "protocol_sanity"
    if "foundation" in lowered:
        return "foundation"
    if "core_v2" in lowered:
        return "core_v2"
    if "core" in lowered:
        return "core_v1"
    return "other"


def _proportion_groups(
    rows: Sequence[dict[str, Any]],
    dimensions: tuple[str, ...],
    metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in dimensions)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        successes = sum(bool(row.get(metric)) for row in values)
        lower, upper = wilson_interval(successes, len(values))
        output.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "metric": metric,
                "successes": successes,
                "n": len(values),
                "rate": successes / len(values),
                "lower": lower,
                "upper": upper,
                "run_ids": sorted(str(row["run_id"]) for row in values),
            }
        )
    return output


def _protocol_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    definitions = (
        ("json_validity", "first_attempt_json_valid"),
        ("schema_validity", "first_attempt_schema_valid"),
        ("decision_validity", "first_attempt_decision_valid"),
        ("executable_decision", "final_executable_decisions"),
    )
    output: list[dict[str, Any]] = []
    for model, values in sorted(grouped.items()):
        total = sum(int(row["first_attempt_count"]) for row in values)
        metrics: dict[str, Any] = {}
        for name, numerator_key in definitions:
            numerator = sum(int(row[numerator_key]) for row in values)
            if total:
                lower, upper = wilson_interval(min(numerator, total), total)
                metrics[name] = {
                    "numerator": numerator,
                    "denominator": total,
                    "rate": numerator / total,
                    "lower": lower,
                    "upper": upper,
                }
            else:
                metrics[name] = {
                    "numerator": numerator,
                    "denominator": 0,
                    "rate": None,
                    "lower": None,
                    "upper": None,
                }
        output.append(
            {
                "model": model,
                "episode_count": len(values),
                "metrics": metrics,
                "run_ids": sorted(str(row["run_id"]) for row in values),
            }
        )
    return output


def _suite_layer_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _proportion_groups(
        rows,
        ("model_configuration", "suite_layer", "suite_label"),
        "success",
    )
    for group in groups:
        group["model"] = group["model_configuration"]
    return groups


def _mean_groups(
    rows: Sequence[dict[str, Any]],
    dimensions: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in dimensions)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        summaries: dict[str, Any] = {}
        for metric in metrics:
            samples = [
                float(row[metric]) for row in values if row.get(metric) is not None
            ]
            summaries[metric] = (
                {
                    "n": len(samples),
                    "mean": statistics.fmean(samples),
                    "median": statistics.median(samples),
                    "minimum": min(samples),
                    "maximum": max(samples),
                }
                if samples
                else None
            )
        output.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "episode_count": len(values),
                "metrics": summaries,
                "run_ids": sorted(str(row["run_id"]) for row in values),
            }
        )
    return output


def _cloud_cost_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("cloud_scope") == "cloud" and row.get("accounted_cost_usd") is not None:
            grouped[str(row["model"])].append(row)
    output = []
    for model, values in sorted(grouped.items()):
        costs = [float(row["accounted_cost_usd"]) for row in values]
        successes = sum(bool(row.get("success")) for row in values)
        total = sum(costs)
        output.append(
            {
                "model": model,
                "episode_count": len(values),
                "metrics": {
                    "cost_per_episode_usd": {
                        "n": len(values),
                        "mean": total / len(values),
                        "median": statistics.median(costs),
                        "minimum": min(costs),
                        "maximum": max(costs),
                    },
                    "cost_per_successful_episode_usd": (
                        {
                            "n": successes,
                            "mean": total / successes,
                            "median": total / successes,
                            "minimum": total / successes,
                            "maximum": total / successes,
                        }
                        if successes
                        else None
                    ),
                    "total_campaign_cost_usd": {
                        "n": len(values),
                        "mean": total,
                        "median": total,
                        "minimum": total,
                        "maximum": total,
                    },
                },
                "run_ids": sorted(str(row["run_id"]) for row in values),
            }
        )
    return output


class _PlotWriter:
    def __init__(self, output_dir: Path, pyplot: Any, manifest: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.pyplot = pyplot
        self.manifest = manifest

    def save(
        self,
        figure: Any,
        plot_id: str,
        title: str,
        *,
        metrics: Sequence[str],
        grouping: Sequence[str],
        filters: dict[str, Any],
        uncertainty: str,
        run_ids: Sequence[str],
    ) -> None:
        png = self.output_dir / f"{plot_id}.png"
        svg = self.output_dir / f"{plot_id}.svg"
        with warnings.catch_warnings(record=True) as layout_warnings:
            warnings.simplefilter("always", UserWarning)
            figure.tight_layout()
        figure.savefig(png, dpi=180, bbox_inches="tight")
        figure.savefig(svg, bbox_inches="tight")
        self.pyplot.close(figure)
        entry = {
            "plot_id": plot_id,
            "title": title,
            "files": {"png": str(png), "svg": str(svg)},
            "metric_names": list(metrics),
            "grouping": list(grouping),
            "filters": filters,
            "uncertainty_method": uncertainty,
            "source_run_ids": sorted(set(run_ids)),
            "sample_count": len(set(run_ids)),
        }
        if layout_warnings:
            entry["layout_warnings"] = [str(item.message) for item in layout_warnings]
        self.manifest["plots"].append(entry)

    def skip(self, plot_id: str, reason: str) -> None:
        self.manifest["skipped_plots"].append({"plot_id": plot_id, "reason": reason})


def _plot_success_by_model_family(data: dict[str, Any], writer: _PlotWriter) -> None:
    groups = data["success_by_model_family"]
    if not groups:
        writer.skip("success_by_model_family", "No reasoning-family episodes")
        return
    title = "Task success by model configuration and family"
    figure, axis = writer.pyplot.subplots(figsize=(10.5, 4.8))
    _proportion_dot_plot(
        axis,
        groups,
        x_key="family",
        series_key="model",
        x_order=data["families"],
        title=title,
        ylabel="Task success rate",
    )
    writer.save(
        figure,
        "success_by_model_family",
        title,
        metrics=("general.task_success",),
        grouping=(
            "model_configuration",
            "suite_name",
            "suite_version",
            "task_family",
        ),
        filters={"task_families": list(FAMILY_ORDER)},
        uncertainty=WILSON_95,
        run_ids=_group_run_ids(groups),
    )


def _plot_success_by_level(
    data: dict[str, Any], family: str, writer: _PlotWriter
) -> None:
    plot_id = f"success_by_level_{family}"
    groups = [
        group
        for group in data["success_by_model_level_family"]
        if group["family"] == family
    ]
    if not groups:
        writer.skip(plot_id, f"No {family} episodes")
        return
    levels = _ordered_values({group["level"] for group in groups}, LEVEL_ORDER)
    title = f"{_display(family)} success by categorical level"
    figure, axis = writer.pyplot.subplots(figsize=(8.2, 4.4))
    _proportion_line_plot(axis, groups, levels, title)
    writer.save(
        figure,
        plot_id,
        title,
        metrics=("general.task_success",),
        grouping=(
            "model_configuration",
            "suite_name",
            "suite_version",
            "task_family",
            "difficulty_level",
        ),
        filters={"task_family": family},
        uncertainty=WILSON_95,
        run_ids=_group_run_ids(groups),
    )


def _plot_model_family_heatmap(data: dict[str, Any], writer: _PlotWriter) -> None:
    groups = data["success_by_model_family"]
    if not groups:
        writer.skip("model_family_heatmap", "No reasoning-family episodes")
        return
    models = sorted({str(group["model"]) for group in groups})
    families = data["families"]
    lookup = {(group["model"], group["family"]): group for group in groups}
    matrix = [
        [lookup.get((model, family), {}).get("rate", math.nan) for family in families]
        for model in models
    ]
    figure, axis = writer.pyplot.subplots(
        figsize=(max(8.0, 1.35 * len(families)), max(3.2, 0.65 * len(models) + 1.8))
    )
    image = axis.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        aspect="auto",
        interpolation="none",
    )
    axis.set_xticks(range(len(families)), [_display(item) for item in families])
    axis.set_yticks(range(len(models)), [_display(item) for item in models])
    axis.tick_params(axis="x", rotation=30)
    for row_index, model in enumerate(models):
        for column_index, family in enumerate(families):
            group = lookup.get((model, family))
            label = (
                "missing" if group is None else f"{group['rate']:.0%}\nn={group['n']}"
            )
            axis.text(
                column_index, row_index, label, ha="center", va="center", fontsize=8
            )
    title = "Model-family task-success heatmap"
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Task success rate")
    writer.save(
        figure,
        "model_family_heatmap",
        title,
        metrics=("general.task_success",),
        grouping=(
            "model_configuration",
            "suite_name",
            "suite_version",
            "task_family",
        ),
        filters={"task_families": list(FAMILY_ORDER)},
        uncertainty="None; cells show raw proportions and n",
        run_ids=_group_run_ids(groups),
    )


def _plot_protocol_compliance(data: dict[str, Any], writer: _PlotWriter) -> None:
    groups = data["protocol_by_model"]
    if not groups or not any(
        metric["denominator"]
        for group in groups
        for metric in group["metrics"].values()
    ):
        writer.skip("protocol_compliance", "No protocol decision counts available")
        return
    definitions = (
        ("json_validity", "JSON valid"),
        ("schema_validity", "Schema valid"),
        ("decision_validity", "Decision valid"),
        ("executable_decision", "Executable"),
    )
    title = "Protocol compliance by model configuration"
    figure, axis = writer.pyplot.subplots(figsize=(9.2, 4.6))
    _protocol_bar_plot(axis, groups, definitions, title)
    writer.save(
        figure,
        "protocol_compliance",
        title,
        metrics=tuple(name for name, _label in definitions),
        grouping=("model_configuration", "suite_name", "suite_version"),
        filters={},
        uncertainty=WILSON_95,
        run_ids=_group_run_ids(groups),
    )


def _plot_safety(data: dict[str, Any], writer: _PlotWriter) -> None:
    groups = data["safety_by_model_family"]
    if not groups:
        writer.skip("safety_by_model_family", "No reasoning-family episodes")
        return
    title = "Episodes with safety violations by model and family"
    figure, axis = writer.pyplot.subplots(figsize=(10.5, 4.8))
    _proportion_dot_plot(
        axis,
        groups,
        x_key="family",
        series_key="model",
        x_order=data["families"],
        title=title,
        ylabel="Safety-violation episode rate",
    )
    writer.save(
        figure,
        "safety_by_model_family",
        title,
        metrics=("general.safety_violations > 0",),
        grouping=(
            "model_configuration",
            "suite_name",
            "suite_version",
            "task_family",
        ),
        filters={"task_families": list(FAMILY_ORDER)},
        uncertainty=WILSON_95,
        run_ids=_group_run_ids(groups),
    )


def _plot_efficiency(data: dict[str, Any], writer: _PlotWriter) -> None:
    groups = data["efficiency_by_model"]
    action_metrics = (
        ("accepted_actions", "Accepted"),
        ("invalid_actions", "Invalid"),
        ("unnecessary_actions", "Unnecessary"),
        ("excess_actions_vs_reference", "Excess vs reference"),
    )
    interaction_metrics = (("interactions", "LLM/policy interactions"),)
    _mean_metric_plot(
        writer,
        groups,
        action_metrics,
        "action_efficiency_by_model",
        "Action efficiency by model configuration",
        "Mean actions per episode",
    )
    _mean_metric_plot(
        writer,
        groups,
        interaction_metrics,
        "interaction_efficiency_by_model",
        "Interaction efficiency by model configuration",
        "Mean interactions per episode",
    )


def _plot_family_metrics(
    data: dict[str, Any], family: str, writer: _PlotWriter
) -> None:
    plot_id = f"family_metrics_{family}"
    groups = data["family_metrics"][family]
    definitions = FAMILY_METRICS[family]
    available = [
        definition
        for definition in definitions
        if any(group["metrics"].get(definition[0]) for group in groups)
    ]
    if not groups or not available:
        writer.skip(plot_id, f"No reported {family} family metrics")
        return
    models = sorted(group["model"] for group in groups)
    lookup = {group["model"]: group for group in groups}
    subplot_width = max(3.5, 1.4 * len(models))
    figure, axes = writer.pyplot.subplots(
        1,
        len(available),
        figsize=(max(5.5, subplot_width * len(available)), 4.6),
        squeeze=False,
    )
    for axis, (metric, label) in zip(axes[0], available, strict=True):
        values = [
            lookup[model]["metrics"].get(metric, {}).get("mean", math.nan)
            if lookup[model]["metrics"].get(metric)
            else math.nan
            for model in models
        ]
        bars = axis.bar(range(len(models)), values, color=COLORS[: len(models)])
        axis.set_xticks(range(len(models)), models, rotation=25, ha="right")
        axis.set_title(label)
        axis.set_ylabel("Mean per episode")
        axis.grid(axis="y", alpha=0.25)
        for model, bar in zip(models, bars, strict=True):
            summary = lookup[model]["metrics"].get(metric)
            if summary:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"n={summary['n']}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    title = f"{_display(family)} family metrics"
    figure.suptitle(title)
    writer.save(
        figure,
        plot_id,
        title,
        metrics=tuple(f"task_specific.{metric}" for metric, _label in available),
        grouping=("model_configuration", "suite_name", "suite_version"),
        filters={"task_family": family},
        uncertainty="None; arithmetic means and n shown",
        run_ids=_group_run_ids(groups),
    )


def _plot_latency_and_tokens(data: dict[str, Any], writer: _PlotWriter) -> None:
    _mean_metric_plot(
        writer,
        data["latency_by_model"],
        (("total_latency_seconds", "Episode latency"),),
        "latency_by_model",
        "Episode latency by model configuration",
        "Mean seconds per episode",
        note="Runtime and hardware provenance: plot_manifest.json",
    )
    _mean_metric_plot(
        writer,
        data["tokens_by_model"],
        (("total_tokens", "Total tokens"),),
        "token_usage_by_model",
        "Token usage by model configuration",
        "Mean tokens per episode",
    )
    _mean_metric_plot(
        writer,
        data["latency_by_family"],
        (("total_latency_seconds", "Episode latency"),),
        "latency_by_family",
        "Episode latency by reasoning family",
        "Mean seconds per episode",
        dimension="family_series",
        note="Runtime and hardware provenance: plot_manifest.json",
    )


def _plot_cloud_operations(data: dict[str, Any], writer: _PlotWriter) -> None:
    _mean_metric_plot(
        writer,
        data["cloud_tokens_by_model"],
        (
            ("prompt_tokens", "Prompt tokens"),
            ("completion_tokens", "Completion tokens"),
        ),
        "cloud_prompt_completion_tokens",
        "Cloud API tokens by model configuration",
        "Mean provider-reported tokens per episode",
        note="Provider tokenizers differ; token counts are not identical compute units.",
    )
    definitions = (
        (
            "cost_per_episode_usd",
            "Cost per episode",
            "cloud_cost_per_episode",
            "Cloud API cost per episode",
        ),
        (
            "cost_per_successful_episode_usd",
            "Cost per successful episode",
            "cloud_cost_per_successful_episode",
            "Cloud API cost per successful episode",
        ),
        (
            "total_campaign_cost_usd",
            "Total cost",
            "cloud_total_cost",
            "Total cloud API cost represented in these episodes",
        ),
    )
    for metric, label, plot_id, title in definitions:
        _mean_metric_plot(
            writer,
            data["cloud_cost_by_model"],
            ((metric, label),),
            plot_id,
            title,
            "USD",
            note="Operational cost is separate from benchmark reasoning quality.",
        )


def _plot_suite_layer_comparison(data: dict[str, Any], writer: _PlotWriter) -> None:
    groups = data["suite_layer_by_model"]
    layers = {group["suite_layer"] for group in groups}
    if not {"foundation", "core_v2"}.issubset(layers):
        writer.skip(
            "suite_layer_comparison",
            "Requires both foundation and core_v2 episode exports",
        )
        return
    labels = sorted(
        {group["suite_label"] for group in groups},
        key=lambda value: (_layer_rank(value.split(":", 1)[0]), value),
    )
    title = "Protocol, foundation, and core-v2 task success"
    figure, axis = writer.pyplot.subplots(figsize=(max(9.0, len(labels) * 2.0), 4.8))
    _proportion_dot_plot(
        axis,
        groups,
        x_key="suite_label",
        series_key="model",
        x_order=labels,
        title=title,
        ylabel="Task success rate",
    )
    writer.save(
        figure,
        "suite_layer_comparison",
        title,
        metrics=("general.task_success",),
        grouping=(
            "model_configuration",
            "suite_layer",
            "suite_name",
            "suite_version",
        ),
        filters={"required_layers": ["foundation", "core_v2"]},
        uncertainty=WILSON_95,
        run_ids=_group_run_ids(groups),
    )


def _proportion_dot_plot(
    axis: Any,
    groups: Sequence[dict[str, Any]],
    *,
    x_key: str,
    series_key: str,
    x_order: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    series = sorted({str(group[series_key]) for group in groups})
    lookup = {(group[series_key], group[x_key]): group for group in groups}
    width = 0.7 / max(1, len(series))
    center = (len(series) - 1) / 2.0
    for index, value in enumerate(series):
        x_values: list[float] = []
        y_values: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        labels: list[tuple[float, float, int]] = []
        for position, category in enumerate(x_order):
            group = lookup.get((value, category))
            if not group:
                continue
            x = position + (index - center) * width
            x_values.append(x)
            y_values.append(group["rate"])
            lower_errors.append(max(0.0, group["rate"] - group["lower"]))
            upper_errors.append(max(0.0, group["upper"] - group["rate"]))
            labels.append((x, group["upper"], group["n"]))
        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            capsize=3,
            color=COLORS[index % len(COLORS)],
            label=_display(value),
        )
        for x, upper, count in labels:
            axis.text(
                x, min(1.06, upper + 0.025), f"n={count}", ha="center", fontsize=7
            )
    axis.set_xticks(range(len(x_order)), [_display(item) for item in x_order])
    axis.tick_params(axis="x", rotation=28)
    axis.set_ylim(-0.03, 1.12)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="best")


def _proportion_line_plot(
    axis: Any,
    groups: Sequence[dict[str, Any]],
    levels: Sequence[str],
    title: str,
) -> None:
    models = sorted({group["model"] for group in groups})
    lookup = {(group["model"], group["level"]): group for group in groups}
    for index, model in enumerate(models):
        available = [
            (position, lookup[(model, level)])
            for position, level in enumerate(levels)
            if (model, level) in lookup
        ]
        x_values = [item[0] for item in available]
        y_values = [item[1]["rate"] for item in available]
        lower = [max(0.0, item[1]["rate"] - item[1]["lower"]) for item in available]
        upper = [max(0.0, item[1]["upper"] - item[1]["rate"]) for item in available]
        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower, upper],
            marker="o",
            capsize=3,
            color=COLORS[index % len(COLORS)],
            label=model,
        )
        for position, group in available:
            axis.text(
                position,
                min(1.06, group["upper"] + 0.025),
                f"n={group['n']}",
                ha="center",
                fontsize=7,
            )
    axis.set_xticks(range(len(levels)), [_display(level) for level in levels])
    axis.set_ylim(-0.03, 1.12)
    axis.set_xlabel("Categorical difficulty level")
    axis.set_ylabel("Task success rate")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="best")


def _protocol_bar_plot(
    axis: Any,
    groups: Sequence[dict[str, Any]],
    definitions: Sequence[tuple[str, str]],
    title: str,
) -> None:
    models = [str(group["model"]) for group in groups]
    width = 0.78 / max(1, len(models))
    center = (len(models) - 1) / 2.0
    for model_index, group in enumerate(groups):
        x_values = [
            index + (model_index - center) * width for index in range(len(definitions))
        ]
        values = [group["metrics"][name]["rate"] or 0.0 for name, _label in definitions]
        bars = axis.bar(
            x_values,
            values,
            width=width,
            color=COLORS[model_index % len(COLORS)],
            label=group["model"],
        )
        lower_errors = []
        upper_errors = []
        for name, _label in definitions:
            metric = group["metrics"][name]
            rate = metric["rate"] or 0.0
            lower_errors.append(max(0.0, rate - (metric["lower"] or 0.0)))
            upper_errors.append(max(0.0, (metric["upper"] or rate) - rate))
        axis.errorbar(
            x_values,
            values,
            yerr=[lower_errors, upper_errors],
            fmt="none",
            ecolor="#222222",
            capsize=3,
            linewidth=1,
        )
        for bar, (name, _label), upper_error in zip(
            bars, definitions, upper_errors, strict=True
        ):
            metric = group["metrics"][name]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.07, bar.get_height() + upper_error + 0.02),
                f"n={metric['denominator']}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axis.set_xticks(range(len(definitions)), [label for _name, label in definitions])
    axis.set_ylim(0.0, 1.12)
    axis.set_ylabel("Compliance rate")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, loc="best")


def _mean_metric_plot(
    writer: _PlotWriter,
    groups: Sequence[dict[str, Any]],
    definitions: Sequence[tuple[str, str]],
    plot_id: str,
    title: str,
    ylabel: str,
    *,
    dimension: str = "model",
    note: str | None = None,
) -> None:
    available = [
        definition
        for definition in definitions
        if any(group["metrics"].get(definition[0]) for group in groups)
    ]
    if not groups or not available:
        writer.skip(plot_id, "Metric is unavailable in the supplied episode exports")
        return
    categories = [str(group[dimension]) for group in groups]
    width = 0.78 / len(available)
    center = (len(available) - 1) / 2.0
    figure, axis = writer.pyplot.subplots(
        figsize=(max(7.2, len(categories) * 1.4), 4.4)
    )
    for metric_index, (metric, label) in enumerate(available):
        x_values = [
            index + (metric_index - center) * width for index in range(len(categories))
        ]
        values = [
            group["metrics"].get(metric, {}).get("mean", math.nan)
            if group["metrics"].get(metric)
            else math.nan
            for group in groups
        ]
        bars = axis.bar(
            x_values,
            values,
            width=width,
            color=COLORS[metric_index % len(COLORS)],
            label=label,
        )
        for group, bar in zip(groups, bars, strict=True):
            summary = group["metrics"].get(metric)
            if summary:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"n={summary['n']}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    axis.set_xticks(range(len(categories)), [_display(item) for item in categories])
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    if note:
        axis.text(
            0.99,
            0.02,
            note,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#4d4d4d",
            bbox={
                "facecolor": "white",
                "edgecolor": "#bdbdbd",
                "alpha": 0.9,
                "pad": 2,
            },
        )
    if len(available) > 1:
        axis.legend(fontsize=8, loc="best")
    writer.save(
        figure,
        plot_id,
        title,
        metrics=tuple(metric for metric, _label in available),
        grouping=(
            ("model_configuration", "suite_name", "suite_version")
            if dimension == "model"
            else ("task_family", "suite_name", "suite_version")
        ),
        filters={},
        uncertainty="None; arithmetic means and n shown",
        run_ids=_group_run_ids(groups),
    )


def _render_foundation_report(
    data: dict[str, Any], plots: Sequence[dict[str, Any]]
) -> str:
    lines = [
        "# CPS Reason Bench Foundation Results",
        "",
        "This report is reconstructed from public episode exports. Protocol sanity is "
        "reported separately from CPS reasoning success, and no global intelligence "
        "score is calculated.",
        "",
        "## Data",
        "",
        f"- Episodes: {data['record_count']}",
        f"- Model configurations: {', '.join(f'`{item}`' for item in data['model_configurations'])}",
        f"- Suites: {', '.join(f'`{item}`' for item in data['suite_versions'])}",
        "",
    ]
    if data["protocol_sanity_by_model"]:
        lines.extend(
            [
                "## Protocol Sanity",
                "",
                "| Model configuration / suite | Episodes | JSON valid | Schema valid | Decision valid | Executable |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group in data["protocol_sanity_by_model"]:
            values = group["metrics"]
            lines.append(
                f"| `{group['model']}` | {group['episode_count']} | "
                f"{_rate_cell(values['json_validity'])} | "
                f"{_rate_cell(values['schema_validity'])} | "
                f"{_rate_cell(values['decision_validity'])} | "
                f"{_rate_cell(values['executable_decision'])} |"
            )
        lines.append("")
    if data["foundation_success_by_model"]:
        lines.extend(
            [
                "## Foundation Success",
                "",
                "| Model configuration / suite | Successes | Runs | Rate | Wilson 95% CI | Safety-violation episode rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        safety = {group["model"]: group for group in data["foundation_safety_by_model"]}
        for group in data["foundation_success_by_model"]:
            safety_group = safety.get(group["model"])
            safety_text = (
                f"{safety_group['rate']:.1%} ({safety_group['successes']}/{safety_group['n']})"
                if safety_group
                else "not available"
            )
            lines.append(
                f"| `{group['model']}` | {group['successes']} | {group['n']} | "
                f"{group['rate']:.1%} | {group['lower']:.1%}-{group['upper']:.1%} | "
                f"{safety_text} |"
            )
        lines.extend(["", "### By Family", ""])
        lines.extend(
            _proportion_markdown_table(
                data["foundation_success_by_model_family"], "family"
            )
        )
        lines.extend(["", "### By Entry Level", ""])
        lines.extend(
            _proportion_markdown_table(
                data["foundation_success_by_model_level"], "level"
            )
        )
        lines.extend(
            [
                "",
                "The level table is descriptive. Entry and bridge labels define an "
                "intended order, but they are categorical and are not treated as evenly "
                "spaced numeric measurements.",
                "",
            ]
        )
    comparison = [
        group
        for group in data["suite_layer_by_model"]
        if group["suite_layer"] in {"protocol_sanity", "foundation", "core_v2"}
    ]
    if comparison:
        lines.extend(
            [
                "## Suite-Layer Comparison",
                "",
                "| Model configuration | Explicit suite | Successes | Runs | Rate |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for group in comparison:
            lines.append(
                f"| `{group['model']}` | `{group['suite_label']}` | "
                f"{group['successes']} | {group['n']} | {group['rate']:.1%} |"
            )
        lines.append("")
    if plots:
        lines.extend(
            [
                "## Plots",
                "",
                "| Figure | PNG | SVG |",
                "| --- | --- | --- |",
            ]
        )
        for plot in plots:
            png = Path(plot["files"]["png"]).name
            svg = Path(plot["files"]["svg"]).name
            lines.append(f"| {plot['title']} | [{png}]({png}) | [{svg}]({svg}) |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "- Protocol compliance indicates interface use; it is not reasoning success.",
            "- Foundation, smoke, protocol-sanity, and core-v2 suites retain separate identities; their episodes are not merged into an unlabeled score.",
            "- Wilson intervals quantify binomial sampling uncertainty only. They do not capture scenario-selection or model-decoding uncertainty.",
            "- Latency is hardware- and runtime-dependent and should be interpreted with the stored experiment provenance.",
            "- Random and scripted policies are controls, not directly comparable language-model capability estimates.",
            "- Difficulty trends require real repeated model runs; this generated report does not assume that performance must decrease monotonically.",
            "",
        ]
    )
    return "\n".join(lines)


def _proportion_markdown_table(
    groups: Sequence[dict[str, Any]], dimension: str
) -> list[str]:
    output = [
        f"| Model configuration / suite | {_display(dimension)} | Successes | Runs | Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        output.append(
            f"| `{group['model']}` | `{group[dimension]}` | "
            f"{group['successes']} | {group['n']} | {group['rate']:.1%} |"
        )
    return output


def _rate_cell(metric: dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "not available"
    return f"{metric['rate']:.1%} ({metric['numerator']}/{metric['denominator']})"


def _resolve_matplotlib(
    loader: Callable[[], tuple[Any, Any]] | None,
) -> tuple[Any, Any]:
    try:
        return (loader or _load_matplotlib)()
    except (ImportError, ModuleNotFoundError) as exc:
        raise PlottingUnavailable(
            'Matplotlib is required for plots; install it with pip install -e ".[plots]"'
        ) from exc


def _load_matplotlib() -> tuple[Any, Any]:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "cps-reason-bench-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    return matplotlib, pyplot


def _configure_matplotlib(matplotlib: Any) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(source: Path) -> dict[str, Any]:
    manifest_path = source.parent / "manifest.json"
    if not manifest_path.is_file():
        return {
            "experiment_manifest": None,
            "runtime_provenance": None,
            "provider_provenance": None,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "experiment_manifest": str(manifest_path.resolve()),
            "runtime_provenance": None,
            "provider_provenance": None,
        }
    provenance = manifest.get("provenance", {})
    runtime = provenance.get("runtime") if isinstance(provenance, dict) else None
    safe_runtime = (
        {key: runtime.get(key) for key in RUNTIME_PROVENANCE_KEYS if key in runtime}
        if isinstance(runtime, dict)
        else None
    )
    ollama = provenance.get("ollama", {}) if isinstance(provenance, dict) else {}
    selected = ollama.get("selected_models", []) if isinstance(ollama, dict) else []
    models = [
        {
            key: item.get(key)
            for key in (
                "identifier",
                "digest",
                "parameter_size",
                "quantization_level",
            )
        }
        for item in selected
        if isinstance(item, dict)
    ]
    return {
        "experiment_manifest": str(manifest_path.resolve()),
        "provenance_id": (
            provenance.get("provenance_id") if isinstance(provenance, dict) else None
        ),
        "runtime_provenance": safe_runtime,
        "provider_provenance": {
            "ollama_version": ollama.get("version"),
            "selected_models": models,
        }
        if isinstance(ollama, dict)
        else None,
    }


def _total_tokens(value: Any) -> float | None:
    if not isinstance(value, dict) or not value:
        return None
    if _optional_numeric(value.get("total_tokens")) is not None:
        return _optional_numeric(value.get("total_tokens"))
    components = [
        _optional_numeric(value.get(key))
        for key in ("prompt_tokens", "completion_tokens")
    ]
    present = [item for item in components if item is not None]
    return sum(present) if present else None


def _capability_available(policy: Any, capability: str) -> bool:
    if not isinstance(policy, dict):
        return True
    capabilities = policy.get("capabilities")
    if not isinstance(capabilities, dict):
        return True
    return capabilities.get(capability) is not False


def _numeric(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _optional_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _ordered_values(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    unique = set(values)
    return [item for item in preferred if item in unique] + sorted(
        unique - set(preferred)
    )


def _group_run_ids(groups: Sequence[dict[str, Any]]) -> list[str]:
    return sorted(
        {str(run_id) for group in groups for run_id in group.get("run_ids", [])}
    )


def _display(value: str) -> str:
    return str(value).replace(" | ", "\n").replace("_", " ")


def _layer_rank(value: str) -> int:
    return {
        "protocol_sanity": 0,
        "foundation": 1,
        "core_v2": 2,
        "core_v1": 3,
        "other": 4,
    }.get(value, 5)
