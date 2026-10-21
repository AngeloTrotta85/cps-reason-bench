"""Locate source-checkout and installed example data."""

from __future__ import annotations

import sys
import os
from pathlib import Path


def examples_root() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parent / "data" / "examples",
        module_path.parents[2] / "examples",
        module_path.parents[1] / "share" / "cps-reason-bench",
        Path(sys.prefix) / "share" / "cps-reason-bench",
    ]
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def project_root() -> Path:
    configured = os.environ.get("CPS_BENCH_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    module_path = Path(__file__).resolve()
    source_root = module_path.parents[2]
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd().resolve()


def default_model_registry_path() -> Path:
    candidates = [
        project_root() / "models.yaml",
        Path.cwd() / "models.yaml",
        Path(sys.prefix) / "share" / "cps-reason-bench" / "models.yaml",
        examples_root() / "models.yaml",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())


def default_launcher_config_path() -> Path:
    candidates = [
        project_root() / "local_experiments.yaml",
        Path.cwd() / "local_experiments.yaml",
        examples_root() / "local_experiments.yaml",
        Path(sys.prefix) / "share" / "cps-reason-bench" / "local_experiments.yaml",
    ]
    return next(
        (path.resolve() for path in candidates if path.is_file()),
        candidates[0].resolve(),
    )


def bundled_scenario_path(task_family: str) -> Path:
    filenames = {
        "prospective_reasoning": "prospective_basic.yaml",
        "planning": "planning_basic.yaml",
        "interference": "interference_basic.yaml",
        "conditional_safety": "conditional_safety_basic.yaml",
        "adaptation": "adaptation_basic.yaml",
        "temporal_tracking": "temporal_tracking_basic.yaml",
    }
    try:
        filename = filenames[task_family]
    except KeyError as exc:
        raise ValueError(f"No bundled scenario template for family {task_family!r}") from exc
    return examples_root() / "scenarios" / filename


def bundled_foundation_scenario_path(task_family: str) -> Path:
    filenames = {
        "prospective_reasoning": "prospective_entry.yaml",
        "planning": "planning_entry.yaml",
        "interference": "interference_entry.yaml",
        "conditional_safety": "conditional_entry.yaml",
        "adaptation": "adaptation_entry.yaml",
        "temporal_tracking": "temporal_entry.yaml",
    }
    try:
        filename = filenames[task_family]
    except KeyError as exc:
        raise ValueError(
            f"No bundled foundation template for family {task_family!r}"
        ) from exc
    return examples_root() / "scenarios" / "foundation" / filename
