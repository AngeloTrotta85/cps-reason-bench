"""Exercise the source checker without an installed benchmark package or site setup."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    destination = tmp_path / "checkout"
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "*.egg-info",
        "build", "dist", "runs",
    ))
    return destination


def run_checker(checkout, tmp_path, *, pythonpath=None, args=()):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-S", str(checkout / "scripts/check_release.py"), *args],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("foreign_package", [False, True])
def test_checker_uses_checkout_without_installation(checkout, tmp_path, foreign_package):
    dependencies = tmp_path / "dependencies"
    # Supply only PyYAML; no installed benchmark or editable-install .pth is visible.
    shutil.copytree(Path(yaml.__file__).parent, dependencies / "yaml",
        ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyd"))
    if foreign_package:
        package = dependencies / "cps_reason_bench"
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('foreign package imported')\n")
    generated = run_checker(checkout, tmp_path, pythonpath=dependencies, args=("--write-manifest",))
    assert generated.returncode == 0, generated.stdout + generated.stderr
    checked = run_checker(checkout, tmp_path, pythonpath=dependencies)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "PASS:" in checked.stdout
    assert not list(checkout.rglob("__pycache__"))


def test_checker_reports_missing_pyyaml_without_traceback(checkout, tmp_path):
    original_manifest = (checkout / "release_manifest.json").read_bytes()
    checked = run_checker(checkout, tmp_path, args=("--write-manifest",))
    assert checked.returncode == 2
    assert "PyYAML is missing" in checked.stderr
    assert "python -m pip install" in checked.stderr
    assert "Traceback" not in checked.stderr
    assert (checkout / "release_manifest.json").read_bytes() == original_manifest


def test_checker_reports_incomplete_checkout(checkout, tmp_path):
    (checkout / "src/cps_reason_bench/__init__.py").unlink()
    checked = run_checker(checkout, tmp_path)
    assert checked.returncode == 2
    assert "Release source package is missing" in checked.stderr
    assert "Traceback" not in checked.stderr
