"""Offline release checks; no git initialization, network, or model requests.

Run from a source checkout: python scripts/check_release.py [--write-manifest].
The package need not be installed; this checkout's src/ is checked directly.
The selected Python interpreter must have the runtime dependency PyYAML.
Only --write-manifest updates the release's own byte-hash manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", "runs", ".mypy_cache", ".ruff_cache"}


def release_files() -> list[Path]:
    result = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Release must not depend on symlinks: {relative}")
        if path.is_file() and path.name != "release_manifest.json":
            result.append(path)
    return sorted(result)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(files: list[Path]) -> dict:
    from cps_reason_bench.public_config import BUILTIN_SUITES, DATA
    from cps_reason_bench.suites import load_suite
    from cps_reason_bench.versions import PACKAGE_VERSION
    return {
        "release_manifest_schema_version": 1, "package_version": PACKAGE_VERSION,
        "public_config_version": 1,
        "generator": {"id": "cps_reason_graph_v2_generator", "version": "1.0.0"},
        "suites": [load_suite(DATA / "benchmark/suites" / value).identity for value in BUILTIN_SUITES.values()],
        "hash_algorithm": "sha256",
        "scope": "Standalone source release; excludes this self-referential manifest and ignored runtime/build/cache directories",
        "files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)} for path in files],
    }


def scan(files: list[Path]) -> list[str]:
    issues = []
    # Match probable real key material, not ordinary env-var references.
    patterns = {
        "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "provider-key": re.compile(r"\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{24,}"),
        "aws-key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
        "personal-path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
        "url-credential": re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    }
    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 2_000_000:
            issues.append(f"Oversized unintended file: {relative}")
        if path.name.startswith(".env") and path.name != ".env.example":
            issues.append(f"Environment file: {relative}")
        if path.suffix in {".pem", ".key", ".gguf", ".safetensors", ".pyc", ".pdf"}:
            issues.append(f"Unexpected release file type: {relative}")
        text = path.read_text(encoding="utf-8")
        for kind, pattern in patterns.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                # The one URL credential fixture is intentionally rejected by validation.
                if kind == "url-credential" and str(relative) == "tests/test_public_release.py":
                    continue
                issues.append(f"{kind}: {relative}:{line} (value withheld)")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                if not (path.parent / target).exists():
                    issues.append(f"Broken local documentation link: {relative} -> {target}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    source = ROOT / "src"
    if not (source / "cps_reason_bench" / "__init__.py").is_file():
        parser.error("Release source package is missing; run the checker from a complete source checkout")
    # Check this tree even when another checkout or wheel is on the interpreter path.
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    try:
        importlib.import_module("yaml")
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        parser.error(
            "PyYAML is missing from this Python environment. Activate the release "
            "environment or install it with: python -m pip install 'PyYAML>=6.0'"
        )
    files = release_files()
    issues = scan(files)
    for path in files:
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)), "exec")
        if path.is_relative_to(ROOT / "src/cps_reason_bench") and path.suffix == ".py":
            module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
            if module.endswith(".__main__"):
                continue
            imported = importlib.import_module(module)
            if not Path(imported.__file__).resolve().is_relative_to(ROOT / "src"):
                issues.append(f"Import escaped release source: {module}")
    manifest = make_manifest(files)
    destination = ROOT / "release_manifest.json"
    if args.write_manifest and not issues:
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif not destination.exists():
        issues.append("Missing release_manifest.json; run --write-manifest after reviewing changes")
    elif json.loads(destination.read_text()) != manifest:
        issues.append("Release manifest mismatch: files or suite identities have changed")
    if issues:
        print("\n".join(issues))
        return 1
    print(f"PASS: {len(files)} source files; hashes, syntax/import isolation, links, size/path/secret-pattern checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
