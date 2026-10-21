"""Best-effort, privacy-conscious runtime and local Ollama provenance."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[tuple[str, ...]], CommandResult]
HttpClient = Callable[[str, str, dict[str, Any] | None], dict[str, Any]]


def collect_experiment_provenance(
    ollama_models: Iterable[str] = (),
    *,
    command_runner: CommandRunner | None = None,
    http_client: HttpClient | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    """Collect bounded metadata without making optional detection a run dependency."""

    runner = command_runner or _run_command
    models = tuple(dict.fromkeys(str(model) for model in ollama_models))
    payload = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": collect_runtime_provenance(command_runner=runner),
        "ollama": collect_ollama_provenance(
            models,
            command_runner=runner,
            http_client=http_client,
            base_url=ollama_base_url,
        ),
    }
    payload["provenance_id"] = _fingerprint(
        {key: value for key, value in payload.items() if key != "collected_at_utc"}
    )
    return payload


def collect_runtime_provenance(
    *, command_runner: CommandRunner | None = None
) -> dict[str, Any]:
    runner = command_runner or _run_command
    source = _source_revision(runner)
    gpu = _gpu_metadata(runner)
    try:
        os_name = platform.freedesktop_os_release().get("PRETTY_NAME")
    except (AttributeError, OSError):
        os_name = platform.system() or None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "operating_system": os_name,
        "kernel_version": platform.release() or None,
        "machine_architecture": platform.machine() or None,
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "total_system_ram_bytes": _total_ram_bytes(),
        "gpu_models": gpu,
        "source_revision": source,
        "collection_scope": "Documented runtime, hardware, source, and model fields only.",
    }


def collect_ollama_provenance(
    model_ids: Iterable[str],
    *,
    command_runner: CommandRunner | None = None,
    http_client: HttpClient | None = None,
    base_url: str = "http://localhost:11434",
) -> dict[str, Any]:
    runner = command_runner or _run_command
    client = http_client or _urllib_json_client(base_url)
    requested = tuple(dict.fromkeys(str(model) for model in model_ids))
    version: str | None = None
    running = False
    errors: list[str] = []
    tags: dict[str, Any] = {}

    try:
        version_payload = client("GET", "/api/version", None)
        version = _nullable_string(version_payload.get("version"))
        running = True
    except Exception as exc:  # optional local service
        errors.append(f"Ollama version API unavailable: {_safe_error(exc)}")
        command = _safe_command(runner, ("ollama", "--version"))
        if command and command.returncode == 0:
            version = _parse_ollama_version(command.stdout)

    installed: list[dict[str, Any]] = []
    if running:
        try:
            tags = client("GET", "/api/tags", None)
        except Exception as exc:  # optional local service
            errors.append(f"Ollama tags API unavailable: {_safe_error(exc)}")

    tag_records = tags.get("models", []) if isinstance(tags, dict) else []
    if isinstance(tag_records, list):
        for item in tag_records:
            if isinstance(item, dict):
                installed.append(_tag_record(item))

    selected: list[dict[str, Any]] = []
    by_name = {
        str(record.get("identifier")): record
        for record in installed
        if record.get("identifier")
    }
    for model_id in requested:
        record = dict(by_name.get(model_id, {"identifier": model_id}))
        record["available"] = model_id in by_name
        if running and record["available"]:
            try:
                shown = client("POST", "/api/show", {"model": model_id})
                record.update(_show_record(shown))
            except Exception as exc:  # metadata failure must not block inference
                record["show_metadata_error"] = _safe_error(exc)
        selected.append(record)

    return {
        "running": running,
        "version": version,
        "base_url": base_url,
        "installed_models": installed,
        "selected_models": selected,
        "errors": errors,
    }


def compact_episode_provenance(
    provenance: dict[str, Any], model_id: str | None
) -> dict[str, Any]:
    """Return a small episode pointer plus exact selected-model identity."""

    compact: dict[str, Any] = {
        "provenance_id": provenance.get("provenance_id"),
        "manifest_path": "../manifest.json",
    }
    ollama = provenance.get("ollama", {})
    if model_id and isinstance(ollama, dict):
        selected = next(
            (
                item
                for item in ollama.get("selected_models", [])
                if isinstance(item, dict) and item.get("identifier") == model_id
            ),
            None,
        )
        if selected:
            compact["ollama_version"] = ollama.get("version")
            compact["model"] = {
                key: selected.get(key)
                for key in (
                    "identifier",
                    "digest",
                    "family",
                    "families",
                    "parameter_size",
                    "quantization_level",
                    "context_length",
                    "available",
                )
            }
    return compact


def _source_revision(runner: CommandRunner) -> dict[str, Any]:
    inside = _safe_command(runner, ("git", "rev-parse", "--is-inside-work-tree"))
    if not inside or inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"git_commit": None, "working_tree_dirty": None, "status": "unavailable"}
    commit = _safe_command(runner, ("git", "rev-parse", "HEAD"))
    status = _safe_command(runner, ("git", "status", "--porcelain"))
    return {
        "git_commit": (
            commit.stdout.strip() if commit and commit.returncode == 0 else None
        ),
        "working_tree_dirty": (
            bool(status.stdout.strip()) if status and status.returncode == 0 else None
        ),
        "status": "available",
    }


def _gpu_metadata(runner: CommandRunner) -> list[dict[str, Any]]:
    query = _safe_command(
        runner,
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ),
    )
    if query and query.returncode == 0 and query.stdout.strip():
        output: list[dict[str, Any]] = []
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                output.append(
                    {
                        "model": parts[0],
                        "memory_total_bytes": _mib_to_bytes(parts[1]),
                        "memory_available_bytes": _mib_to_bytes(parts[2]),
                        "detection": "nvidia-smi",
                    }
                )
        if output:
            return output

    pci = _safe_command(runner, ("lspci", "-mm"))
    if pci and pci.returncode == 0:
        found = []
        for line in pci.stdout.splitlines():
            lowered = line.lower()
            if "vga compatible controller" in lowered or "3d controller" in lowered:
                fields = [field.strip('"') for field in line.split("\t")]
                model = " ".join(fields[2:4]).strip() if len(fields) >= 4 else line
                found.append(
                    {
                        "model": model or None,
                        "memory_total_bytes": None,
                        "memory_available_bytes": None,
                        "detection": "lspci",
                    }
                )
        if found:
            return found
    return []


def _tag_record(item: dict[str, Any]) -> dict[str, Any]:
    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    return {
        "identifier": item.get("name") or item.get("model"),
        "digest": item.get("digest"),
        "size_bytes": item.get("size"),
        "modified_at": item.get("modified_at"),
        "format": details.get("format"),
        "family": details.get("family"),
        "families": details.get("families"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }


def _show_record(payload: dict[str, Any]) -> dict[str, Any]:
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    model_info = (
        payload.get("model_info") if isinstance(payload.get("model_info"), dict) else {}
    )
    context_length = next(
        (
            int(value)
            for key, value in model_info.items()
            if str(key).endswith(".context_length") and isinstance(value, (int, float))
        ),
        None,
    )
    return {
        "family": details.get("family"),
        "families": details.get("families"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "format": details.get("format"),
        "context_length": context_length,
        "capabilities": payload.get("capabilities"),
        "model_info": {
            str(key): value
            for key, value in model_info.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }


def _urllib_json_client(base_url: str) -> HttpClient:
    root = base_url.rstrip("/")

    def request_json(
        method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{root}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(str(exc)) from exc
        if not isinstance(result, dict):
            raise RuntimeError("Ollama API response was not a JSON object")
        return result

    return request_json


def _run_command(args: tuple[str, ...]) -> CommandResult:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(127, "", type(exc).__name__)
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _safe_command(
    runner: CommandRunner, args: tuple[str, ...]
) -> CommandResult | None:
    try:
        return runner(args)
    except Exception:
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def _total_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _mib_to_bytes(value: str) -> int | None:
    try:
        return int(float(value) * 1024 * 1024)
    except ValueError:
        return None


def _parse_ollama_version(value: str) -> str | None:
    for token in value.replace(",", " ").split():
        if token and token[0].isdigit():
            return token
    return None


def _nullable_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
