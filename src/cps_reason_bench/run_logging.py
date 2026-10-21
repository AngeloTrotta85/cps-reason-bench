"""Structured JSONL episode logging."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, output_dir: str | Path, run_id: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.events_path = self.output_dir / f"{run_id}.jsonl"
        self.summary_path = self.output_dir / f"{run_id}_summary.json"
        self._stream = self.events_path.open("w", encoding="utf-8")
        self.sequence = 0

    def write(self, record_type: str, payload: dict[str, Any]) -> None:
        record = {
            "sequence": self.sequence,
            "record_type": record_type,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._stream.write(
            json.dumps(record, sort_keys=True, default=_json_default) + "\n"
        )
        self._stream.flush()
        self.sequence += 1

    def write_summary(self, summary: dict[str, Any]) -> None:
        # The completion trace must be durable before its summary becomes visible.
        self._stream.flush()
        os.fsync(self._stream.fileno())
        atomic_write_text(
            self.summary_path,
            json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_run_id(
    scenario_name: str, policy_name: str, seed: int, configuration: dict[str, Any]
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()[:8]
    return "_".join(
        [timestamp, _slug(scenario_name), _slug(policy_name), f"seed{seed}", digest]
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "run"


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Write a complete file and expose it only through an atomic rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
