"""One-pass response-envelope normalization. Never edits the enclosed payload."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from cps_reason_bench.config import ConfigError
from cps_reason_bench.versions import content_hash


NORMALIZATION_VERSION = "1.0.0"
STRICT_FENCE = "markdown_single_fence"
DIAGNOSTIC_EXTRACTION = "markdown_fenced_extraction"
_FENCE_LINES = re.compile(r"(?m)^[ \t]*`{3,}[^\r\n]*(?:\r?\n|$)")
_OPENING = re.compile(r"[ \t]*```(?:json)?[ \t]*\r?\n\Z")
_CLOSING = re.compile(r"[ \t]*```[ \t]*(?:\r?\n)?\Z")


def response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalization_config(value: Mapping[str, Any] | None) -> dict[str, str]:
    if value is None:
        return {"mode": "none"}
    if not isinstance(value, Mapping) or set(value) - {"mode"}:
        raise ConfigError("response_normalization accepts only a mode field")
    mode = value.get("mode", "none")
    if mode not in {"none", STRICT_FENCE}:
        raise ConfigError("response_normalization.mode must be none or markdown_single_fence")
    return {"mode": mode}


def normalization_identity(mode: str) -> dict[str, Any]:
    if mode not in {"none", STRICT_FENCE, DIAGNOSTIC_EXTRACTION}:
        raise ValueError(f"Unknown normalization mode: {mode}")
    definition = {
        "mode": mode,
        "version": NORMALIZATION_VERSION,
        "opening_fence": "exactly three backticks, optional lowercase json, LF or CRLF",
        "closing_fence": "exactly three backticks on their own line",
        "outside_content": "whitespace_only" if mode == STRICT_FENCE else "diagnostic_extraction_only",
        "payload": "unchanged substring with surrounding whitespace trimmed",
        "passes": 1,
        "plain_response": "unchanged",
        "semantic_changes": False,
        "diagnostic_only": mode == DIAGNOSTIC_EXTRACTION,
    }
    return {**definition, "rule_hash": content_hash(definition)}


@dataclass(frozen=True)
class NormalizationResult:
    raw_response: str
    normalized_response: str
    mode: str
    applied: bool
    success: bool
    reason: str

    def metadata(self) -> dict[str, Any]:
        return {
            "normalized_response": self.normalized_response,
            "normalization_applied": self.applied,
            "normalization_rule": self.mode,
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_rule_hash": normalization_identity(self.mode)["rule_hash"],
            "normalization_success": self.success,
            "normalization_reason": self.reason,
            "raw_response_hash": response_hash(self.raw_response),
            "normalized_response_hash": response_hash(self.normalized_response),
        }


def normalize_response(raw: str, mode: str = "none") -> NormalizationResult:
    """Success describes envelope handling; the strict parser still decides validity.

    Extraction mode is callable only for offline diagnostics, never via model config.
    On failure the original string is returned so strict validation rejects it.
    """
    if mode not in {"none", STRICT_FENCE, DIAGNOSTIC_EXTRACTION}:
        raise ValueError(f"Unknown normalization mode: {mode}")

    def unchanged(success: bool, reason: str) -> NormalizationResult:
        return NormalizationResult(raw, raw, mode, False, success, reason)

    if mode == "none":
        return unchanged(True, "disabled")
    fences = list(_FENCE_LINES.finditer(raw))
    if not fences:
        return unchanged(True, "no_fence")
    if len(fences) != 2:
        return unchanged(False, "multiple_or_unbalanced_fences")
    opening, closing = fences
    if not _OPENING.fullmatch(opening.group()) or not _CLOSING.fullmatch(closing.group()):
        return unchanged(False, "unsupported_fence")
    outside = bool(raw[:opening.start()].strip() or raw[closing.end():].strip())
    if outside and mode == STRICT_FENCE:
        return unchanged(False, "outside_prose")
    interior = raw[opening.end():closing.start()].strip()
    if mode == DIAGNOSTIC_EXTRACTION:
        try:
            json.loads(interior)
        except json.JSONDecodeError:
            return unchanged(False, "invalid_json_inside_fence")
    return NormalizationResult(
        raw, interior, mode, True, True,
        "single_fence_removed" if not outside else "fenced_json_extracted_with_outside_prose",
    )
