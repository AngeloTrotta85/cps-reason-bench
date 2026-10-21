"""Secret detection and redaction shared by provider-facing components."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit


REDACTED = "[REDACTED]"
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "password",
    "secret",
    "cookie",
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def is_secret_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    return any(part in normalized for part in _SECRET_KEY_PARTS) or normalized == "token"


def redact_secrets(value: Any, secret_values: Iterable[str] = ()) -> Any:
    secrets = tuple(item for item in secret_values if item)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, nested in value.items():
            text_key = str(key)
            output[text_key] = (
                REDACTED
                if is_secret_key(text_key)
                else redact_secrets(nested, secrets)
            )
        return output
    if isinstance(value, list):
        return [redact_secrets(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item, secrets) for item in value)
    if isinstance(value, str):
        result = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", value)
        for secret in secrets:
            result = result.replace(secret, REDACTED)
        return result
    return value


def sanitize_url(value: str) -> str:
    """Remove credentials, query strings, and fragments from a logged endpoint."""

    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def safe_header_names(headers: Mapping[str, str]) -> list[str]:
    return sorted(
        key for key in (str(item) for item in headers) if not is_secret_key(key)
    )
