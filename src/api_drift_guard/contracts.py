"""OpenAPI contract comparison via the oasdiff CLI.

oasdiff output format (verified against oasdiff 1.33.0, ``changelog --format json``):
a JSON list of objects with ``id``, ``text``, ``level`` (1=INFO, 2=WARN, 3=ERR),
``operation``, ``path``, ``section`` and ``fingerprint``. Property names are only
available inside ``text`` (e.g. "removed the required property `user_id` from the
response with the `200` status"), so we pin ``--lang en`` and parse that text only
for the specific check IDs listed below.

Consumer-impact rule: oasdiff rates removal of an *optional* response property as
INFO, but a consumer doing ``data["field"]`` still fails, so every removed response
property is treated as breaking here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from api_drift_guard.models import ChangeType, ContractChange

logger = logging.getLogger(__name__)

OASDIFF_ENV_VAR = "OASDIFF_BIN"
OASDIFF_TIMEOUT_SECONDS = 60
OASDIFF_INSTALL_HINT = (
    "Install it from https://github.com/oasdiff/oasdiff/releases and put it on PATH, "
    f"or set {OASDIFF_ENV_VAR} to the binary's location."
)

LEVEL_WARN = 2

REMOVED_PROPERTY_IDS = frozenset(
    {"response-required-property-removed", "response-optional-property-removed"}
)
ADDED_PROPERTY_IDS = frozenset(
    {"response-required-property-added", "response-optional-property-added"}
)

_OPENAPI_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})

_PROPERTY_RE = re.compile(r"property `([^`]+)`")
_STATUS_RE = re.compile(r"with the `([^`]+)` status")


class ContractError(Exception):
    """Base error for contract comparison failures."""


class OasdiffNotFoundError(ContractError):
    pass


class OasdiffExecutionError(ContractError):
    pass


class SpecError(ContractError):
    pass


class OasdiffChange(BaseModel):
    """One raw entry from ``oasdiff changelog --format json``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    text: str = ""
    level: int
    operation: str = ""
    path: str = ""


_CHANGES_ADAPTER = TypeAdapter(list[OasdiffChange])


def find_oasdiff() -> str:
    """Return the oasdiff binary path from $OASDIFF_BIN or PATH."""
    configured = os.environ.get(OASDIFF_ENV_VAR)
    if configured:
        if Path(configured).is_file() and os.access(configured, os.X_OK):
            return configured
        raise OasdiffNotFoundError(
            f"{OASDIFF_ENV_VAR} is set to {configured!r}, which is not an executable file. "
            + OASDIFF_INSTALL_HINT
        )
    found = shutil.which("oasdiff")
    if found is None:
        raise OasdiffNotFoundError("oasdiff was not found on PATH. " + OASDIFF_INSTALL_HINT)
    return found


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=OASDIFF_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise OasdiffExecutionError(
            f"oasdiff timed out after {OASDIFF_TIMEOUT_SECONDS}s: {' '.join(cmd)}"
        ) from exc
    except OSError as exc:
        raise OasdiffExecutionError(f"Could not execute oasdiff at {cmd[0]!r}: {exc}") from exc


def get_oasdiff_version(binary: str) -> str:
    proc = _run([binary, "--version"])
    if proc.returncode != 0:
        raise OasdiffExecutionError(f"'oasdiff --version' failed: {proc.stderr.strip()}")
    # Output looks like "oasdiff version 1.33.0".
    return proc.stdout.strip().removeprefix("oasdiff version").strip()


def load_spec(path: Path) -> dict[str, Any]:
    """Load ``path`` and fail early with a clear message unless it is an OpenAPI 3.x document."""
    if not path.is_file():
        raise SpecError(f"OpenAPI spec not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SpecError(f"Could not parse OpenAPI spec {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise SpecError(f"OpenAPI spec {path} must be a JSON/YAML object")
    version = document.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        found = f"openapi: {version!r}" if version is not None else "no 'openapi' field"
        if "swagger" in document:
            found = f"swagger: {document['swagger']!r}"
        raise SpecError(f"{path} is not an OpenAPI 3.x spec ({found}); only OpenAPI 3.x is supported")
    return document


def list_operations(document: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (METHOD, path template) pairs declared in an OpenAPI document."""
    paths = document.get("paths") or {}
    return sorted(
        (method.upper(), path)
        for path, item in paths.items()
        if isinstance(item, dict)
        for method in item
        if method in _OPENAPI_METHODS
    )


def run_oasdiff_changelog(old_spec: Path, new_spec: Path, binary: str) -> list[OasdiffChange]:
    cmd = [binary, "changelog", str(old_spec), str(new_spec), "--format", "json", "--lang", "en"]
    logger.debug("running oasdiff", extra={"cmd": cmd})
    proc = _run(cmd)
    if proc.returncode != 0:
        raise OasdiffExecutionError(
            f"oasdiff exited with code {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        data = json.loads(proc.stdout or "[]")
        return _CHANGES_ADAPTER.validate_python(data or [])
    except (json.JSONDecodeError, ValidationError) as exc:
        raise OasdiffExecutionError(
            f"Unexpected oasdiff output format (expected a JSON list of changes): {exc}"
        ) from exc


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def parse_change(raw: OasdiffChange) -> ContractChange:
    """Normalise one oasdiff entry; unknown or unparseable entries become UNSUPPORTED."""
    common = {
        "endpoint": raw.path or None,
        "method": raw.operation.upper() or None,
        "response_status": _first_match(_STATUS_RE, raw.text),
        "oasdiff_id": raw.id,
        "oasdiff_level": raw.level,
        "description": raw.text,
    }

    if raw.id in REMOVED_PROPERTY_IDS or raw.id in ADDED_PROPERTY_IDS:
        property_path = _first_match(_PROPERTY_RE, raw.text)
        if property_path:
            removed = raw.id in REMOVED_PROPERTY_IDS
            return ContractChange(
                **common,
                change_type=(
                    ChangeType.REMOVED_RESPONSE_PROPERTY if removed else ChangeType.ADDED_RESPONSE_PROPERTY
                ),
                property_name=property_path.rsplit("/", 1)[-1],
                property_path=property_path,
                breaking=removed,
            )
        logger.warning(
            "could not extract property name from oasdiff change; reporting it as unsupported",
            extra={"oasdiff_id": raw.id, "text": raw.text},
        )

    return ContractChange(
        **common,
        change_type=ChangeType.UNSUPPORTED,
        breaking=raw.level >= LEVEL_WARN,
    )


def detect_changes(old_spec: Path, new_spec: Path, binary: str) -> list[ContractChange]:
    """Compare two specs and return normalised changes in a stable order."""
    changes = [parse_change(raw) for raw in run_oasdiff_changelog(old_spec, new_spec, binary)]
    changes.sort(
        key=lambda c: (c.endpoint or "", c.method or "", c.change_type, c.property_path or "", c.oasdiff_id)
    )
    logger.info(
        "contract comparison complete",
        extra={"changes": len(changes), "breaking": sum(c.breaking for c in changes)},
    )
    return changes
