from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from api_drift_guard.contracts import (
    OasdiffChange,
    OasdiffExecutionError,
    OasdiffNotFoundError,
    SpecError,
    detect_changes,
    find_oasdiff,
    list_operations,
    parse_change,
    run_oasdiff_changelog,
    load_spec,
)
from api_drift_guard.models import ChangeType
from tests.conftest import FIXTURES, requires_oasdiff

# Entries below are verbatim oasdiff 1.33.0 `changelog --format json` output.


def _raw(**fields) -> OasdiffChange:
    return OasdiffChange.model_validate(
        {"operation": "GET", "path": "/users/{user_id}", "section": "paths", **fields}
    )


def test_removed_required_property_is_breaking() -> None:
    change = parse_change(
        _raw(
            id="response-required-property-removed",
            text="removed the required property `user_id` from the response with the `200` status",
            level=3,
        )
    )
    assert change.change_type == ChangeType.REMOVED_RESPONSE_PROPERTY
    assert change.property_name == "user_id"
    assert change.property_path == "user_id"
    assert change.response_status == "200"
    assert change.endpoint == "/users/{user_id}"
    assert change.method == "GET"
    assert change.operation == "GET /users/{user_id}"
    assert change.breaking is True


def test_removed_optional_property_is_breaking_despite_info_level() -> None:
    change = parse_change(
        _raw(
            id="response-optional-property-removed",
            text="removed the optional property `email` from the response with the `200` status",
            level=1,
        )
    )
    assert change.change_type == ChangeType.REMOVED_RESPONSE_PROPERTY
    assert change.breaking is True
    assert change.oasdiff_level == 1


@pytest.mark.parametrize(
    ("property_path", "leaf"),
    [("address/zip", "zip"), ("items/user_id", "user_id"), ("items/tags/items/label", "label")],
)
def test_nested_and_array_property_paths(property_path: str, leaf: str) -> None:
    change = parse_change(
        _raw(
            id="response-optional-property-removed",
            text=f"removed the optional property `{property_path}` from the response with the `200` status",
            level=1,
        )
    )
    assert change.property_path == property_path
    assert change.property_name == leaf


@pytest.mark.parametrize("check_id", ["response-required-property-added", "response-optional-property-added"])
def test_added_property_is_not_breaking(check_id: str) -> None:
    change = parse_change(
        _raw(id=check_id, text="added the optional property `phone` to the response with the `200` status", level=1)
    )
    assert change.change_type == ChangeType.ADDED_RESPONSE_PROPERTY
    assert change.property_name == "phone"
    assert change.breaking is False


def test_other_breaking_change_is_unsupported_but_kept() -> None:
    change = parse_change(
        _raw(id="api-path-removed-without-deprecation", text="api path removed without deprecation", level=3, path="/legacy")
    )
    assert change.change_type == ChangeType.UNSUPPORTED
    assert change.breaking is True
    assert change.property_name is None
    assert change.response_status is None


def test_other_info_change_is_unsupported_and_not_breaking() -> None:
    change = parse_change(_raw(id="response-success-status-added", text="added the success response with the status `201`", level=1))
    assert change.change_type == ChangeType.UNSUPPORTED
    assert change.breaking is False


def test_unparseable_property_text_falls_back_to_unsupported() -> None:
    change = parse_change(_raw(id="response-required-property-removed", text="something unexpected", level=3))
    assert change.change_type == ChangeType.UNSUPPORTED
    assert change.breaking is True


def test_spec_level_change_has_no_endpoint() -> None:
    change = parse_change(
        OasdiffChange.model_validate(
            {"id": "api-major-version-not-bumped", "text": "a breaking change was detected but the major version did not increase, from `1.0.0` to `1.1.0`", "level": 1, "section": "info"}
        )
    )
    assert change.endpoint is None
    assert change.method is None
    assert change.change_type == ChangeType.UNSUPPORTED
    assert change.breaking is False


def test_find_oasdiff_missing_from_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OASDIFF_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(OasdiffNotFoundError, match="not found on PATH"):
        find_oasdiff()


def test_find_oasdiff_bad_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OASDIFF_BIN", str(tmp_path / "missing"))
    with pytest.raises(OasdiffNotFoundError, match="not an executable file"):
        find_oasdiff()


def test_find_oasdiff_from_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "oasdiff"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("OASDIFF_BIN", str(binary))
    assert find_oasdiff() == str(binary)


def _fake_oasdiff(tmp_path: Path, stdout: str, exit_code: int = 0) -> str:
    """A stand-in binary that prints fixed output, for exercising error handling."""
    out = tmp_path / "out.txt"
    out.write_text(stdout)
    script = tmp_path / "fake-oasdiff"
    script.write_text(f"#!/bin/sh\ncat '{out}'\necho 'boom' >&2\nexit {exit_code}\n")
    script.chmod(0o755)
    return str(script)


def test_run_oasdiff_nonzero_exit(tmp_path: Path) -> None:
    binary = _fake_oasdiff(tmp_path, "", exit_code=102)
    with pytest.raises(OasdiffExecutionError, match="exited with code 102: boom"):
        run_oasdiff_changelog(Path("a.json"), Path("b.json"), binary)


@pytest.mark.parametrize("stdout", ["not json", '{"id": "x"}', '[{"text": "no level or id"}]'])
def test_run_oasdiff_unexpected_output(tmp_path: Path, stdout: str) -> None:
    binary = _fake_oasdiff(tmp_path, stdout)
    with pytest.raises(OasdiffExecutionError, match="Unexpected oasdiff output format"):
        run_oasdiff_changelog(Path("a.json"), Path("b.json"), binary)


def test_load_spec_accepts_openapi3_json_and_yaml(tmp_path: Path) -> None:
    load_spec(FIXTURES / "openapi_before.json")
    yaml_spec = tmp_path / "spec.yaml"
    yaml_spec.write_text("openapi: 3.0.3\ninfo: {title: t, version: '1'}\npaths: {}\n")
    load_spec(yaml_spec)


def test_list_operations() -> None:
    document = load_spec(FIXTURES / "openapi_before.json")
    assert list_operations(document) == [
        ("GET", "/users"),
        ("GET", "/users/{user_id}"),
        ("GET", "/users/{user_id}/preferences"),
    ]
    assert list_operations(
        {"paths": {"/a": {"parameters": [], "post": {}, "delete": {}, "summary": "x"}}}
    ) == [("DELETE", "/a"), ("POST", "/a")]
    assert list_operations({"openapi": "3.1.0"}) == []


def test_load_spec_rejects_swagger2(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"swagger": "2.0", "info": {}, "paths": {}}))
    with pytest.raises(SpecError, match="not an OpenAPI 3.x spec .swagger: '2.0'"):
        load_spec(spec)


def test_load_spec_rejects_missing_and_malformed(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SpecError, match="Could not parse"):
        load_spec(bad)


@requires_oasdiff
def test_detect_changes_on_fixtures() -> None:
    changes = detect_changes(
        FIXTURES / "openapi_before.json", FIXTURES / "openapi_after.json", find_oasdiff()
    )
    summary = {(c.method, c.endpoint, c.change_type.value, c.property_path, c.breaking) for c in changes}
    # Scenario 1: user_id renamed to userId (fixtures are generated from the FastAPI demo).
    assert summary == {
        ("GET", "/users", "removed_response_property", "items/user_id", True),
        ("GET", "/users", "added_response_property", "items/userId", False),
        ("GET", "/users/{user_id}", "removed_response_property", "user_id", True),
        ("GET", "/users/{user_id}", "added_response_property", "userId", False),
    }


@requires_oasdiff
@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("openapi_after_email_renamed.json", {("/users", "items/email"), ("/users/{user_id}", "email")}),
        ("openapi_after_phone_added.json", set()),
        ("openapi_after_newsletter_removed.json", {("/users/{user_id}/preferences", "newsletter")}),
        ("openapi_after_locale_removed.json", {("/users", "items/locale"), ("/users/{user_id}", "locale")}),
    ],
)
def test_breaking_changes_per_scenario(fixture: str, expected: set) -> None:
    changes = detect_changes(FIXTURES / "openapi_before.json", FIXTURES / fixture, find_oasdiff())
    assert {(c.endpoint, c.property_path) for c in changes if c.breaking} == expected


@requires_oasdiff
def test_detect_changes_identical_specs_is_empty() -> None:
    spec = FIXTURES / "openapi_before.json"
    assert detect_changes(spec, spec, find_oasdiff()) == []
