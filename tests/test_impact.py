from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from api_drift_guard.config import ConsumerConfig
from api_drift_guard.discovery import Operation
from api_drift_guard.impact import assess_consumer, assess_consumers, most_severe
from api_drift_guard.models import ChangeType, Classification, ContractChange, ImpactFinding

OPERATIONS = [Operation("GET", "/users/{user_id}"), Operation("GET", "/users"), Operation("GET", "/orders/{id}")]


def removed(prop: str, endpoint: str = "/users/{user_id}") -> ContractChange:
    return ContractChange(
        endpoint=endpoint,
        method="GET",
        response_status="200",
        change_type=ChangeType.REMOVED_RESPONSE_PROPERTY,
        property_name=prop.rsplit("/", 1)[-1],
        property_path=prop,
        breaking=True,
        oasdiff_id="response-required-property-removed",
        oasdiff_level=3,
        description=f"removed the required property `{prop}` from the response with the `200` status",
    )


def added(prop: str) -> ContractChange:
    return removed(prop).model_copy(
        update={"change_type": ChangeType.ADDED_RESPONSE_PROPERTY, "breaking": False, "oasdiff_id": "response-required-property-added"}
    )


def consumer(tmp_path: Path, code: str, name: str = "svc") -> ConsumerConfig:
    root = tmp_path / name
    root.mkdir()
    (root / "client.py").write_text(textwrap.dedent(code))
    return ConsumerConfig(name=name, path=root)


def assess(tmp_path: Path, code: str, change: ContractChange, extra: list[ContractChange] | None = None) -> ImpactFinding:
    impact = assess_consumer(consumer(tmp_path, code), OPERATIONS, [change], [change, *(extra or [])])
    (finding,) = impact.findings
    return finding


HEADER = 'import httpx\nBASE = "http://user-service"\n'


def test_direct_subscript_is_affected_with_evidence(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def f(uid):\n    data = httpx.get(f"{BASE}/users/{uid}").json()\n    return data["user_id"]\n'
        'def g(uid):\n    return httpx.get(f"{BASE}/users/{uid}").json()["name"]\n',
        removed("user_id"),
        extra=[added("userId")],
    )
    assert finding.classification == Classification.AFFECTED
    assert finding.analysis_method == "static"
    assert finding.confidence is None
    assert [(e.file, e.line, e.snippet) for e in finding.evidence] == [
        ("client.py", 5, 'return data["user_id"]'),
        ("client.py", 4, 'data = httpx.get(f"{BASE}/users/{uid}").json()'),
    ]
    assert "`userId`" in (finding.suggested_action or "")


def test_other_fields_only_is_no_relevant_usage(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def f(uid):\n    data = httpx.get(f"{BASE}/users/{uid}").json()\n    return data["email"]\n',
        removed("user_id"),
    )
    assert finding.classification == Classification.NO_RELEVANT_USAGE_FOUND
    assert [e.line for e in finding.evidence] == [4]  # the matched request
    assert finding.suggested_action is None


def test_not_calling_the_endpoint_is_no_relevant_usage(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def f(i):\n    return httpx.get(f"{BASE}/orders/{i}").json()["user_id"]\n',
        removed("user_id"),
    )
    assert finding.classification == Classification.NO_RELEVANT_USAGE_FOUND
    assert finding.evidence == []
    assert "No httpx request matching GET /users/{user_id}" in finding.reason


@pytest.mark.parametrize(
    "body",
    ['return data.get("user_id")', 'return "user_id" in data'],
)
def test_tolerant_reads_are_potentially_affected(tmp_path: Path, body: str) -> None:
    finding = assess(
        tmp_path,
        HEADER + f'def f(uid):\n    data = httpx.get(f"{{BASE}}/users/{{uid}}").json()\n    {body}\n',
        removed("user_id"),
    )
    assert finding.classification == Classification.POTENTIALLY_AFFECTED


def test_direct_read_on_unresolved_url_is_potentially_affected(tmp_path: Path) -> None:
    finding = assess(tmp_path, HEADER + 'def f(url):\n    return httpx.get(url).json()["user_id"]\n', removed("user_id"))
    assert finding.classification == Classification.POTENTIALLY_AFFECTED
    assert any("could not be resolved" in e.note for e in finding.evidence)


def test_dynamic_access_is_uncertain(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'FIELDS = ("name",)\ndef f(uid):\n    user = httpx.get(f"{BASE}/users/{uid}").json()\n'
        "    return {k: user[k] for k in FIELDS}\n",
        removed("user_id"),
    )
    assert finding.classification == Classification.UNCERTAIN
    assert any("non-literal key" in e.note for e in finding.evidence)


def test_returned_json_is_uncertain_and_cites_untracked_reads(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def get_user(uid):\n    return httpx.get(f"{BASE}/users/{uid}").json()\n'
        'def owner(uid):\n    return get_user(uid)["user_id"]\n',
        removed("user_id"),
    )
    assert finding.classification == Classification.UNCERTAIN
    notes = [e.note for e in finding.evidence]
    assert any("returned from get_user()" in n for n in notes)
    assert any("not linked to a tracked response" in n for n in notes)


def test_escape_of_unrelated_leaf_does_not_cause_uncertainty(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def f(uid):\n    data = httpx.get(f"{BASE}/users/{uid}").json()\n    return {"name": data["name"]}\n',
        removed("user_id"),
    )
    assert finding.classification == Classification.NO_RELEVANT_USAGE_FOUND


def test_escape_of_parent_object_is_uncertain_for_nested_property(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def f(uid):\n    data = httpx.get(f"{BASE}/users/{uid}").json()\n    return data["address"]\n',
        removed("address/zip"),
    )
    assert finding.classification == Classification.UNCERTAIN


def test_array_property_paths(tmp_path: Path) -> None:
    finding = assess(
        tmp_path,
        HEADER + 'def f():\n    return [u["user_id"] for u in httpx.get(f"{BASE}/users").json()]\n',
        removed("items/user_id", endpoint="/users"),
    )
    assert finding.classification == Classification.AFFECTED


def test_parse_errors_turn_no_usage_into_uncertain(tmp_path: Path) -> None:
    config = consumer(tmp_path, HEADER + 'def f(uid):\n    return httpx.get(f"{BASE}/users/{uid}").json()["email"]\n')
    (config.path / "broken.py").write_text("def broken(:\n")
    impact = assess_consumer(config, OPERATIONS, [removed("user_id")], [removed("user_id")])
    assert impact.findings[0].classification == Classification.UNCERTAIN
    assert impact.parse_errors and impact.files_scanned == 1


def test_unsupported_change_on_called_endpoint_is_uncertain(tmp_path: Path) -> None:
    change = removed("user_id").model_copy(
        update={"change_type": ChangeType.UNSUPPORTED, "property_name": None, "property_path": None, "oasdiff_id": "response-body-type-changed"}
    )
    finding = assess(tmp_path, HEADER + 'def f(uid):\n    httpx.get(f"{BASE}/users/{uid}")\n', change)
    assert finding.classification == Classification.UNCERTAIN
    assert "response-body-type-changed" in finding.reason


def test_consumer_status_is_most_severe_finding(tmp_path: Path) -> None:
    config = consumer(
        tmp_path,
        HEADER + 'def f(uid):\n    data = httpx.get(f"{BASE}/users/{uid}").json()\n    return data["email"], data.get("name")\n',
    )
    changes = [removed("user_id"), removed("name"), removed("items/user_id", endpoint="/users")]
    impact = assess_consumer(config, OPERATIONS, changes, changes)
    assert [f.classification for f in impact.findings] == [
        Classification.NO_RELEVANT_USAGE_FOUND,
        Classification.POTENTIALLY_AFFECTED,
        Classification.NO_RELEVANT_USAGE_FOUND,
    ]
    assert impact.status == Classification.POTENTIALLY_AFFECTED
    assert impact.http_calls == 1
    assert impact.unresolved_http_calls == 0


def test_no_breaking_changes_skips_consumer_analysis(tmp_path: Path) -> None:
    # The consumer path does not exist: proves it is never read.
    missing = ConsumerConfig(name="svc", path=tmp_path / "missing")
    assert assess_consumers([missing], OPERATIONS, [added("phone")]) == []


def test_most_severe() -> None:
    assert most_severe([Classification.NO_RELEVANT_USAGE_FOUND, Classification.UNCERTAIN]) == Classification.UNCERTAIN
    assert most_severe([Classification.UNCERTAIN, Classification.AFFECTED]) == Classification.AFFECTED
    assert most_severe([]) == Classification.NO_RELEVANT_USAGE_FOUND
