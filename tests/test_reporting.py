from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from api_drift_guard.models import ChangeType, CompatibilityReport, ContractChange
from api_drift_guard.reporting import build_report, render_json, write_report


def _change(change_type: ChangeType, breaking: bool, prop: str | None = "x") -> ContractChange:
    return ContractChange(
        endpoint="/users/{user_id}",
        method="GET",
        response_status="200",
        change_type=change_type,
        property_name=prop,
        property_path=prop,
        breaking=breaking,
        oasdiff_id="some-id",
        oasdiff_level=3 if breaking else 1,
        description="text",
    )


def _report(changes: list[ContractChange]) -> CompatibilityReport:
    return build_report(
        provider="user-service",
        old_spec=Path("old.json"),
        new_spec=Path("new.json"),
        oasdiff_version="1.33.0",
        changes=changes,
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_summary_counts_and_status() -> None:
    report = _report(
        [
            _change(ChangeType.REMOVED_RESPONSE_PROPERTY, True),
            _change(ChangeType.ADDED_RESPONSE_PROPERTY, False),
            _change(ChangeType.ADDED_RESPONSE_PROPERTY, False),
            _change(ChangeType.UNSUPPORTED, True, prop=None),
        ]
    )
    assert report.status == "breaking_changes_detected"
    assert report.summary.model_dump() == {
        "total_changes": 4,
        "breaking_changes": 2,
        "removed_response_properties": 1,
        "added_response_properties": 2,
        "unsupported_changes": 1,
    }


def test_no_changes_report() -> None:
    report = _report([])
    assert report.status == "no_breaking_changes"
    assert report.summary.total_changes == 0
    assert report.limitations


def test_only_additions_is_not_breaking() -> None:
    assert _report([_change(ChangeType.ADDED_RESPONSE_PROPERTY, False)]).status == "no_breaking_changes"


def test_json_round_trip(tmp_path: Path) -> None:
    report = _report([_change(ChangeType.REMOVED_RESPONSE_PROPERTY, True)])
    out = tmp_path / "nested" / "report.json"
    write_report(report, out)

    data = json.loads(out.read_text())
    assert data["changes"][0]["change_type"] == "removed_response_property"
    assert data["changes"][0]["property_name"] == "x"
    assert CompatibilityReport.model_validate(data) == report
    assert json.loads(render_json(report)) == data


def test_markdown_for_report_without_breaking_changes() -> None:
    from api_drift_guard.reporting import render_markdown

    text = render_markdown(_report([_change(ChangeType.ADDED_RESPONSE_PROPERTY, False)]))
    assert text.startswith("# API Compatibility Report\n\nProvider: user-service")
    assert "- Result: No breaking changes detected" in text
    assert "- Added response field: `x`" in text
    assert "consumer code was not analyzed and no investigation agents were needed" in text
    assert "- Affected consumers: 0" in text
    assert "## LLM Investigation" not in text


def test_markdown_consumer_sections() -> None:
    from api_drift_guard.models import Classification, ConsumerImpact, Evidence, ImpactFinding, InvestigationRecord
    from api_drift_guard.reporting import render_markdown

    llm_finding = ImpactFinding(
        consumer="svc", endpoint="GET /users/{user_id}", property_path="x", change="removed x",
        classification=Classification.AFFECTED, confidence=0.8, reason="Reads x.",
        evidence=[Evidence(file="a.py", line=3, snippet='v = d["x"]  # `odd`', note="cited by LLM investigation")],
        suggested_action="Stop reading x.", analysis_method="llm",
        static_classification=Classification.UNCERTAIN, notes=["1 reference discarded."],
    )
    unused = ImpactFinding(
        consumer="svc", endpoint="GET /users", property_path="items/x", change="removed",
        classification=Classification.NO_RELEVANT_USAGE_FOUND, reason="No httpx request matching GET /users was found.",
    )
    consumer = ConsumerImpact(
        consumer="svc", path="/tmp/svc", status=Classification.AFFECTED, files_scanned=1, http_calls=1,
        unresolved_http_calls=0, findings=[llm_finding, unused],
    )
    record = InvestigationRecord(
        consumer="svc", agent_name="investigator_svc", model="groq/m", items=1, status="completed",
        attempts=1, prompt_tokens=10, completion_tokens=5, duration_seconds=0.1,
    )
    report = build_report(
        provider="user-service", old_spec=Path("o.json"), new_spec=Path("n.json"), oasdiff_version="1",
        changes=[_change(ChangeType.REMOVED_RESPONSE_PROPERTY, True)], consumers=[consumer],
        analysis_mode="agentic", investigations=[record],
    )
    text = render_markdown(report)
    assert "### svc\n\nStatus: **Affected**" in text
    assert "#### GET /users/{user_id}: removed `x`" in text
    assert '- `a.py:3`: `` v = d["x"]  # `odd` `` (cited by LLM investigation)' in text
    assert "model confidence 0.80" in text and "static result: Uncertain" in text
    assert "- Note: 1 reference discarded." in text
    assert "No relevant usage found:\n\n- GET /users: removed `items/x`." in text
    assert "Agents spawned: 1 · LLM requests: 1 · Tokens: 10 prompt / 5 completion" in text
    assert "| investigator_svc | groq/m | 1 | completed | 1 | 0 |" in text
