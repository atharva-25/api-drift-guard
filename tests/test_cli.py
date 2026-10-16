from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from api_drift_guard.cli import app
from tests.conftest import FIXTURES, REPO_ROOT, requires_oasdiff

runner = CliRunner()

CONFIG = "provider:\n  name: user-service\nconsumers:\n  - name: order-service\n    path: ./order\n"


def _analyze(old: Path, new: Path, config: Path, *extra: str):
    return runner.invoke(
        app, ["analyze", "--old", str(old), "--new", str(new), "--config", str(config), *extra]
    )


@requires_oasdiff
def test_analyze_writes_json_report(write_config, tmp_path: Path) -> None:
    (tmp_path / "order").mkdir()
    out = tmp_path / "report.json"
    result = _analyze(
        FIXTURES / "openapi_before.json", FIXTURES / "openapi_after.json", write_config(CONFIG), "-o", str(out)
    )
    assert result.exit_code == 0, result.output

    report = json.loads(out.read_text())
    assert report["provider"] == "user-service"
    assert report["status"] == "breaking_changes_detected"
    assert report["summary"]["removed_response_properties"] == 2
    assert report["summary"]["breaking_changes"] == 2
    removed = {c["property_name"] for c in report["changes"] if c["breaking"]}
    assert removed == {"user_id"}


@requires_oasdiff
def test_analyze_prints_json_to_stdout_without_output_option(write_config) -> None:
    spec = FIXTURES / "openapi_before.json"
    result = _analyze(spec, spec, write_config(CONFIG), "--log-level", "ERROR")
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["status"] == "no_breaking_changes"
    assert report["changes"] == []


@requires_oasdiff
def test_analyze_demo_consumers(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = _analyze(
        FIXTURES / "openapi_before.json",
        FIXTURES / "openapi_after.json",
        REPO_ROOT / "config.example.yaml",
        "-o",
        str(out),
    )
    assert result.exit_code == 0, result.output

    report = json.loads(out.read_text())
    statuses = {c["consumer"]: c["status"] for c in report["consumers"]}
    assert statuses == {
        "order-service": "affected",
        "billing-service": "no_relevant_usage_found",
        "notification-service": "uncertain",
    }
    assert report["consumer_summary"] == {
        "affected": 1,
        "potentially_affected": 0,
        "no_relevant_usage_found": 1,
        "uncertain": 1,
    }
    order = next(c for c in report["consumers"] if c["consumer"] == "order-service")
    affected = [f for f in order["findings"] if f["classification"] == "affected"]
    assert [(e["file"], e["line"]) for e in affected[0]["evidence"]] == [
        ("order_service/users_client.py", 12),
        ("order_service/users_client.py", 9),
    ]


@requires_oasdiff
def test_analyze_reports_missing_consumer_path(write_config) -> None:
    config = write_config("provider:\n  name: p\nconsumers:\n  - name: gone\n    path: ./does-not-exist\n")
    result = _analyze(FIXTURES / "openapi_before.json", FIXTURES / "openapi_after.json", config)
    assert result.exit_code == 1
    assert "Consumer path does not exist" in result.output


def test_agentic_mode_without_credentials_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env to pick up
    spec = FIXTURES / "openapi_before.json"
    result = _analyze(spec, spec, REPO_ROOT / "config.example.yaml", "--mode", "agentic")
    assert result.exit_code == 1
    assert "Missing credentials" in result.output and "GROQ_API_KEY" in result.output
    assert "--mode static" in result.output


@requires_oasdiff
def test_agentic_mode_on_demo_with_mocked_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.fake_llm import FakeLiteLLMClient, for_all_items

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    client = FakeLiteLLMClient(
        for_all_items(
            "no_relevant_usage_found",
            confidence=0.9,
            evidence=[("notification_service/notifier.py", 6, 'TEMPLATE_FIELDS = ("name", "email", "locale")')],
            reason="TEMPLATE_FIELDS only selects name and email.",
        )
    )
    monkeypatch.setattr("api_drift_guard.cli.make_llm_client", lambda: client)
    out = tmp_path / "report.json"
    result = _analyze(
        FIXTURES / "openapi_before.json",
        FIXTURES / "openapi_after.json",
        REPO_ROOT / "config.example.yaml",
        "--mode",
        "agentic",
        "-o",
        str(out),
    )
    assert result.exit_code == 0, result.output

    report = json.loads(out.read_text())
    assert report["analysis_mode"] == "agentic"
    # Only notification-service was uncertain statically, so exactly one agent ran.
    assert [r["agent_name"] for r in report["investigations"]] == ["investigator_notification_service"]
    assert report["llm_usage"] == {"agents_spawned": 1, "llm_requests": 1, "prompt_tokens": 100, "completion_tokens": 20}
    statuses = {c["consumer"]: c["status"] for c in report["consumers"]}
    assert statuses == {
        "order-service": "affected",
        "billing-service": "no_relevant_usage_found",
        "notification-service": "no_relevant_usage_found",
    }
    notifier = next(c for c in report["consumers"] if c["consumer"] == "notification-service")
    finding = next(f for f in notifier["findings"] if f["analysis_method"] == "llm")
    assert finding["static_classification"] == "uncertain"
    assert finding["evidence"][0]["line"] == 6
    assert len(client.calls) == 1


@requires_oasdiff
def test_analyze_writes_markdown_report(tmp_path: Path) -> None:
    md, js = tmp_path / "report.md", tmp_path / "report.json"
    result = _analyze(
        FIXTURES / "openapi_before.json", FIXTURES / "openapi_after.json", REPO_ROOT / "config.example.yaml",
        "--markdown", str(md), "-o", str(js),
    )
    assert result.exit_code == 0, result.output
    text = md.read_text()
    assert text.startswith("# API Compatibility Report")
    assert "### order-service\n\nStatus: **Affected**" in text
    assert "`order_service/users_client.py:12`: `return user[\"user_id\"]`" in text
    assert "- Affected consumers: 1" in text
    assert json.loads(js.read_text())["status"] == "breaking_changes_detected"
    assert result.stdout == ""  # nothing on stdout when files are requested


@requires_oasdiff
def test_evaluate_command_static(tmp_path: Path) -> None:
    out = tmp_path / "evaluation.json"
    result = runner.invoke(app, ["evaluate", "--dataset", str(REPO_ROOT / "evaluation/scenarios.yaml"), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# API Drift Guard Evaluation")
    assert "| Agentic (static + LLM) | not run |" in result.stdout
    data = json.loads(out.read_text())
    assert data["mode"] == "static" and data["agentic"] is None


def test_evaluate_agentic_requires_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["evaluate", "--dataset", str(REPO_ROOT / "evaluation/scenarios.yaml"), "--mode", "agentic"]
    )
    assert result.exit_code == 1
    assert "Missing credentials" in result.output


def test_analyze_reports_config_error(tmp_path: Path) -> None:
    spec = FIXTURES / "openapi_before.json"
    result = _analyze(spec, spec, tmp_path / "missing.yaml")
    assert result.exit_code == 1
    assert "Config file not found" in result.output


def test_analyze_reports_missing_oasdiff(write_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OASDIFF_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    spec = FIXTURES / "openapi_before.json"
    result = _analyze(spec, spec, write_config(CONFIG))
    assert result.exit_code == 1
    assert "oasdiff was not found on PATH" in result.output


def test_analyze_rejects_non_openapi3_spec(write_config, tmp_path: Path) -> None:
    swagger = tmp_path / "swagger.json"
    swagger.write_text('{"swagger": "2.0"}')
    result = _analyze(swagger, FIXTURES / "openapi_after.json", write_config(CONFIG))
    assert result.exit_code == 1
    assert "only OpenAPI 3.x is supported" in result.output
