from __future__ import annotations

from pathlib import Path

import pytest

from api_drift_guard.config import ConfigError, LlmSettings, LlmSetupError
from api_drift_guard.evaluation import (
    CaseResult,
    classify_outcome,
    compute_metrics,
    load_dataset,
    render_evaluation_markdown,
    run_evaluation,
)
from api_drift_guard.models import Classification as C
from api_drift_guard.pipeline import Mode
from tests.conftest import REPO_ROOT, requires_oasdiff
from tests.fake_llm import FakeLiteLLMClient, verdict

DATASET = REPO_ROOT / "evaluation" / "scenarios.yaml"


@pytest.mark.parametrize(
    ("expected", "predicted", "outcome"),
    [
        (C.AFFECTED, C.AFFECTED, "correct"),
        (C.NO_RELEVANT_USAGE_FOUND, C.AFFECTED, "false_positive"),
        (C.NO_RELEVANT_USAGE_FOUND, C.POTENTIALLY_AFFECTED, "false_positive"),
        (C.AFFECTED, C.NO_RELEVANT_USAGE_FOUND, "false_negative"),
        (C.POTENTIALLY_AFFECTED, C.NO_RELEVANT_USAGE_FOUND, "false_negative"),
        (C.AFFECTED, C.UNCERTAIN, "uncertain"),
        (C.AFFECTED, C.POTENTIALLY_AFFECTED, "mismatch"),
        (C.UNCERTAIN, C.AFFECTED, "mismatch"),
    ],
)
def test_classify_outcome(expected: C, predicted: C, outcome: str) -> None:
    assert classify_outcome(expected, predicted) == outcome


def test_compute_metrics() -> None:
    def case(expected: C, predicted: C, *, missing: int = 0, invalid: int = 0) -> CaseResult:
        return CaseResult(
            scenario="s", consumer="c", expected=expected, predicted=predicted,
            outcome=classify_outcome(expected, predicted), expected_evidence=["a:1", "a:2"],
            missing_evidence=["a:1", "a:2"][:missing], evidence_references=3, invalid_evidence=["x:1"] * invalid,
        )

    metrics = compute_metrics(
        [case(C.AFFECTED, C.AFFECTED), case(C.AFFECTED, C.UNCERTAIN, missing=1), case(C.NO_RELEVANT_USAGE_FOUND, C.AFFECTED, invalid=1)]
    )
    assert metrics.model_dump() == {
        "cases": 3, "correct": 1, "accuracy": 0.333, "false_positives": 1, "false_negatives": 0,
        "uncertain": 1, "mismatches": 0, "evidence_references": 9, "invalid_evidence_references": 1,
        "expected_evidence": 6, "expected_evidence_found": 5,
    }
    assert compute_metrics([]).accuracy == 0.0


def test_dataset_paths_resolve_relative_to_file() -> None:
    dataset = load_dataset(DATASET)
    assert dataset.config == REPO_ROOT / "config.example.yaml"
    assert [s.id for s in dataset.scenarios] == [
        "1_user_id_renamed", "2_email_renamed", "3_phone_added", "4_newsletter_removed", "5_locale_removed",
    ]
    assert all(s.new_spec.is_file() for s in dataset.scenarios)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("config: c.yaml\nold_spec: o.json\nscenarios: []\nextra: 1\n", "Invalid evaluation dataset"),
        (
            "config: c.yaml\nold_spec: o.json\nscenarios:\n"
            + "  - {id: a, description: d, new_spec: n.json, expected_status: no_breaking_changes, expected_agents: 0, consumers: {}}\n" * 2,
            "unique",
        ),
    ],
)
def test_invalid_datasets(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "d.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=match):
        load_dataset(path)
    with pytest.raises(ConfigError, match="not found"):
        load_dataset(tmp_path / "missing.yaml")


def test_unknown_consumer_in_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "d.yaml"
    path.write_text(
        f"config: {REPO_ROOT / 'config.example.yaml'}\nold_spec: {REPO_ROOT / 'fixtures/openapi_before.json'}\n"
        "scenarios:\n  - id: a\n    description: d\n    new_spec: x.json\n    expected_status: no_breaking_changes\n"
        "    expected_agents: 0\n    consumers: {ghost-service: {expected: affected}}\n"
    )
    with pytest.raises(ConfigError, match="unknown consumers: ghost-service"):
        run_evaluation(path, mode=Mode.STATIC)


@requires_oasdiff
def test_static_evaluation_on_demo_dataset() -> None:
    report = run_evaluation(DATASET, mode=Mode.STATIC)
    assert report.agentic is None
    assert report.static.model_dump() == {
        "cases": 15, "correct": 12, "accuracy": 0.8, "false_positives": 0, "false_negatives": 0,
        "uncertain": 3, "mismatches": 0, "evidence_references": 17, "invalid_evidence_references": 0,
        "expected_evidence": 5, "expected_evidence_found": 5,
    }
    assert report.scenarios_with_expected_status == 5
    assert report.scenarios_with_expected_agents == 5
    uncertain = [(c.scenario, c.consumer) for s in report.scenarios for c in s.static_cases if c.outcome == "uncertain"]
    assert uncertain == [
        ("1_user_id_renamed", "notification-service"),
        ("2_email_renamed", "notification-service"),
        ("5_locale_removed", "notification-service"),
    ]
    assert all(s.llm_requests == 0 for s in report.scenarios)
    assert "LLM-assisted accuracy was not measured" in " ".join(report.notes)


@requires_oasdiff
def test_agentic_evaluation_scores_static_and_llm_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the evaluator's bookkeeping with a scripted LLM. This checks the
    evaluation mechanics only; it is not a measurement of real LLM accuracy."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    line17 = ("notification_service/notifier.py", 17, "return {field: user[field] for field in TEMPLATE_FIELDS}")

    def responder(call):
        removed = call.payload["items"][0]["removed_property"]
        affected = removed in ("email", "locale")
        return {
            "verdicts": [
                verdict(
                    item["item_id"],
                    "affected" if affected else "no_relevant_usage_found",
                    evidence=[line17] if affected else [],
                )
                for item in call.payload["items"]
            ]
        }

    client = FakeLiteLLMClient(responder)
    report = run_evaluation(DATASET, mode=Mode.AGENTIC, llm_settings=LlmSettings(escalation_model=None), llm_client=client)
    assert report.static.correct == 12 and report.static.uncertain == 3  # unchanged by the LLM
    assert report.agentic is not None
    assert (report.agentic.correct, report.agentic.false_negatives, report.agentic.uncertain) == (15, 0, 0)
    assert report.agentic.invalid_evidence_references == 0
    assert [s.agents for s in report.scenarios] == [1, 1, 0, 0, 1]
    assert len(client.calls) == 3
    assert sum(s.llm_requests for s in report.scenarios) == 3
    markdown = render_evaluation_markdown(report)
    assert "| Agentic (static + LLM) | 15/15 (100%)" in markdown


def test_agentic_evaluation_without_credentials_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LlmSetupError, match="GROQ_API_KEY"):
        run_evaluation(DATASET, mode=Mode.AGENTIC)
