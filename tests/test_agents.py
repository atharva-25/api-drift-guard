from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from api_drift_guard.agents import (
    AgentReport,
    agent_name,
    backoff_delay,
    check_llm_environment,
    create_investigation_agent,
    investigate,
    plan_investigations,
)
from api_drift_guard.config import ConsumerConfig, LlmSettings, LlmSetupError
from api_drift_guard.discovery import Operation
from api_drift_guard.impact import assess_consumer
from api_drift_guard.models import ChangeType, Classification, ConsumerImpact, ContractChange
from tests.fake_llm import FakeLiteLLMClient, auth_error, for_all_items, rate_limit_error, verdict

OPERATIONS = [Operation("GET", "/users/{user_id}")]
REMOVED = ContractChange(
    endpoint="/users/{user_id}",
    method="GET",
    response_status="200",
    change_type=ChangeType.REMOVED_RESPONSE_PROPERTY,
    property_name="user_id",
    property_path="user_id",
    breaking=True,
    oasdiff_id="response-required-property-removed",
    oasdiff_level=3,
    description="removed the required property `user_id` from the response with the `200` status",
)

# Static result: uncertain (dynamic key on line 7).
DYNAMIC = """\
import httpx
BASE = "http://user-service"
FIELDS = ("name", "email")

def payload(uid):
    user = httpx.get(f"{BASE}/users/{uid}").json()
    return {k: user[k] for k in FIELDS}
"""
# Static result: potentially_affected (.get on line 4).
TOLERANT = 'import httpx\ndef f(uid):\n    data = httpx.get(f"http://u/users/{uid}").json()\n    return data.get("user_id")\n'
# Static result: affected (direct read on line 3).
DIRECT = 'import httpx\ndef f(uid):\n    return httpx.get(f"http://u/users/{uid}").json()["user_id"]\n'


def settings(**overrides) -> LlmSettings:
    return LlmSettings(**{"escalation_model": None, "max_concurrency": 1, **overrides})


def static_impact(tmp_path: Path, name: str, code: str, extra_files: dict[str, str] | None = None) -> ConsumerImpact:
    root = tmp_path / name
    root.mkdir()
    (root / "client.py").write_text(textwrap.dedent(code))
    for rel, text in (extra_files or {}).items():
        (root / rel).write_text(text)
    return assess_consumer(ConsumerConfig(name=name, path=root), OPERATIONS, [REMOVED], [REMOVED])


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def run(consumers, llm_settings, client, sleep=None):
    return asyncio.run(investigate(consumers, llm_settings, llm_client=client, sleep=sleep or Sleeps()))


# --- planning / spawning decisions -------------------------------------------------------


def test_static_fixture_classifications(tmp_path: Path) -> None:
    assert static_impact(tmp_path, "a", DYNAMIC).status == Classification.UNCERTAIN
    assert static_impact(tmp_path, "b", TOLERANT).status == Classification.POTENTIALLY_AFFECTED
    assert static_impact(tmp_path, "c", DIRECT).status == Classification.AFFECTED


def test_case_a_nothing_to_investigate_spawns_zero_agents(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(for_all_items("affected"))
    assert run([], settings(), client) == ([], [])
    settled = [static_impact(tmp_path, "direct", DIRECT), static_impact(tmp_path, "other", DIRECT.replace("user_id", "name"))]
    assert plan_investigations(settled, 6000) == []
    consumers, records = run(settled, settings(), client)
    assert records == []
    assert consumers == settled
    assert client.calls == []


def test_case_b_one_unsettled_consumer_spawns_one_agent(tmp_path: Path) -> None:
    consumers = [static_impact(tmp_path, "direct", DIRECT), static_impact(tmp_path, "notifier", DYNAMIC)]
    client = FakeLiteLLMClient(for_all_items("uncertain", confidence=0.9))
    _, records = run(consumers, settings(), client)
    assert [r.agent_name for r in records] == ["investigator_notifier"]
    assert len(client.calls) == 1
    assert client.calls[0].payload["consumer"] == "notifier"


@pytest.mark.parametrize("max_concurrency", [1, 2, 3])
def test_case_c_three_consumers_spawn_three_agents_within_concurrency(tmp_path: Path, max_concurrency: int) -> None:
    consumers = [static_impact(tmp_path, f"svc-{i}", DYNAMIC) for i in range(3)]
    client = FakeLiteLLMClient(for_all_items("uncertain"))
    _, records = run(consumers, settings(max_concurrency=max_concurrency), client)
    assert sorted(r.agent_name for r in records) == ["investigator_svc_0", "investigator_svc_1", "investigator_svc_2"]
    assert {c.payload["consumer"] for c in client.calls} == {"svc-0", "svc-1", "svc-2"}
    assert client.max_in_flight <= max_concurrency
    if max_concurrency == 1:
        assert client.max_in_flight == 1


def test_only_unsettled_findings_are_sent(tmp_path: Path) -> None:
    impact = static_impact(tmp_path, "svc", DYNAMIC)
    (task,) = plan_investigations([impact], 6000)
    assert task.item_indexes == [0]
    item = __import__("json").loads(task.prompt())["items"][0]
    assert item["removed_property"] == "user_id"
    assert item["static_classification"] == "uncertain"


def test_prompt_contains_compact_snippets_not_the_repository(tmp_path: Path) -> None:
    unrelated = "SECRET_MARKER = 1\n" + "\n".join(f"def f{i}(): return {i}" for i in range(200))
    impact = static_impact(tmp_path, "svc", DYNAMIC, extra_files={"unrelated.py": unrelated})
    (task,) = plan_investigations([impact], 6000)
    prompt = task.prompt()
    assert "SECRET_MARKER" not in prompt
    assert {s.file for s in task.snippets} == {"client.py"}
    code = "\n".join(s.code for s in task.snippets)
    assert '   3| FIELDS = ("name", "email")' in code  # referenced module constant
    assert "   7|     return {k: user[k] for k in FIELDS}" in code
    assert len(prompt) < 2000


# --- agent factory -------------------------------------------------------------------------


def test_agent_factory_creates_independent_structured_agents() -> None:
    first = create_investigation_agent(agent_name("order-service"), "groq/openai/gpt-oss-20b", settings())
    second = create_investigation_agent(agent_name("billing service!"), "groq/openai/gpt-oss-120b", settings())
    assert isinstance(first, LlmAgent) and first is not second
    assert first.name == "investigator_order_service"
    assert second.name == "investigator_billing_service"
    assert agent_name("x", escalation=True) == "investigator_x_escalated"
    assert isinstance(first.model, LiteLlm) and first.model.model == "groq/openai/gpt-oss-20b"
    assert first.output_schema is AgentReport
    assert first.tools == []
    assert first.include_contents == "none"
    assert "{" not in first.instruction  # ADK would treat braces as state placeholders


def test_request_parameters_sent_to_litellm(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(for_all_items("uncertain"))
    run([static_impact(tmp_path, "svc", DYNAMIC)], settings(max_output_tokens=777), client)
    (call,) = client.calls
    assert call.model == "groq/openai/gpt-oss-20b"
    assert call.tools is None
    assert call.kwargs["num_retries"] == 0
    assert call.kwargs["max_tokens"] == 777
    assert call.kwargs["reasoning_effort"] == "low"
    response_format = call.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "AgentReport"


# --- merging and evidence validation ---------------------------------------------------------


def test_verified_llm_verdict_replaces_static_uncertainty(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(
        for_all_items(
            "no_relevant_usage_found",
            confidence=0.85,
            evidence=[("client.py", 3, 'FIELDS = ("name", "email")')],
            reason="FIELDS only contains name and email.",
        )
    )
    (impact,), (record,) = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(), client)
    (finding,) = impact.findings
    assert impact.status == Classification.NO_RELEVANT_USAGE_FOUND
    assert finding.classification == Classification.NO_RELEVANT_USAGE_FOUND
    assert finding.analysis_method == "llm"
    assert finding.static_classification == Classification.UNCERTAIN
    assert finding.confidence == 0.85
    assert finding.reason == "FIELDS only contains name and email."
    assert finding.evidence[0].model_dump() == {
        "file": "client.py",
        "line": 3,
        "snippet": 'FIELDS = ("name", "email")',
        "note": "cited by LLM investigation",
    }
    assert record.status == "completed" and record.rejected_evidence == 0


def test_invented_evidence_is_rejected_and_unsupported_claim_downgraded(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(
        for_all_items(
            "affected",
            evidence=[
                ("client.py", 99, 'return user["user_id"]'),  # line does not exist
                ("client.py", 6, 'return user["user_id"]'),  # wrong text for this line
                ("../../etc/passwd.py", 1, "root:x:0:0"),  # outside the repository
                ("ghost.py", 1, "import httpx"),  # file does not exist
            ],
        )
    )
    (impact,), (record,) = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(), client)
    (finding,) = impact.findings
    assert finding.classification == Classification.UNCERTAIN
    assert record.rejected_evidence == 4
    assert all(e.note != "cited by LLM investigation" for e in finding.evidence)
    assert any("did not match the repository" in n for n in finding.notes)
    assert any("without verifiable evidence" in n for n in finding.notes)


def test_llm_cannot_hide_a_read_found_statically(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(for_all_items("no_relevant_usage_found"))
    (impact,), _ = run([static_impact(tmp_path, "svc", TOLERANT)], settings(), client)
    (finding,) = impact.findings
    assert finding.classification == Classification.POTENTIALLY_AFFECTED
    assert any("static analysis found a read" in n for n in finding.notes)


def test_llm_can_confirm_affected_with_verified_evidence(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(
        for_all_items("affected", evidence=[("client.py", 4, 'return data.get("user_id")')], confidence=1.7)
    )
    (impact,), _ = run([static_impact(tmp_path, "svc", TOLERANT)], settings(), client)
    assert impact.findings[0].classification == Classification.AFFECTED
    assert impact.findings[0].confidence == 1.0  # clamped


def test_missing_verdict_keeps_static_result(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(lambda call: {"verdicts": [verdict("item-999", "affected")]})
    static = static_impact(tmp_path, "svc", DYNAMIC)
    (impact,), (record,) = run([static], settings(), client)
    (finding,) = impact.findings
    assert finding.classification == Classification.UNCERTAIN
    assert finding.analysis_method == "static"
    assert any("no verdict" in n for n in finding.notes)
    assert record.status == "completed"


# --- rate limits, retries and budgets ------------------------------------------------------


def test_rate_limit_is_retried_honouring_retry_after(tmp_path: Path) -> None:
    responses = iter([rate_limit_error(retry_after=7), {"verdicts": []}])
    client = FakeLiteLLMClient(lambda call: next(responses))
    sleeps = Sleeps()
    _, (record,) = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(), client, sleeps)
    assert record.status == "completed"
    assert record.attempts == 2
    assert sleeps.delays == [7.0]


def test_backoff_is_exponential_and_capped() -> None:
    s = settings(initial_backoff_seconds=2, max_backoff_seconds=10)
    assert [backoff_delay(a, s) for a in range(4)] == [2, 4, 8, 10]
    assert backoff_delay(0, s, retry_after=5) == 5
    assert backoff_delay(0, s, retry_after=600) == 10


def test_retries_are_bounded_and_failure_keeps_static_result(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(lambda call: rate_limit_error())
    sleeps = Sleeps()
    static = static_impact(tmp_path, "svc", DYNAMIC)
    (impact,), (record,) = run([static], settings(max_retries=2), client, sleeps)
    assert len(client.calls) == 3
    assert sleeps.delays == [2.0, 4.0]
    assert record.status == "failed" and record.attempts == 3
    assert "RateLimitError" in (record.error or "")
    assert impact.findings[0].classification == static.findings[0].classification
    assert any("LLM investigation failed" in n for n in impact.findings[0].notes)


def test_non_retryable_errors_are_not_retried(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(lambda call: auth_error())
    sleeps = Sleeps()
    _, (record,) = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(max_retries=3), client, sleeps)
    assert len(client.calls) == 1
    assert sleeps.delays == []
    assert record.status == "failed" and "AuthenticationError" in (record.error or "")


@pytest.mark.parametrize("bad_output", ["not json", '{"verdicts": [{"item_id": "item-0"}]}', ""])
def test_invalid_output_is_retried_then_reported(tmp_path: Path, bad_output: str) -> None:
    client = FakeLiteLLMClient(lambda call: bad_output)
    _, (record,) = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(max_retries=1), client)
    assert len(client.calls) == 2
    assert record.status == "failed"
    assert "invalid agent output" in (record.error or "")


def test_request_budget_caps_total_llm_calls(tmp_path: Path) -> None:
    consumers = [static_impact(tmp_path, f"svc-{i}", DYNAMIC) for i in range(3)]
    client = FakeLiteLLMClient(for_all_items("uncertain"))
    _, records = run(consumers, settings(max_llm_requests=2), client)
    assert len(client.calls) == 2
    failed = [r for r in records if r.status == "failed"]
    assert len(failed) == 1 and "budget exhausted" in (failed[0].error or "")
    assert failed[0].attempts == 0


def test_token_usage_is_recorded(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(for_all_items("uncertain"), prompt_tokens=321, completion_tokens=45)
    _, (record,) = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(), client)
    assert (record.prompt_tokens, record.completion_tokens) == (321, 45)


# --- escalation -----------------------------------------------------------------------------


def test_low_confidence_verdicts_escalate_to_larger_model(tmp_path: Path) -> None:
    def responder(call):
        if call.model.endswith("120b"):
            return for_all_items("no_relevant_usage_found", confidence=0.9, evidence=[("client.py", 3, 'FIELDS = ("name", "email")')])(call)
        return for_all_items("uncertain", confidence=0.3)(call)

    client = FakeLiteLLMClient(responder)
    (impact,), records = run(
        [static_impact(tmp_path, "svc", DYNAMIC)], settings(escalation_model="groq/openai/gpt-oss-120b"), client
    )
    assert [c.model for c in client.calls] == ["groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"]
    assert [(r.agent_name, r.escalation) for r in records] == [
        ("investigator_svc", False),
        ("investigator_svc_escalated", True),
    ]
    assert impact.findings[0].classification == Classification.NO_RELEVANT_USAGE_FOUND
    assert impact.findings[0].static_classification == Classification.UNCERTAIN


def test_confident_verdicts_do_not_escalate(tmp_path: Path) -> None:
    client = FakeLiteLLMClient(
        for_all_items("no_relevant_usage_found", confidence=0.95, evidence=[("client.py", 3, 'FIELDS = ("name", "email")')])
    )
    _, records = run([static_impact(tmp_path, "svc", DYNAMIC)], settings(escalation_model="groq/openai/gpt-oss-120b"), client)
    assert len(records) == 1 and len(client.calls) == 1


# --- credentials ----------------------------------------------------------------------------


def test_missing_credentials_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LlmSetupError, match="GROQ_API_KEY"):
        check_llm_environment(settings())
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    check_llm_environment(settings(escalation_model="groq/openai/gpt-oss-120b"))
