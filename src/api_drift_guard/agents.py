"""LLM investigation of findings that static analysis could not settle.

A plain-Python coordinator around Google ADK (verified against google-adk 2.12):

- ``plan_investigations`` (deterministic) selects consumers that have ``uncertain`` or
  ``potentially_affected`` findings. Static ``affected`` findings already have direct
  evidence and never reach an LLM; when nothing needs investigation, zero agents run.
- ``create_investigation_agent`` instantiates one ``LlmAgent`` per consumer at runtime.
  Agents have no tools: deterministic code pre-extracts compact snippets, and the agent
  returns one structured ``AgentReport`` via ``output_schema``.
- Each agent runs in its own ``Runner`` + ``InMemorySessionService`` session (independent
  context). An ``asyncio.Semaphore`` bounds concurrency and a per-run budget caps requests.
- Rate limits, transient errors and invalid output are retried a bounded number of times
  with exponential backoff that honours ``retry-after``.
- ``merge_verdicts`` checks every cited file:line against the repository and never lets an
  LLM verdict hide a field read that static analysis found.
- Verdicts still uncertain or below a confidence threshold are re-investigated once by an
  agent using the escalation model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import litellm
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, ValidationError

from api_drift_guard.config import LlmSettings, LlmSetupError
from api_drift_guard.evidence import CodeSnippet, build_snippets, verify_reference
from api_drift_guard.impact import SEVERITY_ORDER, dedupe_evidence, most_severe
from api_drift_guard.models import (
    Classification,
    ConsumerImpact,
    Evidence,
    ImpactFinding,
    InvestigationRecord,
)

logger = logging.getLogger(__name__)

# LiteLLM otherwise prints help banners to stdout on errors; we report errors ourselves.
litellm.suppress_debug_info = True

APP_NAME = "api_drift_guard"
USER_ID = "api-drift-guard"
INVESTIGATED = frozenset({Classification.UNCERTAIN, Classification.POTENTIALLY_AFFECTED})

_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
)

Sleep = Callable[[float], Awaitable[None]]

# No curly braces: ADK treats {name} in instructions as session-state placeholders.
INSTRUCTION = """\
You investigate whether one consumer service will break after a provider REST API change.

The user message is a JSON object with:
- consumer: the consumer service name.
- items: findings to decide. Each has item_id, endpoint, change, removed_property,
  static_classification, static_reason and static_evidence. Property paths use "/" for
  nesting and "items" for array elements, e.g. "items/user_id".
- code_snippets: the only code you may rely on. Each has file, start_line, end_line and
  code, where every line is prefixed with its line number like "  12| code".

For each item choose a classification:
- affected: the code reads the removed property from this endpoint's response in a way
  that fails or misbehaves without it (KeyError, TypeError, wrong value).
- potentially_affected: the code reads the property but may tolerate its absence (for
  example .get with a default), or the link to this endpoint is plausible but not certain.
- no_relevant_usage_found: the snippets show how the response is used and the removed
  property is not read, for example dynamic keys that resolve only to other fields.
- uncertain: the snippets are not sufficient to decide.

Rules:
- Use only the provided snippets. Never assume code you cannot see; prefer uncertain.
- Every evidence entry must cite a file and line number shown in code_snippets, and its
  snippet must be the exact code on that line, without the line-number prefix.
- confidence is your own estimate between 0 and 1.
- Return exactly one verdict per item_id.
- suggested_action is one short, concrete sentence, or an empty string.
"""


class AgentOutputError(Exception):
    pass


# --- structured output (plain types: providers' strict JSON-schema modes reject constraints)


class AgentEvidence(BaseModel):
    file: str
    line: int
    snippet: str


class AgentVerdict(BaseModel):
    item_id: str
    classification: Classification
    confidence: float
    reason: str
    evidence: list[AgentEvidence]
    suggested_action: str


class AgentReport(BaseModel):
    verdicts: list[AgentVerdict]


# --- planning -----------------------------------------------------------------------


@dataclass(frozen=True)
class InvestigationTask:
    consumer: str
    root: Path
    findings: list[ImpactFinding]  # the consumer's full findings list
    item_indexes: list[int]  # indexes of the findings this task investigates
    snippets: list[CodeSnippet]
    snippets_truncated: bool

    @staticmethod
    def item_id(index: int) -> str:
        return f"item-{index}"

    def prompt(self) -> str:
        items = []
        for index in self.item_indexes:
            finding = self.findings[index]
            items.append(
                {
                    "item_id": self.item_id(index),
                    "endpoint": finding.endpoint,
                    "change": finding.change,
                    "removed_property": finding.property_path,
                    "static_classification": (finding.static_classification or finding.classification).value,
                    "static_reason": finding.reason,
                    "static_evidence": [f"{e.file}:{e.line} {e.note}" for e in finding.evidence],
                }
            )
        payload = {
            "consumer": self.consumer,
            "items": items,
            "code_snippets": [
                {"file": s.file, "start_line": s.start_line, "end_line": s.end_line, "code": s.code}
                for s in self.snippets
            ],
            "snippets_truncated": self.snippets_truncated,
        }
        return json.dumps(payload, indent=1)


def build_task(
    consumer: str, root: Path, findings: list[ImpactFinding], indexes: list[int], max_chars: int
) -> InvestigationTask:
    locations = list(dict.fromkeys((e.file, e.line) for i in indexes for e in findings[i].evidence))
    snippets, truncated = build_snippets(root, locations, max_chars)
    return InvestigationTask(consumer, root, findings, indexes, snippets, truncated)


def plan_investigations(consumers: list[ConsumerImpact], max_context_chars: int) -> list[InvestigationTask]:
    """Decide deterministically which consumers need an investigation agent (one each)."""
    tasks = []
    for impact in consumers:
        indexes = [
            i for i, f in enumerate(impact.findings) if f.classification in INVESTIGATED and f.evidence
        ]
        if indexes:
            tasks.append(build_task(impact.consumer, Path(impact.path), impact.findings, indexes, max_context_chars))
    return tasks


# --- agent factory and execution ---------------------------------------------------------


def agent_name(consumer: str, escalation: bool = False) -> str:
    slug = re.sub(r"\W+", "_", consumer).strip("_").lower() or "consumer"
    return f"investigator_{slug}" + ("_escalated" if escalation else "")


def create_investigation_agent(
    name: str, model: str, settings: LlmSettings, llm_client: LiteLLMClient | None = None
) -> LlmAgent:
    kwargs: dict[str, object] = {
        "num_retries": 0,  # retries are handled (and bounded) by run_agent
        "timeout": settings.request_timeout_seconds,
        "max_tokens": settings.max_output_tokens,
        "temperature": settings.temperature,
        "drop_params": True,
    }
    if settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    if llm_client is not None:
        kwargs["llm_client"] = llm_client
    return LlmAgent(
        name=name,
        model=LiteLlm(model=model, **kwargs),
        description="Investigates how one consumer service uses a changed API response.",
        instruction=INSTRUCTION,
        output_schema=AgentReport,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


@dataclass
class RequestBudget:
    limit: int
    used: int = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


@dataclass(frozen=True)
class AgentRun:
    report: AgentReport | None
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    error: str | None


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None) or getattr(
        exc, "litellm_response_headers", None
    ) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def backoff_delay(attempt: int, settings: LlmSettings, retry_after: float | None = None) -> float:
    delay = settings.initial_backoff_seconds * 2**attempt
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, settings.max_backoff_seconds)


def _parse_report(text: str) -> AgentReport:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return AgentReport.model_validate_json(text)


async def _run_once(agent: LlmAgent, prompt: str) -> tuple[str | None, int, int]:
    """Run the agent once in a fresh session; returns (final text, prompt tokens, output tokens)."""
    sessions = InMemorySessionService()
    final_text: str | None = None
    error: str | None = None
    prompt_tokens = completion_tokens = 0
    async with Runner(agent=agent, app_name=APP_NAME, session_service=sessions) as runner:
        session = await sessions.create_session(app_name=APP_NAME, user_id=USER_ID)
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=message):
            if event.usage_metadata:
                prompt_tokens += event.usage_metadata.prompt_token_count or 0
                completion_tokens += event.usage_metadata.candidates_token_count or 0
            if event.error_code:
                # Provider exceptions (e.g. 429) are re-raised by ADK after this event.
                error = f"{event.error_code}: {event.error_message}"
            elif event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(p.text for p in event.content.parts if p.text and not p.thought)
    if error and not final_text:
        raise AgentOutputError(f"model returned an error: {error}")
    return final_text, prompt_tokens, completion_tokens


async def run_agent(
    agent: LlmAgent, prompt: str, settings: LlmSettings, budget: RequestBudget, sleep: Sleep = asyncio.sleep
) -> AgentRun:
    """Run an agent with bounded retries; never raises for LLM/provider failures."""
    attempts = prompt_tokens = completion_tokens = 0
    error: str | None = None
    for attempt in range(settings.max_retries + 1):
        if not budget.take():
            error = f"LLM request budget exhausted (max_llm_requests={budget.limit})"
            break
        attempts += 1
        try:
            text, used_prompt, used_completion = await _run_once(agent, prompt)
            prompt_tokens += used_prompt
            completion_tokens += used_completion
            if not text:
                raise AgentOutputError("agent returned no final response")
            return AgentRun(_parse_report(text), attempts, prompt_tokens, completion_tokens, None)
        except _RETRYABLE_ERRORS as exc:
            error = f"{type(exc).__name__}: {exc}"
            delay = backoff_delay(attempt, settings, _retry_after(exc))
        except (AgentOutputError, ValidationError, json.JSONDecodeError) as exc:
            error = f"invalid agent output: {exc}"
            delay = backoff_delay(attempt, settings)
        except (litellm.AuthenticationError, litellm.PermissionDeniedError, litellm.NotFoundError, litellm.BadRequestError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        except Exception as exc:  # noqa: BLE001 - one consumer's failure must not abort the run
            logger.exception("unexpected error during agent run", extra={"agent": agent.name})
            error = f"unexpected {type(exc).__name__}: {exc}"
            break
        if attempt < settings.max_retries:
            logger.warning(
                "agent attempt failed; retrying",
                extra={"agent": agent.name, "attempt": attempts, "delay_seconds": delay, "error": error},
            )
            await sleep(delay)
    return AgentRun(None, attempts, prompt_tokens, completion_tokens, error)


# --- merging -------------------------------------------------------------------------------


def merge_verdicts(
    task: InvestigationTask, report: AgentReport, escalation_confidence: float
) -> tuple[list[ImpactFinding], int, list[int]]:
    """Apply verified LLM verdicts to the task's findings.

    Returns (updated findings, rejected evidence count, indexes worth escalating).
    """
    verdicts: dict[str, AgentVerdict] = {}
    for verdict in report.verdicts:
        verdicts.setdefault(verdict.item_id, verdict)

    findings = list(task.findings)
    rejected_total = 0
    escalate: list[int] = []
    for index in task.item_indexes:
        current = findings[index]
        static_classification = current.static_classification or current.classification
        verdict = verdicts.get(task.item_id(index))
        if verdict is None:
            findings[index] = current.model_copy(
                update={"notes": [*current.notes, "LLM investigation returned no verdict for this finding; previous result kept."]}
            )
            escalate.append(index)
            continue

        verified: list[Evidence] = []
        rejected = 0
        for cited in verdict.evidence:
            checked = verify_reference(task.root, cited.file, cited.line, cited.snippet)
            if checked is None:
                rejected += 1
            else:
                verified.append(Evidence(file=checked[0], line=cited.line, snippet=checked[1], note="cited by LLM investigation"))
        rejected_total += rejected

        notes = list(current.notes)
        if rejected:
            notes.append(f"{rejected} LLM evidence reference(s) did not match the repository and were discarded.")
        classification = verdict.classification
        if classification in (Classification.AFFECTED, Classification.POTENTIALLY_AFFECTED) and not verified:
            notes.append(f"LLM classified this as {classification.value} without verifiable evidence; reported as uncertain.")
            classification = Classification.UNCERTAIN
        floor = Classification.POTENTIALLY_AFFECTED
        if static_classification == floor and SEVERITY_ORDER.index(classification) > SEVERITY_ORDER.index(floor):
            notes.append(
                f"LLM suggested {classification.value}, but static analysis found a read of the property; "
                f"kept {floor.value}."
            )
            classification = floor

        confidence = min(max(verdict.confidence, 0.0), 1.0)
        findings[index] = current.model_copy(
            update={
                "classification": classification,
                "confidence": confidence,
                "reason": verdict.reason.strip() or current.reason,
                "evidence": dedupe_evidence(verified + current.evidence),
                "suggested_action": verdict.suggested_action.strip() or current.suggested_action,
                "analysis_method": "llm",
                "static_classification": static_classification,
                "notes": notes,
            }
        )
        if classification == Classification.UNCERTAIN or confidence < escalation_confidence:
            escalate.append(index)
    return findings, rejected_total, escalate


# --- coordinator ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Runtime:
    settings: LlmSettings
    semaphore: asyncio.Semaphore
    budget: RequestBudget
    llm_client: LiteLLMClient | None
    sleep: Sleep


async def _run_task(
    task: InvestigationTask, model: str, escalation: bool, runtime: _Runtime
) -> tuple[list[ImpactFinding], InvestigationRecord, list[int]]:
    name = agent_name(task.consumer, escalation)
    agent = create_investigation_agent(name, model, runtime.settings, runtime.llm_client)
    started = time.monotonic()
    async with runtime.semaphore:
        logger.info(
            "investigation agent started",
            extra={"agent": name, "model": model, "items": len(task.item_indexes), "snippets": len(task.snippets)},
        )
        run = await run_agent(agent, task.prompt(), runtime.settings, runtime.budget, runtime.sleep)

    rejected = 0
    escalate: list[int] = []
    if run.report is None:
        note = f"LLM investigation failed ({run.error}); previous result kept."
        findings = list(task.findings)
        for index in task.item_indexes:
            findings[index] = findings[index].model_copy(update={"notes": [*findings[index].notes, note]})
        logger.error("investigation agent failed", extra={"agent": name, "error": run.error})
    else:
        findings, rejected, escalate = merge_verdicts(task, run.report, runtime.settings.escalation_confidence)

    record = InvestigationRecord(
        consumer=task.consumer,
        agent_name=name,
        model=model,
        escalation=escalation,
        items=len(task.item_indexes),
        status="completed" if run.report is not None else "failed",
        attempts=run.attempts,
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        rejected_evidence=rejected,
        error=run.error,
        duration_seconds=round(time.monotonic() - started, 3),
    )
    return findings, record, escalate


async def _investigate_consumer(
    task: InvestigationTask, runtime: _Runtime
) -> tuple[list[ImpactFinding], list[InvestigationRecord]]:
    findings, record, escalate = await _run_task(task, runtime.settings.model, False, runtime)
    records = [record]
    escalation_model = runtime.settings.escalation_model
    if escalation_model and escalate and record.status == "completed":
        escalated = build_task(task.consumer, task.root, findings, escalate, runtime.settings.max_context_chars)
        findings, record, _ = await _run_task(escalated, escalation_model, True, runtime)
        records.append(record)
    return findings, records


async def investigate(
    consumers: list[ConsumerImpact],
    settings: LlmSettings,
    *,
    llm_client: LiteLLMClient | None = None,
    sleep: Sleep = asyncio.sleep,
) -> tuple[list[ConsumerImpact], list[InvestigationRecord]]:
    """Investigate unsettled findings with dynamically created agents."""
    tasks = plan_investigations(consumers, settings.max_context_chars)
    if not tasks:
        logger.info("no findings need LLM investigation; no agents spawned")
        return consumers, []

    logger.info("spawning investigation agents", extra={"agents": len(tasks), "max_concurrency": settings.max_concurrency})
    runtime = _Runtime(settings, asyncio.Semaphore(settings.max_concurrency), RequestBudget(settings.max_llm_requests), llm_client, sleep)
    results = await asyncio.gather(*(_investigate_consumer(task, runtime) for task in tasks))

    updated = {task.consumer: findings for task, (findings, _) in zip(tasks, results)}
    records = [record for _, task_records in results for record in task_records]
    merged = []
    for impact in consumers:
        findings = updated.get(impact.consumer)
        if findings is None:
            merged.append(impact)
        else:
            status = most_severe([f.classification for f in findings])
            merged.append(impact.model_copy(update={"findings": findings, "status": status}))
    return merged, records


def run_investigations(
    consumers: list[ConsumerImpact], settings: LlmSettings, *, llm_client: LiteLLMClient | None = None
) -> tuple[list[ConsumerImpact], list[InvestigationRecord]]:
    return asyncio.run(investigate(consumers, settings, llm_client=llm_client))


def check_llm_environment(settings: LlmSettings) -> None:
    """Fail fast, before any analysis, if provider credentials are missing."""
    for model in filter(None, [settings.model, settings.escalation_model]):
        try:
            status = litellm.validate_environment(model=model)
        except Exception as exc:  # noqa: BLE001 - surface any provider lookup problem as setup error
            raise LlmSetupError(f"Cannot use model {model!r}: {exc}") from exc
        if not status.get("keys_in_environment"):
            missing = ", ".join(status.get("missing_keys") or ["provider credentials"])
            raise LlmSetupError(
                f"Missing credentials for model {model!r}: set {missing} in the environment or a .env "
                "file, or run with --mode static."
            )
