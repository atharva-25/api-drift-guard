"""Pydantic data models shared across the analysis pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ChangeType(StrEnum):
    REMOVED_RESPONSE_PROPERTY = "removed_response_property"
    ADDED_RESPONSE_PROPERTY = "added_response_property"
    # Detected by oasdiff but outside the MVP's consumer-impact analysis.
    UNSUPPORTED = "unsupported"


class ContractChange(BaseModel):
    """A single normalised API contract change derived from oasdiff output."""

    # None for spec-level changes (e.g. oasdiff's `info` section checks).
    endpoint: str | None = None
    method: str | None = None
    response_status: str | None = None
    change_type: ChangeType
    # Leaf property name, e.g. "zip" for "address/zip".
    property_name: str | None = None
    # Full property path as reported by oasdiff, e.g. "items/user_id".
    property_path: str | None = None
    # Whether the change is likely to break consumers (our rule, see contracts.py).
    breaking: bool
    oasdiff_id: str
    # oasdiff severity: 1=INFO, 2=WARN, 3=ERR.
    oasdiff_level: int
    description: str

    @property
    def operation(self) -> str | None:
        return f"{self.method} {self.endpoint}" if self.endpoint and self.method else None


class ChangeSummary(BaseModel):
    total_changes: int
    breaking_changes: int
    removed_response_properties: int
    added_response_properties: int
    unsupported_changes: int


class Classification(StrEnum):
    AFFECTED = "affected"
    POTENTIALLY_AFFECTED = "potentially_affected"
    NO_RELEVANT_USAGE_FOUND = "no_relevant_usage_found"
    UNCERTAIN = "uncertain"


class Evidence(BaseModel):
    file: str  # relative to the consumer root
    line: int = Field(ge=1)
    snippet: str
    note: str = ""


class ImpactFinding(BaseModel):
    """Impact of one breaking change on one consumer."""

    consumer: str
    endpoint: str  # e.g. "GET /users/{user_id}"
    property_path: str | None = None
    change: str
    classification: Classification
    # Model estimate set by LLM investigation; None for static findings.
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_action: str | None = None
    analysis_method: Literal["static", "llm"] = "static"
    # The deterministic verdict, kept when an LLM investigation replaced it.
    static_classification: Classification | None = None
    # Validation/merge notes, e.g. rejected LLM evidence.
    notes: list[str] = Field(default_factory=list)


class ConsumerImpact(BaseModel):
    consumer: str
    path: str
    # Most severe classification across findings.
    status: Classification
    files_scanned: int
    http_calls: int
    unresolved_http_calls: int
    parse_errors: list[str] = Field(default_factory=list)
    findings: list[ImpactFinding]


class ConsumerSummary(BaseModel):
    affected: int = 0
    potentially_affected: int = 0
    no_relevant_usage_found: int = 0
    uncertain: int = 0


class InvestigationRecord(BaseModel):
    """One dynamically created investigation agent and its execution."""

    consumer: str
    agent_name: str
    model: str
    escalation: bool = False
    items: int
    status: Literal["completed", "failed"]
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rejected_evidence: int = 0
    error: str | None = None
    duration_seconds: float


class LlmUsage(BaseModel):
    agents_spawned: int = 0
    llm_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CompatibilityReport(BaseModel):
    tool_version: str
    generated_at: datetime
    provider: str
    old_spec: str
    new_spec: str
    oasdiff_version: str
    analysis_mode: Literal["static", "agentic"] = "static"
    status: Literal["no_breaking_changes", "breaking_changes_detected"]
    changes: list[ContractChange]
    summary: ChangeSummary
    # Empty when there are no breaking changes: consumers are not analyzed then.
    consumers: list[ConsumerImpact] = Field(default_factory=list)
    consumer_summary: ConsumerSummary = Field(default_factory=ConsumerSummary)
    investigations: list[InvestigationRecord] = Field(default_factory=list)
    llm_usage: LlmUsage = Field(default_factory=LlmUsage)
    limitations: list[str]
