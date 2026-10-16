"""Compatibility report assembly and rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from api_drift_guard import __version__
from api_drift_guard.models import (
    ChangeSummary,
    ChangeType,
    Classification,
    CompatibilityReport,
    ConsumerImpact,
    ConsumerSummary,
    ContractChange,
    ImpactFinding,
    InvestigationRecord,
    LlmUsage,
)

CLASSIFICATION_LABELS = {
    Classification.AFFECTED: "Affected",
    Classification.POTENTIALLY_AFFECTED: "Potentially affected",
    Classification.NO_RELEVANT_USAGE_FOUND: "No relevant usage found",
    Classification.UNCERTAIN: "Uncertain",
}

LIMITATIONS = [
    "Contract changes are detected by oasdiff. Every removed response property is treated "
    "as breaking for consumers, even when oasdiff rates it INFO (optional property), because "
    "direct field access fails when the field is absent.",
    "Renamed properties are reported as a removal plus an addition.",
    "Changes other than response property additions and removals are listed as "
    "'unsupported'; consumers calling those endpoints are reported as uncertain.",
    "Consumer analysis is static and limited to httpx calls and JSON field reads within a "
    "single function. Response data returned from or passed to other functions is reported "
    "as uncertain, not followed.",
    "Requests are matched to endpoints by HTTP method and path only; hosts are not compared.",
    "'No relevant usage found' means static inspection found no read of the removed field. "
    "It does not guarantee compatibility.",
    "In agentic mode, LLM agents only see compact snippets and re-examine uncertain or "
    "potentially affected findings. Their confidence values are model estimates, not "
    "calibrated probabilities. Every file:line they cite is verified against the "
    "repository, and unverifiable references are discarded.",
]


def summarize(changes: list[ContractChange]) -> ChangeSummary:
    def count(change_type: ChangeType) -> int:
        return sum(c.change_type == change_type for c in changes)

    return ChangeSummary(
        total_changes=len(changes),
        breaking_changes=sum(c.breaking for c in changes),
        removed_response_properties=count(ChangeType.REMOVED_RESPONSE_PROPERTY),
        added_response_properties=count(ChangeType.ADDED_RESPONSE_PROPERTY),
        unsupported_changes=count(ChangeType.UNSUPPORTED),
    )


def summarize_consumers(consumers: list[ConsumerImpact]) -> ConsumerSummary:
    summary = ConsumerSummary()
    for consumer in consumers:
        setattr(summary, consumer.status.value, getattr(summary, consumer.status.value) + 1)
    return summary


def build_report(
    *,
    provider: str,
    old_spec: Path,
    new_spec: Path,
    oasdiff_version: str,
    changes: list[ContractChange],
    consumers: list[ConsumerImpact] | None = None,
    analysis_mode: Literal["static", "agentic"] = "static",
    investigations: list[InvestigationRecord] | None = None,
    generated_at: datetime | None = None,
) -> CompatibilityReport:
    summary = summarize(changes)
    consumers = consumers or []
    investigations = investigations or []
    usage = LlmUsage(
        agents_spawned=len(investigations),
        llm_requests=sum(r.attempts for r in investigations),
        prompt_tokens=sum(r.prompt_tokens for r in investigations),
        completion_tokens=sum(r.completion_tokens for r in investigations),
    )
    return CompatibilityReport(
        tool_version=__version__,
        generated_at=generated_at or datetime.now(timezone.utc),
        provider=provider,
        old_spec=str(old_spec),
        new_spec=str(new_spec),
        oasdiff_version=oasdiff_version,
        analysis_mode=analysis_mode,
        status="breaking_changes_detected" if summary.breaking_changes else "no_breaking_changes",
        changes=changes,
        summary=summary,
        consumers=consumers,
        consumer_summary=summarize_consumers(consumers),
        investigations=investigations,
        llm_usage=usage,
        limitations=LIMITATIONS,
    )


def render_json(report: CompatibilityReport) -> str:
    return report.model_dump_json(indent=2)


def write_report(report: CompatibilityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(report) + "\n", encoding="utf-8")


# --- Markdown -------------------------------------------------------------------------------


def code(text: str) -> str:
    """Inline Markdown code that survives backticks in ``text``."""
    if "`" not in text:
        return f"`{text}`"
    return f"`` {text} ``"


def _change_line(change: ContractChange) -> str:
    if change.change_type == ChangeType.REMOVED_RESPONSE_PROPERTY:
        line = f"- Removed response field: {code(change.property_path or '')}"
    elif change.change_type == ChangeType.ADDED_RESPONSE_PROPERTY:
        line = f"- Added response field: {code(change.property_path or '')}"
    else:
        line = f"- Not analyzed ({code(change.oasdiff_id)}): {change.description}"
    return line + (" **(breaking)**" if change.breaking else "")


def _changes_section(changes: list[ContractChange]) -> list[str]:
    lines = ["## Detected Changes", ""]
    if not changes:
        return lines + ["No contract changes detected.", ""]
    groups: dict[str, list[ContractChange]] = {}
    for change in changes:
        groups.setdefault(change.operation or "Specification-level", []).append(change)
    for operation, group in groups.items():
        ordered = sorted(group, key=lambda c: not c.breaking)  # breaking first, otherwise stable
        lines += [f"### {operation}", "", *(_change_line(c) for c in ordered), ""]
    return lines


def _subject(finding: ImpactFinding) -> str:
    return f"{finding.endpoint}: removed {code(finding.property_path)}" if finding.property_path else finding.endpoint


def _analysis_line(finding: ImpactFinding) -> str:
    if finding.analysis_method == "static":
        return "- Analysis: static"
    parts = ["LLM investigation"]
    if finding.confidence is not None:
        parts.append(f"model confidence {finding.confidence:.2f} (an estimate, not a calibrated probability)")
    if finding.static_classification is not None:
        parts.append(f"static result: {CLASSIFICATION_LABELS[finding.static_classification]}")
    return "- Analysis: " + "; ".join(parts)


def _finding_detail(finding: ImpactFinding) -> list[str]:
    lines = [
        f"#### {_subject(finding)}",
        "",
        f"- Status: {CLASSIFICATION_LABELS[finding.classification]}",
        f"- Reason: {finding.reason}",
        "- Evidence:",
    ]
    if finding.evidence:
        lines += [
            f"  - {code(f'{e.file}:{e.line}')}: {code(e.snippet)}" + (f" ({e.note})" if e.note else "")
            for e in finding.evidence
        ]
    else:
        lines.append("  - No direct usage of the removed property identified.")
    if finding.suggested_action:
        lines.append(f"- Suggested action: {finding.suggested_action}")
    lines.append(_analysis_line(finding))
    lines += [f"- Note: {note}" for note in finding.notes]
    return lines + [""]


def _consumer_section(consumer: ConsumerImpact) -> list[str]:
    lines = [f"### {consumer.consumer}", "", f"Status: **{CLASSIFICATION_LABELS[consumer.status]}**", ""]
    unused = [f for f in consumer.findings if f.classification == Classification.NO_RELEVANT_USAGE_FOUND]
    for finding in consumer.findings:
        if finding.classification != Classification.NO_RELEVANT_USAGE_FOUND:
            lines += _finding_detail(finding)
    if unused:
        lines += ["No relevant usage found:", ""]
        for finding in unused:
            checked = ", ".join(code(f"{e.file}:{e.line}") for e in finding.evidence)
            suffix = f" Requests checked: {checked}." if checked else ""
            lines.append(f"- {_subject(finding)}. {finding.reason}{suffix}")
        lines.append("")
    if consumer.parse_errors:
        lines += ["Files that could not be parsed:", "", *(f"- {code(e)}" for e in consumer.parse_errors), ""]
    return lines


def _investigation_section(report: CompatibilityReport) -> list[str]:
    usage = report.llm_usage
    lines = [
        "## LLM Investigation",
        "",
        f"Agents spawned: {usage.agents_spawned} · LLM requests: {usage.llm_requests} · "
        f"Tokens: {usage.prompt_tokens} prompt / {usage.completion_tokens} completion",
        "",
    ]
    if not report.investigations:
        return lines + ["No findings required LLM investigation.", ""]
    lines += [
        "| Agent | Model | Items | Status | Attempts | Rejected evidence |",
        "|---|---|---|---|---|---|",
    ]
    for r in report.investigations:
        status = r.status if r.error is None else f"{r.status}: {r.error}"
        lines.append(
            f"| {r.agent_name} | {r.model} | {r.items} | {status} | {r.attempts} | {r.rejected_evidence} |"
        )
    return lines + [""]


def render_markdown(report: CompatibilityReport) -> str:
    breaking = report.summary.breaking_changes
    result = f"{breaking} breaking change(s) detected" if breaking else "No breaking changes detected"
    lines = [
        "# API Compatibility Report",
        "",
        f"Provider: {report.provider}",
        "",
        f"- Compared: {code(report.old_spec)} → {code(report.new_spec)}",
        f"- Analysis mode: {report.analysis_mode}",
        f"- Result: {result}",
        "",
        *_changes_section(report.changes),
        "## Consumer Impact",
        "",
    ]
    if not report.consumers:
        reason = "No breaking changes were detected" if not breaking else "No consumers are configured"
        lines += [f"{reason}, so consumer code was not analyzed and no investigation agents were needed.", ""]
    for consumer in report.consumers:
        lines += _consumer_section(consumer)

    summary = report.consumer_summary
    lines += [
        "## Summary",
        "",
        f"- Affected consumers: {summary.affected}",
        f"- Potentially affected: {summary.potentially_affected}",
        f"- No relevant usage found: {summary.no_relevant_usage_found}",
        f"- Uncertain: {summary.uncertain}",
        "",
    ]
    if report.analysis_mode == "agentic":
        lines += _investigation_section(report)
    lines += ["## Limitations", "", *(f"- {item}" for item in report.limitations), ""]
    return "\n".join(lines)


def write_markdown(report: CompatibilityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
