"""Deterministic consumer impact classification from static analysis results.

For each (consumer, breaking change) pair, in order of precedence:

1. ``affected``: a direct ``data["field"]`` read of the removed property on a response
   that unambiguously matches the changed endpoint.
2. ``potentially_affected``: the property is read in a tolerant way (``.get``, ``in``) or
   on a request whose endpoint is ambiguous or unresolved.
3. ``uncertain``: the response JSON is used in ways static analysis cannot follow (dynamic
   keys, passed to/returned from other code), files failed to parse, or the change type
   is not analyzed by this tool.
4. ``no_relevant_usage_found``: otherwise. This is never reported as "safe".
"""

from __future__ import annotations

import logging
from functools import partial

from api_drift_guard.config import ConsumerConfig
from api_drift_guard.discovery import CallMatch, Operation, is_resolved, match_calls
from api_drift_guard.models import (
    ChangeType,
    Classification,
    ConsumerImpact,
    ContractChange,
    Evidence,
    ImpactFinding,
)
from api_drift_guard.source_analysis import CodeLocation, RepoAnalysis, analyze_repository

logger = logging.getLogger(__name__)

SEVERITY_ORDER = (
    Classification.AFFECTED,
    Classification.POTENTIALLY_AFFECTED,
    Classification.UNCERTAIN,
    Classification.NO_RELEVANT_USAGE_FOUND,
)
_MAX_EVIDENCE_PER_KIND = 5


def most_severe(classifications: list[Classification]) -> Classification:
    return min(classifications, key=SEVERITY_ORDER.index, default=Classification.NO_RELEVANT_USAGE_FOUND)


def _evidence(location: CodeLocation, note: str) -> Evidence:
    return Evidence(file=location.file, line=location.line, snippet=location.snippet, note=note)


def dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Collapse evidence on the same line, keeping every distinct note."""
    unique: dict[tuple[str, int], Evidence] = {}
    for item in evidence:
        key = (item.file, item.line)
        if key not in unique:
            unique[key] = item
        elif item.note and item.note not in unique[key].note:
            unique[key] = unique[key].model_copy(update={"note": f"{unique[key].note}; {item.note}"})
    return list(unique.values())


def _call_note(match: CallMatch, operation: str) -> str:
    if match.unresolved:
        return f"request URL `{match.call.url_source}` could not be resolved; it may target {operation}"
    if match.ambiguous:
        return f"request matches several endpoints: {', '.join(match.operations)}"
    return f"request to {operation}"


def _parent(path: str) -> str:
    return path.rpartition("/")[0]


def _replacement_hint(change: ContractChange, all_changes: list[ContractChange]) -> str:
    assert change.property_path is not None
    added = [
        c.property_path
        for c in all_changes
        if c.change_type == ChangeType.ADDED_RESPONSE_PROPERTY
        and c.operation == change.operation
        and c.property_path is not None
        and _parent(c.property_path) == _parent(change.property_path)
    ]
    if not added:
        return ""
    return " Fields added to the same response (possible replacements): " + ", ".join(f"`{p}`" for p in added) + "."


def _assess_change(
    consumer: str,
    repo: RepoAnalysis,
    matches: list[CallMatch],
    change: ContractChange,
    all_changes: list[ContractChange],
) -> ImpactFinding:
    operation = change.operation
    assert operation is not None
    finding = partial(
        ImpactFinding,
        consumer=consumer,
        endpoint=operation,
        property_path=change.property_path,
        change=change.description,
    )
    relevant = [m for m in matches if operation in m.operations]
    by_call = {m.call.call_id: m for m in relevant}

    def call_evidence(call_ids: set[int] | None = None) -> list[Evidence]:
        """Evidence for the matched requests, optionally only those behind a finding."""
        chosen = [m for m in relevant if call_ids is None or m.call.call_id in call_ids]
        return [_evidence(m.call.location, _call_note(m, operation)) for m in chosen[:_MAX_EVIDENCE_PER_KIND]]

    if not relevant:
        if repo.parse_errors:
            return finding(
                classification=Classification.UNCERTAIN,
                reason=f"No request matching {operation} was found, but {len(repo.parse_errors)} "
                "file(s) could not be parsed and may contain one.",
                suggested_action="Fix or manually review the files that failed to parse.",
            )
        return finding(
            classification=Classification.NO_RELEVANT_USAGE_FOUND,
            reason=f"No httpx request matching {operation} was found.",
        )

    if change.change_type != ChangeType.REMOVED_RESPONSE_PROPERTY or change.property_path is None:
        return finding(
            classification=Classification.UNCERTAIN,
            reason=f"Calls {operation}, which has a change this tool does not analyze "
            f"({change.oasdiff_id}): {change.description}",
            evidence=call_evidence(),
            suggested_action="Review the consumer's use of this endpoint manually.",
        )

    prop = change.property_path
    prop_path = tuple(prop.split("/"))
    accesses = [a for a in repo.accesses if a.call_id in by_call and a.path == prop_path]
    direct = [a for a in accesses if a.kind == "subscript" and not by_call[a.call_id].ambiguous]

    if direct:
        evidence = [_evidence(a.location, f'direct read of removed field "{prop}"') for a in direct]
        return finding(
            classification=Classification.AFFECTED,
            reason=f"Reads `{prop}` from the {operation} response by direct subscript, which raises "
            "KeyError once the field is removed.",
            evidence=dedupe_evidence(evidence[:_MAX_EVIDENCE_PER_KIND] + call_evidence({a.call_id for a in direct})),
            suggested_action=f"Update {consumer} to stop reading `{prop}`, or keep the field in the provider "
            "response until consumers migrate." + _replacement_hint(change, all_changes),
        )

    if accesses:
        kinds = {"subscript": "subscript on an ambiguously matched request", "get": ".get()", "membership": "`in` check"}
        evidence = [_evidence(a.location, f'reads "{prop}" via {kinds[a.kind]}') for a in accesses]
        return finding(
            classification=Classification.POTENTIALLY_AFFECTED,
            reason=f"Reads `{prop}` from a response that may come from {operation}, either tolerantly "
            "(.get()/`in`, which yields None/False instead of failing) or from a request whose endpoint "
            "is not certain.",
            evidence=dedupe_evidence(evidence[:_MAX_EVIDENCE_PER_KIND] + call_evidence({a.call_id for a in accesses})),
            suggested_action=f"Check that {consumer} behaves correctly when `{prop}` is absent."
            + _replacement_hint(change, all_changes),
        )

    opaque = [
        o for o in repo.opaque_uses
        if o.call_id in by_call and len(o.path) < len(prop_path) and prop_path[: len(o.path)] == o.path
    ]
    if opaque:
        evidence = [_evidence(o.location, o.detail) for o in opaque[:_MAX_EVIDENCE_PER_KIND]]
        reads = repo.untracked_key_reads.get(change.property_name or "", [])
        evidence += [
            _evidence(loc, f'reads key "{change.property_name}" on a value not linked to a tracked response')
            for loc in reads[:_MAX_EVIDENCE_PER_KIND]
        ]
        return finding(
            classification=Classification.UNCERTAIN,
            reason=f"The {operation} response is used in ways static analysis cannot follow (dynamic keys "
            f"or passed to other code), so a read of `{prop}` cannot be ruled out.",
            evidence=dedupe_evidence(evidence + call_evidence({o.call_id for o in opaque})),
            suggested_action="Review the referenced code to determine whether it relies on "
            f"`{prop}`." + _replacement_hint(change, all_changes),
        )

    if repo.parse_errors:
        return finding(
            classification=Classification.UNCERTAIN,
            reason=f"Calls {operation} without reading `{prop}` in the analyzed code, but "
            f"{len(repo.parse_errors)} file(s) could not be parsed.",
            evidence=call_evidence(),
            suggested_action="Fix or manually review the files that failed to parse.",
        )

    return finding(
        classification=Classification.NO_RELEVANT_USAGE_FOUND,
        reason=f"Calls {operation}, but no read of `{prop}` from its response was found.",
        evidence=call_evidence(),
    )


def assess_consumer(
    consumer: ConsumerConfig,
    operations: list[Operation],
    breaking_changes: list[ContractChange],
    all_changes: list[ContractChange],
) -> ConsumerImpact:
    repo = analyze_repository(consumer.path)
    matches = match_calls(repo.calls, operations)
    findings = [_assess_change(consumer.name, repo, matches, change, all_changes) for change in breaking_changes]
    impact = ConsumerImpact(
        consumer=consumer.name,
        path=str(consumer.path),
        status=most_severe([f.classification for f in findings]),
        files_scanned=repo.files_scanned,
        http_calls=len(repo.calls),
        unresolved_http_calls=sum(not is_resolved(c) for c in repo.calls),
        parse_errors=repo.parse_errors,
        findings=findings,
    )
    logger.info("consumer assessed", extra={"consumer": consumer.name, "status": impact.status.value})
    return impact


def assess_consumers(
    consumers: list[ConsumerConfig],
    operations: list[Operation],
    changes: list[ContractChange],
) -> list[ConsumerImpact]:
    """Statically assess every consumer; returns [] when there are no breaking changes."""
    breaking = [c for c in changes if c.breaking and c.operation is not None]
    if not breaking:
        logger.info("no breaking endpoint changes; consumer analysis skipped")
        return []
    return [assess_consumer(consumer, operations, breaking, changes) for consumer in consumers]
