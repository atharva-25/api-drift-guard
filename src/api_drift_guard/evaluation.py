"""Evaluation of consumer impact classification against a ground-truth dataset.

Static and LLM-assisted results are scored separately. In agentic mode every finding
keeps its deterministic verdict in ``static_classification``, so one agentic run yields
both score sets. A successful run is not evidence of correctness; only agreement with
the dataset's expected classifications is.

Outcomes per (scenario, consumer):

- ``correct``: prediction equals the expected classification.
- ``false_positive``: expected ``no_relevant_usage_found``, predicted (potentially) affected.
- ``false_negative``: expected (potentially) affected, predicted ``no_relevant_usage_found``.
- ``uncertain``: predicted ``uncertain`` where the truth is known (an abstention).
- ``mismatch``: any other disagreement, e.g. affected vs potentially_affected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from api_drift_guard.config import ConfigError, LlmSettings, load_config
from api_drift_guard.evidence import verify_reference
from api_drift_guard.impact import most_severe
from api_drift_guard.models import Classification, CompatibilityReport, ConsumerImpact, Evidence
from api_drift_guard.pipeline import Mode, run_analysis
from api_drift_guard.reporting import CLASSIFICATION_LABELS

Outcome = Literal["correct", "false_positive", "false_negative", "uncertain", "mismatch"]
_AFFECTED = frozenset({Classification.AFFECTED, Classification.POTENTIALLY_AFFECTED})
_LLM_NOTE = "cited by LLM investigation"


# --- dataset -----------------------------------------------------------------------------


class ExpectedConsumer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: Classification
    # "file:line" references a correct detection is expected to cite.
    evidence: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    new_spec: Path
    expected_status: Literal["no_breaking_changes", "breaking_changes_detected"]
    # Investigation agents the coordinator should spawn (before any escalation).
    expected_agents: int = Field(ge=0)
    consumers: dict[str, ExpectedConsumer]


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: Path
    old_spec: Path
    scenarios: list[Scenario]

    @field_validator("scenarios")
    @classmethod
    def _unique_ids(cls, scenarios: list[Scenario]) -> list[Scenario]:
        ids = [s.id for s in scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        return scenarios


def load_dataset(path: Path) -> Dataset:
    """Load a dataset; its paths are resolved relative to the dataset file."""
    if not path.is_file():
        raise ConfigError(f"Evaluation dataset not found: {path}")
    try:
        dataset = Dataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"Invalid evaluation dataset {path}: {exc}") from exc
    base = path.parent.resolve()
    dataset.config = (base / dataset.config).resolve()
    dataset.old_spec = (base / dataset.old_spec).resolve()
    for scenario in dataset.scenarios:
        scenario.new_spec = (base / scenario.new_spec).resolve()
    return dataset


# --- results -----------------------------------------------------------------------------


class CaseResult(BaseModel):
    scenario: str
    consumer: str
    expected: Classification
    predicted: Classification
    outcome: Outcome
    expected_evidence: list[str]
    missing_evidence: list[str]
    evidence_references: int
    invalid_evidence: list[str]


class Metrics(BaseModel):
    cases: int = 0
    correct: int = 0
    accuracy: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0
    uncertain: int = 0
    mismatches: int = 0
    evidence_references: int = 0
    invalid_evidence_references: int = 0
    expected_evidence: int = 0
    expected_evidence_found: int = 0


class ScenarioResult(BaseModel):
    id: str
    description: str
    expected_status: str
    status: str
    expected_agents: int
    # Static mode: agents the coordinator would spawn. Agentic: agents actually spawned.
    agents: int
    escalation_agents: int = 0
    llm_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    static_cases: list[CaseResult]
    agentic_cases: list[CaseResult] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    mode: Literal["static", "agentic"]
    generated_at: datetime
    dataset: str
    static: Metrics
    agentic: Metrics | None = None
    scenarios_with_expected_status: int
    scenarios_with_expected_agents: int
    scenarios: list[ScenarioResult]
    notes: list[str]


def classify_outcome(expected: Classification, predicted: Classification) -> Outcome:
    if predicted == expected:
        return "correct"
    if predicted == Classification.UNCERTAIN:
        return "uncertain"
    if expected == Classification.NO_RELEVANT_USAGE_FOUND and predicted in _AFFECTED:
        return "false_positive"
    if expected in _AFFECTED and predicted == Classification.NO_RELEVANT_USAGE_FOUND:
        return "false_negative"
    return "mismatch"


def _static_status(consumer: ConsumerImpact) -> Classification:
    return most_severe([f.static_classification or f.classification for f in consumer.findings])


def _evaluate_case(
    scenario: str,
    name: str,
    expected: ExpectedConsumer,
    consumer: ConsumerImpact | None,
    *,
    static: bool,
) -> CaseResult:
    if consumer is None:
        # Not analyzed because there were no breaking changes.
        predicted, evidence, root = Classification.NO_RELEVANT_USAGE_FOUND, [], None
    else:
        root = Path(consumer.path)
        predicted = _static_status(consumer) if static else consumer.status
        evidence: list[Evidence] = [
            e for f in consumer.findings for e in f.evidence if not (static and e.note == _LLM_NOTE)
        ]
    cited = {f"{e.file}:{e.line}" for e in evidence}
    invalid = [
        f"{e.file}:{e.line}"
        for e in evidence
        if root is None or verify_reference(root, e.file, e.line, e.snippet) is None
    ]
    return CaseResult(
        scenario=scenario,
        consumer=name,
        expected=expected.expected,
        predicted=predicted,
        outcome=classify_outcome(expected.expected, predicted),
        expected_evidence=expected.evidence,
        missing_evidence=[ref for ref in expected.evidence if ref not in cited],
        evidence_references=len(evidence),
        invalid_evidence=invalid,
    )


def compute_metrics(cases: list[CaseResult]) -> Metrics:
    def count(outcome: Outcome) -> int:
        return sum(c.outcome == outcome for c in cases)

    correct = count("correct")
    return Metrics(
        cases=len(cases),
        correct=correct,
        accuracy=round(correct / len(cases), 3) if cases else 0.0,
        false_positives=count("false_positive"),
        false_negatives=count("false_negative"),
        uncertain=count("uncertain"),
        mismatches=count("mismatch"),
        evidence_references=sum(c.evidence_references for c in cases),
        invalid_evidence_references=sum(len(c.invalid_evidence) for c in cases),
        expected_evidence=sum(len(c.expected_evidence) for c in cases),
        expected_evidence_found=sum(len(c.expected_evidence) - len(c.missing_evidence) for c in cases),
    )


def _evaluate_scenario(
    scenario: Scenario, report: CompatibilityReport, mode: Mode, llm_settings: LlmSettings
) -> ScenarioResult:
    consumers = {c.consumer: c for c in report.consumers}
    static_cases = [
        _evaluate_case(scenario.id, name, exp, consumers.get(name), static=True)
        for name, exp in scenario.consumers.items()
    ]
    agentic_cases = []
    if mode == Mode.AGENTIC:
        agentic_cases = [
            _evaluate_case(scenario.id, name, exp, consumers.get(name), static=False)
            for name, exp in scenario.consumers.items()
        ]
        agents = sum(not r.escalation for r in report.investigations)
        escalations = sum(r.escalation for r in report.investigations)
    else:
        from api_drift_guard.agents import plan_investigations

        agents = len(plan_investigations(report.consumers, llm_settings.max_context_chars))
        escalations = 0
    return ScenarioResult(
        id=scenario.id,
        description=scenario.description,
        expected_status=scenario.expected_status,
        status=report.status,
        expected_agents=scenario.expected_agents,
        agents=agents,
        escalation_agents=escalations,
        llm_requests=report.llm_usage.llm_requests,
        prompt_tokens=report.llm_usage.prompt_tokens,
        completion_tokens=report.llm_usage.completion_tokens,
        static_cases=static_cases,
        agentic_cases=agentic_cases,
    )


def run_evaluation(
    dataset_path: Path, *, mode: Mode, llm_settings: LlmSettings | None = None, llm_client: Any = None
) -> EvaluationReport:
    dataset = load_dataset(dataset_path)
    config = load_config(dataset.config)
    settings = llm_settings or config.llm
    configured = {c.name for c in config.consumers}
    for scenario in dataset.scenarios:
        unknown = sorted(set(scenario.consumers) - configured)
        if unknown:
            raise ConfigError(f"Scenario {scenario.id!r} references unknown consumers: {', '.join(unknown)}")

    results = []
    for scenario in dataset.scenarios:
        report = run_analysis(
            old=dataset.old_spec, new=scenario.new_spec, config=config, mode=mode,
            llm_settings=settings, llm_client=llm_client,
        )
        results.append(_evaluate_scenario(scenario, report, mode, settings))

    notes = [
        "Accuracy is measured against the dataset's expected classifications; a successful "
        "tool run alone says nothing about correctness.",
        "'uncertain' predictions are abstentions: neither false positives nor false negatives.",
    ]
    if mode == Mode.STATIC:
        notes.append(
            "LLM-assisted accuracy was not measured. Run with --mode agentic (requires provider "
            "credentials) to score it; 'agents' are the agents the coordinator would spawn."
        )
    return EvaluationReport(
        mode=mode.value,
        generated_at=datetime.now(timezone.utc),
        dataset=str(dataset_path),
        static=compute_metrics([c for r in results for c in r.static_cases]),
        agentic=compute_metrics([c for r in results for c in r.agentic_cases]) if mode == Mode.AGENTIC else None,
        scenarios_with_expected_status=sum(r.status == r.expected_status for r in results),
        scenarios_with_expected_agents=sum(r.agents == r.expected_agents for r in results),
        scenarios=results,
        notes=notes,
    )


# --- rendering ---------------------------------------------------------------------------


def _metrics_rows(name: str, m: Metrics) -> str:
    return (
        f"| {name} | {m.correct}/{m.cases} ({m.accuracy:.0%}) | {m.false_positives} | {m.false_negatives} "
        f"| {m.uncertain} | {m.mismatches} | {m.expected_evidence_found}/{m.expected_evidence} "
        f"| {m.invalid_evidence_references}/{m.evidence_references} |"
    )


def render_evaluation_markdown(report: EvaluationReport) -> str:
    lines = [
        "# API Drift Guard Evaluation",
        "",
        f"- Mode: {report.mode}",
        f"- Scenarios: {len(report.scenarios)} "
        f"(status as expected: {report.scenarios_with_expected_status}, "
        f"agent count as expected: {report.scenarios_with_expected_agents})",
        "",
        "## Classification",
        "",
        "| Analysis | Correct | False positives | False negatives | Uncertain | Mismatches "
        "| Expected evidence found | Invalid evidence |",
        "|---|---|---|---|---|---|---|---|",
        _metrics_rows("Static", report.static),
        _metrics_rows("Agentic (static + LLM)", report.agentic) if report.agentic else "| Agentic (static + LLM) | not run | | | | | | |",
        "",
        "## Scenarios",
        "",
        "| Scenario | Consumer | Expected | Static | Agentic | Outcome |",
        "|---|---|---|---|---|---|",
    ]
    for scenario in report.scenarios:
        agentic = {c.consumer: c for c in scenario.agentic_cases}
        for case in scenario.static_cases:
            llm_case = agentic.get(case.consumer)
            outcome = (llm_case or case).outcome
            lines.append(
                f"| {scenario.id} | {case.consumer} | {CLASSIFICATION_LABELS[case.expected]} "
                f"| {CLASSIFICATION_LABELS[case.predicted]} "
                f"| {CLASSIFICATION_LABELS[llm_case.predicted] if llm_case else '-'} | {outcome} |"
            )
    lines += [
        "",
        "## Agents and LLM usage",
        "",
        "| Scenario | Expected agents | Agents | Escalations | LLM requests | Tokens (prompt/completion) |",
        "|---|---|---|---|---|---|",
    ]
    for s in report.scenarios:
        lines.append(
            f"| {s.id} | {s.expected_agents} | {s.agents} | {s.escalation_agents} | {s.llm_requests} "
            f"| {s.prompt_tokens}/{s.completion_tokens} |"
        )
    lines += ["", "## Notes", "", *(f"- {n}" for n in report.notes), ""]
    return "\n".join(lines)
