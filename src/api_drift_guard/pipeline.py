"""End-to-end analysis: specs -> contract changes -> consumer impact -> report.

Shared by the ``analyze`` and ``evaluate`` commands. Raises ConfigError, ContractError,
SourceAnalysisError or LlmSetupError for problems the caller should report.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from api_drift_guard.config import AppConfig, LlmSettings
from api_drift_guard.contracts import detect_changes, find_oasdiff, get_oasdiff_version, list_operations, load_spec
from api_drift_guard.discovery import Operation
from api_drift_guard.impact import assess_consumers
from api_drift_guard.models import CompatibilityReport, InvestigationRecord
from api_drift_guard.reporting import build_report


class Mode(StrEnum):
    STATIC = "static"
    AGENTIC = "agentic"


def run_analysis(
    *,
    old: Path,
    new: Path,
    config: AppConfig,
    mode: Mode,
    llm_settings: LlmSettings,
    llm_client: Any = None,
) -> CompatibilityReport:
    if mode == Mode.AGENTIC:
        # Imported lazily so static mode does not pay the ADK/LiteLLM import cost.
        from api_drift_guard.agents import check_llm_environment

        check_llm_environment(llm_settings)

    old_document = load_spec(old)
    load_spec(new)
    binary = find_oasdiff()
    oasdiff_version = get_oasdiff_version(binary)
    changes = detect_changes(old, new, binary)
    # Consumers were written against the previous contract, so match calls to its operations.
    operations = [Operation(method, path) for method, path in list_operations(old_document)]
    consumers = assess_consumers(config.consumers, operations, changes)

    investigations: list[InvestigationRecord] = []
    if mode == Mode.AGENTIC:
        from api_drift_guard.agents import run_investigations

        consumers, investigations = run_investigations(consumers, llm_settings, llm_client=llm_client)

    return build_report(
        provider=config.provider.name,
        old_spec=old,
        new_spec=new,
        oasdiff_version=oasdiff_version,
        changes=changes,
        consumers=consumers,
        analysis_mode=mode.value,
        investigations=investigations,
    )
