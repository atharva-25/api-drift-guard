"""Command-line interface."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import find_dotenv, load_dotenv

from api_drift_guard.config import ConfigError, LlmSettings, LlmSetupError, apply_env_overrides, load_config
from api_drift_guard.contracts import ContractError
from api_drift_guard.pipeline import Mode, run_analysis
from api_drift_guard.reporting import render_json, write_markdown, write_report
from api_drift_guard.source_analysis import SourceAnalysisError

logger = logging.getLogger("api_drift_guard")

app = typer.Typer(add_completion=False, help="Detect API contract drift and its impact on consumers.")

_EXPECTED_ERRORS = (ConfigError, ContractError, SourceAnalysisError, LlmSetupError)

ModeOption = Annotated[
    Mode,
    typer.Option(
        "--mode",
        help="static: deterministic analysis only (no credentials needed). "
        "agentic: also investigate uncertain findings with LLM agents.",
    ),
]
LogLevelOption = Annotated[str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")]


def make_llm_client() -> Any:
    """LiteLLM client used by agents; None means ADK's default. Patched in tests."""
    return None


class KeyValueFormatter(logging.Formatter):
    """Formats records as `ts level logger msg key=value...`, including `extra` fields."""

    _STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            f"level={record.levelname}",
            f"logger={record.name}",
            f"msg={record.getMessage()!r}",
        ]
        parts += [f"{k}={v!r}" for k, v in vars(record).items() if k not in self._STANDARD_ATTRS]
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KeyValueFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def _llm_settings(base: LlmSettings, mode: Mode) -> LlmSettings:
    if mode != Mode.AGENTIC:
        return base
    load_dotenv(find_dotenv(usecwd=True))
    return apply_env_overrides(base)


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"Error: {exc}", err=True)
    return typer.Exit(code=1)


@app.callback()
def main() -> None:
    """API Drift Guard."""


@app.command()
def analyze(
    old: Annotated[Path, typer.Option("--old", help="Previous OpenAPI 3.x specification.")],
    new: Annotated[Path, typer.Option("--new", help="Updated OpenAPI 3.x specification.")],
    config: Annotated[Path, typer.Option("--config", help="Consumer configuration YAML.")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write the JSON report here.")] = None,
    markdown: Annotated[Path | None, typer.Option("--markdown", help="Write the Markdown report here.")] = None,
    mode: ModeOption = Mode.STATIC,
    log_level: LogLevelOption = "INFO",
) -> None:
    """Compare two OpenAPI specs and report the impact on consumers.

    Prints the JSON report to stdout unless --output or --markdown is given.
    """
    configure_logging(log_level)
    try:
        app_config = load_config(config)
        report = run_analysis(
            old=old,
            new=new,
            config=app_config,
            mode=mode,
            llm_settings=_llm_settings(app_config.llm, mode),
            llm_client=make_llm_client(),
        )
    except _EXPECTED_ERRORS as exc:
        raise _fail(exc) from exc

    if output is None and markdown is None:
        typer.echo(render_json(report))
    if output is not None:
        write_report(report, output)
        logger.info("JSON report written", extra={"path": str(output), "status": report.status})
    if markdown is not None:
        write_markdown(report, markdown)
        logger.info("Markdown report written", extra={"path": str(markdown)})


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option("--dataset", help="Evaluation dataset YAML.")] = Path(
        "evaluation/scenarios.yaml"
    ),
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write the JSON results here.")] = None,
    markdown: Annotated[Path | None, typer.Option("--markdown", help="Write the Markdown summary here.")] = None,
    mode: ModeOption = Mode.STATIC,
    log_level: LogLevelOption = "WARNING",
) -> None:
    """Score impact classification against a ground-truth scenario dataset.

    Prints the Markdown summary to stdout unless --markdown is given.
    """
    from api_drift_guard.evaluation import load_dataset, render_evaluation_markdown, run_evaluation

    configure_logging(log_level)
    try:
        base_settings = load_config(load_dataset(dataset).config).llm
        report = run_evaluation(
            dataset, mode=mode, llm_settings=_llm_settings(base_settings, mode), llm_client=make_llm_client()
        )
    except _EXPECTED_ERRORS as exc:
        raise _fail(exc) from exc

    summary = render_evaluation_markdown(report)
    if markdown is None:
        typer.echo(summary)
    else:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(summary, encoding="utf-8")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
