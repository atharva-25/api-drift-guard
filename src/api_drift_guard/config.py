"""Loading and validation of the YAML analysis configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


class LlmSetupError(Exception):
    """Raised when the LLM provider cannot be used, e.g. missing credentials."""


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class ConsumerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    path: Path


class LlmSettings(BaseModel):
    """LLM investigation settings. Defaults are sized for Groq's free tier (8K tokens/min)."""

    model_config = ConfigDict(extra="forbid")

    # LiteLLM model strings; the provider's API key is read from the environment by LiteLLM.
    model: str = "groq/openai/gpt-oss-20b"
    # Used to re-investigate uncertain or low-confidence verdicts; None disables escalation.
    escalation_model: str | None = "groq/openai/gpt-oss-120b"
    escalation_confidence: float = Field(default=0.6, ge=0, le=1)
    max_concurrency: int = Field(default=1, ge=1, le=8)
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_backoff_seconds: float = Field(default=2.0, gt=0)
    max_backoff_seconds: float = Field(default=60.0, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    # Hard cap on LLM requests per run, including retries and escalations.
    max_llm_requests: int = Field(default=20, ge=1)
    # Budget for code snippets sent to one agent.
    max_context_chars: int = Field(default=6000, ge=500)
    max_output_tokens: int = Field(default=2048, ge=64)
    temperature: float = Field(default=0.0, ge=0, le=2)
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"


ENV_MODEL = "API_DRIFT_GUARD_MODEL"
ENV_ESCALATION_MODEL = "API_DRIFT_GUARD_ESCALATION_MODEL"
ENV_MAX_CONCURRENCY = "API_DRIFT_GUARD_MAX_CONCURRENCY"


def apply_env_overrides(settings: LlmSettings, environ: Mapping[str, str] | None = None) -> LlmSettings:
    """Environment variables override the config file's model selection and concurrency."""
    env = os.environ if environ is None else environ
    updates: dict[str, object] = {}
    if env.get(ENV_MODEL):
        updates["model"] = env[ENV_MODEL]
    if ENV_ESCALATION_MODEL in env:
        value = env[ENV_ESCALATION_MODEL].strip()
        updates["escalation_model"] = None if value.lower() in ("", "none") else value
    if env.get(ENV_MAX_CONCURRENCY):
        updates["max_concurrency"] = env[ENV_MAX_CONCURRENCY]
    try:
        return LlmSettings.model_validate({**settings.model_dump(), **updates})
    except ValidationError as exc:
        raise ConfigError(f"Invalid LLM settings from environment:\n{_format_validation_error(exc)}") from exc


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfig
    consumers: list[ConsumerConfig] = Field(default_factory=list)
    llm: LlmSettings = Field(default_factory=LlmSettings)

    @field_validator("consumers")
    @classmethod
    def _unique_consumer_names(cls, consumers: list[ConsumerConfig]) -> list[ConsumerConfig]:
        names = [c.name for c in consumers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate consumer names: {', '.join(duplicates)}")
        return consumers


def _format_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  - {location}: {item['msg']}")
    return "\n".join(lines)


def load_config(path: Path) -> AppConfig:
    """Load a config file; consumer paths are resolved relative to the file's directory."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level")

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{_format_validation_error(exc)}") from exc

    base_dir = path.parent.resolve()
    config.consumers = [
        consumer.model_copy(update={"path": (base_dir / consumer.path).resolve()})
        for consumer in config.consumers
    ]
    return config
