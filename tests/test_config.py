from __future__ import annotations

from pathlib import Path

import pytest

from api_drift_guard.config import ConfigError, LlmSettings, apply_env_overrides, load_config


def test_load_valid_config_resolves_paths_relative_to_config_file(write_config, tmp_path: Path) -> None:
    path = write_config(
        "provider:\n  name: user-service\n"
        "consumers:\n"
        "  - name: order-service\n    path: ./demo/order\n"
        "  - name: billing-service\n    path: ../elsewhere/billing\n"
    )
    config = load_config(path)

    assert config.provider.name == "user-service"
    assert [c.name for c in config.consumers] == ["order-service", "billing-service"]
    assert config.consumers[0].path == (tmp_path / "demo/order").resolve()
    assert config.consumers[1].path == (tmp_path.parent / "elsewhere/billing").resolve()


def test_consumers_default_to_empty(write_config) -> None:
    config = load_config(write_config("provider:\n  name: user-service\n"))
    assert config.consumers == []


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(write_config) -> None:
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write_config("provider: [unclosed\n"))


def test_top_level_must_be_mapping(write_config) -> None:
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(write_config("- just\n- a list\n"))


def test_missing_provider(write_config) -> None:
    with pytest.raises(ConfigError, match="provider"):
        load_config(write_config("consumers: []\n"))


def test_unknown_keys_are_rejected(write_config) -> None:
    with pytest.raises(ConfigError, match="consumer"):
        load_config(write_config("provider:\n  name: a\nconsumer: []\n"))


def test_duplicate_consumer_names(write_config) -> None:
    with pytest.raises(ConfigError, match="duplicate consumer names: svc"):
        load_config(
            write_config(
                "provider:\n  name: a\nconsumers:\n"
                "  - {name: svc, path: ./a}\n  - {name: svc, path: ./b}\n"
            )
        )


def test_llm_settings_default_and_override(write_config) -> None:
    assert load_config(write_config("provider:\n  name: a\n")).llm == LlmSettings()
    config = load_config(
        write_config("provider:\n  name: a\nllm:\n  model: groq/other\n  escalation_model: null\n  max_concurrency: 2\n")
    )
    assert config.llm.model == "groq/other"
    assert config.llm.escalation_model is None
    assert config.llm.max_concurrency == 2


@pytest.mark.parametrize("llm", ["max_concurrency: 0", "max_retries: 99", "unknown_key: 1", "reasoning_effort: extreme"])
def test_invalid_llm_settings(write_config, llm: str) -> None:
    with pytest.raises(ConfigError, match="llm"):
        load_config(write_config(f"provider:\n  name: a\nllm:\n  {llm}\n"))


def test_env_overrides() -> None:
    settings = apply_env_overrides(
        LlmSettings(),
        {"API_DRIFT_GUARD_MODEL": "groq/x", "API_DRIFT_GUARD_ESCALATION_MODEL": "none", "API_DRIFT_GUARD_MAX_CONCURRENCY": "3"},
    )
    assert (settings.model, settings.escalation_model, settings.max_concurrency) == ("groq/x", None, 3)
    assert apply_env_overrides(LlmSettings(), {}) == LlmSettings()
    with pytest.raises(ConfigError, match="max_concurrency"):
        apply_env_overrides(LlmSettings(), {"API_DRIFT_GUARD_MAX_CONCURRENCY": "100"})
