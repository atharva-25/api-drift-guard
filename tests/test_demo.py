"""The demo provider is a real FastAPI app and the fixtures are generated from it."""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from tests.conftest import REPO_ROOT

PROVIDER = REPO_ROOT / "demo" / "user_service"


@pytest.fixture
def provider_modules(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(PROVIDER))
    from user_service.app import create_app
    from user_service.contracts import CONTRACTS

    return create_app, CONTRACTS


def test_fixtures_match_the_fastapi_provider() -> None:
    result = subprocess.run(
        [sys.executable, str(PROVIDER / "export_specs.py"), "--check"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_provider_serves_its_contract(provider_modules) -> None:
    create_app, contracts = provider_modules
    client = TestClient(create_app(contracts["v1"]))
    assert client.get("/users/1").json() == {
        "user_id": 1, "name": "Ada Lovelace", "email": "ada@example.com", "locale": "en-GB",
    }
    assert client.get("/users/2").status_code == 404
    assert client.get("/users/1/preferences").json() == {"newsletter": True, "theme": "dark"}

    renamed = TestClient(create_app(contracts["user_id_renamed"]))
    body = renamed.get("/users/1").json()
    assert "userId" in body and "user_id" not in body
