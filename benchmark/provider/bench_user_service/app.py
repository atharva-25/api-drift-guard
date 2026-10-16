"""Benchmark provider: a user-service with a richer response schema than the demo.

Each contract variant is the baseline after one deliberate change. Specs are exported
by ``benchmark/provider/export_specs.py``; nothing here is imported by the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import create_model

Fields = dict[str, tuple[Any, Any]]


@dataclass(frozen=True)
class Contract:
    version: str
    user: Fields
    address: Fields


ADDRESS_V1: Fields = {"city": (str, ...), "zip": (str, ...)}
USER_V1: Fields = {
    "user_id": (int, ...),
    "name": (str, ...),
    "email": (str, ...),
    "nickname": (str | None, None),
    "plan": (str, ...),
    "created_at": (str, ...),
}


def _without(fields: Fields, name: str) -> Fields:
    return {k: v for k, v in fields.items() if k != name}


def _renamed(fields: Fields, old: str, new: str) -> Fields:
    return {(new if k == old else k): v for k, v in fields.items()}


CONTRACTS: dict[str, Contract] = {
    "v1": Contract("1.0.0", USER_V1, ADDRESS_V1),
    "plan_renamed": Contract("2.0.0", _renamed(USER_V1, "plan", "tier"), ADDRESS_V1),
    "email_removed": Contract("2.0.0", _without(USER_V1, "email"), ADDRESS_V1),
    "nickname_removed": Contract("2.0.0", _without(USER_V1, "nickname"), ADDRESS_V1),
    "zip_removed": Contract("2.0.0", USER_V1, _without(ADDRESS_V1, "zip")),
    "created_at_removed": Contract("2.0.0", _without(USER_V1, "created_at"), ADDRESS_V1),
    "user_id_renamed": Contract("2.0.0", _renamed(USER_V1, "user_id", "id"), ADDRESS_V1),
    "avatar_added": Contract("1.1.0", {**USER_V1, "avatar_url": (str | None, None)}, ADDRESS_V1),
}


def create_app(contract: Contract) -> FastAPI:
    Address = create_model("Address", **contract.address)  # type: ignore[call-overload]  # noqa: N806
    User = create_model("User", **contract.user, address=(Address, ...))  # type: ignore[call-overload]  # noqa: N806

    app = FastAPI(title="user-service", version=contract.version)

    @app.get("/users", response_model=list[User])
    def list_users() -> list[Any]:
        return []

    @app.get("/users/{user_id}", response_model=User)
    def get_user(user_id: int) -> Any:
        raise HTTPException(status_code=404, detail="User not found")

    return app
