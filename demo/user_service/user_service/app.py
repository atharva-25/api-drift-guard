"""user-service: a small FastAPI provider whose response contract varies by version."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import create_model

from user_service.contracts import CONTRACTS, Contract

_SAMPLE_VALUES: dict[str, Any] = {
    "user_id": 1,
    "userId": 1,
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "contact_email": "ada@example.com",
    "locale": "en-GB",
    "phone": None,
    "newsletter": True,
    "theme": "dark",
}


def create_app(contract: Contract) -> FastAPI:
    # Models are always named "User"/"Preferences" so versions differ only in fields.
    User = create_model("User", **contract.user)  # type: ignore[call-overload]  # noqa: N806
    Preferences = create_model("Preferences", **contract.preferences)  # type: ignore[call-overload]  # noqa: N806

    users = {1: User(**{name: _SAMPLE_VALUES[name] for name in contract.user})}
    preferences = {1: Preferences(**{name: _SAMPLE_VALUES[name] for name in contract.preferences})}

    app = FastAPI(title="user-service", version=contract.version)

    @app.get("/users", response_model=list[User])
    def list_users() -> list[Any]:
        return list(users.values())

    @app.get("/users/{user_id}", response_model=User)
    def get_user(user_id: int) -> Any:
        if user_id not in users:
            raise HTTPException(status_code=404, detail="User not found")
        return users[user_id]

    @app.get("/users/{user_id}/preferences", response_model=Preferences)
    def get_preferences(user_id: int) -> Any:
        if user_id not in preferences:
            raise HTTPException(status_code=404, detail="User not found")
        return preferences[user_id]

    return app


app = create_app(CONTRACTS["v1"])
