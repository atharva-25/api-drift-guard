"""Renders profile titles via attribute access on a SimpleNamespace."""

from types import SimpleNamespace

import httpx

BASE = "http://user-service"


def load_profile(user_id: int) -> SimpleNamespace:
    response = httpx.get(f"{BASE}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    return SimpleNamespace(**response.json())


def profile_title(user_id: int) -> str:
    profile = load_profile(user_id)
    return f"#{profile.user_id} {profile.name}"
