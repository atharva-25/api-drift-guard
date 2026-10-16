"""Chooses the label shown on a user's badge."""

import httpx

BASE = "http://user-service"


def badge_label(user_id: int) -> str:
    data = httpx.get(f"{BASE}/users/{user_id}", timeout=5.0).json()
    if "nickname" in data:
        return data["nickname"]
    return data["name"]
