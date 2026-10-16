"""Greets users by nickname, falling back to their name."""

import httpx

BASE = "http://user-service"


def greeting(user_id: int) -> str:
    profile = httpx.get(f"{BASE}/users/{user_id}", timeout=5.0).json()
    display = profile.get("nickname") or profile["name"]
    return f"Hi {display}!"
