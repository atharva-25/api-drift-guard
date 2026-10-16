"""Normalizes user emails for deduplication."""

import httpx

BASE = "http://user-service"


def normalized_email(user_id: int) -> str:
    data = httpx.get(f"{BASE}/users/{user_id}", timeout=5.0).json()
    return data.get("email").strip().lower()
