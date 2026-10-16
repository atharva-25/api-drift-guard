"""Reads plan tiers from the legacy directory service (not user-service)."""

import httpx

LEGACY_DIRECTORY = "http://legacy-directory"


def legacy_tier(user_id: int) -> str:
    record = httpx.get(f"{LEGACY_DIRECTORY}/users/{user_id}", timeout=5.0).json()
    return f"{record['user_id']}:{record['plan']}"
