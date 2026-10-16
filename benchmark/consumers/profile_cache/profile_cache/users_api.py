"""Thin client for user-service."""

import httpx

BASE_URL = "http://user-service"


def fetch_user(user_id: int) -> dict:
    response = httpx.get(f"{BASE_URL}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    return response.json()
