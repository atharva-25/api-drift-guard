"""Fetches order owners from user-service."""

import httpx

USER_SERVICE_URL = "http://user-service"


def get_order_owner_id(user_id: int) -> int:
    response = httpx.get(f"{USER_SERVICE_URL}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    user = response.json()
    return user["user_id"]


def get_order_owner_name(user_id: int) -> str:
    response = httpx.get(f"{USER_SERVICE_URL}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    return response.json()["name"]


def get_confirmation_recipient(user_id: int) -> str:
    response = httpx.get(f"{USER_SERVICE_URL}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    user = response.json()
    return user["email"]
