"""Builds shipping labels from the user's postal address."""

import httpx

USER_SERVICE_URL = "http://user-service"


def shipping_label(user_id: int) -> str:
    response = httpx.get(f"{USER_SERVICE_URL}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    user = response.json()
    address = user["address"]
    return f"{user['name']}, {address['city']} {address['zip']}"
