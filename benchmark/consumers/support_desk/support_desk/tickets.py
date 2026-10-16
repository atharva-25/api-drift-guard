"""Finds the contact email for support tickets (uses requests, not httpx)."""

import requests

USER_URL = "http://user-service/users/{user_id}"


def ticket_contact(user_id: int) -> str:
    response = requests.get(USER_URL.format(user_id=user_id), timeout=5)
    response.raise_for_status()
    return response.json()["email"]
