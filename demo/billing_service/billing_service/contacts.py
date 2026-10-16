"""Looks up billing contact details in user-service."""

import httpx

USER_SERVICE_URL = "http://user-service"


def get_billing_contact(user_id: int) -> dict[str, str]:
    with httpx.Client(base_url=USER_SERVICE_URL, timeout=5.0) as client:
        response = client.get(f"/users/{user_id}")
        response.raise_for_status()
        data = response.json()
    return {"name": data["name"], "email": data["email"]}
