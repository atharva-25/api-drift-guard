"""Builds notification payloads from user-service profiles."""

import httpx

USER_SERVICE_URL = "http://user-service"
TEMPLATE_FIELDS = ("name", "email", "locale")


class Notifier:
    def __init__(self) -> None:
        self.client = httpx.Client(base_url=USER_SERVICE_URL, timeout=5.0)

    def build_payload(self, user_id: int) -> dict[str, str]:
        response = self.client.get(f"/users/{user_id}")
        response.raise_for_status()
        user = response.json()
        return {field: user[field] for field in TEMPLATE_FIELDS}
