"""Maps user-service fields onto billing record fields."""

import httpx

USER_SERVICE_URL = "http://user-service"
FIELD_MAP = {"account_id": "user_id", "tier": "plan"}


def to_billing_record(user_id: int) -> dict:
    response = httpx.get(f"{USER_SERVICE_URL}/users/{user_id}", timeout=5.0)
    response.raise_for_status()
    source = response.json()
    return {target: source[field] for target, field in FIELD_MAP.items()}
