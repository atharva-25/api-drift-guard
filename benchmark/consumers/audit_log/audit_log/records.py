"""Builds account-age audit records asynchronously."""

import httpx

BASE_URL = "http://user-service"


async def account_age_record(user_id: int) -> dict[str, str]:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5.0) as client:
        response = await client.get(f"/users/{user_id}")
        response.raise_for_status()
        payload = response.json()
    return {"user": str(payload["user_id"]), "created": payload["created_at"]}
