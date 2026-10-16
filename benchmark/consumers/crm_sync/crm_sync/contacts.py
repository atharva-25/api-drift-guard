"""Loads CRM contacts from user-service, validated by a Pydantic model."""

import httpx
from pydantic import BaseModel

USER_SERVICE = "http://user-service"


class CrmContact(BaseModel):
    user_id: int
    email: str
    plan: str


def load_contact(user_id: int) -> CrmContact:
    with httpx.Client(base_url=USER_SERVICE, timeout=5.0) as client:
        response = client.get(f"/users/{user_id}")
        response.raise_for_status()
        return CrmContact.model_validate(response.json())
