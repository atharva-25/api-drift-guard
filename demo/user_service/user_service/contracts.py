"""Response contract versions of user-service used by the demo scenarios.

``v1`` is the baseline that every consumer was written against; each other version
is the provider after one deliberate change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Contract:
    version: str
    description: str
    # field name -> (type, default); ``...`` marks a required field
    user: dict[str, tuple[Any, Any]]
    preferences: dict[str, tuple[Any, Any]]


_USER_V1: dict[str, tuple[Any, Any]] = {
    "user_id": (int, ...),
    "name": (str, ...),
    "email": (str, ...),
    "locale": (str, ...),
}
_PREFERENCES_V1: dict[str, tuple[Any, Any]] = {"newsletter": (bool, ...), "theme": (str, ...)}


def _without(fields: dict[str, tuple[Any, Any]], name: str) -> dict[str, tuple[Any, Any]]:
    return {k: v for k, v in fields.items() if k != name}


def _renamed(fields: dict[str, tuple[Any, Any]], old: str, new: str) -> dict[str, tuple[Any, Any]]:
    return {(new if k == old else k): v for k, v in fields.items()}


CONTRACTS: dict[str, Contract] = {
    "v1": Contract("1.0.0", "Baseline contract", _USER_V1, _PREFERENCES_V1),
    "user_id_renamed": Contract(
        "2.0.0", "user_id renamed to userId", _renamed(_USER_V1, "user_id", "userId"), _PREFERENCES_V1
    ),
    "email_renamed": Contract(
        "2.0.0", "email renamed to contact_email", _renamed(_USER_V1, "email", "contact_email"), _PREFERENCES_V1
    ),
    "phone_added": Contract(
        "1.1.0", "optional phone field added", {**_USER_V1, "phone": (str | None, None)}, _PREFERENCES_V1
    ),
    "newsletter_removed": Contract(
        "2.0.0", "newsletter removed from preferences", _USER_V1, _without(_PREFERENCES_V1, "newsletter")
    ),
    "locale_removed": Contract("2.0.0", "locale removed from users", _without(_USER_V1, "locale"), _PREFERENCES_V1),
}
