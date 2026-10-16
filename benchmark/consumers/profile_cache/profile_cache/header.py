"""Caches the profile fields shown in the account header."""

from profile_cache.users_api import fetch_user

_CACHE: dict[int, dict[str, str]] = {}


def header_profile(user_id: int) -> dict[str, str]:
    if user_id not in _CACHE:
        user = fetch_user(user_id)
        _CACHE[user_id] = {"display_name": user["name"], "plan": user["plan"]}
    return _CACHE[user_id]
