"""Exports the user directory as CSV."""

import csv
import io

import httpx

BASE = "http://user-service"


def export_directory() -> str:
    users = httpx.get(f"{BASE}/users", timeout=10.0).json()
    rows = [(u["user_id"], u["name"], u["created_at"]) for u in users]
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()
