from __future__ import annotations

import pytest

from api_drift_guard.discovery import Operation, is_resolved, match_call, url_path_segments
from api_drift_guard.source_analysis import CodeLocation, HttpCall

OPERATIONS = [
    Operation("GET", "/users"),
    Operation("GET", "/users/{user_id}"),
    Operation("GET", "/users/me"),
    Operation("DELETE", "/users/{user_id}"),
    Operation("GET", "/users/{user_id}/orders"),
]


def call(url: tuple | None, method: str | None = "GET") -> HttpCall:
    return HttpCall(1, CodeLocation("c.py", 1, ""), method, url, "")


@pytest.mark.parametrize(
    ("url", "segments"),
    [
        (("http://user-service/users/", None), ("users", None)),
        (("https://h:8080/users",), ("users",)),
        (("http://host",), ()),
        ((None, "/users/", None), ("users", None)),
        (("/users/", None, "/orders?limit=5"), ("users", None, "orders")),
        (("/users/",), ("users",)),
        (("users/1#frag",), ("users", "1")),
        (("http://", None, "/users"), ("users",)),
        (("/users/u-", None), ("users", None)),
        ((None,), None),
        ((None, "users"), None),
    ],
)
def test_url_path_segments(url: tuple, segments: tuple | None) -> None:
    assert url_path_segments(url) == segments


@pytest.mark.parametrize(
    ("url", "method", "expected"),
    [
        (("http://user-service/users/", None), "GET", ("GET /users/{user_id}",)),
        (("http://user-service/users/42",), "GET", ("GET /users/{user_id}",)),
        (("http://user-service/users/me",), "GET", ("GET /users/me",)),
        (("/users",), "GET", ("GET /users",)),
        (("/users/", None, "/orders"), "GET", ("GET /users/{user_id}/orders",)),
        (("/users/", None), "DELETE", ("DELETE /users/{user_id}",)),
    ],
)
def test_match_call_picks_most_specific_operation(url: tuple, method: str, expected: tuple) -> None:
    match = match_call(call(url, method), OPERATIONS)
    assert match is not None
    assert match.operations == expected
    assert not match.ambiguous


def test_dynamic_method_is_ambiguous() -> None:
    match = match_call(call(("/users/", None), method=None), OPERATIONS)
    assert match is not None
    assert set(match.operations) == {"GET /users/{user_id}", "DELETE /users/{user_id}"}
    assert match.ambiguous


@pytest.mark.parametrize(
    ("url", "method"),
    [(("/orders/1",), "GET"), (("/users",), "POST"), (("/users/1/2/3",), "GET")],
)
def test_no_match(url: tuple, method: str) -> None:
    assert match_call(call(url, method), OPERATIONS) is None


def test_unresolved_url_may_target_any_operation_with_the_method() -> None:
    match = match_call(call(None, "DELETE"), OPERATIONS)
    assert match is not None
    assert match.unresolved and match.ambiguous
    assert match.operations == ("DELETE /users/{user_id}",)
    assert not is_resolved(match.call)
    assert match_call(call(None, "PUT"), OPERATIONS) is None
