from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from api_drift_guard.source_analysis import (
    RepoAnalysis,
    SourceAnalysisError,
    analyze_repository,
    normalize_url,
)


def analyze(tmp_path: Path, code: str, filename: str = "client.py") -> RepoAnalysis:
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code))
    return analyze_repository(tmp_path)


def reads(result: RepoAnalysis) -> set[tuple[tuple[str, ...], str, int]]:
    return {(a.path, a.kind, a.location.line) for a in result.accesses}


def opaque(result: RepoAnalysis) -> set[tuple[tuple[str, ...], str]]:
    return {(o.path, o.kind) for o in result.opaque_uses}


# --- request detection and URL extraction ------------------------------------


def test_module_function_with_fstring_and_constant_base(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        BASE = "http://user-service"

        def f(user_id):
            return httpx.get(f"{BASE}/users/{user_id}", timeout=5)
        """,
    )
    (call,) = result.calls
    assert call.method == "GET"
    assert call.url == ("http://user-service/users/", None)
    assert call.url_source == 'f"{BASE}/users/{user_id}"'
    assert call.location.file == "client.py"
    assert call.location.line == 6
    assert call.location.snippet == 'return httpx.get(f"{BASE}/users/{user_id}", timeout=5)'


@pytest.mark.parametrize(
    ("code", "method", "url"),
    [
        ('import httpx as hx\nhx.post("http://svc/users")', "POST", ("http://svc/users",)),
        ('from httpx import get\nget("/users/1")', "GET", ("/users/1",)),
        ('import httpx\nhttpx.request("delete", "/users/" + str(uid))', "DELETE", ("/users/", None)),
        ('import httpx\nhttpx.request(m, url="/users")', None, ("/users",)),
        ('import httpx\nhttpx.get(url="/users/{}".format(uid))', "GET", ("/users/", None)),
        ('import httpx\nhttpx.get(some_url)', "GET", None),
        ('import httpx\ndef f(uid):\n    url = f"/users/{uid}"\n    httpx.get(url)', "GET", ("/users/", None)),
    ],
)
def test_request_forms(tmp_path: Path, code: str, method: str | None, url: tuple | None) -> None:
    (call,) = analyze(tmp_path, code).calls
    assert call.method == method
    assert call.url == url


def test_client_with_base_url_in_with_block(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        def f(uid):
            with httpx.Client(base_url="http://user-service") as client:
                return client.get(f"/users/{uid}")
        """,
    )
    (call,) = result.calls
    assert call.url == ("http://user-service/users/", None)


def test_module_level_async_client_and_await(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        from httpx import AsyncClient
        client = AsyncClient(base_url=SERVICE_URL)
        async def f(uid):
            resp = await client.get(f"/users/{uid}")
            return resp.json()["name"]
        """,
    )
    (call,) = result.calls
    # Unknown base URL is kept as a leading dynamic part.
    assert call.url == (None, "/users/", None)
    assert reads(result) == {(("name",), "subscript", 6)}


def test_self_client_assigned_in_init(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        class C:
            def __init__(self):
                self.http = httpx.Client(base_url="http://svc")
            def f(self, uid):
                return self.http.get(f"/users/{uid}").json()["email"]
        """,
    )
    (call,) = result.calls
    assert call.url == ("http://svc/users/", None)
    assert reads(result) == {(("email",), "subscript", 7)}


def test_absolute_url_ignores_client_base(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        'import httpx\nc = httpx.Client(base_url="http://a")\nc.get("http://b/users")\n',
    )
    assert result.calls[0].url == ("http://b/users",)


def test_non_httpx_calls_are_ignored(tmp_path: Path) -> None:
    result = analyze(tmp_path, 'import requests\nrequests.get("/users/1").json()["user_id"]\n')
    assert result.calls == []
    assert result.accesses == []


# --- JSON field tracking -------------------------------------------------------


def test_assignment_chain_and_nested_reads(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        def f():
            response = httpx.get("/users/1")
            data = response.json()
            uid = data["user_id"]
            city = data["address"]["city"]
            email = data.get("email", "")
            has_phone = "phone" in data
        """,
    )
    assert reads(result) == {
        (("user_id",), "subscript", 6),
        (("address",), "subscript", 7),
        (("address", "city"), "subscript", 7),
        (("email",), "get", 8),
        (("phone",), "membership", 9),
    }
    assert result.opaque_uses == []


def test_arrays_loops_comprehensions_and_indexes(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        def f():
            users = httpx.get("/users").json()
            for user in users:
                print(user["user_id"])
            names = [u["name"] for u in users]
            first = users[0]["email"]
            tail = users[1:]
            last = tail[-1]["phone"]
        """,
    )
    assert {a.path for a in result.accesses} == {
        ("items", "user_id"),
        ("items", "name"),
        ("items", "email"),
        ("items", "phone"),
    }


def test_module_constant_key_is_resolved(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        'import httpx\nKEY = "user_id"\ndef f():\n    return httpx.get("/u").json()[KEY]\n',
    )
    assert {a.path for a in result.accesses} == {("user_id",)}


def test_dynamic_keys_are_opaque(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        FIELDS = ("name", "email")
        def f(key):
            data = httpx.get("/users/1").json()
            a = data[key]
            b = {k: data[k] for k in FIELDS}
            c = data.get(key)
            for k, v in data.items():
                pass
        """,
    )
    assert result.accesses == []
    assert opaque(result) == {((), "dynamic_key")}
    assert len(result.opaque_uses) == 4


@pytest.mark.parametrize(
    ("statement", "detail_fragment"),
    [
        ("return data", "returned from f()"),
        ("process(data)", "passed to process()"),
        ("User(**data)", "unpacked with ** into User()"),
        ("self_cache[1] = data", "stored in self_cache[1]"),
        ("out = {'user': data}", "stored in a dict literal"),
        ("items.append(data)", "passed to items.append()"),
        ("httpx.post('/audit', json=data)", "sent in httpx.post()"),
        ("handle(response)", "response object passed to handle()"),
    ],
)
def test_escapes_are_opaque(tmp_path: Path, statement: str, detail_fragment: str) -> None:
    result = analyze(
        tmp_path,
        f"""
        import httpx
        def f():
            response = httpx.get("/users/1")
            data = response.json()
            {statement}
        """,
    )
    details = [o.detail for o in result.opaque_uses if o.kind == "escape"]
    assert any(detail_fragment in d for d in details), details


def test_harmless_uses_are_not_escapes(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx, json, logging
        logger = logging.getLogger(__name__)
        def f():
            data = httpx.get("/users/1").json()
            print(data)
            logger.info("got %s", data)
            json.dumps(data)
            if len(data) and isinstance(data, dict):
                pass
        """,
    )
    assert result.opaque_uses == []


def test_escape_of_a_leaf_value_keeps_its_path(tmp_path: Path) -> None:
    result = analyze(tmp_path, 'import httpx\ndef f():\n    return httpx.get("/u").json()["name"]\n')
    assert opaque(result) == {(("name",), "escape")}


def test_reassignment_stops_tracking(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        'import httpx\ndef f():\n    data = httpx.get("/u").json()\n    data = {}\n    data["user_id"]\n',
    )
    assert result.accesses == []
    assert "user_id" in result.untracked_key_reads


def test_scopes_do_not_leak_between_functions(tmp_path: Path) -> None:
    result = analyze(
        tmp_path,
        """
        import httpx
        def a():
            data = httpx.get("/u").json()
        def b(data):
            return data["user_id"]
        """,
    )
    assert result.accesses == []
    assert [loc.line for loc in result.untracked_key_reads["user_id"]] == [6]


def test_untracked_key_reads(tmp_path: Path) -> None:
    result = analyze(tmp_path, 'def f(user):\n    return user["user_id"], user.get("email")\n')
    assert set(result.untracked_key_reads) == {"user_id", "email"}


# --- repository walking ---------------------------------------------------------


def test_skips_virtualenvs_and_records_parse_errors(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "x.py").write_text('import httpx\nhttpx.get("/u")\n')
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "ok.py").write_text('import httpx\nhttpx.get("/u")\n')
    (tmp_path / "pkg" / "broken.py").write_text("def f(:\n")

    result = analyze_repository(tmp_path)
    assert result.files_scanned == 1
    assert [c.location.file for c in result.calls] == ["pkg/ok.py"]
    assert len(result.parse_errors) == 1
    assert result.parse_errors[0].startswith("pkg/broken.py:")


def test_missing_repository(tmp_path: Path) -> None:
    with pytest.raises(SourceAnalysisError, match="does not exist"):
        analyze_repository(tmp_path / "missing")


def test_normalize_url() -> None:
    assert normalize_url(("a", "b", None, None, "", "c", "d")) == ("ab", None, "cd")
    assert normalize_url((None, "")) == (None,)
