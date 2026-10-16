from __future__ import annotations

from pathlib import Path

import pytest

from api_drift_guard.evidence import build_snippets, verify_reference

SOURCE = """\
import httpx

BASE = "http://user-service"
FIELDS = ("name", "email")
UNUSED = "x"


def payload(uid):
    user = httpx.get(f"{BASE}/users/{uid}").json()
    return {k: user[k] for k in FIELDS}


def other():
    return 1
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "client.py").write_text(SOURCE)
    (tmp_path / "notes.txt").write_text("not python\n")
    return tmp_path


def test_snippet_is_enclosing_function_plus_referenced_constants(repo: Path) -> None:
    snippets, truncated = build_snippets(repo, [("pkg/client.py", 10)], 10_000)
    assert not truncated
    assert [(s.start_line, s.end_line) for s in snippets] == [(3, 4), (8, 10)]
    assert snippets[0].code == '   3| BASE = "http://user-service"\n   4| FIELDS = ("name", "email")'
    assert snippets[1].code.splitlines()[-1] == "  10|     return {k: user[k] for k in FIELDS}"
    assert all("UNUSED" not in s.code and "def other" not in s.code for s in snippets)


def test_module_level_location_uses_a_small_window(repo: Path) -> None:
    snippets, _ = build_snippets(repo, [("pkg/client.py", 5)], 10_000)
    assert [(s.start_line, s.end_line) for s in snippets] == [(2, 8)]


def test_overlapping_locations_are_merged(repo: Path) -> None:
    snippets, _ = build_snippets(repo, [("pkg/client.py", 9), ("pkg/client.py", 10)], 10_000)
    assert [(s.start_line, s.end_line) for s in snippets] == [(3, 4), (8, 10)]


def test_budget_truncates(repo: Path) -> None:
    snippets, truncated = build_snippets(repo, [("pkg/client.py", 10)], 80)
    assert truncated
    assert sum(len(s.code) for s in snippets) <= 80


def test_unreadable_or_out_of_range_locations_are_skipped(repo: Path) -> None:
    assert build_snippets(repo, [("missing.py", 1), ("pkg/client.py", 500)], 10_000) == ([], False)


@pytest.mark.parametrize(
    ("file", "line", "snippet"),
    [
        ("pkg/client.py", 10, "return {k: user[k] for k in FIELDS}"),
        ("pkg/client.py", 10, "  return   {k: user[k]   for k in FIELDS}  "),  # whitespace
        ("pkg/client.py", 10, "  10|     return {k: user[k] for k in FIELDS}"),  # numbered prefix
        ("pkg/client.py", 10, "{k: user[k] for k in FIELDS}"),  # fragment
        ("pkg/./client.py", 10, "return {k: user[k] for k in FIELDS}\nnext line ignored"),
    ],
)
def test_verify_reference_accepts_real_references(repo: Path, file: str, line: int, snippet: str) -> None:
    assert verify_reference(repo, file, line, snippet) == ("pkg/client.py", "return {k: user[k] for k in FIELDS}")


@pytest.mark.parametrize(
    ("file", "line", "snippet"),
    [
        ("pkg/client.py", 9, "return {k: user[k] for k in FIELDS}"),  # wrong line
        ("pkg/client.py", 0, "import httpx"),
        ("pkg/client.py", 99, "import httpx"),
        ("pkg/missing.py", 1, "import httpx"),
        ("notes.txt", 1, "not python"),
        ("../outside.py", 1, "import httpx"),
        ("/etc/hosts", 1, "localhost"),
        ("pkg/client.py", 10, "("),  # trivial fragment
        ("pkg/client.py", 10, "   "),
    ],
)
def test_verify_reference_rejects_invented_references(repo: Path, file: str, line: int, snippet: str) -> None:
    (repo.parent / "outside.py").write_text("import httpx\n")
    assert verify_reference(repo, file, line, snippet) is None
