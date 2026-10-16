"""Deterministic evidence handling: compact code context for agents, and verification
of every file:line reference an agent returns against the repository on disk."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_WINDOW_LINES = 40
_FALLBACK_RADIUS = 3
_MAX_CONSTANT_LINES = 5
_MIN_FRAGMENT_CHARS = 10


@dataclass(frozen=True)
class CodeSnippet:
    file: str
    start_line: int
    end_line: int
    code: str  # each line prefixed with its line number: "  12| code"


@dataclass(frozen=True)
class _ParsedFile:
    lines: list[str]
    tree: ast.Module | None


def _parse(root: Path, rel_path: str) -> _ParsedFile | None:
    try:
        source = (root / rel_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        tree = None
    return _ParsedFile(source.splitlines(), tree)


def _enclosing_function(tree: ast.Module, line: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno is not None:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            if start <= line <= node.end_lineno and (best is None or node.lineno > best.lineno):
                best = node
    return best


def _window(parsed: _ParsedFile, line: int) -> tuple[int, int]:
    """Line range around ``line``: its enclosing function, clipped to a maximum size."""
    function = _enclosing_function(parsed.tree, line) if parsed.tree else None
    if function is None:
        start, end = line - _FALLBACK_RADIUS, line + _FALLBACK_RADIUS
    else:
        start = min([function.lineno, *(d.lineno for d in function.decorator_list)])
        end = function.end_lineno or line
        if end - start + 1 > _MAX_WINDOW_LINES:
            half = _MAX_WINDOW_LINES // 2
            start, end = max(start, line - half), min(end, line + half)
    return max(1, start), min(len(parsed.lines), end)


def _referenced_module_assignments(parsed: _ParsedFile, start: int, end: int) -> list[tuple[int, int]]:
    """Ranges of short module-level assignments to names used within [start, end]."""
    if parsed.tree is None:
        return []
    used = {
        node.id
        for node in ast.walk(parsed.tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and start <= node.lineno <= end
    }
    ranges = []
    for stmt in parsed.tree.body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        stmt_end = stmt.end_lineno or stmt.lineno
        if names & used and stmt_end - stmt.lineno < _MAX_CONSTANT_LINES:
            ranges.append((stmt.lineno, stmt_end))
    return ranges


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _render(parsed: _ParsedFile, rel_path: str, start: int, end: int) -> CodeSnippet:
    code = "\n".join(f"{n:>4}| {parsed.lines[n - 1]}" for n in range(start, end + 1))
    return CodeSnippet(rel_path, start, end, code)


def build_snippets(
    root: Path, locations: list[tuple[str, int]], max_chars: int
) -> tuple[list[CodeSnippet], bool]:
    """Build numbered code snippets around ``locations`` (in priority order).

    Returns the snippets and whether some were dropped to stay within ``max_chars``.
    """
    ranges_by_file: dict[str, list[tuple[int, int]]] = {}
    parsed_files: dict[str, _ParsedFile] = {}
    for rel_path, line in locations:
        if rel_path not in parsed_files:
            parsed = _parse(root, rel_path)
            if parsed is None:
                continue
            parsed_files[rel_path] = parsed
        parsed = parsed_files[rel_path]
        if not 1 <= line <= len(parsed.lines):
            continue
        start, end = _window(parsed, line)
        ranges = ranges_by_file.setdefault(rel_path, [])
        ranges.append((start, end))
        ranges.extend(_referenced_module_assignments(parsed, start, end))

    snippets: list[CodeSnippet] = []
    used = 0
    truncated = False
    for rel_path, ranges in ranges_by_file.items():
        for start, end in _merge_ranges(ranges):
            snippet = _render(parsed_files[rel_path], rel_path, start, end)
            if used + len(snippet.code) > max_chars:
                truncated = True
                continue
            snippets.append(snippet)
            used += len(snippet.code)
    return snippets, truncated


def _normalize(text: str) -> str:
    return " ".join(text.split())


def verify_reference(root: Path, file: str, line: int, snippet: str) -> tuple[str, str] | None:
    """Return (normalized relative path, actual source line) if the reference is real, else None.

    A reference is real when ``file`` is a Python file inside ``root``, ``line`` exists,
    and the cited snippet's first non-blank line matches that source line (ignoring
    whitespace; a cited fragment of the line is accepted).
    """
    root = root.resolve()
    try:
        path = (root / file).resolve()
    except (OSError, ValueError):
        return None
    if not path.is_relative_to(root) or path.suffix != ".py" or not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if not 1 <= line <= len(lines):
        return None

    actual = lines[line - 1]
    cited_lines = [ln for ln in snippet.splitlines() if ln.strip()]
    if not cited_lines:
        return None
    cited = cited_lines[0]
    # Tolerate the "  12| " prefix used in the snippets we sent.
    prefix, sep, rest = cited.partition("|")
    if sep and prefix.strip().isdigit():
        cited = rest
    cited_norm, actual_norm = _normalize(cited), _normalize(actual)
    # A fragment must be substantial, so trivial text like "(" cannot validate a line.
    long_enough = len(cited_norm) >= min(len(actual_norm), _MIN_FRAGMENT_CHARS)
    if cited_norm and long_enough and cited_norm in actual_norm:
        return path.relative_to(root).as_posix(), actual.strip()
    return None
