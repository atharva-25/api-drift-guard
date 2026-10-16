"""Lightweight static analysis of httpx usage in Python consumer code.

This is deliberately not a data-flow engine. Within a single function it tracks:

- httpx requests: ``httpx.get(...)``, ``httpx.request("GET", ...)`` and request methods
  on ``httpx.Client``/``AsyncClient`` instances (local, module-level or ``self.<attr>``);
- the response object and its ``.json()`` value through plain assignments, ``with``,
  ``for`` loops and comprehensions;
- field reads on that JSON: ``data["field"]``, ``data.get("field")``, ``"field" in data``.

Whatever it cannot follow is recorded as an ``OpaqueUse`` (dynamic keys, JSON passed to,
returned from or stored outside the function) so that classification can report
uncertainty instead of silently assuming a field is unused.
"""

from __future__ import annotations

import ast
import itertools
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Path segment for "an element of an array"; matches oasdiff's property path notation.
ITEMS = "items"

HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})
_REQUEST_FUNCTIONS = HTTP_METHODS | {"request"}
_CLIENT_CLASSES = frozenset({"Client", "AsyncClient"})
SKIP_DIRS = frozenset(
    {".git", ".hg", ".venv", "venv", "__pycache__", "node_modules", "build", "dist",
     ".tox", ".mypy_cache", ".pytest_cache", "site-packages"}
)
# Callees that receive JSON without reading specific fields; passing JSON to them is not an escape.
_HARMLESS_FUNCTIONS = frozenset({"print", "len", "bool", "str", "repr", "isinstance", "type", "id"})
_LOGGER_NAMES = frozenset({"logger", "log", "logging"})
_SNIPPET_MAX_CHARS = 200
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_FORMAT_FIELD_RE = re.compile(r"\{[^{}]*\}")

UrlPart = str | None  # None marks a part whose value is unknown statically
UrlTemplate = tuple[UrlPart, ...]


class SourceAnalysisError(Exception):
    pass


@dataclass(frozen=True)
class CodeLocation:
    file: str  # POSIX path relative to the repository root
    line: int
    snippet: str


@dataclass(frozen=True)
class HttpCall:
    call_id: int
    location: CodeLocation
    method: str | None  # None when dynamic, e.g. client.request(method, url)
    url: UrlTemplate | None  # None when nothing about the URL is known
    url_source: str


@dataclass(frozen=True)
class FieldAccess:
    call_id: int
    location: CodeLocation
    path: tuple[str, ...]
    kind: Literal["subscript", "get", "membership"]


@dataclass(frozen=True)
class OpaqueUse:
    call_id: int
    location: CodeLocation
    # JSON path of the value that could not be followed; () is the whole body.
    path: tuple[str, ...]
    kind: Literal["dynamic_key", "escape"]
    detail: str


@dataclass
class RepoAnalysis:
    root: Path
    files_scanned: int = 0
    calls: list[HttpCall] = field(default_factory=list)
    accesses: list[FieldAccess] = field(default_factory=list)
    opaque_uses: list[OpaqueUse] = field(default_factory=list)
    # Literal-key reads (x["k"], x.get("k")) on values not linked to a tracked response.
    untracked_key_reads: dict[str, list[CodeLocation]] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)


# Abstract values tracked per scope.
@dataclass(frozen=True)
class _Client:
    base_url: UrlTemplate | None


@dataclass(frozen=True)
class _Response:
    call_id: int


@dataclass(frozen=True)
class _Json:
    call_id: int
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Url:
    parts: UrlTemplate


_Value = _Client | _Response | _Json | _Url | None
_Env = dict[str, _Value]

_INDEX = object()  # marker: integer index or slice into an array


def _is_self_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"


def _is_format_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
    )


def _element(value: _Value) -> _Value:
    return _Json(value.call_id, value.path + (ITEMS,)) if isinstance(value, _Json) else None


def _first_tracked(values: list[_Value]) -> _Value:
    return next((v for v in values if isinstance(v, (_Json, _Response))), None)


def normalize_url(parts: tuple[UrlPart, ...]) -> UrlTemplate:
    """Merge adjacent literal parts, collapse adjacent unknown parts and drop empty strings."""
    merged: list[UrlPart] = []
    for part in parts:
        if part == "":
            continue
        if merged and part is not None and merged[-1] is not None:
            merged[-1] += part
        elif not (merged and part is None and merged[-1] is None):
            merged.append(part)
    return tuple(merged)


def _is_absolute(url: UrlTemplate) -> bool:
    return bool(url) and isinstance(url[0], str) and _SCHEME_RE.match(url[0]) is not None


class _FileAnalyzer:
    def __init__(self, rel_path: str, source: str, tree: ast.Module, ids: Iterator[int], result: RepoAnalysis):
        self.file = rel_path
        self.source = source
        self.lines = source.splitlines()
        self.tree = tree
        self.ids = ids
        self.result = result
        self.module_aliases: set[str] = set()
        self.function_aliases: dict[str, str] = {}
        self.client_aliases: set[str] = set()
        self.constants: dict[str, str] = {}
        self.function_stack: list[str] = []
        self._collect_imports()
        self._collect_constants()

    def run(self) -> None:
        self._visit_body(self.tree.body, {})

    # --- setup -------------------------------------------------------------

    def _collect_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "httpx":
                        self.module_aliases.add(alias.asname or "httpx")
            elif isinstance(node, ast.ImportFrom) and node.module == "httpx":
                for alias in node.names:
                    local = alias.asname or alias.name
                    if alias.name in _REQUEST_FUNCTIONS:
                        self.function_aliases[local] = alias.name
                    elif alias.name in _CLIENT_CLASSES:
                        self.client_aliases.add(local)

    def _collect_constants(self) -> None:
        for stmt in self.tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target, value = stmt.targets[0], stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                target, value = stmt.target, stmt.value
            else:
                continue
            if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                self.constants[target.id] = value.value

    # --- recording ---------------------------------------------------------

    def _loc(self, node: ast.AST) -> CodeLocation:
        line = getattr(node, "lineno", 0)
        text = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        return CodeLocation(self.file, line, text[:_SNIPPET_MAX_CHARS])

    def _source_text(self, node: ast.AST | None) -> str:
        """The node's code as written (falls back to unparsing)."""
        if node is None:
            return ""
        return ast.get_source_segment(self.source, node) or ast.unparse(node)

    def _access(self, owner: _Json, key: str, kind: Literal["subscript", "get", "membership"], node: ast.AST) -> _Json:
        path = owner.path + (key,)
        self.result.accesses.append(FieldAccess(owner.call_id, self._loc(node), path, kind))
        return _Json(owner.call_id, path)

    def _opaque(self, owner: _Json, node: ast.AST, kind: Literal["dynamic_key", "escape"], detail: str) -> None:
        self.result.opaque_uses.append(OpaqueUse(owner.call_id, self._loc(node), owner.path, kind, detail))

    def _escape(self, value: _Value, node: ast.AST, detail: str) -> None:
        if isinstance(value, _Json):
            self._opaque(value, node, "escape", detail)
        elif isinstance(value, _Response):
            self._opaque(_Json(value.call_id), node, "escape", f"response object {detail}")

    def _untracked_read(self, key: str, node: ast.AST) -> None:
        self.result.untracked_key_reads.setdefault(key, []).append(self._loc(node))

    # --- statements --------------------------------------------------------

    def _visit_body(self, body: list[ast.stmt], env: _Env) -> None:
        for stmt in body:
            self._visit_stmt(stmt, env)

    def _visit_stmt(self, stmt: ast.stmt, env: _Env) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in stmt.decorator_list:
                self._eval(decorator, env)
            self.function_stack.append(stmt.name)
            self._visit_body(stmt.body, {k: v for k, v in env.items() if isinstance(v, _Client)})
            self.function_stack.pop()
        elif isinstance(stmt, ast.ClassDef):
            class_env = {k: v for k, v in env.items() if isinstance(v, _Client)}
            class_env.update(self._self_clients(stmt))
            self._visit_body(stmt.body, class_env)
        elif isinstance(stmt, ast.Assign):
            value = self._eval(stmt.value, env)
            for target in stmt.targets:
                self._bind(target, value, env, stmt)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._bind(stmt.target, self._eval(stmt.value, env), env, stmt)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                value = self._eval(item.context_expr, env)
                if item.optional_vars is not None:
                    self._bind(item.optional_vars, value, env, stmt)
            self._visit_body(stmt.body, env)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._bind(stmt.target, _element(self._eval(stmt.iter, env)), env, stmt)
            self._visit_body(stmt.body, env)
            self._visit_body(stmt.orelse, env)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                where = f"{self.function_stack[-1]}()" if self.function_stack else "module"
                self._escape(self._eval(stmt.value, env), stmt, f"returned from {where}")
        else:
            self._visit_children(stmt, env)

    def _visit_children(self, node: ast.AST, env: _Env) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._visit_stmt(child, env)
            elif isinstance(child, ast.expr):
                self._eval(child, env)
            else:
                self._visit_children(child, env)

    def _self_clients(self, cls: ast.ClassDef) -> _Env:
        """Find ``self.<attr> = httpx.Client(...)`` anywhere in the class's methods."""
        clients: _Env = {}
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                client = self._client_from_call(node.value, {})
                for target in node.targets:
                    if client is not None and _is_self_attr(target):
                        clients[f"self.{target.attr}"] = client
        return clients

    def _bind(self, target: ast.expr, value: _Value, env: _Env, node: ast.AST) -> None:
        if isinstance(target, ast.Name):
            if value is None:
                env.pop(target.id, None)
            else:
                env[target.id] = value
            return
        if _is_self_attr(target) and isinstance(value, _Client):
            env[f"self.{target.attr}"] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind(elt.value if isinstance(elt, ast.Starred) else elt, None, env, node)
        else:
            # Evaluate reads inside the target, e.g. the key in out[data["id"]] = ...
            self._visit_children(target, env)
        self._escape(value, node, f"stored in {ast.unparse(target)}")

    # --- expressions -------------------------------------------------------

    def _eval(self, node: ast.expr, env: _Env) -> _Value:
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Attribute):
            key = f"self.{node.attr}"
            if _is_self_attr(node) and key in env:
                return env[key]
            self._eval(node.value, env)
            return None
        if isinstance(node, ast.Await):
            return self._eval(node.value, env)
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        if isinstance(node, ast.Subscript):
            return self._eval_subscript(node, env)
        if isinstance(node, ast.Compare):
            self._eval_compare(node, env)
            return None
        if isinstance(node, ast.BoolOp):
            return _first_tracked([self._eval(v, env) for v in node.values])
        if isinstance(node, ast.IfExp):
            self._eval(node.test, env)
            return _first_tracked([self._eval(node.body, env), self._eval(node.orelse, env)])
        if isinstance(node, ast.NamedExpr):
            value = self._eval(node.value, env)
            self._bind(node.target, value, env, node)
            return value
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            self._eval_comprehension(node, env)
            return None
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    self._eval(key, env)
                detail = "unpacked with **" if key is None else "stored in a dict literal"
                self._escape(self._eval(value, env), value, detail)
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                inner = elt.value if isinstance(elt, ast.Starred) else elt
                self._escape(self._eval(inner, env), elt, "stored in a container literal")
            return None
        if isinstance(node, ast.Lambda):
            return None
        self._visit_children(node, env)
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            return self._url_value(node, env)
        return None

    def _eval_call(self, node: ast.Call, env: _Env) -> _Value:
        func = node.func
        request_fn = self._httpx_function(func)
        if request_fn is not None:
            return self._record_request(node, request_fn, None, env)
        client = self._client_from_call(node, env)
        if client is not None:
            self._eval_args(node, env, harmless=True)
            return client
        if _is_format_call(node):
            self._eval_args(node, env, harmless=True)
            return self._url_value(node, env)

        if isinstance(func, ast.Attribute):
            owner = self._eval(func.value, env)
            if isinstance(owner, _Client) and func.attr in _REQUEST_FUNCTIONS:
                return self._record_request(node, func.attr, owner.base_url, env)
            if isinstance(owner, _Response) and func.attr == "json":
                self._eval_args(node, env, harmless=True)
                return _Json(owner.call_id)
            if isinstance(owner, _Json):
                return self._eval_json_method(node, owner, env)
            if func.attr == "get" and node.args:
                key = self._constant_key(node.args[0], env)
                if isinstance(key, str):
                    self._untracked_read(key, node)
        elif not isinstance(func, ast.Name):
            self._eval(func, env)
        self._eval_args(node, env, harmless=self._is_harmless(func))
        return None

    def _eval_args(self, node: ast.Call, env: _Env, *, harmless: bool) -> None:
        callee = ast.unparse(node.func)
        for arg in node.args:
            value = self._eval(arg.value if isinstance(arg, ast.Starred) else arg, env)
            if not harmless:
                self._escape(value, arg, f"passed to {callee}()")
        for kw in node.keywords:
            value = self._eval(kw.value, env)
            if not harmless:
                self._escape(value, kw.value, f"passed to {callee}()" if kw.arg else f"unpacked with ** into {callee}()")

    def _eval_json_method(self, node: ast.Call, owner: _Json, env: _Env) -> _Value:
        attr = node.func.attr  # type: ignore[attr-defined]
        if attr == "get":
            key = self._constant_key(node.args[0], env) if node.args else None
            if isinstance(key, str):
                for extra in node.args[1:]:
                    self._eval(extra, env)
                return self._access(owner, key, "get", node)
            self._eval_args(node, env, harmless=True)
            self._opaque(owner, node, "dynamic_key", f".get() with non-literal key in {ast.unparse(node)}")
            return None
        if attr in ("keys", "values", "items"):
            self._opaque(owner, node, "dynamic_key", f"iterates .{attr}()")
            return None
        if attr == "copy":
            return owner
        self._eval_args(node, env, harmless=True)
        return None

    def _eval_subscript(self, node: ast.Subscript, env: _Env) -> _Value:
        base = self._eval(node.value, env)
        key = self._constant_key(node.slice, env)
        if not isinstance(base, _Json):
            if isinstance(key, str):
                self._untracked_read(key, node)
            else:
                self._eval(node.slice, env)
            return None
        if isinstance(node.slice, ast.Slice):
            return base
        if key is _INDEX:
            return _Json(base.call_id, base.path + (ITEMS,))
        if isinstance(key, str):
            return self._access(base, key, "subscript", node)
        self._eval(node.slice, env)
        self._opaque(base, node, "dynamic_key", f"subscript with non-literal key `{ast.unparse(node.slice)}`")
        return None

    def _eval_compare(self, node: ast.Compare, env: _Env) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            container = self._eval(node.comparators[0], env)
            key = self._constant_key(node.left, env)
            if isinstance(container, _Json) and isinstance(key, str):
                self._access(container, key, "membership", node)
            else:
                self._eval(node.left, env)
            return
        self._visit_children(node, env)

    def _eval_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp, env: _Env
    ) -> None:
        comp_env = dict(env)
        for gen in node.generators:
            self._bind(gen.target, _element(self._eval(gen.iter, comp_env)), comp_env, node)
            for condition in gen.ifs:
                self._eval(condition, comp_env)
        if isinstance(node, ast.DictComp):
            self._eval(node.key, comp_env)
            self._eval(node.value, comp_env)
        else:
            self._eval(node.elt, comp_env)

    def _constant_key(self, node: ast.expr, env: _Env) -> str | object | None:
        if isinstance(node, ast.Slice):
            return _INDEX
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return _INDEX
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return _INDEX if isinstance(node.operand.value, int) else None
        if isinstance(node, ast.Name):
            bound = env.get(node.id)
            if isinstance(bound, _Url) and len(bound.parts) == 1 and isinstance(bound.parts[0], str):
                return bound.parts[0]
            if node.id not in env:
                return self.constants.get(node.id)
        return None

    # --- httpx ---------------------------------------------------------------

    def _httpx_function(self, func: ast.expr) -> str | None:
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self.module_aliases
            and func.attr in _REQUEST_FUNCTIONS
        ):
            return func.attr
        if isinstance(func, ast.Name):
            return self.function_aliases.get(func.id)
        return None

    def _client_from_call(self, node: ast.Call, env: _Env) -> _Client | None:
        func = node.func
        is_client = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in self.module_aliases
            and func.attr in _CLIENT_CLASSES
        ) or (isinstance(func, ast.Name) and func.id in self.client_aliases)
        if not is_client:
            return None
        base = next((kw.value for kw in node.keywords if kw.arg == "base_url"), None)
        return _Client(None if base is None else (self._url(base, env) or (None,)))

    def _is_harmless(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in _HARMLESS_FUNCTIONS
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.value.id in _LOGGER_NAMES or (func.value.id == "json" and func.attr == "dumps")
        return False

    def _record_request(self, node: ast.Call, fn: str, base_url: UrlTemplate | None, env: _Env) -> _Response:
        args = list(node.args)
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        unpacked = [kw.value for kw in node.keywords if kw.arg is None]

        method: str | None = fn.upper()
        if fn == "request":
            method_node = args.pop(0) if args else keywords.pop("method", None)
            method = None
            if isinstance(method_node, ast.Constant) and isinstance(method_node.value, str):
                method = method_node.value.upper()
            elif method_node is not None:
                self._eval(method_node, env)

        url_node = args.pop(0) if args else keywords.pop("url", None)
        url: UrlTemplate | None = None
        if url_node is not None:
            url = self._url(url_node, env)
            self._eval(url_node, env)
            if url is not None and base_url is not None and not _is_absolute(url):
                url = normalize_url(base_url + url)

        callee = ast.unparse(node.func)
        for value_node in [*args, *keywords.values(), *unpacked]:
            self._escape(self._eval(value_node, env), value_node, f"sent in {callee}()")

        call_id = next(self.ids)
        self.result.calls.append(
            HttpCall(
                call_id=call_id,
                location=self._loc(node),
                method=method,
                url=url,
                url_source=self._source_text(url_node),
            )
        )
        return _Response(call_id)

    # --- URL templates -------------------------------------------------------

    def _url_value(self, node: ast.expr, env: _Env) -> _Url | None:
        parts = self._url_parts(node, env)
        return _Url(normalize_url(parts)) if parts is not None else None

    def _url(self, node: ast.expr, env: _Env) -> UrlTemplate | None:
        parts = self._url_parts(node, env)
        if parts is None or all(p is None for p in parts):
            return None
        return normalize_url(parts)

    def _url_parts(self, node: ast.expr, env: _Env) -> tuple[UrlPart, ...] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        if isinstance(node, ast.Name):
            bound = env.get(node.id)
            if isinstance(bound, _Url):
                return bound.parts
            return (self.constants[node.id],) if node.id in self.constants and node.id not in env else None
        if isinstance(node, ast.JoinedStr):
            parts: list[UrlPart] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    parts.extend(self._url_parts(value.value, env) or (None,))
            return tuple(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._url_parts(node.left, env), self._url_parts(node.right, env)
            if left is None and right is None:
                return None
            return (left or (None,)) + (right or (None,))
        if _is_format_call(node):
            template: str = node.func.value.value  # type: ignore[attr-defined]
            pieces = _FORMAT_FIELD_RE.split(template)
            return tuple(itertools.chain.from_iterable((p, None) for p in pieces))[:-1]
        return None


def iter_python_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts[:-1])
    )


def analyze_repository(root: Path) -> RepoAnalysis:
    """Statically analyze every Python file under ``root``."""
    if not root.is_dir():
        raise SourceAnalysisError(f"Consumer path does not exist or is not a directory: {root}")

    result = RepoAnalysis(root=root)
    ids = itertools.count(1)
    for path in iter_python_files(root):
        rel_path = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
        except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as exc:
            logger.warning("could not parse consumer file", extra={"file": rel_path, "error": str(exc)})
            result.parse_errors.append(f"{rel_path}: {exc}")
            continue
        result.files_scanned += 1
        _FileAnalyzer(rel_path, source, tree, ids, result).run()

    logger.info(
        "consumer source analyzed",
        extra={"root": str(root), "files": result.files_scanned, "http_calls": len(result.calls)},
    )
    return result
