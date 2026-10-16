"""Executable ground-truth oracle.

Runs a consumer's real code with HTTP intercepted: requests to the provider host get
a JSON body generated from an OpenAPI spec's response schema (every declared property
present, deterministic example values); other hosts get fixed bodies from the
benchmark file. Running a consumer against the old and the new spec shows whether the
contract change actually breaks it.

Independent of the analyzer under test: it imports nothing from ``api_drift_guard``.
Invoked in a fresh subprocess per run, so consumer modules never leak between cases:

    python benchmark/oracle.py --request '<json>'   ->  prints {"ok": bool, "error": str|null}
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- example generation from OpenAPI schemas ------------------------------------------


def _resolve(schema: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        node: Any = spec
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        schema = node
    return schema


def example_value(schema: dict[str, Any], spec: dict[str, Any], name: str = "value") -> Any:
    """A deterministic example that satisfies ``schema``, including all optional properties."""
    schema = _resolve(schema, spec)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [s for s in schema[key] if _resolve(s, spec).get("type") != "null"]
            return example_value(options[0] if options else schema[key][0], spec, name)
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for part in schema["allOf"]:
            merged.update(example_value(part, spec, name))
        return merged
    kind = schema.get("type", "object" if "properties" in schema else "string")
    if kind == "object":
        return {prop: example_value(sub, spec, prop) for prop, sub in schema.get("properties", {}).items()}
    if kind == "array":
        return [example_value(schema.get("items", {}), spec, name)]
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.5
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return f"{name}-example"


def _segments(path: str) -> list[str]:
    return [s for s in path.split("/") if s]


def find_operation(spec: dict[str, Any], method: str, path: str) -> dict[str, Any] | None:
    """Most specific operation whose path template matches ``path``."""
    request = _segments(path)
    best: tuple[int, dict[str, Any]] | None = None
    for template, item in spec.get("paths", {}).items():
        operation = item.get(method.lower())
        parts = _segments(template)
        if operation is None or len(parts) != len(request):
            continue
        literal = 0
        for tpl, seg in zip(parts, request):
            if tpl.startswith("{") and tpl.endswith("}"):
                continue
            if tpl != seg:
                break
            literal += 1
        else:
            if best is None or literal > best[0]:
                best = (literal, operation)
    return best[1] if best else None


def provider_response(spec: dict[str, Any], method: str, path: str) -> tuple[int, Any]:
    operation = find_operation(spec, method, path)
    if operation is None:
        return 404, {"detail": "Not Found"}
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    return 200, example_value(schema, spec)


# --- running a consumer ------------------------------------------------------------------


@dataclass(frozen=True)
class OracleRequest:
    spec: str
    consumer_root: str
    module: str
    expr: str
    provider_hosts: list[str]
    external_responses: dict[str, Any]


def _install_http_fakes(request: OracleRequest) -> None:
    import httpx

    spec = json.loads(Path(request.spec).read_text(encoding="utf-8"))

    def respond(host: str, method: str, path: str) -> tuple[int, Any]:
        if host in request.provider_hosts:
            return provider_response(spec, method, path)
        if host in request.external_responses:
            return 200, request.external_responses[host]
        return 503, {"detail": f"unknown host {host}"}

    def handle(self: Any, req: httpx.Request) -> httpx.Response:
        status, body = respond(req.url.host, req.method, req.url.path)
        return httpx.Response(status, json=body, request=req)

    async def handle_async(self: Any, req: httpx.Request) -> httpx.Response:
        return handle(self, req)

    httpx.HTTPTransport.handle_request = handle  # type: ignore[method-assign]
    httpx.AsyncHTTPTransport.handle_async_request = handle_async  # type: ignore[method-assign]

    try:
        import requests
        from urllib.parse import urlsplit
    except ImportError:  # pragma: no cover - requests is optional
        return

    def send(self: Any, prepared: Any, **kwargs: Any) -> Any:
        parts = urlsplit(prepared.url)
        status, body = respond(parts.hostname or "", prepared.method, parts.path)
        response = requests.models.Response()
        response.status_code = status
        response._content = json.dumps(body).encode()
        response.headers["Content-Type"] = "application/json"
        response.url = prepared.url
        response.request = prepared
        return response

    requests.Session.send = send  # type: ignore[method-assign]


def run_in_process(request: OracleRequest) -> dict[str, Any]:
    _install_http_fakes(request)
    sys.path.insert(0, request.consumer_root)
    try:
        module = importlib.import_module(request.module)
        result = eval(request.expr, vars(module))  # noqa: S307 - benchmark-authored expressions only
        if inspect.iscoroutine(result):
            result = asyncio.run(result)
    except Exception as exc:  # noqa: BLE001 - any failure means the consumer broke
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "error": None, "result": repr(result)[:200]}


def run_oracle(request: OracleRequest, timeout: float = 60.0) -> dict[str, Any]:
    """Run one oracle check in a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, __file__, "--request", json.dumps(request.__dict__)],
        capture_output=True, text=True, timeout=timeout, check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": f"oracle process failed: {proc.stderr.strip()[-500:]}"}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    print(json.dumps(run_in_process(OracleRequest(**json.loads(args.request)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
