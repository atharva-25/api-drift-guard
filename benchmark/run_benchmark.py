"""API Drift Guard consumer-impact benchmark runner.

    python benchmark/run_benchmark.py verify                      # check ground truth with the oracle
    python benchmark/run_benchmark.py run --mode deterministic    # OpenAPI diff + AST only
    python benchmark/run_benchmark.py run --mode full             # + Google ADK/Groq agents
    python benchmark/run_benchmark.py report                      # results/comparison.md

The analyzer is treated as a black box: each scenario is one ``api-drift-guard analyze``
subprocess, and predictions are read from its JSON report. Nothing here imports
``api_drift_guard``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (  # noqa: E402
    classification_metrics,
    confusion_matrix,
    evidence_metrics,
    tag_breakdown,
    uncertain_rate,
)
from oracle import OracleRequest, run_oracle  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
CASES_FILE = BENCHMARK_DIR / "cases.yaml"
RESULTS_DIR = BENCHMARK_DIR / "results"
TOOL_MODE = {"deterministic": "static", "full": "agentic"}
SCENARIO_TIMEOUT_SECONDS = 900


# --- benchmark definition --------------------------------------------------------------


class Consumer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Path
    module: str
    expr: str


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    consumer: str
    truth: Literal["affected", "not_affected"]
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _affected_cases_name_the_breaking_line(self) -> Case:
        if self.truth == "affected" and not self.evidence:
            raise ValueError(f"{self.consumer}: affected cases must list the breaking line(s)")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    old_spec: Path
    new_spec: Path
    change: str
    tags: list[str] = Field(default_factory=list)
    cases: list[Case]


class OracleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_hosts: list[str]
    external_responses: dict[str, Any] = Field(default_factory=dict)


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    tool_llm_settings: dict[str, Any]
    oracle: OracleSettings
    consumers: dict[str, Consumer]
    scenarios: list[Scenario]

    @model_validator(mode="after")
    def _consistent(self) -> Benchmark:
        ids = [s.id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        for scenario in self.scenarios:
            names = [c.consumer for c in scenario.cases]
            if len(names) != len(set(names)):
                raise ValueError(f"{scenario.id}: a consumer appears twice")
            unknown = set(names) - set(self.consumers)
            if unknown:
                raise ValueError(f"{scenario.id}: unknown consumers {sorted(unknown)}")
        return self

    def resolve(self, path: Path) -> Path:
        return (CASES_FILE.parent / path).resolve()


def load_benchmark() -> Benchmark:
    return Benchmark.model_validate(yaml.safe_load(CASES_FILE.read_text(encoding="utf-8")))


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


# --- ground-truth verification ------------------------------------------------------------


def verify_ground_truth(bench: Benchmark) -> dict[str, Any]:
    """Run every consumer against the old and new contract; compare with the labels."""
    checks = []
    for scenario in bench.scenarios:
        for case in scenario.cases:
            consumer = bench.consumers[case.consumer]

            def run(spec: Path) -> dict[str, Any]:
                return run_oracle(
                    OracleRequest(
                        spec=str(bench.resolve(spec)),
                        consumer_root=str(bench.resolve(consumer.path)),
                        module=consumer.module,
                        expr=consumer.expr,
                        provider_hosts=bench.oracle.provider_hosts,
                        external_responses=bench.oracle.external_responses,
                    )
                )

            before, after = run(scenario.old_spec), run(scenario.new_spec)
            oracle_truth = "affected" if before["ok"] and not after["ok"] else "not_affected"
            checks.append(
                {
                    "scenario": scenario.id,
                    "consumer": case.consumer,
                    "label": case.truth,
                    "works_on_old_contract": before["ok"],
                    "old_error": before["error"],
                    "new_error": after["error"],
                    "oracle": oracle_truth if before["ok"] else "invalid",
                    "agrees": before["ok"] and oracle_truth == case.truth,
                }
            )
    return {
        "cases": len(checks),
        "agreements": sum(c["agrees"] for c in checks),
        "disagreements": [c for c in checks if not c["agrees"]],
        "checks": checks,
    }


# --- running the analyzer -----------------------------------------------------------------


def _tool_command() -> list[str]:
    script = Path(sys.executable).parent / "api-drift-guard"
    if script.is_file():
        return [str(script)]
    found = shutil.which("api-drift-guard")
    return [found] if found else [sys.executable, "-m", "api_drift_guard.cli"]


def _tool_env() -> dict[str, str]:
    """Environment for the analyzer, without overrides that would change its settings."""
    return {k: v for k, v in os.environ.items() if not k.startswith("API_DRIFT_GUARD_")}


def _write_config(bench: Benchmark, scenario: Scenario, path: Path) -> None:
    config = {
        "provider": {"name": "user-service"},
        "consumers": [
            {"name": c.consumer, "path": str(bench.resolve(bench.consumers[c.consumer].path))} for c in scenario.cases
        ],
        "llm": bench.tool_llm_settings,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _check_reference(root: Path, file: str, line: int, snippet: str) -> bool:
    """Independent evidence check: real file inside the consumer, real line, matching text."""
    path = (root / file).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= line <= len(lines):
        return False
    cited = _normalize(snippet)
    return bool(cited) and cited in _normalize(lines[line - 1])


def _most_severe(classifications: list[str]) -> str:
    order = ["affected", "potentially_affected", "uncertain", "no_relevant_usage_found"]
    return min(classifications, key=order.index) if classifications else "no_relevant_usage_found"


def _predictions(bench: Benchmark, scenario: Scenario, report: dict[str, Any] | None) -> list[dict[str, Any]]:
    consumers = {c["consumer"]: c for c in (report or {}).get("consumers", [])}
    rows = []
    for case in scenario.cases:
        root = bench.resolve(bench.consumers[case.consumer].path)
        impact = consumers.get(case.consumer)
        findings = impact["findings"] if impact else []
        if report is None:
            predicted = "error"
        elif impact is not None:
            predicted = impact["status"]
        elif report.get("status") == "no_breaking_changes":
            predicted = "not_analyzed"
        else:
            predicted = "error"
        evidence = [
            {
                "file": e["file"], "line": e["line"], "snippet": e["snippet"], "note": e["note"],
                "valid": _check_reference(root, e["file"], e["line"], e["snippet"]),
            }
            for f in findings for e in f["evidence"]
        ]
        cited = {f"{e['file']}:{e['line']}" for e in evidence}
        rows.append(
            {
                "scenario": scenario.id,
                "consumer": case.consumer,
                "truth": case.truth,
                "predicted": predicted,
                "static_prediction": _most_severe([f.get("static_classification") or f["classification"] for f in findings])
                if findings else predicted,
                "analysis_methods": sorted({f["analysis_method"] for f in findings}),
                "confidences": [f["confidence"] for f in findings if f.get("confidence") is not None],
                "tags": sorted(set(case.tags) | set(scenario.tags)),
                "expected_evidence": case.evidence,
                "expected_evidence_hit": any(ref in cited for ref in case.evidence) if case.evidence else None,
                "evidence": evidence,
                "reasons": [f["reason"] for f in findings if f["classification"] != "no_relevant_usage_found"],
            }
        )
    return rows


def run_scenario(bench: Benchmark, scenario: Scenario, mode: str, raw_dir: Path) -> dict[str, Any]:
    config_path = raw_dir / f"{scenario.id}.config.yaml"
    report_path = raw_dir / f"{scenario.id}.report.json"
    _write_config(bench, scenario, config_path)
    report_path.unlink(missing_ok=True)
    cmd = [
        *_tool_command(), "analyze",
        "--old", str(bench.resolve(scenario.old_spec)),
        "--new", str(bench.resolve(scenario.new_spec)),
        "--config", str(config_path),
        "--mode", TOOL_MODE[mode],
        "-o", str(report_path),
        "--log-level", "WARNING",
    ]
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=_tool_env(), capture_output=True, text=True,
            timeout=SCENARIO_TIMEOUT_SECONDS, check=False,
        )
        exit_code, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, stderr = None, f"timed out after {SCENARIO_TIMEOUT_SECONDS}s: {exc}"
    duration = round(time.perf_counter() - started, 3)

    report = json.loads(report_path.read_text()) if exit_code == 0 and report_path.is_file() else None
    investigations = (report or {}).get("investigations", [])
    return {
        "id": scenario.id,
        "change": scenario.change,
        "duration_seconds": duration,
        "exit_code": exit_code,
        "error": None if report is not None else (stderr.strip()[-2000:] or "no report produced"),
        "tool_version": (report or {}).get("tool_version"),
        "oasdiff_version": (report or {}).get("oasdiff_version"),
        "report_status": (report or {}).get("status"),
        "llm_usage": (report or {}).get("llm_usage"),
        "investigations": [
            {k: i[k] for k in ("consumer", "agent_name", "model", "escalation", "status", "attempts",
                               "prompt_tokens", "completion_tokens", "rejected_evidence", "error", "duration_seconds")}
            for i in investigations
        ],
        "rate_limit_failures": sum("RateLimit" in (i.get("error") or "") for i in investigations),
        "failed_investigations": sum(i["status"] == "failed" for i in investigations),
        "predictions": _predictions(bench, scenario, report),
    }


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _credentials_available() -> bool:
    if os.environ.get("GROQ_API_KEY", "").strip():
        return True
    try:
        from dotenv import dotenv_values
    except ImportError:
        return False
    return bool((dotenv_values(REPO_ROOT / ".env").get("GROQ_API_KEY") or "").strip())


def summarize(predictions: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    runs = len(scenarios)
    total_time = sum(s["duration_seconds"] for s in scenarios)
    usage = [s["llm_usage"] or {} for s in scenarios]
    by_model: dict[str, dict[str, int]] = {}
    for s in scenarios:
        for i in s["investigations"]:
            model = by_model.setdefault(i["model"], {"agents": 0, "requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
            model["agents"] += 1
            model["requests"] += i["attempts"]
            model["prompt_tokens"] += i["prompt_tokens"]
            model["completion_tokens"] += i["completion_tokens"]
    return {
        "primary": classification_metrics(predictions, "primary"),
        "strict": classification_metrics(predictions, "strict"),
        "uncertain_rate": uncertain_rate(predictions),
        "confusion_matrix": confusion_matrix(predictions),
        "evidence": evidence_metrics(predictions),
        "by_tag": tag_breakdown(predictions),
        "timing": {
            "scenario_runs": runs,
            "total_seconds": round(total_time, 3),
            "mean_seconds_per_scenario_run": round(total_time / runs, 3) if runs else None,
            "mean_seconds_per_case": round(total_time / len(predictions), 3) if predictions else None,
        },
        "llm": {
            "agents_spawned": sum(u.get("agents_spawned", 0) for u in usage),
            "requests": sum(u.get("llm_requests", 0) for u in usage),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "by_model": by_model,
            "failed_investigations": sum(s["failed_investigations"] for s in scenarios),
            "rate_limit_failures": sum(s["rate_limit_failures"] for s in scenarios),
        },
        "scenario_errors": [{"scenario": s["id"], "error": s["error"]} for s in scenarios if s["error"]],
    }


def run_benchmark(mode: str, only: set[str] | None, pause_seconds: float) -> Path:
    bench = load_benchmark()
    if mode == "full" and not _credentials_available():
        raise SystemExit("full mode needs GROQ_API_KEY (environment or .env); not running.")

    verification = verify_ground_truth(bench)
    if verification["disagreements"]:
        print(json.dumps(verification["disagreements"], indent=2), file=sys.stderr)
        raise SystemExit("ground truth and oracle disagree; fix the benchmark before running it.")

    scenarios = [s for s in bench.scenarios if only is None or s.id in only]
    raw_dir = RESULTS_DIR / "raw" / mode
    raw_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    results = []
    for index, scenario in enumerate(scenarios):
        print(f"[{mode}] {scenario.id} ({len(scenario.cases)} cases)...", file=sys.stderr, flush=True)
        result = run_scenario(bench, scenario, mode, raw_dir)
        results.append(result)
        used_llm = bool((result["llm_usage"] or {}).get("llm_requests"))
        if mode == "full" and used_llm and pause_seconds and index < len(scenarios) - 1:
            time.sleep(pause_seconds)  # stay under the provider's tokens-per-minute limit

    predictions = [p for r in results for p in r["predictions"]]
    consumer_files = [p for c in bench.consumers.values() for p in bench.resolve(c.path).rglob("*.py")]
    spec_files = sorted({bench.resolve(s.old_spec) for s in scenarios} | {bench.resolve(s.new_spec) for s in scenarios})
    output = {
        "benchmark_version": bench.version,
        "mode": mode,
        "analyzer_mode": TOOL_MODE[mode],
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "analyzer_command": " ".join(Path(c).name for c in _tool_command()),
            "analyzer_version": next((r["tool_version"] for r in results if r["tool_version"]), None),
            "oasdiff_version": next((r["oasdiff_version"] for r in results if r["oasdiff_version"]), None),
            "git": _git_state(),
        },
        "configuration": {
            "cases_file": _rel(CASES_FILE),
            "cases_file_sha256": _sha256([CASES_FILE]),
            "consumers_sha256": _sha256(consumer_files),
            "specs_sha256": _sha256(spec_files),
            "scenario_filter": sorted(only) if only else None,
            "llm_settings": bench.tool_llm_settings if mode == "full" else None,
            "pause_seconds_between_llm_scenarios": pause_seconds if mode == "full" else None,
            "seed": "not configurable in the analyzer; temperature fixed at "
            f"{bench.tool_llm_settings.get('temperature')}" if mode == "full" else None,
        },
        "ground_truth_verification": {k: verification[k] for k in ("cases", "agreements")},
        "metrics": summarize(predictions, results),
        "scenarios": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{mode}.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check every ground-truth label with the executable oracle")
    run = sub.add_parser("run", help="run the benchmark in one mode")
    run.add_argument("--mode", choices=sorted(TOOL_MODE), required=True)
    run.add_argument("--scenarios", help="comma-separated scenario ids (default: all)")
    run.add_argument("--pause-seconds", type=float, default=20.0,
                     help="full mode: pause after scenarios that used the LLM (default 20)")
    sub.add_parser("report", help="write results/comparison.md from results/*.json")
    args = parser.parse_args()

    if args.command == "verify":
        verification = verify_ground_truth(load_benchmark())
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / "ground_truth_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        print(f"{verification['agreements']}/{verification['cases']} labels confirmed by the oracle")
        for d in verification["disagreements"]:
            print(f"  DISAGREE {d['scenario']} / {d['consumer']}: label={d['label']} oracle={d['oracle']} "
                  f"(old: {d['old_error']}, new: {d['new_error']})")
        return 1 if verification["disagreements"] else 0
    if args.command == "run":
        only = set(args.scenarios.split(",")) if args.scenarios else None
        path = run_benchmark(args.mode, only, args.pause_seconds)
        print(f"wrote {_rel(path)}")
        return 0
    from report import write_comparison

    print(f"wrote {_rel(write_comparison(RESULTS_DIR))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
