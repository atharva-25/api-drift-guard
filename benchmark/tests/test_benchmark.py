"""Benchmark self-tests (run with `pytest benchmark/tests`; separate from the app's tests)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))

from metrics import classification_metrics, confusion_matrix, evidence_metrics, outcome  # noqa: E402
from oracle import OracleRequest, example_value, find_operation, provider_response, run_oracle  # noqa: E402
from run_benchmark import load_benchmark  # noqa: E402

SPEC = json.loads((BENCHMARK / "specs" / "v1.json").read_text())


@pytest.mark.parametrize(
    ("truth", "predicted", "view", "expected"),
    [
        ("affected", "affected", "primary", "TP"),
        ("affected", "potentially_affected", "primary", "TP"),
        ("affected", "potentially_affected", "strict", "abstain"),
        ("not_affected", "potentially_affected", "primary", "FP"),
        ("affected", "no_relevant_usage_found", "primary", "FN"),
        ("affected", "not_analyzed", "primary", "FN"),
        ("not_affected", "not_analyzed", "primary", "TN"),
        ("affected", "uncertain", "primary", "abstain"),
        ("not_affected", "uncertain", "primary", "abstain"),
        ("affected", "error", "primary", "error"),
    ],
)
def test_outcome_mapping(truth: str, predicted: str, view: str, expected: str) -> None:
    assert outcome(truth, predicted, view) == expected


def test_uncertain_is_never_correct_and_counts_against_recall() -> None:
    cases = [
        {"truth": "affected", "predicted": "affected"},
        {"truth": "affected", "predicted": "uncertain"},
        {"truth": "not_affected", "predicted": "uncertain"},
        {"truth": "not_affected", "predicted": "no_relevant_usage_found"},
    ]
    m = classification_metrics(cases)
    assert (m["precision"], m["recall"], m["accuracy"], m["abstention_rate"]) == (1.0, 0.5, 0.5, 0.5)
    assert m["f1"] == round(2 * 1.0 * 0.5 / 1.5, 4)
    assert confusion_matrix(cases)["affected"]["uncertain"] == 1


def test_metrics_with_no_positive_predictions() -> None:
    m = classification_metrics([{"truth": "affected", "predicted": "uncertain"}])
    assert m["precision"] is None and m["recall"] == 0.0 and m["f1"] is None


def test_evidence_metrics() -> None:
    cases = [
        {"truth": "affected", "predicted": "affected", "expected_evidence": ["a.py:1"], "expected_evidence_hit": True,
         "evidence": [{"valid": True}, {"valid": False}]},
        {"truth": "affected", "predicted": "uncertain", "expected_evidence": ["b.py:2"], "expected_evidence_hit": False,
         "evidence": [{"valid": True}]},
    ]
    e = evidence_metrics(cases)
    assert (e["references_cited"], e["references_valid"], e["breaking_line_hit_rate"]) == (3, 2, 1.0)


def test_example_generation_includes_optional_and_nested_fields() -> None:
    status, body = provider_response(SPEC, "GET", "/users/42")
    assert status == 200
    assert body["nickname"] == "nickname-example"  # optional, nullable -> present
    assert body["address"] == {"city": "city-example", "zip": "zip-example"}
    assert isinstance(body["user_id"], int)
    status, listing = provider_response(SPEC, "GET", "/users")
    assert status == 200 and isinstance(listing, list) and listing[0]["plan"] == "plan-example"
    assert provider_response(SPEC, "GET", "/orders/1")[0] == 404
    assert find_operation(SPEC, "POST", "/users") is None
    assert example_value({"anyOf": [{"type": "null"}, {"type": "integer"}]}, SPEC) == 1


def test_oracle_detects_breakage_and_isolates_runs() -> None:
    bench = load_benchmark()
    consumer = bench.consumers["shipping"]

    def run(spec: str) -> dict:
        return run_oracle(OracleRequest(
            spec=str(BENCHMARK / "specs" / spec), consumer_root=str(bench.resolve(consumer.path)),
            module=consumer.module, expr=consumer.expr,
            provider_hosts=bench.oracle.provider_hosts, external_responses=bench.oracle.external_responses,
        ))

    assert run("v1.json") == {"ok": True, "error": None, "result": "'name-example, city-example zip-example'"}
    assert run("zip_removed.json") == {"ok": False, "error": "KeyError: 'zip'"}


def test_benchmark_definition_is_consistent() -> None:
    bench = load_benchmark()
    cases = [c for s in bench.scenarios for c in s.cases]
    assert 30 <= len(cases) <= 40
    assert all(bench.resolve(s.old_spec).is_file() and bench.resolve(s.new_spec).is_file() for s in bench.scenarios)
    assert all(bench.resolve(c.path).is_dir() for c in bench.consumers.values())
    for scenario in bench.scenarios:
        for case in scenario.cases:
            root = bench.resolve(bench.consumers[case.consumer].path)
            for ref in case.evidence:
                file, line = ref.rsplit(":", 1)
                assert int(line) <= len((root / file).read_text().splitlines()), ref


def test_benchmark_does_not_import_the_analyzer() -> None:
    imports = re.compile(r"^\s*(from|import)\s+api_drift_guard\b", re.MULTILINE)
    offenders = [p.name for p in BENCHMARK.rglob("*.py") if imports.search(p.read_text())]
    assert offenders == []
