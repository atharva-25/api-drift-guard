"""Markdown comparison report built only from stored benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODES = ("deterministic", "full")
LABELS = {
    "affected": "affected",
    "potentially_affected": "potentially",
    "uncertain": "uncertain",
    "no_relevant_usage_found": "no usage",
    "not_analyzed": "not analyzed",
    "error": "error",
}
BEYOND_AST = ["dynamic-access", "cross-function", "model-validation", "attribute-access",
              "host-mismatch", "guarded-access", "unsupported-client", "tolerant-access"]


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value:.1%}"
    return f"{value:,}" if isinstance(value, int) else f"{value}"


def _load(results_dir: Path) -> dict[str, dict[str, Any]]:
    loaded = {}
    for mode in MODES:
        path = results_dir / f"{mode}.json"
        if path.is_file():
            loaded[mode] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _row(name: str, values: list[str]) -> str:
    return f"| {name} | " + " | ".join(values) + " |"


def _metric_table(results: dict[str, dict[str, Any]], view: str) -> list[str]:
    modes = [m for m in MODES if m in results]
    m = {mode: results[mode]["metrics"] for mode in modes}
    lines = ["| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for label, key, pct in [
        ("Precision", "precision", True), ("Recall", "recall", True), ("F1", "f1", True),
        ("Accuracy (abstentions count as wrong)", "accuracy", True),
        ("True positives", "true_positives", False), ("False positives", "false_positives", False),
        ("True negatives", "true_negatives", False), ("False negatives", "false_negatives", False),
        ("Abstentions", "abstentions", False), ("…of which truly affected", "abstentions_on_affected", False),
        ("Analyzer errors", "errors", False),
    ]:
        lines.append(_row(label, [_fmt(m[mode][view][key], pct) for mode in modes]))
    return lines


def _confusion(result: dict[str, Any]) -> list[str]:
    matrix = result["metrics"]["confusion_matrix"]
    columns = [p for p in LABELS if any(matrix[t][p] for t in matrix)]
    lines = ["| Truth \\ Predicted | " + " | ".join(LABELS[c] for c in columns) + " |",
             "|---|" + "---|" * len(columns)]
    for truth, row in matrix.items():
        lines.append(_row(truth, [str(row[c]) for c in columns]))
    return lines


def write_comparison(results_dir: Path) -> Path:
    results = _load(results_dir)
    if "deterministic" not in results:
        raise SystemExit("no deterministic results; run `run --mode deterministic` first")
    det = results["deterministic"]
    modes = [m for m in MODES if m in results]
    cases = det["metrics"]["primary"]["cases"]
    affected = det["metrics"]["primary"]["truly_affected"]
    scenarios = len(det["scenarios"])

    lines = [
        "# API Drift Guard Benchmark: Deterministic vs Full Analysis",
        "",
        f"{cases} consumer-impact cases across {scenarios} OpenAPI changes "
        f"({affected} truly affected, {cases - affected} not affected). Every label was "
        f"hand-written and confirmed by executing the consumer against the old and new contract "
        f"({det['ground_truth_verification']['agreements']}/{det['ground_truth_verification']['cases']} agreed).",
        "",
        "- **deterministic**: OpenAPI diff (oasdiff) + Python AST analysis (`--mode static`)",
        "- **full**: the above + Google ADK investigation agents on Groq (`--mode agentic`)",
        "",
    ]
    if "full" not in results:
        lines += ["> The full mode has not been run, so only deterministic results are shown.", ""]

    lines += ["## Run details", "", "| | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for label, getter in [
        ("Finished (UTC)", lambda r: r["finished_at"][:19].replace("T", " ")),
        ("Git commit", lambda r: (r["environment"]["git"]["commit"] or "n/a")[:10]
         + (" (dirty)" if r["environment"]["git"]["dirty"] else "")),
        ("Analyzer / oasdiff", lambda r: f"{r['environment']['analyzer_version']} / {r['environment']['oasdiff_version']}"),
        ("Python", lambda r: r["environment"]["python"]),
        ("Models", lambda r: ", ".join(
            f"`{x}`" for x in filter(None, [(r["configuration"]["llm_settings"] or {}).get("model"),
                                             (r["configuration"]["llm_settings"] or {}).get("escalation_model")])) or "none"),
        ("Temperature / reasoning effort", lambda r: "n/a" if not r["configuration"]["llm_settings"] else
         f"{r['configuration']['llm_settings']['temperature']} / {r['configuration']['llm_settings']['reasoning_effort']}"),
        ("Cases file SHA-256", lambda r: r["configuration"]["cases_file_sha256"][:12]),
    ]:
        lines.append(_row(label, [getter(results[m]) for m in modes]))

    lines += ["", "## Classification (primary view: `potentially_affected` = positive)", "",
              *_metric_table(results, "primary"), "",
              "## Strict view (`potentially_affected` = abstention)", "", *_metric_table(results, "strict"), ""]

    lines += ["## Uncertainty, evidence, cost", "", "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for label, getter in [
        ("Uncertain prediction rate", lambda r: _fmt(r["metrics"]["uncertain_rate"], True)),
        ("Cited references that are real (file, line, text)", lambda r: (
            f"{r['metrics']['evidence']['references_valid']}/{r['metrics']['evidence']['references_cited']} "
            f"({_fmt(r['metrics']['evidence']['validity_rate'], True)})")),
        ("True positives citing the breaking line", lambda r: (
            f"{r['metrics']['evidence']['true_positives_citing_breaking_line']}/"
            f"{r['metrics']['evidence']['true_positives_with_expected_evidence']} "
            f"({_fmt(r['metrics']['evidence']['breaking_line_hit_rate'], True)})")),
        ("Mean time per scenario run (s)", lambda r: _fmt(r["metrics"]["timing"]["mean_seconds_per_scenario_run"])),
        ("Mean time per case (s)", lambda r: _fmt(r["metrics"]["timing"]["mean_seconds_per_case"])),
        ("Agents spawned", lambda r: _fmt(r["metrics"]["llm"]["agents_spawned"])),
        ("LLM requests", lambda r: _fmt(r["metrics"]["llm"]["requests"])),
        ("Tokens (prompt / completion)", lambda r: f"{r['metrics']['llm']['prompt_tokens']:,} / {r['metrics']['llm']['completion_tokens']:,}"),
        ("Failed investigations (rate-limited)", lambda r: f"{r['metrics']['llm']['failed_investigations']} ({r['metrics']['llm']['rate_limit_failures']})"),
        ("Analyzer run errors", lambda r: _fmt(len(r["metrics"]["scenario_errors"]))),
    ]:
        lines.append(_row(label, [getter(results[m]) for m in modes]))
    if "full" in results and results["full"]["metrics"]["llm"]["by_model"]:
        lines += ["", "LLM usage by model (full mode):", "", "| Model | Agents | Requests | Prompt tokens | Completion tokens |",
                  "|---|---|---|---|---|"]
        for model, u in results["full"]["metrics"]["llm"]["by_model"].items():
            lines.append(f"| `{model}` | {u['agents']} | {u['requests']} | {u['prompt_tokens']:,} | {u['completion_tokens']:,} |")

    for mode in modes:
        lines += ["", f"## Confusion matrix: {mode}", "", *_confusion(results[mode])]

    lines += ["", "## By access pattern", "",
              "Correct = TP + TN. Patterns marked † need information that AST analysis of literal key "
              "reads within a single function cannot provide.", "",
              "| Pattern | Cases | " + " | ".join(f"{m} correct | {m} uncertain" for m in modes) + " |",
              "|---|---|" + "---|---|" * len(modes)]
    for tag, d in det["metrics"]["by_tag"].items():
        values = []
        for mode in modes:
            t = results[mode]["metrics"]["by_tag"][tag]
            values += [f"{t['correct']}/{t['cases']}", str(t["uncertain"])]
        lines.append(_row(f"{tag}{' †' if tag in BEYOND_AST else ''}", [str(d["cases"]), *values]))

    lines += ["", "## Per-case predictions", "",
              "| Scenario | Consumer | Truth | " + " | ".join(modes) + " |", "|---|---|---|" + "---|" * len(modes)]
    by_mode = {m: {(p["scenario"], p["consumer"]): p for s in results[m]["scenarios"] for p in s["predictions"]} for m in modes}
    for key, p in by_mode["deterministic"].items():
        cells = []
        for mode in modes:
            q = by_mode[mode].get(key)
            if q is None:
                cells.append("-")
                continue
            ok = q["truth"] == "affected" and q["predicted"] in ("affected", "potentially_affected") or (
                q["truth"] == "not_affected" and q["predicted"] in ("no_relevant_usage_found", "not_analyzed"))
            cells.append(f"{LABELS[q['predicted']]} {'✓' if ok else '✗'}")
        lines.append(_row(key[0], [key[1], p["truth"], *cells]))

    errors = [(m, e) for m in modes for e in results[m]["metrics"]["scenario_errors"]]
    failed = [(m, s["id"], i) for m in modes for s in results[m]["scenarios"] for i in s["investigations"] if i["status"] == "failed"]
    lines += ["", "## Errors and rate-limit failures", ""]
    if not errors and not failed:
        lines.append("None recorded.")
    lines += [f"- {m} / {e['scenario']}: analyzer error: {e['error'][:300]}" for m, e in errors]
    lines += [f"- {m} / {sid}: {i['agent_name']} ({i['model']}) failed after {i['attempts']} attempt(s): {i['error']}"
              for m, sid, i in failed]

    lines += [
        "", "## Scoring rules", "",
        "- Ground truth is binary (`affected` / `not_affected`), hand-labelled and confirmed by the executable oracle.",
        "- `affected` and `potentially_affected` are positive predictions (strict view: `potentially_affected` is an abstention).",
        "- `no_relevant_usage_found` is a negative prediction and is never taken as proof of safety: on an affected "
        "consumer it is a false negative.",
        "- `uncertain` is an abstention: never correct, excluded from precision, counted as a miss for recall, reported as a rate.",
        "- `not_analyzed` (the analyzer found no breaking change) is a negative prediction; analyzer errors are never correct.",
        "", "## Limitations of this benchmark", "",
        "- 39 cases over constructed consumers. Each pattern is realistic, but this is not a sample of real-world code, "
        "so the metrics show the behaviour on these patterns, not production accuracy.",
        "- The full mode was run once. LLM output is non-deterministic even at temperature 0, and the analyzer has no seed setting."
        if "full" in results else "- The full mode has not been run yet.",
        "- Execution time includes analyzer start-up and oasdiff per scenario, and LLM latency in full mode.",
        "",
    ]
    path = results_dir / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
