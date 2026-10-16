"""Benchmark scoring. Pure functions over per-case predictions.

Ground truth is binary: ``affected`` (the consumer's behaviour breaks) or ``not_affected``.
Predictions are the analyzer's consumer status, mapped as follows:

=========================  ==========================================================
Prediction                 Scored as
=========================  ==========================================================
affected                   positive
potentially_affected       positive in the primary view ("flagged for review");
                           an abstention in the secondary (strict) view
no_relevant_usage_found    negative. It is *not* proof of safety: when the truth is
                           affected it is a false negative
not_analyzed               negative (the analyzer reported no breaking changes, so it
                           declared the change compatible for every consumer)
uncertain                  abstention: never correct, excluded from precision, counted
                           against recall when the truth is affected, reported as a rate
error                      the analyzer run failed: never correct, counted like an
                           abstention, and reported separately
=========================  ==========================================================

precision = TP / (TP + FP)
recall    = TP / (number of truly affected cases)     # abstentions and errors are misses
F1        = harmonic mean of precision and recall
accuracy  = (TP + TN) / all cases                     # abstentions and errors are incorrect
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

PREDICTIONS = ("affected", "potentially_affected", "uncertain", "no_relevant_usage_found", "not_analyzed", "error")
TRUTHS = ("affected", "not_affected")
View = Literal["primary", "strict"]
Outcome = Literal["TP", "FP", "TN", "FN", "abstain", "error"]


def outcome(truth: str, predicted: str, view: View = "primary") -> Outcome:
    if truth not in TRUTHS or predicted not in PREDICTIONS:
        raise ValueError(f"unknown truth/prediction: {truth!r}/{predicted!r}")
    if predicted == "error":
        return "error"
    if predicted == "uncertain" or (predicted == "potentially_affected" and view == "strict"):
        return "abstain"
    positive = predicted in ("affected", "potentially_affected")
    if positive:
        return "TP" if truth == "affected" else "FP"
    return "FN" if truth == "affected" else "TN"


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def classification_metrics(cases: list[dict[str, Any]], view: View = "primary") -> dict[str, Any]:
    outcomes = Counter(outcome(c["truth"], c["predicted"], view) for c in cases)
    tp, fp, tn, fn = outcomes["TP"], outcomes["FP"], outcomes["TN"], outcomes["FN"]
    positives = sum(c["truth"] == "affected" for c in cases)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, positives)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "view": view,
        "cases": len(cases),
        "truly_affected": positives,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "abstentions": outcomes["abstain"],
        "abstentions_on_affected": sum(
            outcome(c["truth"], c["predicted"], view) == "abstain" and c["truth"] == "affected" for c in cases
        ),
        "errors": outcomes["error"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _ratio(tp + tn, len(cases)),
        "abstention_rate": _ratio(outcomes["abstain"], len(cases)),
    }


def uncertain_rate(cases: list[dict[str, Any]]) -> float | None:
    return _ratio(sum(c["predicted"] == "uncertain" for c in cases), len(cases))


def confusion_matrix(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {truth: {p: 0 for p in PREDICTIONS} for truth in TRUTHS}
    for case in cases:
        matrix[case["truth"]][case["predicted"]] += 1
    return matrix


def evidence_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Validity of every cited reference, and whether true positives cite the breaking line."""
    cited = sum(len(c["evidence"]) for c in cases)
    valid = sum(sum(e["valid"] for e in c["evidence"]) for c in cases)
    true_positives = [
        c for c in cases if c["truth"] == "affected" and c["predicted"] in ("affected", "potentially_affected")
    ]
    with_expected = [c for c in true_positives if c["expected_evidence"]]
    hits = sum(c["expected_evidence_hit"] for c in with_expected)
    return {
        "references_cited": cited,
        "references_valid": valid,
        "validity_rate": _ratio(valid, cited),
        "true_positives_with_expected_evidence": len(with_expected),
        "true_positives_citing_breaking_line": hits,
        "breaking_line_hit_rate": _ratio(hits, len(with_expected)),
    }


def tag_breakdown(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tags = sorted({t for c in cases for t in c["tags"]})
    breakdown = {}
    for tag in tags:
        subset = [c for c in cases if tag in c["tags"]]
        outcomes = Counter(outcome(c["truth"], c["predicted"]) for c in subset)
        breakdown[tag] = {
            "cases": len(subset),
            "correct": outcomes["TP"] + outcomes["TN"],
            "false_positives": outcomes["FP"],
            "false_negatives": outcomes["FN"],
            "uncertain": sum(c["predicted"] == "uncertain" for c in subset),
            "errors": outcomes["error"],
        }
    return breakdown
