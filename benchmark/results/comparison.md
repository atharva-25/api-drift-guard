# API Drift Guard Benchmark: Deterministic vs Full Analysis

39 consumer-impact cases across 12 OpenAPI changes (17 truly affected, 22 not affected). Every label was hand-written and confirmed by executing the consumer against the old and new contract (39/39 agreed).

- **deterministic**: OpenAPI diff (oasdiff) + Python AST analysis (`--mode static`)
- **full**: the above + Google ADK investigation agents on Groq (`--mode agentic`)

## Run details

| | deterministic | full |
|---|---|---|
| Finished (UTC) | 2026-10-16 16:47:01 | 2026-10-16 16:52:05 |
| Git commit | 12fdb5e8c0 (dirty) | 12fdb5e8c0 (dirty) |
| Analyzer / oasdiff | 0.1.0 / 1.33.0 | 0.1.0 / 1.33.0 |
| Python | 3.13.16 | 3.13.16 |
| Models | none | `groq/openai/gpt-oss-20b`, `groq/openai/gpt-oss-120b` |
| Temperature / reasoning effort | n/a | 0.0 / low |
| Cases file SHA-256 | 83e1e7d3199f | 83e1e7d3199f |

## Classification (primary view: `potentially_affected` = positive)

| Metric | deterministic | full |
|---|---|---|
| Precision | 69.2% | 76.5% |
| Recall | 52.9% | 76.5% |
| F1 | 60.0% | 76.5% |
| Accuracy (abstentions count as wrong) | 56.4% | 74.4% |
| True positives | 9 | 13 |
| False positives | 4 | 4 |
| True negatives | 13 | 16 |
| False negatives | 1 | 1 |
| Abstentions | 12 | 5 |
| …of which truly affected | 7 | 3 |
| Analyzer errors | 0 | 0 |

## Strict view (`potentially_affected` = abstention)

| Metric | deterministic | full |
|---|---|---|
| Precision | 72.7% | 81.2% |
| Recall | 47.1% | 76.5% |
| F1 | 57.1% | 78.8% |
| Accuracy (abstentions count as wrong) | 53.8% | 74.4% |
| True positives | 8 | 13 |
| False positives | 3 | 3 |
| True negatives | 13 | 16 |
| False negatives | 1 | 1 |
| Abstentions | 14 | 6 |
| …of which truly affected | 8 | 3 |
| Analyzer errors | 0 | 0 |

## Uncertainty, evidence, cost

| Metric | deterministic | full |
|---|---|---|
| Uncertain prediction rate | 30.8% | 12.8% |
| Cited references that are real (file, line, text) | 58/58 (100.0%) | 65/65 (100.0%) |
| True positives citing the breaking line | 9/9 (100.0%) | 13/13 (100.0%) |
| Mean time per scenario run (s) | 0.149 | 3.826 |
| Mean time per case (s) | 0.046 | 1.177 |
| Agents spawned | 0 | 18 |
| LLM requests | 0 | 18 |
| Tokens (prompt / completion) | 0 / 0 | 18,873 / 3,739 |
| Failed investigations (rate-limited) | 0 (0) | 1 (0) |
| Analyzer run errors | 0 | 0 |

LLM usage by model (full mode):

| Model | Agents | Requests | Prompt tokens | Completion tokens |
|---|---|---|---|---|
| `groq/openai/gpt-oss-20b` | 14 | 14 | 14,435 | 2,596 |
| `groq/openai/gpt-oss-120b` | 4 | 4 | 4,438 | 1,143 |

## Confusion matrix: deterministic

| Truth \ Predicted | affected | potentially | uncertain | no usage | not analyzed |
|---|---|---|---|---|---|
| affected | 8 | 1 | 7 | 1 | 0 |
| not_affected | 3 | 1 | 5 | 8 | 5 |

## Confusion matrix: full

| Truth \ Predicted | affected | potentially | uncertain | no usage | not analyzed |
|---|---|---|---|---|---|
| affected | 13 | 0 | 3 | 1 | 0 |
| not_affected | 3 | 1 | 2 | 11 | 5 |

## By access pattern

Correct = TP + TN. Patterns marked † need information that AST analysis of literal key reads within a single function cannot provide.

| Pattern | Cases | deterministic correct | deterministic uncertain | full correct | full uncertain |
|---|---|---|---|---|---|
| array-access | 2 | 2/2 | 0 | 2/2 | 0 |
| async-client | 2 | 2/2 | 0 | 2/2 | 0 |
| attribute-access † | 2 | 0/2 | 2 | 1/2 | 1 |
| cross-function † | 5 | 1/5 | 4 | 3/5 | 2 |
| direct-access | 3 | 3/3 | 0 | 3/3 | 0 |
| dynamic-access † | 7 | 2/7 | 5 | 7/7 | 0 |
| guarded-access † | 1 | 0/1 | 0 | 0/1 | 0 |
| host-mismatch † | 2 | 0/2 | 0 | 0/2 | 0 |
| model-validation † | 3 | 0/3 | 3 | 0/3 | 3 |
| multiple-affected | 19 | 9/19 | 7 | 13/19 | 3 |
| nested-access | 2 | 2/2 | 0 | 2/2 | 0 |
| non-breaking | 5 | 5/5 | 0 | 5/5 | 0 |
| not-called | 3 | 3/3 | 0 | 3/3 | 0 |
| reads-other-fields | 6 | 5/6 | 1 | 5/6 | 1 |
| removed | 19 | 10/19 | 6 | 13/19 | 3 |
| renamed | 15 | 7/15 | 6 | 11/15 | 2 |
| tolerant-access † | 3 | 2/3 | 0 | 2/3 | 0 |
| unsupported-client † | 1 | 0/1 | 0 | 0/1 | 0 |

## Per-case predictions

| Scenario | Consumer | Truth | deterministic | full |
|---|---|---|---|---|
| demo-user-id-renamed | order-service | affected | affected ✓ | affected ✓ |
| demo-user-id-renamed | billing-service | not_affected | no usage ✓ | no usage ✓ |
| demo-user-id-renamed | notification-service | not_affected | uncertain ✗ | no usage ✓ |
| demo-email-renamed | order-service | affected | affected ✓ | affected ✓ |
| demo-email-renamed | billing-service | affected | affected ✓ | affected ✓ |
| demo-email-renamed | notification-service | affected | uncertain ✗ | affected ✓ |
| demo-phone-added | order-service | not_affected | not analyzed ✓ | not analyzed ✓ |
| demo-phone-added | billing-service | not_affected | not analyzed ✓ | not analyzed ✓ |
| demo-phone-added | notification-service | not_affected | not analyzed ✓ | not analyzed ✓ |
| demo-newsletter-removed | order-service | not_affected | no usage ✓ | no usage ✓ |
| demo-newsletter-removed | billing-service | not_affected | no usage ✓ | no usage ✓ |
| demo-newsletter-removed | notification-service | not_affected | no usage ✓ | no usage ✓ |
| demo-locale-removed | order-service | not_affected | no usage ✓ | no usage ✓ |
| demo-locale-removed | billing-service | not_affected | no usage ✓ | no usage ✓ |
| demo-locale-removed | notification-service | affected | uncertain ✗ | affected ✓ |
| plan-renamed | profile-cache | affected | uncertain ✗ | affected ✓ |
| plan-renamed | crm-sync | affected | uncertain ✗ | uncertain ✗ |
| plan-renamed | field-mapper | affected | uncertain ✗ | affected ✓ |
| plan-renamed | legacy-directory | not_affected | affected ✗ | affected ✗ |
| plan-renamed | greeting | not_affected | no usage ✓ | no usage ✓ |
| email-removed | crm-sync | affected | uncertain ✗ | uncertain ✗ |
| email-removed | email-normalizer | affected | potentially ✓ | affected ✓ |
| email-removed | support-desk | affected | no usage ✗ | no usage ✗ |
| email-removed | profile-view | not_affected | uncertain ✗ | no usage ✓ |
| email-removed | shipping | not_affected | no usage ✓ | no usage ✓ |
| nickname-removed | greeting | not_affected | potentially ✗ | potentially ✗ |
| nickname-removed | badge | not_affected | affected ✗ | affected ✗ |
| nickname-removed | profile-cache | not_affected | uncertain ✗ | uncertain ✗ |
| zip-removed | shipping | affected | affected ✓ | affected ✓ |
| zip-removed | field-mapper | not_affected | uncertain ✗ | no usage ✓ |
| zip-removed | crm-sync | not_affected | uncertain ✗ | uncertain ✗ |
| created-at-removed | directory-export | affected | affected ✓ | affected ✓ |
| created-at-removed | audit-log | affected | affected ✓ | affected ✓ |
| user-id-renamed | profile-view | affected | uncertain ✗ | uncertain ✗ |
| user-id-renamed | directory-export | affected | affected ✓ | affected ✓ |
| user-id-renamed | audit-log | affected | affected ✓ | affected ✓ |
| user-id-renamed | legacy-directory | not_affected | affected ✗ | affected ✗ |
| avatar-added | profile-cache | not_affected | not analyzed ✓ | not analyzed ✓ |
| avatar-added | field-mapper | not_affected | not analyzed ✓ | not analyzed ✓ |

## Errors and rate-limit failures

- full / plan-renamed: investigator_crm_sync (groq/openai/gpt-oss-20b) failed after 1 attempt(s): BadRequestError: litellm.BadRequestError: GroqException - {"error":{"message":"Failed to generate JSON. Please adjust your prompt. See 'failed_generation' for more details.","type":"invalid_request_error","code":"json_validate_failed","failed_generation":"{\"verdicts\":[{\"item_id\":\"item-1\",\"classification\":\"uncertain\",\"confidence\":0.4,\"reason\":\"The provided code only forwards the entire JSON response to `CrmContact.model_validate`, and the definition of `CrmContact` is not included in the snippets. Therefore, it is unclear whether the `plan` property is accessed or required by the model, making the impact uncertain.\",\"suggested_action\":\"\",\"evidence\":[{\"file\":\"crm_sync/contacts.py\",\"line\":17,\"snippet\":\"response = client.get(f\\\"/users/{user_id}\\\")\"},{\"file\":\"crm_sync/contacts.py\",\"line\":19,\"snippet\":\"return CrmContact.model_validate(response.json())\"}]}}"}}


## Scoring rules

- Ground truth is binary (`affected` / `not_affected`), hand-labelled and confirmed by the executable oracle.
- `affected` and `potentially_affected` are positive predictions (strict view: `potentially_affected` is an abstention).
- `no_relevant_usage_found` is a negative prediction and is never taken as proof of safety: on an affected consumer it is a false negative.
- `uncertain` is an abstention: never correct, excluded from precision, counted as a miss for recall, reported as a rate.
- `not_analyzed` (the analyzer found no breaking change) is a negative prediction; analyzer errors are never correct.

## Limitations of this benchmark

- 39 cases over constructed consumers. Each pattern is realistic, but this is not a sample of real-world code, so the metrics show the behaviour on these patterns, not production accuracy.
- The full mode was run once. LLM output is non-deterministic even at temperature 0, and the analyzer has no seed setting.
- Execution time includes analyzer start-up and oasdiff per scenario, and LLM latency in full mode.
