# API Drift Guard

**Agentic cross-service API compatibility analyzer.** Given two versions of a provider's
OpenAPI spec and a set of Python consumer repositories, it finds the contract changes
and determines which consumers are likely to break. Every claim comes with file:line
evidence that has been verified against the source.

Existing tools stop at "the spec changed". API Drift Guard goes on to check how each
downstream service actually uses the changed fields:

```
user-service removed `user_id` from GET /users/{user_id}

order-service         Affected                 order_service/users_client.py:12  return user["user_id"]
billing-service       No relevant usage found  calls the endpoint, reads only name and email
notification-service  Uncertain → LLM agent    notifier.py:17  {field: user[field] for field in TEMPLATE_FIELDS}
```

## How it works

```
OpenAPI before/after ──► oasdiff changelog ──► breaking-change extraction        (deterministic)
                                                      │
consumer repos ──► Python AST analysis ──► endpoint matching ──► static verdicts  (deterministic)
                                                      │
                          coordinator: which findings are still unsettled?        (deterministic)
                                                      │
               one Google ADK LlmAgent per consumer, created at runtime           (LLM, optional)
                                                      │
                   evidence verification + merge ──► JSON / Markdown report       (deterministic)
```

**Design rule:** LLMs are used only for what deterministic code can't do reliably,
such as interpreting dynamic access or connecting code to runtime behaviour. Parsing,
spec diffing, locating calls, extracting snippets, validating evidence and reporting
are all plain Python.

| Module | Responsibility |
|---|---|
| `contracts.py` | Runs oasdiff, validates OpenAPI 3.x, normalises changes |
| `source_analysis.py` | AST analysis: httpx calls, URLs, `.json()` values, field reads, untrackable uses |
| `discovery.py` | Matches request URLs to OpenAPI path templates |
| `impact.py` | Static classification per (consumer, breaking change) |
| `evidence.py` | Builds compact snippets for agents; verifies cited file:line references |
| `agents.py` | Coordinator, ADK agent factory, bounded retries, merging of verdicts |
| `pipeline.py` | End-to-end run shared by `analyze` and `evaluate` |
| `reporting.py` | JSON and Markdown reports |
| `evaluation.py` | Scores results against a ground-truth dataset |
| `cli.py` / `config.py` / `models.py` | Typer CLI, YAML config, Pydantic models |

## Setup

Requirements: Python 3.11+ (developed with 3.13), and [oasdiff](https://github.com/oasdiff/oasdiff/releases)
on `PATH` or `OASDIFF_BIN` (developed against 1.33.0). Agentic mode also needs a
[Groq API key](https://console.groq.com/keys).

```bash
conda create -n api-drift python=3.13 && conda activate api-drift
pip install -e ".[dev]"
oasdiff --version
cp .env.example .env    # set GROQ_API_KEY (agentic mode only)
```

## Usage

```bash
# Deterministic analysis; no credentials needed
api-drift-guard analyze \
    --old fixtures/openapi_before.json \
    --new fixtures/openapi_after.json \
    --config config.example.yaml \
    --output reports/report.json --markdown reports/report.md

# Also investigate unsettled findings with LLM agents
api-drift-guard analyze ... --mode agentic

# Score the tool against the demo scenarios
api-drift-guard evaluate --dataset evaluation/scenarios.yaml [--mode agentic] \
    [--output reports/evaluation.json] [--markdown reports/evaluation.md]
```

With neither `--output` nor `--markdown`, `analyze` prints JSON to stdout. Logs go
to stderr as `key=value` lines (`--log-level DEBUG` for detail).

Exit codes: `0` completed, `1` config/spec/oasdiff/consumer-path/credential error,
`2` CLI usage error.

### Configuration

```yaml
provider:
  name: user-service

consumers:                  # paths are relative to this file
  - name: order-service
    path: ./demo/order_service

llm:                        # optional; agentic mode only; defaults sized for Groq's free tier
  model: groq/openai/gpt-oss-20b
  escalation_model: groq/openai/gpt-oss-120b   # null disables escalation
  escalation_confidence: 0.6
  max_concurrency: 1
  max_retries: 3
  max_llm_requests: 20
  max_context_chars: 6000
```

`API_DRIFT_GUARD_MODEL`, `API_DRIFT_GUARD_ESCALATION_MODEL` (`none` disables escalation) and
`API_DRIFT_GUARD_MAX_CONCURRENCY` override the file. Any LiteLLM model string works,
with that provider's API key in the environment. API keys are never read from the config file.

## Contract changes

Changes come from `oasdiff changelog --format json --lang en`.

| oasdiff check | `change_type` | `breaking` |
|---|---|---|
| `response-required-property-removed` | `removed_response_property` | yes |
| `response-optional-property-removed` | `removed_response_property` | yes. oasdiff rates it INFO, but `data["field"]` still fails |
| `response-{required,optional}-property-added` | `added_response_property` | no |
| anything else | `unsupported` | if oasdiff level ≥ WARN |

A rename is a removal plus an addition. Fields added to the same response are suggested
as possible replacements. Consumers calling an endpoint with an `unsupported` breaking
change are reported as `uncertain`.

## Static consumer analysis

Every `.py` file in a consumer (excluding virtualenvs, `build/`, `dist/`, etc.) is parsed
with Python's `ast` module:

1. **Requests:** `httpx.get/post/...`, `httpx.request(...)`, and `Client`/`AsyncClient`
   methods (local, module-level, `self.<attr>`). URLs are resolved from literals,
   f-strings, `+`, `.format()`, module constants and `base_url=`.
2. **Endpoint matching** against the *old* spec, segment by segment: `f"{BASE}/users/{uid}"`
   matches `GET /users/{user_id}`, and `/users/me` beats `/users/{user_id}`.
3. **Field reads** on the response JSON through assignments, `with`, `for` loops and
   comprehensions within a function: `data["f"]`, `data.get("f")`, `"f" in data`,
   including nested and array (`items/…`) paths. Anything that can't be followed
   (dynamic keys, or JSON passed to, returned from or stored outside the function) is
   recorded instead of ignored.

| Classification | Static rule |
|---|---|
| `affected` | Direct `data["field"]` read on a request unambiguously matched to the changed endpoint |
| `potentially_affected` | Tolerant read (`.get`, `in`), or a direct read on a request whose endpoint is ambiguous or unresolved |
| `uncertain` | The response (or the removed field's parent) is used in ways that can't be followed. Also used for parse errors and unsupported change types |
| `no_relevant_usage_found` | None of the above. This is **not** a guarantee of compatibility |

A consumer's status is its most severe finding. Static findings have
`analysis_method: "static"` and `confidence: null`.

## Agentic investigation

- **Coordinator (deterministic).** Only consumers with `uncertain` or `potentially_affected`
  findings get an agent: one agent per consumer, so 0, 1 or N agents. Static `affected`
  findings are never sent to an LLM.
- **Agent factory.** Each agent is a Google ADK `LlmAgent` (via `LiteLlm`) created at runtime.
  It runs in its own `Runner` + `InMemorySessionService` session, so contexts are
  independent. It has no tools: deterministic code supplies the context, and the agent
  returns one structured `AgentReport` (`output_schema`).
- **Compact context.** The agent sees the enclosing function of each evidence line, plus
  the module constants it references (e.g. `TEMPLATE_FIELDS`), within `max_context_chars`.
  Repositories are never sent wholesale.
- **Verified evidence.**
  - Each cited file:line must be a `.py` file inside the consumer, the line must exist, and
    the snippet must match it; otherwise the reference is discarded.
  - A verdict of `affected` or `potentially_affected` with no verifiable evidence becomes `uncertain`.
  - An LLM can't downgrade a field read that static analysis found. The original verdict is
    kept in `static_classification`.
- **Escalation.** Verdicts that are still `uncertain` or below `escalation_confidence` are
  re-investigated once by a new agent on the larger model.
- **Free-tier limits** (gpt-oss: 8K tokens/min, 30 requests/min):
  - concurrency is capped (`max_concurrency`, default 1), with a per-run request cap (`max_llm_requests`)
  - bounded exponential backoff honours `retry-after`; there are no LiteLLM-internal retries
  - auth and bad-request errors aren't retried
  - token usage is recorded per agent (`investigations`) and in total (`llm_usage`)
- **Graceful failure.** If an agent fails, its static findings are kept, with a note.

Confidence values are model estimates, not calibrated probabilities.

## Demo and evaluation

`demo/user_service` is a FastAPI provider whose contract versions generate the fixtures
(`python demo/user_service/export_specs.py`; a test checks they are current). The three
consumers each use a different access pattern:

| Consumer | Pattern |
|---|---|
| `order_service` | `httpx.get(f"{BASE}/users/{id}")`; reads `user_id`, `name`, `email` directly |
| `billing_service` | `httpx.Client(base_url=...)`; reads `name`, `email` |
| `notification_service` | `self.client`; `{f: user[f] for f in TEMPLATE_FIELDS}` (dynamic keys) |

`evaluation/scenarios.yaml` holds ground truth, judged from the code, for five scenarios:

| Scenario | Change | Truth |
|---|---|---|
| 1 | `user_id` → `userId` | order affected; others unaffected |
| 2 | `email` → `contact_email` | order, billing and notification affected |
| 3 | optional `phone` added | not breaking; 0 agents |
| 4 | `preferences.newsletter` removed | nobody calls the endpoint; 0 agents |
| 5 | `locale` removed | only notification, via dynamic access |

Static results (`api-drift-guard evaluate`):

| Analysis | Correct | False positives | False negatives | Uncertain | Expected evidence found | Invalid evidence |
|---|---|---|---|---|---|---|
| Static | 12/15 (80%) | 0 | 0 | 3 | 5/5 | 0/17 |
| Agentic | not yet measured | | | | | |

The 3 uncertain cases are all notification-service's dynamic access, the case the
agents exist for. Agent counts matched expectations in all five scenarios (1, 1, 0, 0, 1).
LLM-assisted accuracy hasn't been measured yet: run `api-drift-guard evaluate --mode agentic`
with a Groq key. The evaluator scores static and LLM results separately from the same
run, and treats `uncertain` as an abstention, not an error.

## Tests

```bash
pytest
```

No API key or network access is needed. LLM calls are replaced by a scripted LiteLLM
client (`tests/fake_llm.py`), so the real ADK `Runner` → `LlmAgent` → `LiteLlm` path
runs. Tests that need the oasdiff binary are skipped when it isn't installed.

## Limitations

- Static analysis is intra-function. JSON passed between functions or files is
  reported as `uncertain`, and agents only see the snippets they're given.
- Only httpx is recognised (no `requests` or `aiohttp`), and only JSON responses.
- Requests are matched by method and path; hosts aren't compared. A dynamic URL part is
  assumed to be one path segment.
- A property literally named `items` is ambiguous with oasdiff's array notation.
- Property names are parsed from oasdiff's English text, because its JSON has no
  structured field for them. If the text can't be parsed, the change is reported as
  `unsupported`, never dropped.
- The evaluation dataset is small and constructed. It shows the behaviour on known
  patterns, not accuracy on real codebases.

Out of scope for this MVP: automatic fixes, PR creation or GitHub integration,
repository discovery, non-Python consumers.
