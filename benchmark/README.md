# Consumer-impact benchmark

A reproducible benchmark of how accurately API Drift Guard identifies downstream
Python consumers affected by OpenAPI changes. It compares two modes on the same cases:

| Mode | Analyzer invocation | What runs |
|---|---|---|
| `deterministic` | `api-drift-guard analyze --mode static` | oasdiff + AST analysis |
| `full` | `api-drift-guard analyze --mode agentic` | the above + Google ADK investigation agents on Groq |

Measured results: [`results/comparison.md`](results/comparison.md). Machine-readable
results: `results/deterministic.json` and `results/full.json`.

## Running it

```bash
conda activate api-drift
python benchmark/run_benchmark.py verify                   # ground truth vs the executable oracle
python benchmark/run_benchmark.py run --mode deterministic # no credentials needed
python benchmark/run_benchmark.py run --mode full          # needs GROQ_API_KEY (env or .env)
python benchmark/run_benchmark.py report                   # results/comparison.md
pytest benchmark/tests                                     # benchmark self-tests
```

`run` always re-verifies the ground truth first, and refuses to score if any label
disagrees with the oracle. `--scenarios a,b` runs a subset. In full mode,
`--pause-seconds` (default 20) pauses after each scenario that used the LLM, to stay
under Groq's free-tier limit of 8K tokens per minute.

## Independence from the analyzer

- The runner calls the analyzer **only through its CLI**, one subprocess per scenario,
  and reads its JSON report. No benchmark code imports `api_drift_guard`; a self-test
  enforces this.
- Mock HTTP responses, endpoint matching and evidence verification are implemented
  separately here (`oracle.py`, `run_benchmark.py`).
- The benchmark has its own provider (`provider/`, exporting `specs/`) and consumers
  (`consumers/`). It also reuses the demo fixtures and consumers as-is.
- `API_DRIFT_GUARD_*` environment overrides are removed from the analyzer's environment.
  Every run's analyzer config is generated from `cases.yaml`.

## Cases

39 (scenario, consumer) cases across 12 OpenAPI changes: 5 reused demo scenarios
(`fixtures/`) and 7 from the benchmark provider. 17 are truly affected and 22 are not.
Each new consumer is a distinct access pattern, not a variation of another:

| Pattern tag | Consumer(s) | What it exercises |
|---|---|---|
| `direct-access` | order, billing | `data["field"]` on the matched response |
| `nested-access` | shipping | `user["address"]["zip"]` |
| `array-access` | directory-export | comprehension over `GET /users` |
| `async-client` | audit-log | `httpx.AsyncClient` + `await` |
| `dynamic-access` † | notification, field-mapper | keys from a tuple/dict constant |
| `cross-function` † | profile-cache, profile-view | JSON returned by a helper, read in another function or module |
| `model-validation` † | crm-sync | `PydanticModel.model_validate(resp.json())` |
| `attribute-access` † | profile-view | `SimpleNamespace(**json).user_id` |
| `tolerant-access` † | greeting, email-normalizer | `.get()` with a fallback (safe), and `.get().strip()` (crashes) |
| `guarded-access` † | badge | `if "nickname" in data: data["nickname"]` |
| `host-mismatch` † | legacy-directory | same path shape on a **different** service |
| `unsupported-client` † | support-desk | `requests` instead of httpx |
| `reads-other-fields`, `not-called`, `non-breaking` | various | decoys and additive changes |

† The correct answer needs information beyond literal key reads on the response
within a single function. Scenario tags: `removed`, `renamed`, `multiple-affected`,
`non-breaking`.

## Ground truth

Each case in `cases.yaml` has a hand-written `truth` (`affected` / `not_affected`), a
`rationale`, and, for affected cases, the `file:line` that breaks. Labels are written
from the consumer code and the contract change, not from analyzer output.

They are then confirmed by an **executable oracle** (`oracle.py`). It runs the
consumer's real code in a fresh interpreter with httpx's and requests' transports
patched:
- Requests to `user-service` receive a body generated from the spec's 200 response
  schema, with every declared property present and deterministic example values.
- Requests to other hosts receive fixed bodies.

A case is valid only if the consumer **works on the old contract**. The oracle's
verdict is `affected` if it then fails on the new contract. `verify` must report
39/39 agreements before any scoring happens.

## Scoring

Ground truth is binary. The analyzer's consumer status is mapped as follows:

| Prediction | Primary view | Strict view |
|---|---|---|
| `affected` | positive | positive |
| `potentially_affected` | positive ("flagged") | abstention |
| `no_relevant_usage_found` | negative | negative |
| `not_analyzed` (analyzer found no breaking change) | negative | negative |
| `uncertain` | abstention | abstention |
| analyzer error | error | error |

- **Precision** = TP / (TP + FP). Abstentions are excluded.
- **Recall** = TP / truly affected. Abstentions and errors on affected cases count as misses.
- **F1** = harmonic mean of precision and recall. **Accuracy** = (TP + TN) / all cases.
- `uncertain` is **never counted as correct**. Its rate is reported separately.
- `no_relevant_usage_found` is **not proof that a consumer is unaffected**. It is scored
  as a negative prediction, so on an affected consumer it is a false negative.
  The `support-desk` case exists to test this.
- **Evidence:**
  - every cited reference is checked independently (file inside the consumer, line
    exists, snippet matches the line)
  - for true positives, the benchmark checks whether one of the expected breaking
    lines was cited
- **Time:** wall-clock seconds per analyzer subprocess (start-up, oasdiff, analysis,
  and LLM latency in full mode), also averaged per case.
- **LLM:** agents spawned, requests, and tokens, per model, from the analyzer's report.
  Failed investigations and rate-limit failures are counted.

## Reproducibility

Each results file records:
- start and end time, Python and platform, analyzer and oasdiff versions
- the git commit and whether the tree was dirty
- SHA-256 hashes of `cases.yaml`, the consumer code and the specs
- the fixed LLM settings: models, temperature 0, `reasoning_effort` low, retry and
  budget limits
- per-scenario timings, exit codes and errors, every investigation, and every per-case
  prediction with its evidence

Raw analyzer reports and generated configs are written to `results/raw/`, which git
ignores because they contain absolute local paths.

The analyzer exposes no seed parameter, and the benchmark does not modify the analyzer,
so LLM output may vary between full-mode runs even at temperature 0.

## Limitations

- The consumers are constructed. Each pattern is realistic, but 39 cases are not a
  sample of real-world code: the metrics describe behaviour on these patterns, not
  production accuracy.
- The oracle exercises one code path per consumer, with one example response.
- Mock responses include every declared property. Real providers may omit optional ones.
