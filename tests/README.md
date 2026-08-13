# Tests

GPU-free, docker-free, API-key-free. All tests run against the toy24
`ActionableEnv` (pure stdlib) and `ScriptedClient` (canned LLM responses).
The whole suite (115 tests) finishes in well under a second.

## Run

```bash
pip install -e ".[dev]"
pytest                                  # full suite
pytest tests/test_objectives.py -v      # one file
pytest -k difficulty_zone               # by name pattern
```

## What's covered

| File | Surface tested |
|---|---|
| `test_actionable_env_toy24.py` | Full `ActionableEnv` interface walk-through on `Toy24Env`: `reset` / `step` / `evaluate` / `observe` / `tool_schemas` / `env_state_schema` / `save_state` / `from_state` round-trip. **Template for new-Bridge tests.** |
| `test_envharness_composition.py` | Stacking `Rules` and `Setup` decorators around an `ActionableEnv`: hook firing, pass-through invariants, S₀ replay, save/load of the harness stack. |
| `test_code_loader.py` | Agent-emitted Python code lifecycle: empty body → pass-through; syntax error → `RulesCodeError`; wrong base class → typed error. |
| `test_objectives.py` | `DifficultyZone`: band validation, in-band / too-easy / too-hard branches, axis-weights output. |
| `test_budget.py` | All `BudgetPolicy` implementations: stop conditions and edge cases. |
| `test_baseline_cache.py` | Cache key determinism + load/save round-trip + corrupt-file resilience. |

## Patterns to copy when adding tests

### Adding a Bridge (`ActionableEnv`) for a new benchmark

Model your test on `test_actionable_env_toy24.py`. The `ActionableEnv` ABC
contract is:

- `reset(seed, options) → EnvResetResponse` produces an `Observation`
- `step(Action) → EnvResponse` with `reward / terminated / info`
- `evaluate() → EvaluationResult` (the authoritative success signal)
- `observe() → Observation` works BEFORE the first `step`
- `tool_schemas()` and `env_state_schema()` are non-empty
- `save_state() → dict` / `from_state(dict) → ActionableEnv` round-trip

Pin a known-solvable task via `reset(options=...)` and walk it through to
success in the test.

### Adding a new harness layer

Model your test on `test_envharness_composition.py`. The `EnvHarness` ABC
wraps an inner `ActionableEnv` and:

- forwards `reset` / `step` / `observe` / `evaluate` / `get_env_state` to
  `inner` by default
- overrides only the methods the layer governs (e.g. `Rules` overrides
  `step` to insert A / O / T hooks)
- exposes its own `save_state` / `from_state` so the persistence walker
  can serialize the whole stack

### Adding a new Harness Agent implementation

Model on `test_envharness_composition.py` for the scripted agent. The
`HarnessAgent` ABC contract is:

- `propose(ctx) → Candidate` (carries `rules_code` + `in_env_actions`)
- `decide(candidate, traces, ctx) → DecideResult` (ACCEPT / REFINE / REJECT)
- `refine(candidate, traces, ctx) → Candidate`

`HarnessAgentContext` exposes `tool_schemas`, `env_state_schema`,
`objective`, `task_id`, and (when computed) `baseline: BaselineSnapshot`.

### Adding a new `MutationObjective`

Model on `test_objectives.py`. The contract is:

- `evaluate(recent_traces) → ObjectiveSignal`
- The signal carries `score`, `diagnostic`, `suggestion_prompt`, optional `weights`

Test the branches: empty history, "too easy", "too hard", "in band".

### Adding a new `BudgetPolicy`

Model on `test_budget.py`:

- `should_stop(attempts, last_decision, objective_signal) → bool`
- Test each terminating condition independently.

## Why no LLMHarnessAgent / SubprocessRunner / real-bench tests?

These need either an API key, docker daemon, or a long-running runtime
(textworld, playwright). They belong in **integration** tests run via
`scripts/run_harness.py` against a real config — see the per-benchmark
READMEs under `experiments/<bench>/`. The unit tests above pin the typed
interfaces; the integration runs validate behavior on real envs.
