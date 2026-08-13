# EnvHarness on toy24

4-numbers-make-24 puzzle. Pure in-memory Python — no docker, browser or
simulator.

## Environment

```bash
conda create -n eh-toy24 python=3.12 && conda activate eh-toy24
pip install -e .                  # pydantic, pyyaml, litellm

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key

python scripts/check_env.py toy24     # preflight
```

`mutated_smoke.yaml` is `mutated.yaml` at a reduced task count. Both put the
Harness Agent and the Policy on the same model; `MODEL` switches every stage at
once.

## Run

```bash
# All three modes, reduced task count. PY overrides the interpreter.
bash experiments/toy24/reproduce_smoke.sh

# 1. Scripted plumbing (no API key).
python scripts/run_harness.py --scripted --in-process --run-name toy24-baseline

# 2. Original env — Gemini Policy through EnvHarness with NoopMutator.
python scripts/run_harness.py \
    --config experiments/toy24/original.yaml \
    --run-name toy24-original

# 3. Mutated env — LLM Harness Agent + LLM Policy, DifficultyZone [0.3, 0.7].
python scripts/run_harness.py \
    --config experiments/toy24/mutated.yaml \
    --run-name toy24-mutated

# 3b. Gemini-only variant of mode 3.
python scripts/run_harness.py \
    --config experiments/toy24/mutated_smoke.yaml \
    --run-name toy24-mutated-smoke
```

Traces land in `runs/<run-name>/traces.jsonl`.

## Files

| File | Purpose |
|---|---|
| `original.yaml` | NoopMutator config: the env routed through EnvHarness with zero mutations |
| `mutated.yaml` | LLM Harness Agent + DifficultyZone (Anthropic mutator, OpenAI policy) |
| `mutated_smoke.yaml` | same as `mutated.yaml` with both roles on Gemini |
| `reproduce_smoke.sh` | runs all three modes at a reduced task count |
| `../../scripts/run_harness.py` | the bridge-agnostic runner both configs drive |
| `../../envharness/bridges/toy24/` | the toy24 Bridge (`Toy24Env`) |
