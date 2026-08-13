# EnvHarness on ALFWorld

## Environment

ALFWorld pins to Python 3.12 (3.13 hits a latent TextWorld bug).

```bash
conda create -n eh-alfworld python=3.12 && conda activate eh-alfworld
pip install -e .                       # pydantic, pyyaml, litellm

# ALFWorld runtime. On a clean 3.12 env the plain install fails: the
# transitive dependency `visdom` imports pkg_resources at build time, which
# setuptools >= 81 no longer ships. Constrain the build env:
printf 'setuptools<81\n' > /tmp/eh_constraints.txt
PIP_CONSTRAINT=/tmp/eh_constraints.txt pip install "alfworld[full]" textworld==1.7.0

# Game data (~2.3 GB). Honours $ALFWORLD_DATA, else ~/.cache/alfworld.
export ALFWORLD_DATA=~/eh_alfworld_data
alfworld-download

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key

python scripts/check_env.py alfworld    # preflight (python, imports, data)
```

Check the bridge loads:

```bash
python -c "from envharness.bridges.alfworld import AlfworldEnv; AlfworldEnv(); print('OK')"
```

## Run

```bash
# Reduced task count; the values it sets are at the top of the script.
bash experiments/alfworld/reproduce_smoke.sh

# Full run: corpus -> induce -> subset -> eval. Every shard and the eval
# run with the interpreter you launch this with.
python experiments/alfworld/reproduce.py
```

Shards round-robin whatever key pool the provider has; more Gemini keys
spread that per-key quota, and one key (or none, on Vertex) simply shares.

Output lands under `runs/alfworld_headline_<timestamp>/`, with the eval
summary at `$ROOT_RUN/eval/summary.json`:

```json
{
  "rows": [
    {"condition": "<nobank|orig|ours>", "split": "eval_in_distribution",
     "n_won": 0, "n": 140, "sr": 0.0, "start_seed": 0, "round": 1},
    {"condition": "<nobank|orig|ours>", "split": "eval_out_of_distribution",
     "n_won": 0, "n": 134, "sr": 0.0, "start_seed": 0, "round": 1}
  ]
}
```

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-4.1-mini` | model for **all** stages — corpus policy, harness agent, induction, eval. Any provider (`openai/…`, `vertex_ai/claude-…`, `gemini/…`) |
| `<PROVIDER>_API_KEY(S)` | per `MODEL` | `$OPENAI_API_KEY`, `$GEMINI_API_KEYS`, or ADC on Vertex. Shards round-robin the pool; key 0 induces, the last one evals |
| `PY` | the launching interpreter | interpreter for every stage |
| `N_TASKS_TOTAL` | `100` | corpus size |
| `N_SHARDS` | `4` | parallel corpus shards |
| `TASK_BASE_OFFSET` | `0` | first task_id |
| `EVAL_START_SEEDS` | `0,1000,2000` | one eval round per start seed |
| `EVAL_CONCURRENCY` | `24` | inner parallelism of `reasoning_bank_eval.py` |
| `EVAL_N_INDIST` / `EVAL_N_OOD` | all (140 / 134) | cap eval tasks per split |
| `ROOT_RUN` | `runs/alfworld_headline_<ts>` | output dir |
| `SKIP_CORPUS=1` | — | reuse `$ROOT_RUN/corpus/traces.jsonl` |
| `SKIP_INDUCE=1` | — | reuse `$ROOT_RUN/banks/*_full.jsonl` |
| `SKIP_SUBSET=1` | — | reuse `$ROOT_RUN/banks/*_subset*.jsonl` |
| `SKIP_EVAL=1` | — | stop after Stage 3 |

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | one-command entry point (chains stages 1-4) |
| `reproduce_smoke.sh` | the same four stages at a reduced task count |
| `corpus.yaml` | Stage 1 Orchestrator + Harness Agent config (per-shard) |
| `reasoning_bank_eval.py` | Stage 4 eval driver (`AlfworldEnv` + Bank + LiteLLM) |
| `reasoning_bank_eval.yaml` | Stage 4 eval config (model, retrieval top-K, splits) |
| `../../scripts/run_harness.py` | bridge-agnostic Stage 1 worker (one shard) |
| `../../scripts/induce_pair.py` | Stage 2 induction |
| `../../scripts/subset.py` | Stage 3 1-per-task subset |
| `../../envharness/bridges/alfworld/` | the ALFWorld Bridge (`AlfworldEnv`) |
| `../../envharness/prompts/alfworld_skill_prompt.py` | skill prompt + retrieval helpers used by `reasoning_bank_eval.py` |
