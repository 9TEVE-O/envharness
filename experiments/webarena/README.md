# EnvHarness on WebArena

Tasks run through `browsergym` + `playwright` against a local WebArena docker
stack, across 4 sites (reddit / shopping / shopping_admin / gitlab).

## Environment

WebArena needs a dedicated python env (the vendored `third_party/reasoning_bank_agent` import chain
is heavy) and a local docker stack (4 sites × 3 replicas for parallel eval).

```bash
# 1. Python env
conda create -n eh-webarena python=3.11
~/miniconda3/envs/eh-webarena/bin/pip install -e .
~/miniconda3/envs/eh-webarena/bin/pip install \
    browsergym==0.14.1 browsergym-webarena==0.14.1 playwright \
    litellm pyyaml langchain-core langchain_community langchain_anthropic \
    google-genai openai transformers huggingface_hub joblib tiktoken nltk
~/miniconda3/envs/eh-webarena/bin/python -m nltk.downloader punkt_tab
~/miniconda3/envs/eh-webarena/bin/python -m playwright install chromium

# 2. Local docker stack (downloads the 4 official image tars ~207 GB once,
#    then loads + starts + health-checks all 12 containers)
IMAGE_DIR=~/webarena_images bash experiments/webarena/setup_stack.sh

# 3. Site URLs (optional -- reproduce.py falls back to these 127.0.0.1 defaults)
export WA_SHOPPING="http://localhost:17770"
export WA_SHOPPING_ADMIN="http://localhost:17780/admin"
export WA_REDDIT="http://localhost:19999"
export WA_GITLAB="http://localhost:18023"

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key

# 4. Preflight -- verifies the full import chain builds and 12/12 ports serve
~/miniconda3/envs/eh-webarena/bin/python scripts/check_env.py webarena
```

## Run

```bash
# Reduced task count; the values it sets are at the top of the script.
bash experiments/webarena/reproduce_smoke.sh

# Full run: corpus -> banks -> eval. Run it with the env that has the
# browsergym + reasoning_bank_agent stack.
python experiments/webarena/reproduce.py
```

`WEBARENA_PYTHON` overrides the interpreter used for every stage; it
defaults to the one you launch `reproduce.py` with.

Stage 3 launches one `dispatcher.sh` per site, which fans that site's tasks
over its 3 containers, one `worker.py` process per task. Per-site rows land
in `$ROOT_RUN/eval/<condition>_<site>.jsonl` with `task_id` and `success`.

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-4.1-mini` | model for **all** stages — corpus policy, harness agent, induction, eval. Any provider (`openai/…`, `vertex_ai/claude-…`, `gemini/…`) |
| `<PROVIDER>_API_KEY(S)` | per `MODEL` | `$OPENAI_API_KEY`, `$GEMINI_API_KEYS`, or ADC on Vertex. A Gemini pool round-robins across workers |
| `WEBARENA_PYTHON` | the launching interpreter | python binary for every stage |
| `ROOT_RUN` | `experiments/webarena/runs/reproduce_<ts>` | output dir |
| `N_CORPUS_WORKERS` | `4` | parallel corpus workers |
| `N_CORPUS_PER_SITE` | `20` | corpus tasks per site (`ALL` for the full list) |
| `N_EVAL_PER_SITE` | all | cap eval tasks per site (writes a truncated copy under the run dir; `tasks/` is never modified) |
| `CORPUS_YAML` | `experiments/webarena/corpus.yaml` | Stage 1 config |
| `TOPK` | `5` | eval retrieval top-K |
| `SKIP_CORPUS=1` | — | reuse `$ROOT_RUN/corpus/all_traces.jsonl` |
| `SKIP_INDUCE=1` | — | reuse `$ROOT_RUN/banks/*.jsonl` |
| `SKIP_EVAL=1` | — | stop after Stage 2 |

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | one-command entry point (chains stages 1-3) |
| `reproduce_smoke.sh` | the same three stages at a reduced task count |
| `corpus.yaml` | Stage 1 Orchestrator + Harness Agent config (per-site) |
| `corpus_smoke.yaml` | reduced-scale Stage 1 config |
| `build_reasoning_bank.py` | Stage 2 cascade + baseline bank induction |
| `dispatcher.sh` + `worker.py` | Stage 3 eval: per-site queue dispatch (3 containers/site), one GenericAgent episode per task |
| `setup_stack.sh` | one-time docker stack bring-up (12 containers) |
| `tasks/` | per-site task id lists |
| `reasoning_bank_agent/` | the WebArena browser agent; vendored third-party code (Apache-2.0) |
| `../../scripts/run_harness.py` | bridge-agnostic Stage 1 worker |
| `../../envharness/bridges/webarena/` | the WebArena Bridge (`WebArenaEnv`) |
