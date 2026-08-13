# EnvHarness on SWE-bench

## Environment

Each rollout resolves a real GitHub issue inside a per-task Docker container,
and the official SWE-bench scorer grades the patch.

```bash
conda create -n eh-swebench python=3.12 && conda activate eh-swebench
pip install -e .                              # EnvHarness core
pip install swebench datasets docker litellm

# Claude on Vertex (MODEL=vertex_ai/claude-...) additionally needs the SDK
# litellm's vertex_ai provider imports, plus ADC and a project:
pip install "google-cloud-aiplatform>=1.38"
export GOOGLE_CLOUD_PROJECT='...'             # gcloud auth application-default login

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key

python scripts/check_env.py swebench          # preflight
```

`swebench` must be importable by the SAME interpreter that runs the
pipeline: the scorer is invoked as
`sys.executable -m swebench.harness.run_evaluation`.

Docker must be reachable from the launching process — check with `docker
info`. If that fails only because your shell session predates `usermod -aG
docker $USER` (group membership is resolved at login), wrap the command in
`sg docker -c "..."`.

## Run

```bash
# Reduced task count; the values it sets are at the top of the script.
bash experiments/swebench/reproduce_smoke.sh

# Full run: corpus -> banks -> 3-condition eval. Run it with the env that
# has swebench installed -- the scorer runs in this same interpreter.
python experiments/swebench/reproduce.py

```

Output lands under `runs/swebench_headline_<timestamp>/`: per-condition rows
in `eval/<nobank|orig|ours>.jsonl` (`instance_id`, `success`, `submitted`,
`submitted_patch_len`, `error`) plus a `summary.json`. `reproduce.py` resumes
a condition on a nonzero exit, so re-running it retries rows that failed on
infra (rate-limit, docker race).

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `<PROVIDER>_API_KEY(S)` | per `MODEL` | `$OPENAI_API_KEY`, `$GEMINI_API_KEYS`, or ADC on Vertex. A Gemini pool round-robins across workers |
| `PY` | the launching interpreter | interpreter for every stage |
| `MODEL` | `openai/gpt-4.1-mini` | model for **all** stages — corpus policy, harness agent, induction, eval. Any provider (`gemini/…`, `openai/…`, `vertex_ai/claude-…`) |
| `EH_EMBED_MODEL` | provider default | embedding model for bank retrieval |
| `CLAUDE_THINKING_BUDGET` | — | thinking budget when the model is Claude |
| `ROOT_RUN` | `runs/swebench_headline_<ts>` | output dir |
| `N_TASKS` | `100` | corpus size sampled from the Lite pool |
| `LITE_POOL` | `300` | Lite split size sampled from |
| `TASK_SEED` | `20260727` | seed for the task-offset draw |
| `TASK_OFFSETS` | — | explicit comma-separated Lite offsets (overrides the draw) |
| `CORPUS_YAML` | `experiments/swebench/corpus.yaml` | Stage 1 config |
| `CORPUS_CONCURRENCY` | `4` | parallel per-task corpus runs |
| `CORPUS_RETRIES` | `2` | auto-rerun rounds for failed corpus tasks |
| `EVAL_N` | `407` | held-out episodes per condition |
| `EVAL_CONCURRENCY` | `6` | parallel eval episodes |
| `EVAL_MAX_STEPS` | `250` | eval step budget per episode |
| `RB_REASONING_EFFORT` | `low` | eval-policy reasoning effort (see below) |
| `SKIP_CORPUS=1` / `SKIP_BANKS=1` / `SKIP_EVAL=1` | — | reuse earlier stages |
| `SKIP_ORIG=1` | — | run only the nobank + ours conditions |

### `reasoning_effort` across providers

`RB_REASONING_EFFORT` (default `low`) is passed through
`envharness.infra.model`, which adapts it per provider: Gemini maps it onto
`thinkingLevel`, Claude turns it into a thinking budget (and then requires
`temperature=1`, which the resolver sets), and a model that does not accept
the parameter at all has it dropped. `off`/`none` omits it entirely.

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | one-command entry point (chains stages 1-3) |
| `reproduce_smoke.sh` | the same three stages at a reduced task count |
| `corpus.yaml` | Stage 1 Orchestrator + Harness Agent config (per-task) |
| `corpus_smoke.yaml` | reduced-scale Stage 1 config |
| `bank_distillation/build_orig_bank.py` | Stage 2 orig bank (baseline intra-pairs) |
| `bank_distillation/build_ours_bank.py` | Stage 2 ours bank (cascade, mutation-preferred) |
| `bank_distillation/induce.py` | shared induction prompts/helpers for both builders |
| `reasoning_bank_eval.py` | Stage 3 eval driver (Bridge + Bank + LiteLLM + official scorer) |
| `bash_bridge.py` | single-tool (bash) SWE-bench Bridge subclass |
| `../../scripts/run_harness.py` | bridge-agnostic Stage 1 worker (one task) |
| `../../envharness/bridges/swebench/` | the SWE-bench Bridge (`SWEBenchEnv` + official scorer) |
