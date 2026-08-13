# EnvHarness on SpreadsheetBench

[RUCKBReasoning/SpreadsheetBench](https://github.com/RUCKBReasoning/SpreadsheetBench).
Each episode runs in a per-episode sandbox working directory holding a copy of
the task's input spreadsheet; the policy manipulates it with `run_python(code)`
and declares done with `submit()`. Grading is the official Online-Judge metric
(LibreOffice formula recalculation + cell comparison at `answer_position`).
Skill induction + injection prompts are ported from
[Trace2Skill](https://github.com/Qwen-Applications/Trace2Skill).

## Environment

```bash
conda create -n eh-sb python=3.12 && conda activate eh-sb
pip install -e .                       # pydantic, pyyaml, litellm
pip install openpyxl pandas

# LibreOffice (headless formula recalculation for the OJ metric). The harness
# isolates a per-process user profile, so concurrent headless soffice is fine.
sudo apt install -y libreoffice-calc

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key
```

**Data** (not checked in; gitignored). Both archives ship inside the
SpreadsheetBench repository's `data/` directory. Fetch and unpack them from
the EnvHarness repo root:

```bash
mkdir -p experiments/spreadsheetbench/data/_dl
BASE=https://raw.githubusercontent.com/RUCKBReasoning/SpreadsheetBench/main/data

# eval set (Stage 4) -> data/_dl/all_data_912_v0.1/
curl -L -o /tmp/sb912.tar.gz $BASE/spreadsheetbench_912_v0.1.tar.gz
tar xzf /tmp/sb912.tar.gz -C experiments/spreadsheetbench/data/_dl

# corpus train set (Stage 1) -> data/spreadsheetbench_verified_400/
curl -L -o /tmp/sb400.tar.gz $BASE/spreadsheetbench_verified_400.tar.gz
tar xzf /tmp/sb400.tar.gz -C experiments/spreadsheetbench/data
```

Both tarballs unpack to exactly the directory names the YAMLs expect:

```
experiments/spreadsheetbench/data/_dl/all_data_912_v0.1/            # Stage 4 eval set
experiments/spreadsheetbench/data/spreadsheetbench_verified_400/    # Stage 1 corpus
```

The eval set gives each task 3 test cases (`{n}_{id}_input.xlsx` /
`{n}_{id}_answer.xlsx`); the fixed held-out instance ids ship in
`data/held_out_idx.txt`. The corpus set is the *verified* subset in the
single-test-case layout (`dataset.json` + `spreadsheet/{id}/*_init.xlsx` +
`*_golden.xlsx`), matching `corpus.yaml`'s `reset_options.data_path` — edit
`data_path` if you keep them elsewhere.

Preflight (checks deps, LibreOffice and both datasets):

```bash
python scripts/check_env.py spreadsheetbench
```

Run everything from the repo root, so namespace imports and the relative
`data_path` values in the YAMLs resolve.

## Run

```bash
# Reduced task count; the values it sets are at the top of the script.
bash experiments/spreadsheetbench/reproduce_smoke.sh

# Full run: corpus -> induce -> subset -> eval -> report.
python experiments/spreadsheetbench/reproduce.py
```

Run `reproduce.py` with the interpreter of the env you installed into (it
passes `sys.executable` to `dispatcher.sh` as `SB_PYTHON`).

Stages can also be run by hand:

```bash
# Stage 1
N_WORKERS=6 bash experiments/spreadsheetbench/dispatcher.sh corpus \
    experiments/spreadsheetbench/corpus.yaml sb_corpus_001 0:100 \
    runs/sb_corpus_001/done.jsonl

# Stage 2 (merge the per-task traces first)
cat runs/sb_corpus_001/corpus_task*/traces.jsonl > runs/sb_corpus_001/traces.jsonl
python experiments/spreadsheetbench/induce.py \
    --traces runs/sb_corpus_001/traces.jsonl \
    --out-dir runs/sb_corpus_001/banks --thread-diagnosis

# Stage 4 (one condition; ids = csv list or START:COUNT range)
N_WORKERS=10 bash experiments/spreadsheetbench/dispatcher.sh eval \
    experiments/spreadsheetbench/reasoning_bank_eval.yaml ours \
    runs/sb_corpus_001/banks/ours_subset_matched.jsonl \
    "$(tr -d '\n' < experiments/spreadsheetbench/data/held_out_idx.txt)" \
    runs/sb_eval_001/ours.jsonl
```

Output lands under `runs/<ROOT_RUN>/`. Stage 5 groups eval records by base
task id (an instance id `13-1#2` is test case 2 of base task `13-1`) over the
bases where all three conditions have all 3 test cases, and reports pass@1
(all test cases of a base pass) and mean_score (per-base mean test-case pass
rate).

The two eval-size knobs differ in granularity: `N_HELD` caps raw instance
ids, `N_HELD_BASES` caps whole base tasks (all 3 of their test cases).

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-4.1-mini` | model for **all** stages — corpus policy, harness agent, induction, eval. Any provider (`openai/…`, `vertex_ai/claude-…`, `gemini/…`) |
| `<PROVIDER>_API_KEY(S)` | per `MODEL` | `$OPENAI_API_KEY`, `$GEMINI_API_KEYS`, or ADC on Vertex. Corpus round-robins the pool, induce uses key 0 |
| `ROOT_RUN` | `runs/sb_reproduce_<ts>` | output dir |
| `N_TRAIN` | `100` | train tasks for corpus |
| `CORPUS_WORKERS` | `#keys` | dispatcher workers, Stage 1 |
| `EVAL_WORKERS` | `10` | dispatcher workers, Stage 4 |
| `HELD_IDS_FILE` | `data/held_out_idx.txt` | held-out instance ids |
| `N_HELD` | all | cap on raw held-out instance ids |
| `N_HELD_BASES` | all | cap on complete held-out base tasks |
| `CORPUS_YAML` | `experiments/spreadsheetbench/corpus.yaml` | Stage 1 config |
| `EVAL_YAML` | `experiments/spreadsheetbench/reasoning_bank_eval.yaml` | Stage 4 config |
| `SB_PYTHON` | the launching interpreter | interpreter used by `dispatcher.sh` |
| `SKIP_CORPUS/SKIP_INDUCE/SKIP_SUBSET/SKIP_EVAL=1` | — | resume partial pipelines |

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | one-command entry point chaining all stages |
| `reproduce_smoke.sh` | the same stages at a reduced task count |
| `corpus.yaml` | Stage 1 Orchestrator + Harness Agent config |
| `corpus_smoke.yaml` | reduced-scale Stage 1 config |
| `induce.py` | Stage 2 orig/ours induction (Trace2Skill-ported prompts, `--thread-diagnosis`) |
| `prompts.py` | policy system prompt + skill-injection + skill-induction prompts |
| `reasoning_bank_eval.py` / `reasoning_bank_eval.yaml` | Stage 4 eval driver + config (MMR retrieval, multi-test-case grading) |
| `reasoning_bank_eval_smoke.yaml` | reduced-scale Stage 4 config |
| `worker.py` / `dispatcher.sh` | parallel corpus + eval dispatch (work-stealing queue, key round-robin, resumable) |
| `data/held_out_idx.txt` | fixed held-out instance ids for Stage 4 |
| `data/` | dataset location (download; gitignored) |
| `../../envharness/bridges/spreadsheetbench/` | the SpreadsheetBench Bridge (`SpreadsheetBenchEnv`) |
