# EnvHarness on OfficeQA

Factual document-QA over parsed U.S. Treasury Bulletin tables (the gated
`databricks/officeqa` dataset on Hugging Face). Grading is a teacher-free
normalized exact-match — no LibreOffice, no LLM judge.

## Environment

```bash
conda create -n eh-officeqa python=3.12 && conda activate eh-officeqa
pip install -e .                      # pydantic, pyyaml, litellm

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key
```

**Data (does not ship).** OfficeQA is gated on Hugging Face. Accept access to
[`databricks/officeqa`](https://huggingface.co/datasets/databricks/officeqa),
then materialize two things yourself:

1. **The question/answer payload** — the dataset's `officeqa_full.csv`, placed
   next to the id split:

   ```
   experiments/officeqa/data/officeqa_full.csv
   ```

   Only the fixed id split ships in this repo
   (`data/officeqa_id_split/{train,val,test}/items.json` — 50 / 24 / 172 uids).
   It pins the partition without redistributing the gated payload, and
   `data/officeqa_id_split/split_manifest.json` records the exact source
   revision those ids came from.

2. **The parsed document corpus** — `treasury_bulletins_parsed/`, a directory of
   `.txt` files, anywhere on disk:

   ```bash
   export OFFICEQA_DOCS_DIR=~/officeqa/treasury_bulletins_parsed
   # (or set reset_options.docs_root in corpus.yaml / reasoning_bank_eval.yaml)
   ```

The preflight checks both:

```bash
python scripts/check_env.py officeqa
```

## Run

```bash
# Reduced task count; the values it sets are at the top of the script.
bash experiments/officeqa/reproduce_smoke.sh

# Full run: corpus -> induce -> subset -> eval -> report.
python experiments/officeqa/reproduce.py
```

Run `reproduce.py` with the interpreter of the env you installed into: it
uses `sys.executable` and forwards it to `dispatcher.sh` as `SB_PYTHON`, so
corpus and eval workers land in the same env as the driver.

`reasoning_bank_eval.py` is the underlying eval driver (the dispatcher's
`worker.py` calls its `run_episode`); it also runs standalone with
`--config` / `--bank-overrides`. A stage can be run by hand:

```bash
N_WORKERS=6 bash experiments/officeqa/dispatcher.sh corpus \
    experiments/officeqa/corpus.yaml oqa_corpus_001 0:50 \
    runs/oqa_corpus_001/done.jsonl
```

Everything lands under `runs/<ROOT_RUN>/`; eval records carry the
normalized-exact-match verdict plus a token-level F1 for partial-credit
analysis. Transient API errors are excluded from the denominator rather than
counted as failures.

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-4.1-mini` | model for **all** stages — corpus policy, harness agent, induction, eval. Any provider (`openai/…`, `vertex_ai/claude-…`, `gemini/…`) |
| `<PROVIDER>_API_KEY(S)` | per `MODEL` | `$OPENAI_API_KEY`, `$GEMINI_API_KEYS`, or ADC on Vertex. Corpus round-robins the pool, induce uses key 0 |
| `ROOT_RUN` | `runs/officeqa_reproduce_<ts>` | output dir |
| `N_TASKS` | `50` | train tasks for corpus |
| `CORPUS_WORKERS` | `6` | dispatcher workers, Stage 1 |
| `EVAL_WORKERS` | `18` | dispatcher workers, Stage 4 |
| `EVAL_RANGE` | `0:172` | held-out eval slice, `START:COUNT` |
| `CORPUS_YAML` | `experiments/officeqa/corpus.yaml` | Stage 1 config |
| `EVAL_YAML` | `experiments/officeqa/reasoning_bank_eval.yaml` | Stage 4 config |
| `DOCS_DIR` | `~/officeqa/treasury_bulletins_parsed` | parsed-doc corpus root (exported as `OFFICEQA_DOCS_DIR`) |
| `SB_PYTHON` | the launching interpreter | interpreter used by `dispatcher.sh` |
| `SKIP_CORPUS=1` | — | reuse `$ROOT_RUN/corpus/traces.jsonl` |
| `SKIP_INDUCE=1` | — | reuse `$ROOT_RUN/corpus/banks/*_full.jsonl` |
| `SKIP_SUBSET=1` | — | reuse `$ROOT_RUN/corpus/banks/*_subset*.jsonl` |
| `SKIP_EVAL=1` | — | stop after Stage 3 |

## Files

| File | Role |
|---|---|
| `reproduce.py` | one-command driver chaining all stages |
| `reproduce_smoke.sh` | the same stages at a reduced task count |
| `corpus.yaml` | Stage 1 corpus config |
| `corpus_smoke.yaml` | reduced-scale Stage 1 config |
| `prompts.py` | policy system prompt + skill-induction prompts |
| `induce.py` | Stage 2: corpus traces → orig/ours skill banks (`--thread-diagnosis`) |
| `reasoning_bank_eval.py` / `reasoning_bank_eval.yaml` | Stage 4 eval driver + config |
| `reasoning_bank_eval_smoke.yaml` | reduced-scale Stage 4 config |
| `worker.py` / `dispatcher.sh` | parallel corpus + eval dispatch (work-stealing queue, key round-robin, resumable) |
| `data/officeqa_id_split/` | the fixed train/val/test id split + its source manifest |
| `data/officeqa_full.csv` | the gated QA payload — **not** redistributed here; fetch it yourself |
| `../../envharness/bridges/officeqa/` | the OfficeQA Bridge (`OfficeQAEnv`) |
