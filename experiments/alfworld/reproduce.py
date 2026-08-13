# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reproduce the ALFWorld experiment (corpus -> induce -> 3-round eval).

Single entry point that chains all 4 stages::

  Stage 1  mutation corpus generation         (4 shards, contig task_ids 0..99)
  Stage 2  pair induction                     (scripts/induce_pair.py)
  Stage 3  size-matched per-task subsets       (scripts/subset.py)
  Stage 4  3-round full eval (FULL banks + MMR K=5)  (experiments/alfworld/reasoning_bank_eval.py)

Implementation note: Stage 1 spawns shards via `subprocess.Popen(
start_new_session=True)` so each shard runs in its own process group
and survives independently; `Popen.wait()` then blocks until each
shard finishes.

Environment::

    <PROVIDER>_API_KEYS  comma-separated pool for whatever MODEL names
                       ($GEMINI_API_KEYS, $OPENAI_API_KEYS; Vertex needs
                       none). Shards round-robin the pool, induce uses
                       KEYS[0], eval uses KEYS[-1]. More keys spread a
                       per-key quota; one key simply shares.
    ROOT_RUN           output dir, default runs/alfworld_headline_<timestamp>
    N_TASKS_TOTAL      train tasks for corpus, default 100
    N_SHARDS           corpus shards, default 4
    TASK_BASE_OFFSET   base task_id offset, default 0
    EVAL_START_SEEDS   comma-separated, default "0,1000,2000"; one eval
                       round runs per start seed
    EVAL_CONCURRENCY   eval inner concurrency, default 24
    MODEL              model for every stage (corpus policy, harness agent,
                       induction, eval). Any provider: openai/... , gemini/... ,
                       vertex_ai/claude-... . Exported to children as EH_MODEL.

Skip stages (re-run partial pipelines)::

    SKIP_CORPUS=1   re-use $ROOT_RUN/corpus/traces.jsonl
    SKIP_INDUCE=1   re-use $ROOT_RUN/banks/*_full.jsonl
    SKIP_SUBSET=1   re-use $ROOT_RUN/banks/*_subset*.jsonl
    SKIP_EVAL=1     stop after Stage 3
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
import time
from pathlib import Path

from envharness.infra.model import key_env, key_pool, missing_key_message

# Run from the repo root.
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

# Every stage runs with this interpreter; defaults to the one that launched
# the driver, so activating your env is enough.
PY = os.environ.get("PY") or sys.executable

MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
# Every child process resolves its client through envharness.infra.model,
# which honours EH_MODEL -- so one export reaches stages behind a shard,
# a dispatcher or the subprocess runner.
os.environ["EH_MODEL"] = MODEL

# The key pool belongs to whatever provider MODEL names, so this reads
# $OPENAI_API_KEY on GPT and nothing at all on Vertex. Shards round-robin the
# pool: more keys spread Gemini's per-key quota, fewer just share.
_missing = missing_key_message(MODEL)
if _missing:
    sys.exit(_missing)
KEYS = key_pool(MODEL) or [None]
N_SHARDS = int(os.environ.get("N_SHARDS") or 4)

N_TASKS_TOTAL = int(os.environ.get("N_TASKS_TOTAL") or 100)
TASK_BASE_OFFSET = int(os.environ.get("TASK_BASE_OFFSET") or 0)
EVAL_START_SEEDS = os.environ.get("EVAL_START_SEEDS") or "0,1000,2000"
EVAL_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY") or 24)
# Optional eval-size caps, forwarded to reasoning_bank_eval.py's --n-indist /
# --n-ood. Unset = the full 140 + 134 eval splits.
# Stage 1 / Stage 4 configs.
CORPUS_YAML = os.environ.get("CORPUS_YAML", "experiments/alfworld/corpus.yaml")
EVAL_YAML = os.environ.get(
    "EVAL_YAML", "experiments/alfworld/reasoning_bank_eval.yaml")
EVAL_N_INDIST = os.environ.get("EVAL_N_INDIST")
EVAL_N_OOD = os.environ.get("EVAL_N_OOD")

ROOT_RUN = Path(os.environ.get("ROOT_RUN") or
                 f"runs/alfworld_headline_{_dt.datetime.now():%Y%m%d_%H%M%S}")
ROOT_RUN.mkdir(parents=True, exist_ok=True)
print(f"[run] output root: {ROOT_RUN}", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: corpus generation -- parallel shards via Popen(start_new_session)
# ---------------------------------------------------------------------------

def stage1_corpus() -> Path:
    corpus_dir = ROOT_RUN / "corpus"
    traces_path = corpus_dir / "traces.jsonl"

    if os.environ.get("SKIP_CORPUS") and traces_path.exists():
        print(f"[skip Stage 1] re-using {traces_path}", flush=True)
        return traces_path

    print(f"\n=== Stage 1: corpus generation "
          f"({N_TASKS_TOTAL} tasks, {N_SHARDS} shards) ===", flush=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    # run_harness.py writes to ./runs/{run_name}/..., so run_name must be
    # corpus_dir expressed relative to runs/. os.path.relpath (instead of
    # Path.relative_to) stays valid when ROOT_RUN is absolute or outside
    # runs/ (it yields a ../-prefixed path that still resolves to corpus_dir).
    run_name_base = Path(
        os.path.relpath(corpus_dir.resolve(), (ROOT / "runs").resolve())
    ).as_posix()
    # Distribute exactly N_TASKS_TOTAL contiguous task_ids across shards
    # (base + remainder split; ceil-division would overshoot when
    # N_TASKS_TOTAL % N_SHARDS != 0, e.g. 90/4 -> 4*23 = 92 tasks).
    base, rem = divmod(N_TASKS_TOTAL, N_SHARDS)
    shard_sizes = [base + (1 if i < rem else 0) for i in range(N_SHARDS)]
    procs: list[tuple[int, subprocess.Popen, object]] = []

    offset = TASK_BASE_OFFSET
    for i in range(N_SHARDS):
        n_shard = shard_sizes[i]
        if n_shard == 0:
            continue
        log_path = corpus_dir / f"shard{i}.log"
        env = os.environ.copy()
        env.update(key_env(MODEL, KEYS[i % len(KEYS)]))
        log_f = log_path.open("w")
        p = subprocess.Popen(
            [PY, "scripts/run_harness.py",
              "--config", CORPUS_YAML,
              "--run-name", f"{run_name_base}/shard{i}",
              "--n-tasks", str(n_shard),
              "--task-offset", str(offset)],
            env=env, stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append((i, p, log_f))
        print(f"  shard {i} started (offset={offset}, n={n_shard}, "
              f"pid={p.pid})", flush=True)
        offset += n_shard

    print(f"  waiting for {len(procs)} shards...", flush=True)
    failed = []
    for i, p, log_f in procs:
        rc = p.wait()
        log_f.close()
        if rc != 0:
            print(f"[WARN] shard {i} exited rc={rc}", flush=True)
            failed.append((i, rc))

    if failed:
        sys.exit(f"[FATAL] {len(failed)} shards exited non-zero: {failed}")

    missing, parts = [], []
    for i in range(N_SHARDS):
        if shard_sizes[i] == 0:
            continue    # shard never launched (N_SHARDS > N_TASKS_TOTAL)
        p = corpus_dir / f"shard{i}" / "traces.jsonl"
        if not p.exists():
            missing.append(i)
        else:
            parts.append(p)
    if missing:
        sys.exit(f"[FATAL] shards missing traces: {missing}")
    with traces_path.open("w") as out:
        for p in parts:
            out.write(p.read_text())
    n = sum(1 for _ in traces_path.open())
    print(f"  concat -> {traces_path} ({n} records)", flush=True)
    return traces_path


# ---------------------------------------------------------------------------
# Stage 2: pair induction (RB SUCCESSFUL_SI + FAILED_SI verbatim)
# ---------------------------------------------------------------------------

def stage2_induce(traces: Path) -> tuple[Path, Path]:
    banks_dir = ROOT_RUN / "banks"
    orig_full = banks_dir / "orig_full.jsonl"
    ours_full = banks_dir / "ours_full.jsonl"

    if os.environ.get("SKIP_INDUCE") and orig_full.exists() and ours_full.exists():
        print(f"[skip Stage 2] re-using {orig_full} + {ours_full}", flush=True)
        return orig_full, ours_full

    print(f"\n=== Stage 2: pair induction ===", flush=True)
    env = os.environ.copy()
    env.update(key_env(MODEL, KEYS[0]))
    subprocess.run(
        [PY, "scripts/induce_pair.py",
          "--llm-model", MODEL,
          "--traces", str(traces),
          "--out-dir", str(banks_dir),
          "--concurrency", "8"],
        env=env, check=True,
    )
    return orig_full, ours_full


# ---------------------------------------------------------------------------
# Stage 3: 1-per-task size-matched subsets
# ---------------------------------------------------------------------------

def stage3_subset(orig_full: Path, ours_full: Path) -> tuple[Path, Path]:
    banks_dir = ROOT_RUN / "banks"
    orig_sub = banks_dir / "orig_subset.jsonl"
    ours_sub = banks_dir / "ours_subset_matched.jsonl"

    if os.environ.get("SKIP_SUBSET") and orig_sub.exists() and ours_sub.exists():
        print(f"[skip Stage 3] re-using {orig_sub} + {ours_sub}", flush=True)
        return orig_sub, ours_sub

    print(f"\n=== Stage 3: 1-per-task subset ===", flush=True)
    subprocess.run(
        [PY, "scripts/subset.py",
          "--orig", str(orig_full), "--ours", str(ours_full),
          "--out-dir", str(banks_dir)],
        check=True,
    )
    return orig_sub, ours_sub


# ---------------------------------------------------------------------------
# Stage 4: multi-round eval (start_seeds 0/1000/2000; FULL banks + MMR K=5)
# ---------------------------------------------------------------------------

def stage4_eval(orig_bank: Path, ours_bank: Path) -> Path:
    eval_dir = ROOT_RUN / "eval"

    if os.environ.get("SKIP_EVAL"):
        print(f"[skip Stage 4]", flush=True)
        return eval_dir

    rounds = EVAL_START_SEEDS.split(",")
    print(f"\n=== Stage 4: {len(rounds)}-round eval "
          f"(start_seeds={EVAL_START_SEEDS}, FULL banks + MMR K=5) ===",
          flush=True)
    env = os.environ.copy()
    # Documented protocol: eval uses KEYS[-1] only. reasoning_bank_eval.py
    # resolves the pool variable BEFORE the single-key one, so drop the
    # inherited pool from the child env or the protocol is silently ignored.
    env.update(key_env(MODEL, KEYS[-1]))
    env.pop("GEMINI_API_KEYS", None)
    # Use Popen + start_new_session=True so the eval child has its own
    # process group; if this launcher process gets killed externally,
    # the eval keeps running. Drains stdout/stderr to a log so we don't
    # lose visibility.
    log_path = eval_dir / "stage4.log"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "experiments/alfworld/reasoning_bank_eval.py",
           "--config", EVAL_YAML,
           "--bank-overrides", f"orig={orig_bank},ours={ours_bank}",
           "--concurrency", str(EVAL_CONCURRENCY),
           "--start-seeds", EVAL_START_SEEDS,
           "--out-dir", str(eval_dir)]
    if EVAL_N_INDIST:
        cmd += ["--n-indist", str(EVAL_N_INDIST)]
    if EVAL_N_OOD:
        cmd += ["--n-ood", str(EVAL_N_OOD)]
    with log_path.open("w") as lf:
        p = subprocess.Popen(
            cmd,
            env=env, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rc = p.wait()
    if rc != 0:
        sys.exit(f"[FATAL] Stage 4 reasoning_bank_eval exited rc={rc}; see {log_path}")
    return eval_dir


def main() -> int:
    t0 = time.time()
    traces = stage1_corpus()
    orig_full, ours_full = stage2_induce(traces)
    # Size-matched subsets are still built (secondary reference / ablation),
    # but the HEADLINE eval uses the FULL banks with MMR K=5 retrieval
    # (the default in reasoning_bank_eval.yaml).
    stage3_subset(orig_full, ours_full)
    eval_dir = stage4_eval(orig_full, ours_full)
    print(f"\n=== done ===")
    print(f"  root:   {ROOT_RUN}")
    print(f"  corpus: {traces} ({sum(1 for _ in traces.open())} records)")
    print(f"  banks:  {orig_full} ({sum(1 for _ in orig_full.open())} items)")
    print(f"          {ours_full} ({sum(1 for _ in ours_full.open())} items)")
    print(f"  eval:   {eval_dir}/summary.json")
    print(f"  wall:   {(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
