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

"""One-command SWE-bench reproduce: corpus -> banks -> 3-condition eval.

Protocol:
  Stage 1  CORPUS   Per-task LLMHarnessAgent mutation runs over SWE-bench-Lite
                    (one orchestrator run per task via scripts/run_harness.py
                    + experiments/swebench/corpus.yaml). The task set is a
                    reproducible-random draw of N_TASKS offsets from the full
                    Lite pool (seed TASK_SEED), NOT the first-N contiguous
                    block. Each run's traces.jsonl carries kind=baseline /
                    exploration / accepted rollouts for that task.
  Stage 2  BANKS    orig  = build_orig_bank --pair-mode
                            intra_baseline  (un-mutated Lite traces only)
                    ours  = build_ours_bank --pair-mode cascade
                            (mutated traces preferred, baseline fallback)
                    SAME distillation prompts/pipeline for both arms; the
                    ONLY differential is the trace source.
  Stage 3  EVAL     reasoning_bank_eval.py on Verified-minus-Lite (n=407, built in) x
                    {nobank, orig, ours}. gemini-3.5-flash, think_action,
                    T=0.4, max_steps=250.

Usage (docker group required; wrap with `sg docker -c` if needed):
    export GEMINI_API_KEYS=key1,...   # >=1; round-robin across workers
    python experiments/swebench/reproduce.py

Env knobs:
    N_TASKS             corpus size: how many Lite offsets to sample (default 100)
    LITE_POOL           Lite test-split size to sample from (default 300)
    TASK_SEED           seed for the random offset draw (default 20260727);
                        same seed -> same tasks
    TASK_OFFSETS        explicit comma-separated offsets; overrides the random
                        draw
    CORPUS_CONCURRENCY  parallel per-task corpus runs (default 4)
    CORPUS_RETRIES      auto-rerun failed corpus tasks this many times (default 2)
    EVAL_N              held-out episodes per condition (default 407 = all)
    EVAL_CONCURRENCY    parallel eval episodes (default 6)
    MODEL               default openai/gpt-4.1-mini
    ROOT_RUN            output root (default runs/swebench_headline_<ts>)
    SKIP_CORPUS=1 / SKIP_BANKS=1 / SKIP_EVAL=1   reuse earlier stages

All stages write incrementally; the driver auto-reruns failed corpus tasks
and resumes failed eval conditions instead of aborting the run.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from envharness.infra.model import key_env, key_pool, missing_key_message, pool_env

ROOT = Path(__file__).resolve().parents[2]          # repo root
# Every stage runs with this interpreter; defaults to the one that launched
# the driver, so activating your env is enough.
PY = os.environ.get("PY") or sys.executable

# The key pool belongs to whatever provider MODEL names; workers round-robin
# it so a per-key quota is spread rather than hammered. Vertex needs none.
_MODEL_EARLY = os.environ.get("MODEL", "openai/gpt-4.1-mini")
_missing = missing_key_message(_MODEL_EARLY)
if _missing:
    sys.exit(_missing)
KEYS = key_pool(_MODEL_EARLY) or [None]

N_TASKS = int(os.environ.get("N_TASKS", "100"))
# LITE_POOL: size of the SWE-bench-Lite test split the corpus samples from.
LITE_POOL = int(os.environ.get("LITE_POOL", "300"))
# TASK_SEED: RNG seed for the random task-offset draw. The SAME seed always
# yields the SAME N_TASKS offsets; set a new value for a fresh sample.
TASK_SEED = int(os.environ.get("TASK_SEED", "20260727"))
# TASK_OFFSETS: explicit comma-separated Lite offsets to run (non-contiguous
# allowed). Overrides the random draw -- e.g. an explicit set of
# baseline-fail task offsets instead of the random sample.
# pool_taskNNN dirs are named by offset.
_offsets_env = os.environ.get("TASK_OFFSETS", "").strip()
TASK_OFFSETS = (
    [int(x) for x in _offsets_env.split(",") if x.strip()]
    if _offsets_env
    else sorted(random.Random(TASK_SEED).sample(range(LITE_POOL), N_TASKS)))
CORPUS_CONCURRENCY = int(os.environ.get("CORPUS_CONCURRENCY", "4"))
CORPUS_RETRIES = int(os.environ.get("CORPUS_RETRIES", "2"))
EVAL_N = int(os.environ.get("EVAL_N", "407"))
EVAL_CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "6"))
# EVAL_MAX_STEPS: agent-step cap per eval episode. An episode that hits the
# cap ends without submitting.
EVAL_MAX_STEPS = int(os.environ.get("EVAL_MAX_STEPS", "250"))
# CORPUS_YAML: Stage 1 orchestrator config.
CORPUS_YAML = os.environ.get("CORPUS_YAML", "experiments/swebench/corpus.yaml")
MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
# Children resolve their client through envharness.infra.model, which
# honours EH_MODEL. The CLI flags below are explicit for the stages that
# take one; this covers anything spawned deeper.
os.environ["EH_MODEL"] = MODEL
ROOT_RUN = Path(os.environ.get(
    "ROOT_RUN", f"runs/swebench_headline_{time.strftime('%m%d_%H%M%S')}"))

BANKS = ROOT_RUN / "banks"
EVAL = ROOT_RUN / "eval"
CONDITIONS = [
    ("nobank", None),
    ("orig",   BANKS / "orig_bank.jsonl"),
    ("ours",   BANKS / "ours_bank.jsonl"),
]
# SKIP_ORIG=1: run only the nobank + ours conditions.
if os.environ.get("SKIP_ORIG"):
    CONDITIONS = [c for c in CONDITIONS if c[0] != "orig"]


def _env_with_key(i: int) -> dict:
    env = dict(os.environ)
    env.update(key_env(MODEL, KEYS[i % len(KEYS)]))
    env.pop("GEMINI_API_KEYS", None)
    return env


def _run(cmd: list[str], *, env: dict, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as lf:
        lf.write(f"\n=== {' '.join(cmd)} ===\n"); lf.flush()
        return subprocess.run(cmd, env=env, stdout=lf,
                              stderr=subprocess.STDOUT, cwd=ROOT).returncode


# ---------------------------------------------------------------------------
# Stage 1: corpus (per-task mutation runs over Lite)
# ---------------------------------------------------------------------------

def _pool_dir(idx: int) -> Path:
    return ROOT_RUN / f"pool_task{idx:03d}"


def _corpus_task_ok(idx: int) -> bool:
    tj = ROOT / _pool_dir(idx) / "traces.jsonl"
    if not tj.is_file() or tj.stat().st_size == 0:
        return False
    try:
        with open(tj) as f:
            return any(line.strip() for line in f)
    except OSError:
        return False


def _run_corpus_task(idx: int, slot: int) -> tuple[int, bool]:
    rel = _pool_dir(idx)
    rc = _run(
        [PY, "scripts/run_harness.py",
         "--config", CORPUS_YAML, "--model", MODEL,
         "--run-name", str(rel.relative_to("runs")) if str(rel).startswith("runs") else str(rel),
         "--n-tasks", "1", "--task-offset", str(idx)],
        env=_env_with_key(slot),
        log_path=ROOT / ROOT_RUN / "logs" / f"corpus_task{idx:03d}.log",
    )
    return idx, (rc == 0 and _corpus_task_ok(idx))


def stage_corpus() -> None:
    todo = [i for i in TASK_OFFSETS if not _corpus_task_ok(i)]
    print(f"[corpus] {len(TASK_OFFSETS)} tasks, {len(todo)} to run "
          f"(concurrency={CORPUS_CONCURRENCY})", flush=True)
    for round_i in range(CORPUS_RETRIES + 1):
        if not todo:
            break
        if round_i:
            print(f"[corpus] retry round {round_i}: re-running {len(todo)} "
                  f"failed tasks: {todo}", flush=True)
        failed: list[int] = []
        with cf.ThreadPoolExecutor(max_workers=CORPUS_CONCURRENCY) as pool:
            futs = {pool.submit(_run_corpus_task, i, slot): i
                    for slot, i in enumerate(todo)}
            for fut in cf.as_completed(futs):
                idx, ok = fut.result()
                print(f"[corpus] task {idx:03d} {'ok' if ok else 'FAILED'}",
                      flush=True)
                if not ok:
                    failed.append(idx)
        todo = sorted(failed)
    if todo:
        sys.exit(f"[corpus] FATAL: tasks still failing after "
                 f"{CORPUS_RETRIES} retries: {todo}")
    print("[corpus] done", flush=True)


# ---------------------------------------------------------------------------
# Stage 2: banks (same pipeline, different trace source)
# ---------------------------------------------------------------------------

def _check_bank(path: Path, label: str) -> None:
    p = ROOT / path
    n = sum(1 for ln in open(p)) if p.is_file() else 0
    if n == 0:
        sys.exit(f"[banks] FATAL: {label} bank is empty ({p})")
    print(f"[banks] {label}: {n} items", flush=True)


def stage_banks() -> None:
    (ROOT / BANKS).mkdir(parents=True, exist_ok=True)
    common = ["--pool-dir", str(ROOT_RUN), "--pool-glob", "pool_task*",
              "--llm-model", MODEL, "--concurrency", "4", "--dedup-titles"]
    jobs = [
        ("orig", [PY, "experiments/swebench/bank_distillation/"
                      "build_orig_bank.py",
                  *common, "--pair-mode", "intra_baseline",
                  "--out", str(BANKS / "orig_bank.jsonl")]),
        ("ours", [PY, "experiments/swebench/bank_distillation/"
                      "build_ours_bank.py",
                  *common, "--pair-mode", "cascade",
                  "--out", str(BANKS / "ours_bank.jsonl")]),
    ]
    if os.environ.get("SKIP_ORIG"):
        jobs = [j for j in jobs if j[0] != "orig"]
    # build_bank round-robins the GEMINI_API_KEYS pool across its --concurrency
    # skill-extraction workers, so pass the FULL pool (NOT single-key
    # _env_with_key, which pops GEMINI_API_KEYS -> workers 429-throttle).
    banks_env = dict(os.environ)
    banks_env.update(pool_env(MODEL, KEYS))
    for label, cmd in jobs:
        print(f"[banks] building {label} ...", flush=True)
        rc = _run(cmd, env=banks_env,
                  log_path=ROOT / ROOT_RUN / "logs" / f"bank_{label}.log")
        if rc != 0:
            sys.exit(f"[banks] FATAL: {label} builder exited {rc} "
                     f"(see logs/bank_{label}.log)")
    if not os.environ.get("SKIP_ORIG"):
        _check_bank(BANKS / "orig_bank.jsonl", "orig")
    _check_bank(BANKS / "ours_bank.jsonl", "ours")


# ---------------------------------------------------------------------------
# Stage 3: eval (sequential conditions; auto-resume on nonzero exit)
# ---------------------------------------------------------------------------

def stage_eval() -> None:
    (ROOT / EVAL).mkdir(parents=True, exist_ok=True)
    for cond_i, (label, bank) in enumerate(CONDITIONS):
        out = EVAL / f"{label}.jsonl"
        cmd = [PY, "experiments/swebench/reasoning_bank_eval.py",
               "--out", str(out), "--resume",
               "--n", str(EVAL_N), "--concurrency", str(EVAL_CONCURRENCY),
               "--model", MODEL, "--temperature", "0.4",
               "--max-steps", str(EVAL_MAX_STEPS)]
        if bank is not None:
            cmd += ["--bank", str(bank)]
        # Auto-rerun the condition on nonzero exit (resume keeps finished
        # episodes); 3 attempts before declaring the stage failed.
        # Eval uses reasoning_bank_eval.py, which round-robins the GEMINI_API_KEYS pool
        # across concurrent episodes itself. Pass the FULL pool (NOT the
        # single-key _env_with_key, which pops GEMINI_API_KEYS) -- otherwise
        # all EVAL_CONCURRENCY episodes hammer one key and 429-throttle.
        eval_env = dict(os.environ)
        eval_env.update(pool_env(MODEL, KEYS))
        # reasoning_effort=low by default. envharness.infra.model adapts it to
        # the target provider (Gemini thinkingLevel, Claude thinking budget,
        # dropped where unsupported); "off"/"none" omits it.
        eval_env.setdefault("RB_REASONING_EFFORT", "low")
        for attempt in range(3):
            print(f"[eval] {label} (attempt {attempt + 1})", flush=True)
            rc = _run(cmd, env=eval_env,
                      log_path=ROOT / ROOT_RUN / "logs" / f"eval_{label}.log")
            if rc == 0:
                break
            print(f"[eval] {label} exited {rc}; resuming in 60s", flush=True)
            time.sleep(60)
        else:
            sys.exit(f"[eval] FATAL: {label} kept failing "
                     f"(see logs/eval_{label}.log)")
    _summary()


def _summary() -> None:
    rows = []
    for label, _ in CONDITIONS:
        p = ROOT / EVAL / f"{label}.jsonl"
        raw = [json.loads(ln) for ln in open(p)] if p.is_file() else []
        # Dedup by instance_id, LAST row wins: a re-run of an error
        # datapoint appends a fresh row that must override the stale one,
        # so the denominator stays at the true task count (not inflated by
        # duplicate rows) and the clean result replaces the infra failure.
        by_iid = {}
        for r in raw:
            iid = r.get("instance_id")
            if iid is None:
                by_iid[id(r)] = r          # keep un-keyable rows as-is
            else:
                by_iid[iid] = r
        recs = list(by_iid.values())
        n = len(recs)
        won = sum(1 for r in recs if r.get("success"))
        steps = [r.get("duration_steps") or 0 for r in recs]
        rows.append((label, won, n,
                     (sum(steps) / len(steps)) if steps else 0.0))
    print("\n=== SWE-bench Verified-minus-Lite summary ===")
    for label, won, n, avg in rows:
        sr = won / n * 100 if n else 0.0
        print(f"  {label:8s} SR={won}/{n} = {sr:.2f}%   avg_steps={avg:.1f}")
    (ROOT / ROOT_RUN / "summary.json").write_text(json.dumps(
        [{"condition": l, "n_won": w, "n": n, "sr": (w / n if n else 0.0),
          "avg_steps": a} for l, w, n, a in rows], indent=2))
    print(f"summary: {ROOT_RUN}/summary.json")


def main() -> None:
    (ROOT / ROOT_RUN / "logs").mkdir(parents=True, exist_ok=True)
    # Fail fast if this process can't talk to docker.
    if subprocess.run(["docker", "info"], capture_output=True).returncode:
        sys.exit("docker unreachable from this process -- run via "
                 "`sg docker -c '... reproduce.py'` or fix group membership")
    print(f"[reproduce] ROOT_RUN={ROOT_RUN}  model={MODEL}  "
          f"keys={len(KEYS)}", flush=True)
    if not os.environ.get("SKIP_CORPUS"):
        stage_corpus()
    if not os.environ.get("SKIP_BANKS"):
        stage_banks()
    if not os.environ.get("SKIP_EVAL"):
        stage_eval()


if __name__ == "__main__":
    main()
