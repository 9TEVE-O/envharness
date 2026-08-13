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

"""Reproduce the WebArena experiment (corpus -> banks -> eval, 3 conditions x 4 sites).

Single entry point that chains all 3 stages::

  Stage 1  mutation corpus generation    (per-task dispatch, N_CORPUS_WORKERS parallel)
  Stage 2  cascade bank induction        (V=cascade, A=baseline-only intra-pairs)
  Stage 3  eval (V vs A vs C × 4 sites)  (dispatcher.sh per site, 3 containers each)

Environment::

    GEMINI_API_KEY     single key (or GEMINI_API_KEYS for comma-separated pool)
    WEBARENA_PYTHON    python binary, default: the launching interpreter
    ROOT_RUN           output dir, default experiments/webarena/runs/reproduce_<timestamp>
    N_CORPUS_WORKERS   parallel corpus workers, default 4
    N_CORPUS_PER_SITE  train tasks per site, default 20 (paper=20)
    TOPK               eval retrieval top-K, default 5

    WebArena docker containers must be running:
      forum, forum_1, forum_2         (reddit)
      shopping, shopping_1, shopping_2
      shopping_admin, shopping_admin_1, shopping_admin_2
      gitlab, gitlab_1, gitlab_2

    WA_REDDIT, WA_SHOPPING, WA_SHOPPING_ADMIN, WA_GITLAB env vars should be set,
    or defaults (127.0.0.1 ports 19999/17770/17780/18023) are used.

Skip stages (re-run partial pipelines)::

    SKIP_CORPUS=1   re-use $ROOT_RUN/corpus/all_traces.jsonl
    SKIP_INDUCE=1   re-use $ROOT_RUN/banks/*.jsonl
    SKIP_EVAL=1     stop after Stage 2

Usage (from the repo root)::

    export GEMINI_API_KEY='...'
    python experiments/webarena/reproduce.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from envharness.infra.model import (key_env, key_pool,
                                     missing_key_message, pool_env)

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

# Every stage runs with this interpreter; defaults to the one that launched
# the driver, so activating your env is enough.
PY = os.environ.get("WEBARENA_PYTHON") or sys.executable
EXPDIR = ROOT / "experiments" / "webarena"

MODEL = os.environ.get("MODEL", "openai/gpt-4.1-mini")
# Children (corpus workers, dispatcher, eval workers) resolve their client
# through envharness.infra.model, which honours EH_MODEL.
os.environ["EH_MODEL"] = MODEL

# Keys belong to whatever provider MODEL names: $OPENAI_API_KEY on GPT,
# nothing at all on Vertex (ADC).
_missing = missing_key_message(MODEL)
if _missing:
    sys.exit("ERROR: " + _missing)
KEYS = key_pool(MODEL)


def _child_env() -> dict:
    """os.environ copy carrying the provider's key, and only when non-empty
    (an empty-string key makes every litellm call fail auth)."""
    env = os.environ.copy()
    env["WEBARENA_PYTHON"] = PY
    if KEYS:
        env.update(key_env(MODEL, KEYS[0]))
    else:
        env.pop("GEMINI_API_KEY", None)
    return env

N_CORPUS_WORKERS = int(os.environ.get("N_CORPUS_WORKERS") or 4)
N_CORPUS_PER_SITE = os.environ.get("N_CORPUS_PER_SITE") or "20"
TOPK = int(os.environ.get("TOPK") or 5)
# N_EVAL_PER_SITE: cap how many tasks Stage 3 evaluates per site.
# Unset = the full per-site task list.
N_EVAL_PER_SITE = os.environ.get("N_EVAL_PER_SITE")
# CORPUS_YAML: Stage 1 orchestrator config.
CORPUS_YAML = os.environ.get(
    "CORPUS_YAML", str(Path(__file__).resolve().parent / "corpus.yaml"))

ROOT_RUN = Path(os.environ.get("ROOT_RUN") or
                str(EXPDIR / "runs" / f"reproduce_{_dt.datetime.now():%Y%m%d_%H%M%S}"))
ROOT_RUN.mkdir(parents=True, exist_ok=True)
print(f"[run] output root: {ROOT_RUN}", flush=True)

# WebArena env vars
WA_DEFAULTS = {
    "WA_REDDIT": "http://127.0.0.1:19999",
    "WA_SHOPPING": "http://127.0.0.1:17770",
    "WA_SHOPPING_ADMIN": "http://127.0.0.1:17780/admin",
    "WA_GITLAB": "http://127.0.0.1:18023",
    "WA_WIKIPEDIA": "http://stub",
    "WA_MAP": "http://stub",
    "WA_HOMEPAGE": "http://stub",
}
for k, v in WA_DEFAULTS.items():
    os.environ.setdefault(k, v)
    os.environ.setdefault(k[3:], v)


# ---------------------------------------------------------------------------
# Stage 1: Corpus generation (per-task dispatch)
# ---------------------------------------------------------------------------

def stage1_corpus() -> Path:
    corpus_dir = ROOT_RUN / "corpus"
    traces_path = corpus_dir / "all_traces.jsonl"

    if os.environ.get("SKIP_CORPUS") and traces_path.exists():
        n = sum(1 for _ in traces_path.open())
        print(f"[skip Stage 1] re-using {traces_path} ({n} traces)", flush=True)
        return traces_path

    print(f"\n{'='*60}")
    print(f"  Stage 1: Corpus Generation")
    print(f"  workers={N_CORPUS_WORKERS}, per_site={N_CORPUS_PER_SITE}")
    print(f"  {_dt.datetime.now():%H:%M:%S}")
    print(f"{'='*60}", flush=True)

    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Build task list (sample N per site, seed=42 for reproducibility)
    n_per_site = N_CORPUS_PER_SITE
    task_ids_str = subprocess.check_output(
        [PY, "-c", f"""
import json, os, random
random.seed(42)
n_per_site = '{n_per_site}'
task_dir = '{EXPDIR / "tasks"}'
ids = []
for site in ['reddit','shopping','shopping_admin','gitlab']:
    site_ids = json.load(open(os.path.join(task_dir, site + '.json')))
    random.shuffle(site_ids)
    n = len(site_ids) if n_per_site.upper() == 'ALL' else int(n_per_site)
    ids.extend(site_ids[:n])
random.shuffle(ids)
print(' '.join(str(x) for x in ids))
"""],
        text=True,
    ).strip()
    task_ids = task_ids_str.split()
    total = len(task_ids)

    # Skip already-done tasks
    done_tasks = set()
    for f in corpus_dir.glob("task_*.jsonl"):
        if f.stat().st_size > 0:
            done_tasks.add(f.stem.replace("task_", ""))
    remaining = [t for t in task_ids if t not in done_tasks]
    print(f"  Total: {total}, already done: {len(done_tasks)}, "
          f"remaining: {len(remaining)}", flush=True)

    if remaining:
        # Write queue file
        queue_path = corpus_dir / "_queue.txt"
        queue_path.write_text("\n".join(remaining))

        # Shell-based worker loop over a flock-guarded shared task queue
        env = _child_env()
        worker_script = f"""
set -euo pipefail
QUEUE="{queue_path}"
LOCK="{corpus_dir / '_queue.lock'}"
CORPUS_DIR="{corpus_dir}"
CONFIG="{CORPUS_YAML}"

grab_task() {{
  ( flock -x 200; head -1 "$QUEUE" 2>/dev/null; sed -i '1d' "$QUEUE" 2>/dev/null ) 200>"$LOCK"
}}

while true; do
  TID=$(grab_task)
  [ -z "$TID" ] && exit 0
  OUT="$CORPUS_DIR/task_${{TID}}.jsonl"
  [ -f "$OUT" ] && [ -s "$OUT" ] && continue
  RN="corpus_t${{TID}}_$$"
  {PY} scripts/run_harness.py --config "$CONFIG" --run-name "$RN" --task-ids "$TID" >/dev/null 2>&1 || true
  if [ -f "runs/$RN/traces.jsonl" ]; then
    cp "runs/$RN/traces.jsonl" "$OUT"
    rm -rf "runs/$RN"
  fi
  DONE=$(ls "$CORPUS_DIR"/task_*.jsonl 2>/dev/null | wc -l)
  echo "  task $TID done ($DONE/{total})"
done
"""
        procs = []
        for i in range(min(N_CORPUS_WORKERS, len(remaining))):
            log_path = corpus_dir / f"worker_{i}.log"
            with log_path.open("w") as lf:
                p = subprocess.Popen(
                    ["bash", "-c", worker_script],
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                    start_new_session=True, cwd=str(ROOT),
                )
                procs.append((i, p))
                print(f"  worker {i} started (pid={p.pid})", flush=True)

        print(f"  waiting for {len(procs)} workers...", flush=True)
        for i, p in procs:
            rc = p.wait()
            if rc != 0:
                print(f"  [WARN] worker {i} exited rc={rc}", flush=True)

        # Cleanup
        queue_path.unlink(missing_ok=True)
        (corpus_dir / "_queue.lock").unlink(missing_ok=True)

    # Merge per-task traces
    parts = sorted(corpus_dir.glob("task_*.jsonl"))
    with traces_path.open("w") as out:
        for p in parts:
            out.write(p.read_text())
    n = sum(1 for _ in traces_path.open())
    print(f"  Corpus done: {n} traces from {len(parts)} tasks", flush=True)
    return traces_path


# ---------------------------------------------------------------------------
# Stage 2: Cascade bank induction
# ---------------------------------------------------------------------------

def stage2_induce(traces: Path) -> tuple[Path, Path]:
    banks_dir = ROOT_RUN / "banks"
    ours_path = banks_dir / "ours_full.jsonl"
    orig_path = banks_dir / "orig_full.jsonl"

    if os.environ.get("SKIP_INDUCE") and ours_path.exists() and orig_path.exists():
        n_v = sum(1 for _ in ours_path.open())
        n_a = sum(1 for _ in orig_path.open())
        print(f"[skip Stage 2] re-using banks: V={n_v}, A={n_a}", flush=True)
        return ours_path, orig_path

    print(f"\n{'='*60}")
    print(f"  Stage 2: Cascade Bank Induction")
    print(f"  {_dt.datetime.now():%H:%M:%S}")
    print(f"{'='*60}", flush=True)

    env = _child_env()
    builder = EXPDIR / "build_reasoning_bank.py"
    subprocess.run(
        [PY, str(builder),
         "--traces", str(traces),
         "--out-dir", str(banks_dir),
         "--concurrency", "6"],
        env=env, check=True, cwd=str(ROOT),
    )

    n_v = sum(1 for _ in ours_path.open()) if ours_path.exists() else 0
    n_a = sum(1 for _ in orig_path.open()) if orig_path.exists() else 0
    print(f"  Banks: V={n_v} items, A={n_a} items", flush=True)
    return ours_path, orig_path


# ---------------------------------------------------------------------------
# Stage 3: Eval (V vs A vs C × 4 sites)
# ---------------------------------------------------------------------------

def _eval_tasks_file(site: str, eval_dir: Path) -> Path:
    """Per-site eval task list. Returns the shipped full list unless
    N_EVAL_PER_SITE caps it, in which case a truncated copy is written under
    the run dir (the shipped tasks/ files are never modified)."""
    full = EXPDIR / "tasks" / f"{site}.json"
    if not N_EVAL_PER_SITE:
        return full
    ids = json.load(full.open())[: int(N_EVAL_PER_SITE)]
    sub_dir = eval_dir / "tasks_subset"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub = sub_dir / f"{site}.json"
    sub.write_text(json.dumps(ids))
    return sub


def stage3_eval(ours_bank: Path, orig_bank: Path) -> Path:
    eval_dir = ROOT_RUN / "eval"

    if os.environ.get("SKIP_EVAL"):
        print(f"[skip Stage 3]", flush=True)
        return eval_dir

    print(f"\n{'='*60}")
    print(f"  Stage 3: Evaluation (V vs A vs C × 4 sites)")
    print(f"  V bank: {sum(1 for _ in ours_bank.open())} items")
    print(f"  A bank: {sum(1 for _ in orig_bank.open())} items")
    print(f"  top-K: {TOPK}")
    print(f"  {_dt.datetime.now():%H:%M:%S}")
    print(f"{'='*60}", flush=True)

    eval_dir.mkdir(parents=True, exist_ok=True)
    dispatcher = EXPDIR / "dispatcher.sh"

    env = _child_env()
    if KEYS:
        env.update(pool_env(MODEL, KEYS))

    procs = []
    for site in ["reddit", "shopping", "shopping_admin", "gitlab"]:
        tasks_file = _eval_tasks_file(site, eval_dir)
        log_path = eval_dir / f"{site}.log"
        site_script = f"""
set -euo pipefail
cd "{ROOT}"
echo "== ours / {site} =="
bash "{dispatcher}" {site} "{ours_bank}" "{eval_dir}/ours_{site}.jsonl" "{tasks_file}" {TOPK}
echo "== orig / {site} =="
bash "{dispatcher}" {site} "{orig_bank}" "{eval_dir}/orig_{site}.jsonl" "{tasks_file}" {TOPK}
echo "== nobank / {site} =="
bash "{dispatcher}" {site} none "{eval_dir}/nobank_{site}.jsonl" "{tasks_file}" {TOPK}
"""
        with log_path.open("w") as lf:
            p = subprocess.Popen(
                ["bash", "-c", site_script],
                env=env, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append((site, p))
            print(f"  [launch] {site} (pid={p.pid})", flush=True)

    print(f"  Waiting for eval...", flush=True)
    for site, p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"  [WARN] {site} exited rc={rc}", flush=True)

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    _print_results(eval_dir)
    return eval_dir


def _count_results(eval_dir: Path, cond: str, site: str) -> tuple[int, int]:
    f = eval_dir / f"{cond}_{site}.jsonl"
    n = w = 0
    if f.exists():
        for line in f.open():
            if line.strip():
                n += 1
                try:
                    if json.loads(line).get("success"):
                        w += 1
                except Exception:
                    pass
    return n, w


def _print_results(eval_dir: Path):
    sites = ["reddit", "shopping", "shopping_admin", "gitlab"]
    conds = ["ours", "orig", "nobank"]
    print(f"{'Site':<14} {'V(ours)':>9} {'A(orig)':>9} {'C(none)':>9} "
          f"{'V-A':>7} {'V-C':>7}")
    print("-" * 60)

    per_site: dict[str, dict[str, tuple[int, int]]] = {}
    for site in sites:
        vals = {cond: _count_results(eval_dir, cond, site) for cond in conds}
        per_site[site] = vals
        row = f"{site:<14}"
        for c in conds:
            n, w = vals[c]
            row += f" {w/n*100:>5.1f}%  " if n else "      -  "
        no, wo = vals["ours"]
        na, wa = vals["orig"]
        nc, wc = vals["nobank"]
        row += f" {(wo/no-wa/na)*100:>+5.1f}p" if (no and na) else "      -"
        row += f" {(wo/no-wc/nc)*100:>+5.1f}p" if (no and nc) else "      -"
        print(row)

    # Macro average: only compare means over the SAME site set. Restrict
    # to sites where ALL conditions-with-data have results; annotate when
    # that drops sites (a missing condition file must not skew the macro).
    conds_with_data = [c for c in conds
                       if any(per_site[s][c][0] for s in sites)]
    common = [s for s in sites
              if all(per_site[s][c][0] for c in conds_with_data)]
    if common and "ours" in conds_with_data and "orig" in conds_with_data:
        print("-" * 60)
        mean = lambda c: sum(per_site[s][c][1] / per_site[s][c][0] * 100
                             for s in common) / len(common)
        mv, ma = mean("ours"), mean("orig")
        has_c = "nobank" in conds_with_data
        mc = mean("nobank") if has_c else 0.0
        c_str = f"{mc:>5.1f}%" if has_c else "    -"
        vc_str = f"{mv-mc:>+5.1f}p" if has_c else "     -"
        print(f"{'Macro avg':<14} {mv:>5.1f}%   {ma:>5.1f}%   {c_str}   "
              f"{mv-ma:>+5.1f}p {vc_str}")
        if len(common) < len(sites):
            print(f"  (macro avg restricted to sites with data in all "
                  f"compared conditions: {', '.join(common)})")
    elif not common:
        print("-" * 60)
        print("Macro avg: - (no site has data for all compared conditions)")

    summary_path = eval_dir / "summary.txt"
    with summary_path.open("w") as sf:
        sf.write(f"WebArena Eval Results ({_dt.datetime.now():%Y-%m-%d %H:%M})\n")
        for site in sites:
            vals = per_site[site]
            no, wo = vals["ours"]
            na, wa = vals["orig"]
            nc, wc = vals["nobank"]
            sf.write(f"{site}: V={wo}/{no} A={wa}/{na} C={wc}/{nc}\n")


def main() -> int:
    t0 = time.time()
    traces = stage1_corpus()
    ours_bank, orig_bank = stage2_induce(traces)
    eval_dir = stage3_eval(ours_bank, orig_bank)

    print(f"\n=== experiment complete ===", flush=True)
    print(f"  root:   {ROOT_RUN}")
    print(f"  corpus: {traces}")
    n_v = sum(1 for _ in ours_bank.open()) if ours_bank.exists() else 0
    n_a = sum(1 for _ in orig_bank.open()) if orig_bank.exists() else 0
    print(f"  banks:  V={n_v} items, A={n_a} items")
    print(f"  eval:   {eval_dir}")
    print(f"  wall:   {(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
