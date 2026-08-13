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

"""One-command reproduce for the OfficeQA ReasoningBank result (ours > orig > base).

Chains all stages, mirroring experiments/alfworld/reproduce.py::

  Stage 1  transferability-constrained mutation corpus   (dispatcher.sh corpus, 50 train tasks)
  Stage 2  paired-diff induction WITH mutator-diagnosis  (induce.py --thread-diagnosis)
  Stage 3  combined ours subset (mutated + orig fallback)
  Stage 4  eval base / orig / ours on the 172-task test split (dispatcher.sh eval)
  Stage 5  matched base/orig/ours report

Configuration:
  * The policy model and reasoning_effort are set in the YAMLs.
  * The mutation prompt's constraints live in corpus.yaml.
  * Induction runs with --thread-diagnosis, which threads the Harness Agent's
    diagnosis into the ours paired-diff prompt.

Prerequisites (one-time):
  * A Python 3.12 env with the repo installed -- run with its interpreter.
  * OfficeQA docs materialized (gated databricks/officeqa on HF). Default docs root
    ~/officeqa/treasury_bulletins_parsed ; override with OFFICEQA_DOCS_DIR.
    The dataset payload (officeqa_full.csv + officeqa_id_split) ships under
    experiments/officeqa/data/ .

Environment:
    GEMINI_API_KEYS    comma-separated keys (corpus round-robins; induce uses KEYS[0]).
                       Falls back to GEMINI_API_KEY.
    ROOT_RUN           output dir, default runs/officeqa_reproduce_<timestamp>
    N_TASKS            train tasks for corpus, default 50
    CORPUS_WORKERS     dispatcher workers for corpus, default 6
    EVAL_WORKERS       dispatcher workers for eval, default 18
    DOCS_DIR           docs root, default ~/officeqa/treasury_bulletins_parsed

Skip stages (re-run partial pipelines):
    SKIP_CORPUS=1  re-use $ROOT_RUN/corpus/traces.jsonl
    SKIP_INDUCE=1  re-use $ROOT_RUN/banks/*_full.jsonl
    SKIP_SUBSET=1  re-use $ROOT_RUN/banks/*_subset*.jsonl
    SKIP_EVAL=1    stop after Stage 3

Usage:
    export GEMINI_API_KEYS=k1,k2,k3,k4,k5,k6
    <your-env>/bin/python experiments/officeqa/reproduce.py

"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from envharness.infra.model import key_env, key_pool, missing_key_message, pool_env

ROOT = Path(__file__).resolve().parents[2]           # repo root
EXP = "experiments/officeqa"
PY = sys.executable


def _env(k, default): return os.environ.get(k, default)


def _keys() -> str:
    """Comma-separated key pool for whatever MODEL names.

    Reads $OPENAI_API_KEY on GPT and nothing on Vertex (ADC), so the driver
    is runnable on any provider rather than only on Gemini.
    """
    model = _env("MODEL", "openai/gpt-4.1-mini")
    missing = missing_key_message(model)
    if missing:
        sys.exit(missing)
    return ",".join(key_pool(model))


def _run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        sys.exit(f"stage failed (rc={r.returncode}): {' '.join(cmd)}")


def main() -> int:
    keys = _keys()
    docs = os.path.expanduser(_env("DOCS_DIR", "~/officeqa/treasury_bulletins_parsed"))
    if not os.path.isdir(docs):
        sys.exit(f"OfficeQA docs not found at {docs!r}. Materialize databricks/officeqa "
                 "(treasury_bulletins_parsed) from HF, or set DOCS_DIR.")
    ts = _dt.datetime.now().strftime("%m%d_%H%M%S")   # not used for logic, just a name
    root = Path(_env("ROOT_RUN", f"runs/officeqa_reproduce_{ts}"))
    n_tasks = int(_env("N_TASKS", "50"))
    corpus_w = _env("CORPUS_WORKERS", "6")
    eval_w = _env("EVAL_WORKERS", "18")
    # EVAL_RANGE is "START:COUNT" over the test split; 0:172 is the whole
    # split.
    eval_range = _env("EVAL_RANGE", "0:172")
    eval_n = int(eval_range.split(":")[1])
    # Stage 1 / Stage 4 configs.
    corpus_yaml = _env("CORPUS_YAML", f"{EXP}/corpus.yaml")
    eval_yaml = _env("EVAL_YAML", f"{EXP}/reasoning_bank_eval.yaml")
    corpus_dir = root / "corpus"
    banks = corpus_dir / "banks"
    eval_dir = root / "eval"
    (root).mkdir(parents=True, exist_ok=True)

    # SB_PYTHON must be propagated: dispatcher.sh falls back to a hardcoded
    # interpreter path that need not exist. Pinning it to THIS interpreter
    # keeps corpus + eval workers in the same env as the driver.
    env = {**os.environ, "PYTHONWARNINGS": "ignore::UserWarning", "PYTHONPATH": ".",
           # EH_MODEL reaches every stage the dispatcher spawns.
           "EH_MODEL": _env("MODEL", "openai/gpt-4.1-mini"),
           "HOME": os.path.expanduser("~"), "SB_PYTHON": PY,
           **pool_env(_env("MODEL", "openai/gpt-4.1-mini"),
                      [k for k in keys.split(",") if k]),
           "OFFICEQA_DOCS_DIR": docs}

    # ---- Stage 1: corpus (transferability mutation, low policy) ----
    if _env("SKIP_CORPUS", "") != "1":
        print("=== Stage 1: corpus (transferability mutation) ===")
        (corpus_dir).mkdir(parents=True, exist_ok=True)
        _run(["bash", f"{EXP}/dispatcher.sh", "corpus", corpus_yaml,
              str(corpus_dir).replace("runs/", "", 1) if str(corpus_dir).startswith("runs/") else str(corpus_dir),
              f"0:{n_tasks}", str(corpus_dir / "done.jsonl")],
             {**env, "N_WORKERS": corpus_w})
    # merge traces
    traces = corpus_dir / "traces.jsonl"
    with open(traces, "w") as out:
        for f in sorted(corpus_dir.glob("corpus_task*/traces.jsonl")):
            out.write(f.read_text())
    print(f"[repro] merged {sum(1 for _ in open(traces))} traces")

    # ---- Stage 2: induce (--thread-diagnosis) ----
    if _env("SKIP_INDUCE", "") != "1":
        print("=== Stage 2: induce (paired-diff + #3 diagnosis + quality gate) ===")
        _run([PY, f"{EXP}/induce.py", "--traces", str(traces), "--out-dir", str(banks),
              "--llm-model", _env("MODEL", "openai/gpt-4.1-mini"),
              "--concurrency", "6", "--thread-diagnosis"], env)

    # ---- Stage 3: combined ours subset ----
    if _env("SKIP_SUBSET", "") != "1":
        print("=== Stage 3: combined ours subset (mutated + orig fallback) ===")
        rng = random.Random(42)
        def load(f): return [json.loads(l) for l in open(f)]
        def bt(x):
            g = defaultdict(list)
            for it in x: g[str(it["source"]["task_id"])].append(it)
            return g
        og = bt(load(banks / "orig_full.jsonl")); ug = bt(load(banks / "ours_full.jsonl"))
        osub = {t: rng.choice(og[t]) for t in sorted(og, key=int)}
        (banks / "orig_subset.jsonl").write_text(
            "\n".join(json.dumps(osub[t]) for t in sorted(og, key=int)) + "\n")
        u = [rng.choice(ug[t]) if t in ug else osub[t] for t in sorted(og, key=int)]
        (banks / "ours_subset_matched.jsonl").write_text(
            "\n".join(json.dumps(x) for x in u) + "\n")
        print(f"[repro] subset {len(osub)} skills, ours differ on {len(set(ug) & set(og))} tasks")

    if _env("SKIP_EVAL", "") == "1":
        print("SKIP_EVAL=1 -> stopping after Stage 3."); return 0

    # ---- Stage 4: eval base / orig / ours ----
    print("=== Stage 4: eval base / orig / ours (172-task test split, low) ===")
    eval_dir.mkdir(parents=True, exist_ok=True)
    conds = [("nobank", "none"),
             ("orig", str(banks / "orig_subset.jsonl")),
             ("ours", str(banks / "ours_subset_matched.jsonl"))]
    for cond, bank in conds:
        out = eval_dir / f"{cond}.jsonl"
        prev = -1
        for r in range(1, 7):
            miss = _missing(out, eval_n)
            print(f"[repro] eval {cond} missing={miss} (round {r})")
            if miss <= 0 or miss == prev:
                break
            prev = miss
            nw = eval_w if r < 3 else "6"
            _run(["bash", f"{EXP}/dispatcher.sh", "eval", eval_yaml,
                  cond, bank, eval_range, str(out)], {**env, "N_WORKERS": str(nw)})

    # ---- Stage 5: report ----
    _report(eval_dir)
    return 0


def _missing(out: Path, n: int = 172) -> int:
    if not out.exists():
        return n
    rows = [json.loads(l) for l in open(out) if l.strip()]
    keep = [r for r in rows if not (r.get("error") or "").strip()]
    out.write_text("\n".join(json.dumps(r) for r in keep) + ("\n" if keep else ""))
    done = {r["task_idx"] for r in keep if r.get("task_idx") is not None}
    return len(set(range(n)) - done)


def _report(eval_dir: Path) -> None:
    def L(f):
        p = eval_dir / f
        if not p.exists(): return {}
        return {json.loads(l)["task_idx"]: bool(json.loads(l).get("success"))
                for l in open(p) if l.strip()
                and not (json.loads(l).get("error") or "").strip()
                and json.loads(l).get("task_idx") is not None}
    B, O, U = L("nobank.jsonl"), L("orig.jsonl"), L("ours.jsonl")
    c = sorted(set(B) & set(O) & set(U)); m = len(c)
    if not m:
        print("no matched tasks"); return
    sb = sum(B[k] for k in c) / m * 100
    so = sum(O[k] for k in c) / m * 100
    su = sum(U[k] for k in c) / m * 100
    uo = sum(1 for k in c if U[k] and not O[k]); ou = sum(1 for k in c if O[k] and not U[k])
    print("\n" + "=" * 60)
    print(f"OfficeQA ReasoningBank -- matched n={m}")
    print(f"  base (nobank)  {sb:.1f}%")
    print(f"  orig           {so:.1f}%")
    print(f"  ours           {su:.1f}%")
    print(f"  ours - orig    {su - so:+.1f}   (paired ours/orig {uo}:{ou})")
    print(f"  ours - base    {su - sb:+.1f}")
    print(f"  orig - base    {so - sb:+.1f}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
