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

"""One-command reproduce for the SpreadsheetBench ReasoningBank result.

Full pipeline (mirrors experiments/officeqa/reproduce.py), the MMR+top_k=5 recipe:

  Stage 1  transferability-constrained mutation corpus   (dispatcher.sh corpus, corpus.yaml)
  Stage 2  paired-diff induction WITH mutator-diagnosis  (induce.py --thread-diagnosis)
  Stage 3  combined ours subset (mutated + orig fallback)
  Stage 4  eval base / orig / ours on the fixed held-out (dispatcher.sh eval, reasoning_bank_eval.yaml)
  Stage 5  912-hard report (pass@1 + mean_score, ours vs orig vs base)

Configuration:
  * The policy model and reasoning_effort are set in the YAMLs.
  * The mutation prompt's constraints live in corpus.yaml.
  * Induction runs with --thread-diagnosis.
  * Retrieval mode and top_k are set in reasoning_bank_eval.yaml.

Metrics (group test cases by base id):
  * pass@1     = fraction of base ids whose ALL test cases pass (official hard metric)
  * mean_score = per-base mean test-case pass rate (soft partial credit)

Environment:
    GEMINI_API_KEYS    comma-separated keys (corpus round-robins; induce uses KEYS[0]).
    ROOT_RUN           output dir, default runs/sb_reproduce_<timestamp>
    N_TRAIN            train tasks for corpus, default 100
    CORPUS_WORKERS     dispatcher workers for corpus, default = #keys
    EVAL_WORKERS       dispatcher workers for eval, default 10 (shared-quota sweet spot)
    HELD_IDS_FILE      held-out instance ids, default experiments/spreadsheetbench/data/held_out_idx.txt

Skip stages (resume partial pipelines):
    SKIP_CORPUS=1  reuse $ROOT_RUN/corpus/traces.jsonl
    SKIP_INDUCE=1  reuse $ROOT_RUN/banks/*_full.jsonl
    SKIP_SUBSET=1  reuse $ROOT_RUN/banks/*_subset*.jsonl
    SKIP_EVAL=1    stop after Stage 3

Usage:
    export GEMINI_API_KEYS=k1,k2,k3,k4,k5,k6
    <your-env>/bin/python experiments/spreadsheetbench/reproduce.py
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
EXP = "experiments/spreadsheetbench"
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


def _run(cmd, env) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        sys.exit(f"stage failed (rc={r.returncode}): {' '.join(str(c) for c in cmd)}")


def _held_ids() -> list[int]:
    f = _env("HELD_IDS_FILE", f"{EXP}/data/held_out_idx.txt")
    p = ROOT / f if not os.path.isabs(f) else Path(f)
    if p.exists():
        return [int(x) for x in p.read_text().replace("\n", ",").split(",") if x.strip()]
    # fallback: full 912-multi instance range
    from envharness.bridges.spreadsheetbench.dataset import load_dataset_multi
    n = len(load_dataset_multi(f"{EXP}/data/_dl/all_data_912_v0.1"))
    print(f"[repro] HELD_IDS_FILE missing -> full 912-multi range 0:{n}")
    return list(range(n))


def _missing(out: Path, ids: set[int]) -> int:
    if not out.exists():
        return len(ids)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    keep = [r for r in rows if not (r.get("error") or "").strip()]
    out.write_text("\n".join(json.dumps(r) for r in keep) + ("\n" if keep else ""))
    done = {r["task_idx"] for r in keep if r.get("task_idx") is not None}
    return len(ids - done)


def main() -> int:
    keys = _keys()
    ts = _dt.datetime.now().strftime("%m%d_%H%M%S")
    root = Path(_env("ROOT_RUN", f"runs/sb_reproduce_{ts}"))
    n_train = int(_env("N_TRAIN", "100"))
    corpus_w = _env("CORPUS_WORKERS", str(len([k for k in keys.split(",") if k.strip()])))
    # Stage 1 / Stage 4 configs.
    corpus_yaml = _env("CORPUS_YAML", f"{EXP}/corpus.yaml")
    eval_yaml = _env("EVAL_YAML", f"{EXP}/reasoning_bank_eval.yaml")
    eval_w = _env("EVAL_WORKERS", "10")
    corpus_dir = root / "corpus"
    banks = root / "banks"
    eval_dir = root / "eval"
    root.mkdir(parents=True, exist_ok=True)
    held = _held_ids()
    # N_HELD caps how many held-out instances Stage 4 evaluates. Unset =
    # the full held_out_idx.txt list.
    n_held = _env("N_HELD", "")
    if n_held:
        held = held[:int(n_held)]
    # N_HELD_BASES keeps only the first N *complete* base tasks (all of their
    # test cases). The Stage 5 report groups instances back by base id and
    # needs every test case of a base present in all three conditions, so a
    # raw N_HELD cut can leave partial bases that the report drops.
    # Unset = the full held_out_idx.txt list.
    n_bases = _env("N_HELD_BASES", "")
    if n_bases:
        from envharness.bridges.spreadsheetbench.dataset import load_dataset_multi
        ds = load_dataset_multi(f"{EXP}/data/_dl/all_data_912_v0.1")
        held_set = set(held)
        by_base = defaultdict(list)
        for idx, t in enumerate(ds):
            if idx in held_set:
                by_base[str(t.id).split("#", 1)[0]].append(idx)
        picked, want = [], int(n_bases)
        for base in sorted(by_base, key=lambda b: min(by_base[b])):
            if len(picked) // 3 >= want:
                break
            if len(by_base[base]) >= 3:
                picked.extend(sorted(by_base[base])[:3])
        held = picked or held
        print(f"[repro] N_HELD_BASES={want} -> {len(held)} instances "
              f"({len(held)//3} complete bases)")
    ids_csv = ",".join(str(i) for i in held)
    id_set = set(held)

    env = {**os.environ, "PYTHONWARNINGS": "ignore::UserWarning", "PYTHONPATH": ".",
           # EH_MODEL reaches every stage the dispatcher spawns.
           "EH_MODEL": _env("MODEL", "openai/gpt-4.1-mini"),
           "HOME": os.path.expanduser("~"), "SB_PYTHON": PY,
           **pool_env(_env("MODEL", "openai/gpt-4.1-mini"),
                      [k for k in keys.split(",") if k])}

    print(f"[repro] ROOT_RUN={root}  train=0..{n_train-1}  held-out={len(held)} instances")

    # ---- Stage 1: corpus (transferability mutation, low policy) ----
    if _env("SKIP_CORPUS", "") != "1":
        print("=== Stage 1: corpus (transferability mutation) ===")
        corpus_dir.mkdir(parents=True, exist_ok=True)
        run_root = str(corpus_dir)
        if run_root.startswith("runs/"):
            run_root = run_root[len("runs/"):]
        _run(["bash", f"{EXP}/dispatcher.sh", "corpus", corpus_yaml,
              run_root, f"0:{n_train}", str(corpus_dir / "done.jsonl")],
             {**env, "N_WORKERS": corpus_w})
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

    # ---- Stage 3: combined ours subset (mutated + orig fallback) ----
    if _env("SKIP_SUBSET", "") != "1":
        print("=== Stage 3: combined ours subset (mutated + orig fallback) ===")
        rng = random.Random(42)
        def load(f): return [json.loads(l) for l in open(f)]
        def bt(x):
            g = defaultdict(list)
            for it in x: g[str(it["source"]["task_id"])].append(it)
            return g
        og = bt(load(banks / "orig_full.jsonl")); ug = bt(load(banks / "ours_full.jsonl"))
        keys_sorted = sorted(og, key=lambda t: int(t) if t.isdigit() else t)
        osub = {t: rng.choice(og[t]) for t in keys_sorted}
        (banks / "orig_subset.jsonl").write_text(
            "\n".join(json.dumps(osub[t]) for t in keys_sorted) + "\n")
        u = [rng.choice(ug[t]) if t in ug else osub[t] for t in keys_sorted]
        (banks / "ours_subset_matched.jsonl").write_text(
            "\n".join(json.dumps(x) for x in u) + "\n")
        print(f"[repro] subset {len(osub)} skills, ours differ on {len(set(ug) & set(og))} tasks")

    if _env("SKIP_EVAL", "") == "1":
        print("SKIP_EVAL=1 -> stopping after Stage 3."); return 0

    # ---- Stage 4: eval base / orig / ours (MMR + top_k=5) ----
    print(f"=== Stage 4: eval base / orig / ours (MMR top_k=5, {len(held)} held-out) ===")
    eval_dir.mkdir(parents=True, exist_ok=True)
    conds = [("nobank", "none"),
             ("orig", str(banks / "orig_subset.jsonl")),
             ("ours", str(banks / "ours_subset_matched.jsonl"))]
    for cond, bank in conds:
        out = eval_dir / f"{cond}.jsonl"
        prev = -1
        for r in range(1, 9):
            miss = _missing(out, id_set)
            print(f"[repro] eval {cond} missing={miss} (round {r})")
            if miss <= 0 or miss == prev:
                break
            prev = miss
            nw = eval_w if r < 5 else "8"
            _run(["bash", f"{EXP}/dispatcher.sh", "eval", eval_yaml,
                  cond, bank, ids_csv, str(out)],
                 {**env, "N_WORKERS": str(nw), "SB_SAVE_OUTPUTS": str(eval_dir / "outputs" / cond)})

    # ---- Stage 5: 912-hard report ----
    _report(eval_dir)
    return 0


def _report(eval_dir: Path) -> None:
    def base(i): return str(i).split("#", 1)[0]
    def load(f):
        p = eval_dir / f
        if not p.exists(): return {}, {}
        ok = [r for r in (json.loads(l) for l in open(p) if l.strip())
              if not (r.get("error") or "").strip()]
        g = defaultdict(list); inst = {}
        for r in ok:
            g[base(r.get("task_id", ""))].append(bool(r.get("success")))
            inst[r.get("task_idx")] = bool(r.get("success"))
        return {b: v[:3] for b, v in g.items() if len(v) >= 3}, inst
    B, Bi = load("nobank.jsonl"); O, Oi = load("orig.jsonl"); U, Ui = load("ours.jsonl")
    common = sorted(set(B) & set(O) & set(U)); m = len(common)
    ci = sorted(set(Bi) & set(Oi) & set(Ui)); mi = len(ci)
    print("\n" + "=" * 64)
    print(f"SpreadsheetBench (MMR top_k=5) -- matched bases n={m}, instances n={mi}")
    if not m:
        print("no matched bases"); print("=" * 64); return
    pa = lambda M: sum(1 for k in common if all(M[k])) / m * 100
    ms = lambda M: sum(sum(M[k]) / len(M[k]) for k in common) / m * 100
    print(f"  {'cond':6s} {'pass@1':>8s} {'mean_score':>11s}")
    for nm, M in (("base", B), ("orig", O), ("ours", U)):
        print(f"  {nm:6s} {pa(M):8.2f} {ms(M):11.2f}")
    uo = sum(1 for k in common if all(U[k]) and not all(O[k]))
    ou = sum(1 for k in common if all(O[k]) and not all(U[k]))
    print(f"  ours-orig  pass@1 {pa(U)-pa(O):+.2f}   mean_score {ms(U)-ms(O):+.2f}   paired {uo}:{ou}")
    print(f"  orig-base  pass@1 {pa(O)-pa(B):+.2f}   mean_score {ms(O)-ms(B):+.2f}")
    print("=" * 64)


if __name__ == "__main__":
    raise SystemExit(main())
