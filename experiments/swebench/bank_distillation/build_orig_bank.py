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

"""Balanced bank build — apples-to-apples pairing for cascade vs intra-baseline.

For each task in the pool that has base_fail AND base_succ (the common
universe):

  cascade_pairs = base_fail × (acc_succ if available else expl_succ if
                   available else base_succ)
  intra_pairs   = base_fail × base_succ           [intra-baseline only]
  cap = min(cascade_pairs, intra_pairs)

  cascade mode uses up to `cap` pairs from its cascade source
  intra_baseline mode uses up to `cap` pairs from base_succ

This guarantees identical per-task pair counts. The ONLY differential is
which trace fills the succ slot on tasks where the cascade selects a
mutation tier.

Subsampling is deterministic via seed.

--pair-mode controls which side to emit (cascade or intra_baseline).
Run twice with same seed to get matched-size banks.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import sys

from envharness.infra.model import (key_pool as _key_pool,
                                    missing_key_message)
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from envharness.reasoning_bank import Bank, MemoryItem, embed_texts
from induce import (
    induce_paired_diff_atomic, format_trajectory_swebench,
)


def get_task_id_from_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    pat = re.compile(r"task_idx=0.*task_id=(\d+)")
    for ln in log_path.read_text().splitlines():
        if "baseline_compute" not in ln and "baseline_cache_hit" not in ln:
            continue
        m = pat.search(ln)
        if m:
            return int(m.group(1))
    return None


def extract_mutator_diagnosis(traces: list[dict]) -> tuple[str, str]:
    label = "unspecified"; rationale = "unspecified"
    for t in traces:
        if t.get("kind") != "accepted": continue
        cand = t.get("candidate") or {}
        fa = t.get("failure_analysis") or {}
        if fa.get("label"): label = fa["label"]
        rat = cand.get("rationale")
        if rat and "skip" not in rat.lower(): rationale = rat; break
    return label, rationale


def categorize(traces: list[dict]):
    base_fail, base_succ, acc_succ, expl_succ = [], [], [], []
    for t in traces:
        if not t.get("steps"): continue
        k = t.get("kind"); s = bool(t.get("success"))
        cand = t.get("candidate") or {}
        # Release Trace schema serializes `rules_code`; dev-format traces use
        # `mutation_code`. Accept both.
        has_mut = bool((cand.get("rules_code") or cand.get("mutation_code") or "").strip())
        if k == "baseline":
            (base_succ if s else base_fail).append(t)
        elif k == "accepted" and has_mut and s:
            acc_succ.append(t)
        elif k == "exploration" and has_mut and s:
            expl_succ.append(t)
    return base_fail, base_succ, acc_succ, expl_succ


def select_succ_for_mode(mode: str, acc_succ, expl_succ, base_succ):
    """Return (succ_list, tier_label) per mode."""
    if mode == "cascade":
        if acc_succ: return acc_succ, "1_accepted"
        if expl_succ: return expl_succ, "2_exploration_last"
        if base_succ: return base_succ, "3_baseline_fallback"
        return [], "skip_no_succ"
    elif mode == "intra_baseline":
        if base_succ: return base_succ, "intra_baseline"
        return [], "skip_no_base_succ"
    else:
        raise ValueError(f"unknown mode: {mode}")


def induce_for_pair(args_tuple) -> list[MemoryItem]:
    (task_dir, mode, tier_label, base_tr, mut_tr, iid, problem,
     failure_label, rationale, llm_model, gemini_api_key) = args_tuple
    out: list[MemoryItem] = []
    base_text = format_trajectory_swebench(base_tr.get("steps") or [])
    mut_text = format_trajectory_swebench(mut_tr.get("steps") or [])
    items = induce_paired_diff_atomic(
        problem_statement=problem,
        baseline_trajectory_text=base_text,
        mutated_trajectory_text=mut_text,
        failure_label=failure_label,
        mutation_rationale=rationale,
        llm_model=llm_model,
        gemini_api_key=gemini_api_key,
    )
    if not items: return out
    texts = [f"{it['title']}: {it['description']}" for it in items]
    try: embs = embed_texts(texts)
    except Exception: return out
    for it, emb in zip(items, embs):
        out.append(MemoryItem(
            title=it["title"], description=it["description"],
            content=it["content"], embedding=emb,
            source={
                "task_dir": task_dir, "condition": f"balanced_{mode}",
                "instance_id": iid,
                "induction_mode": f"paired_diff_atomic_balanced_{tier_label}",
                "tier": tier_label,
                "baseline_episode_id": base_tr.get("episode_id"),
                "mutated_episode_id": mut_tr.get("episode_id"),
                "mutator_failure_label": failure_label,
                "task_query": (problem or "")[:200],
            },
        ))
    return out


def dedup_by_title(items):
    seen = set(); out = []
    for it in items:
        k = (it.title or "").strip().lower()
        if k in seen: continue
        seen.add(k); out.append(it)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pool-dir", default="runs")
    p.add_argument("--pool-glob", default="v4-pool-task*")
    p.add_argument("--pair-mode", required=True, choices=["cascade", "intra_baseline"])
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--llm-model", default="openai/gpt-4.1-mini")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dedup-titles", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # The pool belongs to args.llm_model's provider: one key on GPT, none on
    # Vertex. Round-robin below spreads a per-key quota where there is one.
    _missing = missing_key_message(args.llm_model)
    if _missing:
        print(f"ERROR: {_missing}", file=sys.stderr); return 2
    _keys = _key_pool(args.llm_model) or [None]
    gemini_api_key = _keys[0]
    print(f"[load] pair_mode={args.pair_mode} pool={args.pool_glob}", flush=True)
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    iid_for_task_id = {i: r["instance_id"] for i, r in enumerate(ds)}
    problem_for_iid = {r["instance_id"]: r["problem_statement"] for r in ds}

    pool_pattern = os.path.join(args.pool_dir, args.pool_glob)
    task_dirs = sorted([d for d in glob.glob(pool_pattern) if os.path.isdir(d)])
    print(f"[load] found {len(task_dirs)} task dirs", flush=True)

    # Build per-task plan with balanced cap
    work = []
    plan_rows = []
    for d in task_dirs:
        log = Path(d) / "orchestrator.log"
        traces_path = Path(d) / "traces.jsonl"
        if not traces_path.exists(): continue
        task_id = get_task_id_from_log(log)
        if task_id is None: continue
        iid = iid_for_task_id.get(task_id)
        if iid is None: continue
        problem = problem_for_iid.get(iid)
        if not problem: continue
        traces = [json.loads(l) for l in traces_path.read_text().splitlines() if l.strip()]
        bf, bs, accs, expls = categorize(traces)
        if not bf: continue

        # Compute cap = min(cascade_pairs, intra_pairs)
        casc_succ, casc_tier = select_succ_for_mode("cascade", accs, expls, bs)
        intra_succ, intra_tier = select_succ_for_mode("intra_baseline", accs, expls, bs)
        n_casc = len(bf) * len(casc_succ)
        n_intra = len(bf) * len(intra_succ)
        if n_casc == 0 or n_intra == 0:
            # Task not in the common universe; skip
            continue
        cap = min(n_casc, n_intra)

        # Now select pairs for the requested mode
        if args.pair_mode == "cascade":
            sel_succ, tier = casc_succ, casc_tier
            all_pairs = [(b, s) for b in bf for s in sel_succ]
        else:
            sel_succ, tier = intra_succ, intra_tier
            all_pairs = [(b, s) for b in bf for s in sel_succ]

        # Deterministic subsample to cap (sorted then sample with task-specific seed)
        all_pairs.sort(key=lambda p: (p[0].get("episode_id"), p[1].get("episode_id")))
        if len(all_pairs) > cap:
            # zlib.crc32 is stable across processes (unlike str hash(), which
            # PYTHONHASHSEED randomizes) -- keeps the promised cross-run
            # determinism for same-seed subsampling.
            rng = random.Random(args.seed + zlib.crc32(d.encode()) % 10_000)
            all_pairs = rng.sample(all_pairs, cap)

        plan_rows.append((os.path.basename(d), casc_tier, intra_tier,
                          n_casc, n_intra, cap, len(all_pairs)))

        label, rationale = extract_mutator_diagnosis(
            [t for t in traces if t.get("kind") == "accepted"])
        for b, s in all_pairs:
            work.append((d, args.pair_mode, tier, b, s, iid, problem,
                         label, rationale, args.llm_model,
                         _keys[len(work) % len(_keys)]))

    print()
    print("=== Per-task balanced plan ===")
    print(f"  {'task':20s}  casc-tier  intra-tier      casc  intra  cap  using")
    for row in plan_rows:
        print(f"  {row[0]:20s}  {row[1]:18s} {row[2]:18s} {row[3]:4d} {row[4]:5d}"
              f" {row[5]:4d} {row[6]:4d}")
    print(f"\nTOTAL pairs queued for mode={args.pair_mode}: {len(work)}", flush=True)
    if not work:
        Bank().save(Path(args.out)); return 0

    bank = Bank()
    t0 = time.time(); n_done = 0; n_items = 0; empty = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(induce_for_pair, w): w[0] for w in work}
        for fut in cf.as_completed(futures):
            td = futures[fut]
            try: items = fut.result()
            except Exception as e:
                print(f"  {td}: {type(e).__name__}: {e}", file=sys.stderr)
                items = []
            if not items: empty += 1
            bank.add(items)
            n_done += 1; n_items += len(items)
            if n_done % 5 == 0 or n_done == len(work):
                el = time.time() - t0
                rate = n_done / max(el, 1e-3) * 60
                print(f"  [{n_done}/{len(work)}] items={n_items} empty={empty}"
                      f" rate={rate:.1f}/min el={el/60:.1f}m", flush=True)

    if args.dedup_titles:
        before = len(bank.items)
        bank.items = dedup_by_title(bank.items)
        print(f"[dedup] kept {len(bank.items)} / {before} after title dedup")
    bank.save(Path(args.out))
    print(f"\n=== DONE: bank size {len(bank)} out: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
