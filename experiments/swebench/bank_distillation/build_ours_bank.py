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

"""Cascade-fallback bank build (ours).

Per-task succ-source cascade:
  Tier 1: kind=accepted, mutation_code present, success=True  (real mutation)
  Tier 2: kind=exploration, mutation_code present, success=True  (last attempt)
  Tier 3: kind=baseline, success=True  (fall back to intra-baseline)
  Skip:  no base_fail OR all tiers empty

The cascade covers the same task set as the orig (intra-baseline) bank —
it only differs on tasks where Tier 1 or Tier 2 is available, since Tier 3
IS the orig bank's source.

Distill prompt: induce_paired_diff_atomic (same prompt as the orig bank build).
Pair selection: cross-product (base_fail × selected_succ_tier).
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import sys

from envharness.infra.model import (key_pool as _key_pool,
                                    missing_key_message)
import time
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


def _cand_id(t: dict):
    cand = t.get("candidate") or {}
    return (t.get("candidate_id") or cand.get("id")
            or cand.get("candidate_id") or t.get("iteration_id"))


def categorize_with_cascade(traces: list[dict], tier2_last_only: bool = True,
                             max_baseline_rollouts: int | None = 5):
    """Returns (base_fail, selected_succ, tier_label).

    Cascade: acc_succ > expl_succ > base_succ. Filters: with steps,
    mutation_code present for acc/expl, success True.

    Pair-selection controls:
      - tier2_last_only: tier-2 uses ONLY the LAST exploration candidate's
        successes (the "2_exploration_last" intent). Collecting EVERY
        exploration success across all candidates would flood the bank via
        cross-product on easy tasks with many out-of-band successes.
      - max_baseline_rollouts: only consider the FIRST N baseline rollouts
        when deciding base_fail. This keeps the base_fail decision at the
        K=5 baseline protocol even if a corpus was generated with a larger
        baseline K, where extra rollouts can introduce a single noise
        failure that spuriously "unlocks" easy tasks.
    """
    base_fail, base_succ, acc_succ, expl_succ = [], [], [], []
    n_baseline = 0
    for t in traces:
        if not t.get("steps"): continue
        k = t.get("kind"); s = bool(t.get("success"))
        cand = t.get("candidate") or {}
        # Release Trace schema serializes `rules_code`; dev-format traces use
        # `mutation_code`. Accept both.
        has_mut = bool((cand.get("rules_code") or cand.get("mutation_code") or "").strip())
        if k == "baseline":
            n_baseline += 1
            if max_baseline_rollouts is not None and n_baseline > max_baseline_rollouts:
                continue    # ignore extra baseline rollouts beyond the first N
            (base_succ if s else base_fail).append(t)
        elif k == "accepted" and has_mut and s:
            acc_succ.append(t)
        elif k == "exploration" and has_mut and s:
            expl_succ.append(t)
    if not base_fail:
        return None, None, "skip_no_base_fail"
    if acc_succ:
        return base_fail, acc_succ, "1_accepted"
    if expl_succ:
        if tier2_last_only:
            # Keep only the LAST exploration candidate's successes.
            last_cid = _cand_id(expl_succ[-1])
            expl_succ = [t for t in expl_succ if _cand_id(t) == last_cid]
        return base_fail, expl_succ, "2_exploration_last"
    if base_succ:
        return base_fail, base_succ, "3_baseline_fallback"
    return None, None, "skip_no_succ"


def categorize_intra_mutated(traces: list[dict]):
    """Returns list[(succ_trace, fail_trace)] pairs, both from the SAME
    accepted candidate (same mutation_code applied, same env). This is the
    "intra-env" contrast: env is constant, only policy actions differ.

    Per RB-correct distillation theory: cleaner skill extraction because
    env perturbation is held constant, so differences are purely policy-side.
    """
    from collections import defaultdict
    by_cand = defaultdict(lambda: {"succ": [], "fail": []})
    for t in traces:
        if not t.get("steps"): continue
        if t.get("kind") != "accepted": continue
        cand = t.get("candidate") or {}
        if not (cand.get("rules_code") or cand.get("mutation_code") or "").strip(): continue
        cid = (t.get("candidate_id")
                or cand.get("id")
                or cand.get("candidate_id")
                or t.get("iteration_id"))
        if cid is None: continue
        if t.get("success"):
            by_cand[cid]["succ"].append(t)
        else:
            by_cand[cid]["fail"].append(t)
    pairs = []
    for cid, d in by_cand.items():
        if not d["succ"] or not d["fail"]: continue
        for s in d["succ"]:
            for f in d["fail"]:
                pairs.append((s, f, cid))
    return pairs


def induce_for_pair(args_tuple) -> list[MemoryItem]:
    (task_dir, tier_label, base_tr, mut_tr, iid, problem,
     failure_label, rationale, llm_model, gemini_api_key, existing_titles) = args_tuple
    out: list[MemoryItem] = []
    base_text = format_trajectory_swebench(base_tr.get("steps") or [])
    mut_text = format_trajectory_swebench(mut_tr.get("steps") or [])
    items = induce_paired_diff_atomic(
        problem_statement=problem,
        baseline_trajectory_text=base_text,
        mutated_trajectory_text=mut_text,
        failure_label=failure_label,
        mutation_rationale=rationale,
        existing_lesson_titles=existing_titles,
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
                "task_dir": task_dir, "condition": "cascade",
                "instance_id": iid,
                "induction_mode": f"paired_diff_atomic_cascade_{tier_label}",
                "tier": tier_label,
                "baseline_episode_id": base_tr.get("episode_id"),
                "mutated_episode_id": mut_tr.get("episode_id"),
                "baseline_success": False,
                "mutated_success": True,
                "mutator_failure_label": failure_label,
                "task_query": (problem or "")[:200],
            },
        ))
    return out


def dedup_by_title(items: list[MemoryItem]) -> list[MemoryItem]:
    seen: set[str] = set(); out = []
    for it in items:
        k = (it.title or "").strip().lower()
        if k in seen: continue
        seen.add(k); out.append(it)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pool-dir", default="runs")
    p.add_argument("--pool-glob", default="v4-pool-task*")
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--llm-model", default="openai/gpt-4.1-mini")
    p.add_argument("--dedup-titles", action="store_true")
    p.add_argument("--existing-bank", default=None,
                   help="Path to existing bank JSONL. Distillation prompt will see "
                        "existing lesson titles and skip re-emitting them — produces "
                        "ONLY NEW lessons for augment-mode iterative refinement.")
    p.add_argument("--pair-mode", default="cascade",
                   choices=["cascade", "intra_mutated"],
                   help="cascade: base_fail (orig env) × mutated_succ (mutated env). "
                        "intra_mutated: mutated_succ × mutated_fail (same mutated env) — "
                        "cleaner skill extraction, env held constant.")
    # --- pair-selection caps (see categorize_with_cascade) ---
    p.add_argument("--max-pairs-per-task", type=int, default=5,
                   help="Cap base_fail×succ pairs per task. 0 disables. "
                        "Prevents easy-task flooding.")
    p.add_argument("--max-baseline-rollouts", type=int, default=5,
                   help="Only consider the first N baseline rollouts for the "
                        "base_fail decision (default 5 = the corpus baseline "
                        "K). Use a large number to disable.")
    p.add_argument("--no-tier2-last-only", action="store_true",
                   help="Tier-2 uses ALL exploration successes across "
                        "candidates (can flood on easy tasks). Default keeps "
                        "only the LAST exploration candidate.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Key POOL for round-robin across concurrent skill-extraction calls.
    # Pinning all `--concurrency` workers to a single GEMINI_API_KEY throttles
    # them against each other (429). GEMINI_API_KEYS (plural) is a comma-list.
    # The pool belongs to args.llm_model's provider: one key on GPT, none on
    # Vertex. Round-robin below spreads a per-key quota where there is one.
    _missing = missing_key_message(args.llm_model)
    if _missing:
        print(f"ERROR: {_missing}", file=sys.stderr); return 2
    _keys = _key_pool(args.llm_model) or [None]
    gemini_api_key = _keys[0]  # fallback default; per-work rotation below uses the pool

    existing_titles: list[str] = []
    if args.existing_bank:
        existing_bank = Bank.load(Path(args.existing_bank))
        existing_titles = [it.title for it in existing_bank.items]
        print(f"[load] existing-bank: {len(existing_titles)} titles from {args.existing_bank}",
              flush=True)

    print("[load] princeton-nlp/SWE-bench_Lite ...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    iid_for_task_id = {i: r["instance_id"] for i, r in enumerate(ds)}
    problem_for_iid = {r["instance_id"]: r["problem_statement"] for r in ds}

    pool_pattern = os.path.join(args.pool_dir, args.pool_glob)
    task_dirs = sorted([d for d in glob.glob(pool_pattern) if os.path.isdir(d)])
    print(f"[load] found {len(task_dirs)} task dirs", flush=True)

    work = []
    tier_counts = {"1_accepted": 0, "2_exploration_last": 0,
                   "3_baseline_fallback": 0, "skip_no_base_fail": 0,
                   "skip_no_succ": 0, "skip_no_log": 0, "skip_no_iid": 0}
    for d in task_dirs:
        log = Path(d) / "orchestrator.log"
        traces_path = Path(d) / "traces.jsonl"
        if not traces_path.exists():
            tier_counts["skip_no_log"] += 1; continue
        task_id = get_task_id_from_log(log)
        if task_id is None:
            tier_counts["skip_no_log"] += 1; continue
        iid = iid_for_task_id.get(task_id)
        if iid is None: tier_counts["skip_no_iid"] += 1; continue
        problem = problem_for_iid.get(iid)
        if not problem: tier_counts["skip_no_iid"] += 1; continue
        traces = [json.loads(l) for l in traces_path.read_text().splitlines() if l.strip()]
        if args.pair_mode == "intra_mutated":
            # NEW: succ × fail within the same accepted candidate (same mutated env)
            intra_pairs = categorize_intra_mutated(traces)
            if not intra_pairs:
                tier_counts["skip_no_succ"] += 1
                continue
            tier_counts["1_accepted"] = tier_counts.get("1_accepted", 0) + 1
            label, rationale = extract_mutator_diagnosis(
                [t for t in traces if t.get("kind") == "accepted"])
            for succ_t, fail_t, cid in intra_pairs:
                # In intra-mutated mode, the "BASELINE" arg is mutated_fail (still
                # fails in mutated env) and "MUTATED" arg is mutated_succ (succeeds).
                # Same env → all difference comes from policy actions.
                work.append((d, "intra_mutated", fail_t, succ_t, iid, problem,
                             label, rationale, args.llm_model, gemini_api_key,
                             existing_titles))
            continue
        # Default: cascade (cross-env: base_fail × mutated_succ)
        base_fail, sel_succ, tier = categorize_with_cascade(
            traces, tier2_last_only=not args.no_tier2_last_only,
            max_baseline_rollouts=args.max_baseline_rollouts)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if base_fail is None: continue
        label, rationale = extract_mutator_diagnosis(
            [t for t in traces if t.get("kind") == "accepted"])
        # Cap cross-product pairs PER TASK so a single easy task with many
        # successes cannot flood the bank. We take the first N base_fail and
        # first M succ, capping total pairs.
        cap = args.max_pairs_per_task
        n_pairs_task = 0
        for b in base_fail:
            for s in sel_succ:
                if cap and n_pairs_task >= cap:
                    break
                work.append((d, tier, b, s, iid, problem,
                             label, rationale, args.llm_model,
                             _keys[len(work) % len(_keys)],
                             existing_titles))
                n_pairs_task += 1
            if cap and n_pairs_task >= cap:
                break

    print()
    print("=== Cascade tier breakdown ===")
    for k, v in tier_counts.items():
        print(f"  {k}: {v}")
    print(f"\nTOTAL pairs queued: {len(work)}", flush=True)
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
