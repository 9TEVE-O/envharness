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

"""Stage 2: induce orig + ours skill banks from an OfficeQA corpus.

OfficeQA clone of `experiments/spreadsheetbench/induce.py`. Groups the corpus
traces by `rollout_seed`, baseline -> orig, accepted -> ours, paired-diff when
both a success and a failure exist for a task, and uses the OfficeQA
document-navigation prompts + trajectory formatting from
`experiments.officeqa.prompts`. The retrieval query is the task question.

Run from the repo root:
  export GEMINI_API_KEY=...
  python experiments/officeqa/induce.py \
      --traces runs/oqa_corpus_001/traces.jsonl \
      --out-dir runs/oqa_corpus_001/banks --concurrency 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.officeqa.prompts import (
    format_trajectory, induce_failed, induce_paired_diff, induce_success,
    task_query,
)
from envharness.reasoning_bank import Bank, MemoryItem, embed_texts
from envharness.infra.model import (effective_model, key_env, key_pool,
                                    missing_key_message, qualify)

# Toggle for SWE-bench technique #3 (thread Mutator diagnosis into ours' paired-diff).
# Set by main() from --thread-diagnosis; default OFF (clean orig==ours induction).
_THREAD_DIAGNOSIS = False


def _instruction(steps: list[dict]) -> str:
    """Task question = the retrieval query. Read it from the first step's
    observation.data['question'] (the bridge exposes it there); fall back to
    parsing the 'question:' line out of observation.text."""
    if not steps:
        return ""
    obs = (steps[0].get("filtered_observation")
           or steps[0].get("raw_observation") or {})
    data = obs.get("data") if isinstance(obs, dict) else {}
    if isinstance(data, dict) and data.get("question"):
        return str(data["question"])
    text = obs.get("text", "") if isinstance(obs, dict) else ""
    for ln in (text or "").splitlines():
        if ln.lower().startswith("question:"):
            return ln.split(":", 1)[1].strip()
    return ""


def _pick_pair(rollouts: list[dict]) -> tuple[dict | None, dict | None]:
    """One success + one failure; prefer shortest success (cleanest) and
    longest failure (most signal)."""
    succ = [r for r in rollouts if r.get("success") and r.get("steps")]
    fail = [r for r in rollouts if not r.get("success") and r.get("steps")]
    succ.sort(key=lambda r: len(r["steps"]))
    fail.sort(key=lambda r: -len(r["steps"]))
    return (succ[0] if succ else None), (fail[0] if fail else None)


def _items_for_task(tid: str, condition: str, succ: dict | None,
                    fail: dict | None, llm_model: str,
                    gemini_api_key: str | None = None) -> list[dict]:
    instr = task_query(_instruction((succ or fail or {}).get("steps") or []))
    # Optional #3 toggle: thread the Mutator's diagnosis (only present on ours
    # traces) into the paired-diff. Off unless --thread-diagnosis.
    diagnosis = ""
    if _THREAD_DIAGNOSIS:
        for tr in (succ, fail):
            if not tr:
                continue
            cand = tr.get("candidate") or {}
            fa = tr.get("failure_analysis") or {}
            diagnosis = (cand.get("rationale") or fa.get("label") or "").strip()
            if diagnosis:
                break
    out: list[dict] = []
    if succ and fail:
        items = induce_paired_diff(
            instr, format_trajectory(fail["steps"]),
            format_trajectory(succ["steps"]), llm_model=llm_model,
            gemini_api_key=gemini_api_key, diagnosis=diagnosis)
        induction = "paired_diff"
    elif succ:
        items = induce_success(instr, format_trajectory(succ["steps"]),
                               llm_model=llm_model, gemini_api_key=gemini_api_key)
        induction = "single_succ"
    elif fail:
        items = induce_failed(instr, format_trajectory(fail["steps"]),
                              llm_model=llm_model, gemini_api_key=gemini_api_key)
        induction = "single_fail"
    else:
        items = []
        induction = "none"
    for it in items:
        it["source"] = {"task_id": tid, "condition": condition,
                        "induction": induction, "task_query": instr}
        out.append(it)
    return out


def _build_bank(*, condition: str, by_task: dict, llm_model: str,
                concurrency: int, embed_model: str,
                keys: list[str] | None = None) -> Bank:
    items: list[dict] = []
    keys = keys or [None]

    def _one(idx_kv):
        idx, (tid, rollouts) = idx_kv
        succ, fail = _pick_pair(rollouts)
        if not succ and not fail:
            return []
        # Round-robin the API key across tasks so ~hundreds of induction
        # calls spread over all keys instead of hammering one (which
        # rate-limits every call to [] -> empty bank).
        gemini_api_key = keys[idx % len(keys)]
        return _items_for_task(tid, condition, succ, fail, llm_model, gemini_api_key=gemini_api_key)

    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for chunk in pool.map(_one, enumerate(by_task.items())):
            items.extend(chunk)

    # Build-time consolidation (SWE-bench technique #5): drop near-duplicate
    # skills by case-insensitive title so a few generic titles don't flood the
    # bank and dominate retrieval. Keeps the first occurrence.
    _seen: set[str] = set()
    _deduped: list[dict] = []
    for it in items:
        key = (it.get("title") or "").strip().lower()
        if key and key in _seen:
            continue
        _seen.add(key)
        _deduped.append(it)
    n_drop = len(items) - len(_deduped)
    items = _deduped
    print(f"  [{condition}] induced {len(items)} items from {len(by_task)} tasks "
          f"(title-dedup dropped {n_drop})")
    if not items:
        return Bank()
    texts = [f"## {it['title']}\n{it['description']}\n{it['content']}"
             for it in items]
    vecs = embed_texts(texts, model=embed_model)
    bank = Bank()
    for it, v in zip(items, vecs):
        bank.add([MemoryItem(title=it["title"], description=it["description"],
                             content=it["content"], embedding=v,
                             source=it["source"])])
    return bank


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--llm-model", default="openai/gpt-4.1-mini")
    # Unset: embed_texts resolves it from $EH_EMBED_MODEL, else the provider
    # of the run's model. Pinning one provider's embeddings here would build a
    # bank whose vectors the eval side cannot query (dimensions differ).
    p.add_argument("--embed-model", default=None)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--thread-diagnosis", action="store_true",
                   help="SWE #3: thread Mutator diagnosis into ours' paired-diff")
    args = p.parse_args(argv)

    global _THREAD_DIAGNOSIS
    _THREAD_DIAGNOSIS = bool(args.thread_diagnosis)

    import os
    keys = key_pool(args.llm_model)
    if keys:
        # Embeddings read the provider's key from the environment.
        os.environ.update(key_env(args.llm_model, keys[0]))
    print(f"[oqa induce]  model={qualify(effective_model(args.llm_model))}  "
          f"spreading induction across {len(keys)} key(s)")

    by_baseline: dict[str, list] = defaultdict(list)
    by_accepted: dict[str, list] = defaultdict(list)
    n = 0
    with args.traces.open() as f:
        for line in f:
            t = json.loads(line)
            n += 1
            tid = str(t.get("rollout_seed"))
            kind = t.get("kind", "")
            if kind == "baseline":
                by_baseline[tid].append(t)
            elif kind == "accepted":
                by_accepted[tid].append(t)

    print(f"[oqa induce]  {n} records  baseline_tasks={len(by_baseline)}  "
          f"accepted_tasks={len(by_accepted)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for cond, by_task in (("orig", by_baseline), ("ours", by_accepted)):
        bank = _build_bank(condition=cond, by_task=dict(by_task),
                           llm_model=args.llm_model, concurrency=args.concurrency,
                           embed_model=args.embed_model, keys=keys or None)
        out = args.out_dir / f"{cond}_full.jsonl"
        bank.save(out)
        print(f"  -> {out}  ({len(bank)} items)")
    print(f"total wall: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
