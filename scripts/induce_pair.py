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

"""Stage 2: induce pair-banks from a corpus run's traces.jsonl.

For each task in `traces.jsonl` (the corpus generation output):
  - orig: one BASELINE failure trace + one BASELINE success trace
  - ours: CASCADE -- the task's ACCEPTED rollouts when the HarnessAgent
    committed a mutation, otherwise fall back to the SAME task's baseline
    rollouts (mirrors swebench v9 cascade Tier 3). Both banks therefore
    cover the same task set; the ours-vs-orig differential comes only from
    tasks where a mutation was accepted.

The induce_memory_items function (envharness.reasoning_bank)
calls Gemini with the verbatim ReasoningBank SUCCESSFUL_SI / FAILED_SI
prompts. Embeddings are added with gemini-embedding-001.

Run::

  export GEMINI_API_KEY=...
  <your-env>/bin/python scripts/induce_pair.py \\
      --traces runs/<corpus-run>/traces.jsonl \\
      --out-dir runs/<corpus-run>/banks \\
      --concurrency 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envharness.reasoning_bank import (
    MemoryItem, embed_texts, format_trajectory,
    induce_memory_items, induce_paired_diff,
)


def _task_query(steps: list[dict]) -> str:
    if not steps:
        return ""
    first = (steps[0].get("filtered_observation") or {}).get("text", "")
    for ln in first.splitlines():
        if "your task is to:" in ln.lower():
            return ln.split(":", 1)[1].strip()
        if "task:" in ln.lower():
            return ln.split(":", 1)[1].strip()
    return first.splitlines()[0] if first else ""


def _pick_pair(rollouts: list[dict]) -> tuple[dict | None, dict | None]:
    """One success + one failure trace; prefer longest failure (more signal)
    and shortest success (cleanest demonstration)."""
    succ = [r for r in rollouts if r.get("success") and r.get("steps")]
    fail = [r for r in rollouts if not r.get("success") and r.get("steps")]
    succ.sort(key=lambda r: len(r["steps"]))
    fail.sort(key=lambda r: -len(r["steps"]))
    return (succ[0] if succ else None), (fail[0] if fail else None)


def _items_for_task(task_id: str, condition: str, succ: dict | None,
                     fail: dict | None, llm_model: str) -> list[dict]:
    """Per task: if both succ and fail exist, use paired-diff (compare the
    two trajectories and extract what the success did differently). If only
    one exists, fall back to single-trajectory induction."""
    query = _task_query((succ or fail or {}).get("steps") or [])
    out: list[dict] = []

    if succ and fail:
        # Paired-diff: compare fail vs succ on the SAME environment
        for it in induce_paired_diff(
            task_query=query,
            fail_trajectory_text=format_trajectory(fail["steps"]),
            succ_trajectory_text=format_trajectory(succ["steps"]),
            llm_model=llm_model,
        ):
            it["source"] = {"task_id": task_id, "condition": condition,
                              "induction": "paired_diff",
                              "succ_episode_id": succ.get("episode_id", ""),
                              "fail_episode_id": fail.get("episode_id", ""),
                              "task_query": query}
            out.append(it)
    elif succ:
        for it in induce_memory_items(
            task_query=query,
            trajectory_text=format_trajectory(succ["steps"]),
            success=True, llm_model=llm_model,
        ):
            it["source"] = {"task_id": task_id, "condition": condition,
                              "induction": "single_succ",
                              "task_query": query}
            out.append(it)
    elif fail:
        for it in induce_memory_items(
            task_query=query,
            trajectory_text=format_trajectory(fail["steps"]),
            success=False, llm_model=llm_model,
        ):
            it["source"] = {"task_id": task_id, "condition": condition,
                              "induction": "single_fail",
                              "task_query": query}
            out.append(it)
    return out


def _build_bank(*, condition: str, traces_by_task: dict, llm_model: str,
                  concurrency: int, embed_model: str, out_path: Path,
                  embed_chunk_size: int = 20) -> int:
    """Induce items, then embed + append-write the bank JSONL incrementally
    in chunks of `embed_chunk_size`. A crash mid-embedding (e.g. a
    rate-limit streak) preserves every chunk already written, instead of
    losing the whole 30-60 min induction to one post-hoc embedding pass.
    Returns the number of items written to `out_path`."""
    items: list[dict] = []

    def _one(task_id_rollouts):
        task_id, rollouts = task_id_rollouts
        succ, fail = _pick_pair(rollouts)
        if not succ and not fail:
            return []
        return _items_for_task(task_id, condition, succ, fail, llm_model)

    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for chunk in pool.map(_one, traces_by_task.items()):
            items.extend(chunk)

    print(f"  [{condition}] induced {len(items)} items from "
          f"{len(traces_by_task)} tasks")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")   # fresh file; don't append onto a stale bank
    n_written = 0
    with out_path.open("a") as f:
        for i in range(0, len(items), embed_chunk_size):
            batch = items[i:i + embed_chunk_size]
            # Embed in chunks via the same Gemini embedding endpoint the
            # eval uses; append + flush after each chunk.
            texts = [f"## {it['title']}\n{it['description']}\n{it['content']}"
                     for it in batch]
            vecs = embed_texts(texts, model=embed_model)
            for it, v in zip(batch, vecs):
                f.write(json.dumps(MemoryItem(
                    title=it["title"], description=it["description"],
                    content=it["content"], embedding=v, source=it["source"],
                ).to_dict()) + "\n")
            f.flush()
            n_written += len(batch)
            print(f"    [{condition}] embedded + wrote "
                  f"{n_written}/{len(items)} -> {out_path}", flush=True)
    return n_written


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, type=Path,
                    help="traces.jsonl from a corpus run")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--llm-model", default="openai/gpt-4.1-mini")
    # Unset: embed_texts resolves it from $EH_EMBED_MODEL, else the provider
    # of the run's model. Pinning one provider's embeddings here would build a
    # bank whose vectors the eval side cannot query (dimensions differ).
    p.add_argument("--embed-model", default=None)
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args(argv)

    # Group traces by per-task identifier. Use `rollout_seed` because
    # `task_id` in Trace is the orchestrator's config-level LABEL
    # (constant across the whole run); the per-task identifier the
    # orchestrator passes to Bridge.reset is `rollout_seed`.
    # orig items come from kind="baseline"; ours items come from kind="accepted".
    by_task_baseline: dict[str, list] = defaultdict(list)
    by_task_accepted: dict[str, list] = defaultdict(list)
    n_total = 0
    with args.traces.open() as f:
        for line in f:
            t = json.loads(line)
            n_total += 1
            kind = t.get("kind", "")
            tid = str(t.get("rollout_seed"))
            if kind == "baseline":
                by_task_baseline[tid].append(t)
            elif kind == "accepted":
                by_task_accepted[tid].append(t)

    # ours = per-task CASCADE: use the ACCEPTED rollouts when the task has
    # them (the HarnessAgent committed a mutation); otherwise FALL BACK to
    # that task's baseline rollouts. This keeps ours covering the SAME task
    # set as orig -- a task whose mutation never reached ACCEPT contributes
    # its unmutated traces instead of disappearing from the bank (same
    # semantics as swebench's v9 cascade Tier 3).
    by_task_ours: dict[str, list] = {}
    n_fallback = 0
    for tid, rollouts in by_task_baseline.items():
        acc = by_task_accepted.get(tid)
        if acc:
            by_task_ours[tid] = acc
        else:
            by_task_ours[tid] = rollouts
            n_fallback += 1
    for tid, rollouts in by_task_accepted.items():
        by_task_ours.setdefault(tid, rollouts)

    print(f"[induce_pair]  traces.jsonl: {n_total} records")
    print(f"  baseline tasks: {len(by_task_baseline)}")
    print(f"  accepted tasks: {len(by_task_accepted)}")
    print(f"  ours cascade:   {len(by_task_ours)} tasks "
          f"({len(by_task_ours) - n_fallback} accepted + {n_fallback} baseline-fallback)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    empty_conds: list[str] = []
    for cond, by_task in (("orig", by_task_baseline),
                           ("ours", by_task_ours)):
        out = args.out_dir / f"{cond}_full.jsonl"
        n_written = _build_bank(condition=cond, traces_by_task=dict(by_task),
                                  llm_model=args.llm_model,
                                  concurrency=args.concurrency,
                                  embed_model=args.embed_model,
                                  out_path=out)
        print(f"  -> {out}  ({n_written} items)")
        if by_task and n_written == 0:
            empty_conds.append(cond)

    print(f"total wall: {(time.time()-t0)/60:.1f} min")
    if empty_conds:
        print(f"[FATAL] induction produced 0 items for condition(s) "
              f"{empty_conds} despite non-empty trace input -- check "
              f"GEMINI_API_KEY / induction model access; refusing to hand "
              f"empty bank(s) to downstream stages.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
