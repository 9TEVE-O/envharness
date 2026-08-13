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

"""WebArena bank builder — single-trajectory induction, skip mutation failures.

V bank: mutation success > baseline success > baseline fail (skip mutation fail).
A bank: baseline success > baseline fail.
Both use RB-native single-trajectory induction (SUCCESSFUL_SI / FAILED_SI).

Usage::

  PY=<your-webarena-env>/bin/python
  $PY experiments/webarena/build_reasoning_bank.py \
      --traces runs/corpus/all_traces.jsonl \
      --out-dir runs/banks \
      --concurrency 6
"""
from __future__ import annotations

from envharness.infra.model import completion_kwargs

import argparse
import concurrent.futures as cf
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import litellm

from envharness.reasoning_bank import (
    Bank, MemoryItem, embed_texts, parse_memory_items,
)


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------

def format_trajectory(steps: list[dict]) -> str:
    parts = []
    for i, s in enumerate(steps, 1):
        action = (s.get("raw_action") or {}).get("kwargs", {}).get("action_str", "")
        obs = s.get("filtered_observation") or {}
        data = obs.get("data") or {}
        url = data.get("url", "")
        err = data.get("last_action_error", "")
        think_parts = [f"Step {i}."]
        if url:
            think_parts.append(f"Current page: {url}")
        if err:
            think_parts.append(f"Last action error: {err[:200]}")
        think = " ".join(think_parts)
        parts.append(f"<think>\n{think}\n</think>\n<action>\n{action}\n</action>")
    return "\n\n".join(parts)


def extract_task_query(steps: list[dict]) -> str:
    if not steps:
        return ""
    first_obs = (steps[0].get("filtered_observation") or {})
    data = first_obs.get("data") or {}
    goal = data.get("goal_text")
    if isinstance(goal, str) and goal:
        return goal
    text = first_obs.get("text") or ""
    for line in text.splitlines():
        if line.lower().startswith("task:"):
            return line.split(":", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# Prompts (from original ReasoningBank WebArena)
# ---------------------------------------------------------------------------

SUCCESSFUL_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
""".strip()

FAILED_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
""".strip()


# ---------------------------------------------------------------------------
# Single-trajectory induction
# ---------------------------------------------------------------------------

def induce_one(args_tuple) -> list[MemoryItem]:
    (task_id, trace, success, task_query, llm_model, tier) = args_tuple
    traj_text = format_trajectory(trace.get("steps") or [])
    prompt_text = f"**Query:** {task_query}\n\n**Trajectory:**\n{traj_text}"
    system = SUCCESSFUL_SI if success else FAILED_SI
    for attempt in range(4):
        try:
            r = litellm.completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt_text}],
                **completion_kwargs(llm_model, temperature=1.0),
            )
            txt = r.choices[0].message.content or ""
            items = parse_memory_items(txt)
            return _embed_items(items[:3], task_id, tier)
        except Exception:
            if attempt == 3:
                return []
            time.sleep(2 ** attempt)
    return []


def _embed_items(items, task_id, tier) -> list[MemoryItem]:
    if not items:
        return []
    texts = [f"{it['title']}: {it['description']}" for it in items]
    try:
        embs = embed_texts(texts)
    except Exception:
        return []
    return [MemoryItem(
        title=it["title"], description=it["description"],
        content=it["content"], embedding=emb,
        source={"task_id": task_id, "tier": tier},
    ) for it, emb in zip(items, embs)]


def dedup_by_title(items: list[MemoryItem]) -> list[MemoryItem]:
    seen: set[str] = set()
    out = []
    for it in items:
        k = (it.title or "").strip().lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------

def _has_mutation(cand: dict) -> bool:
    if (cand.get("rules_code") or "").strip():
        return True
    if cand.get("in_env_actions"):
        return True
    return False


def select_v(traces: list[dict]) -> list[tuple[dict, bool, str]]:
    """V bank: mutation success > baseline success > baseline fail.
    Skip mutation failures."""
    acc_succ, expl_succ, base_succ, base_fail = [], [], [], []
    for t in traces:
        if not t.get("steps"):
            continue
        k = t.get("kind", "")
        s = bool(t.get("success"))
        cand = t.get("candidate") or {}
        if k == "baseline":
            (base_succ if s else base_fail).append(t)
        elif k in ("accepted", "exploration") and _has_mutation(cand) and s:
            (acc_succ if k == "accepted" else expl_succ).append(t)

    if acc_succ:
        return [(acc_succ[0], True, "accepted_succ")]
    if expl_succ:
        return [(expl_succ[0], True, "exploration_succ")]
    if base_succ:
        return [(base_succ[0], True, "baseline_succ")]
    if base_fail:
        return [(base_fail[0], False, "baseline_fail")]
    return []


def select_a(traces: list[dict]) -> list[tuple[dict, bool, str]]:
    """A bank: baseline success > baseline fail."""
    base_succ, base_fail = [], []
    for t in traces:
        if not t.get("steps"):
            continue
        if t.get("kind") != "baseline":
            continue
        (base_succ if t.get("success") else base_fail).append(t)

    if base_succ:
        return [(base_succ[0], True, "baseline_succ")]
    if base_fail:
        return [(base_fail[0], False, "baseline_fail")]
    return []


# ---------------------------------------------------------------------------
# Build bank
# ---------------------------------------------------------------------------

def build_bank(*, select_fn, traces_by_task: dict, llm_model: str,
               concurrency: int) -> Bank:
    work = []
    tier_counts: dict[str, int] = defaultdict(int)

    for task_id, traces in sorted(traces_by_task.items()):
        selected = select_fn(traces)
        if not selected:
            tier_counts["skip_no_steps"] += 1
            continue

        task_query = ""
        for t in traces:
            if t.get("steps"):
                task_query = extract_task_query(t["steps"])
                if task_query:
                    break
        if not task_query:
            tier_counts["skip_no_query"] += 1
            continue

        for trace, success, tier in selected:
            tier_counts[tier] += 1
            work.append((task_id, trace, success, task_query, llm_model, tier))

    print(f"  Tier breakdown: {dict(tier_counts)}")
    print(f"  Trajectories queued: {len(work)}", flush=True)
    if not work:
        return Bank()

    bank = Bank()
    t0 = time.time()
    n_done = 0
    n_items = 0
    empty = 0
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(induce_one, w): w[0] for w in work}
        for fut in cf.as_completed(futures):
            try:
                items = fut.result()
            except Exception as e:
                print(f"  error: {type(e).__name__}: {e}", file=sys.stderr)
                items = []
            if not items:
                empty += 1
            bank.add(items)
            n_done += 1
            n_items += len(items)
            if n_done % 10 == 0 or n_done == len(work):
                el = time.time() - t0
                print(f"  [{n_done}/{len(work)}] items={n_items} "
                      f"empty={empty} el={el/60:.1f}m", flush=True)

    bank.items = dedup_by_title(bank.items)
    return bank


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, type=Path,
                   help="all_traces.jsonl from corpus generation")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--llm-model", default="openai/gpt-4.1-mini")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args(argv)

    by_task: dict[str, list] = defaultdict(list)
    n_total = 0
    with args.traces.open() as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            n_total += 1
            tid = str(t.get("rollout_seed", t.get("task_id", "")))
            by_task[tid].append(t)

    print(f"[build_bank] {n_total} traces, {len(by_task)} tasks")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("\n=== V bank (mut succ > base succ > base fail) ===")
    v_bank = build_bank(
        select_fn=select_v, traces_by_task=dict(by_task),
        llm_model=args.llm_model, concurrency=args.concurrency,
    )
    v_out = args.out_dir / "ours_full.jsonl"
    v_bank.save(v_out)
    print(f"  -> {v_out} ({len(v_bank)} items)")

    print("\n=== A bank (base succ > base fail) ===")
    a_bank = build_bank(
        select_fn=select_a, traces_by_task=dict(by_task),
        llm_model=args.llm_model, concurrency=args.concurrency,
    )
    a_out = args.out_dir / "orig_full.jsonl"
    a_bank.save(a_out)
    print(f"  -> {a_out} ({len(a_bank)} items)")

    print(f"\ntotal wall: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
