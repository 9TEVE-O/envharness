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

"""SWE-bench + ReasoningBank eval driver.

Three test-time conditions:
    --bank none                       # nobank baseline
    --bank <orig_bank.jsonl>          # orig: intra-baseline paired-diff bank
    --bank <ours_bank.jsonl>          # ours: cascade (EnvHarness) bank

Optional --bank2 for stacked retrieval (top-k from each bank, lessons
concatenated). Two ways to combine banks:
    --bank A --bank2 B  --top-k 5 --top-k2 5    # stacked: 5 from each (10 total)
    Pre-merge into one .jsonl and run with --bank only  # merged: 5 from union

Run::

    export GEMINI_API_KEY=...
    python experiments/swebench/reasoning_bank_eval.py \\
        --bank runs/<run>/banks/ours_bank.jsonl \\
        --top-k 5 \\
        --n 407 --concurrency 4 \\
        --out runs/<run>/eval/ours.jsonl --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

# Make envharness importable when this script is run directly
# (`python experiments/swebench/reasoning_bank_eval.py ...`).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envharness.core.types import Candidate
from envharness.infra.model import (client_spec, missing_key_message,
                                    key_pool as _key_pool)
from envharness.orchestration.runner import (
    EnvSpec, EpisodeSpec, PolicySpec, SubprocessRunner,
)
from envharness.reasoning_bank import Bank


# ---------------------------------------------------------------------------
# Held-out task list: SWE-bench Verified MINUS Lite (~407 tasks)
# ---------------------------------------------------------------------------

def verified_minus_lite() -> list[tuple[str, str, str]]:
    """Return [(instance_id, repo, problem_statement), ...] for instances in
    Verified but NOT in Lite. Lite is what the banks are distilled from, so
    excluding the ~93 overlapping instances prevents leakage."""
    from datasets import load_dataset
    v = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    l = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    lite_ids = {row["instance_id"] for row in l}
    out: list[tuple[str, str, str]] = []
    for row in v:
        if row["instance_id"] in lite_ids:
            continue
        out.append((row["instance_id"], row["repo"], row["problem_statement"]))
    return out


# ---------------------------------------------------------------------------
# Memory block formatting (soft inject + per-step discuss gate)
# ---------------------------------------------------------------------------
#
# ReasoningBank-canonical soft header + the
# per-step "explicitly state whether any insight applies" gate (arXiv
# 2509.25140 Appendix A.2). Without the gate the policy follows retrieved
# lessons blindly; with it the policy reasons about applicability per step.

_PER_STEP_GATE = (
    "In each step, before acting, explicitly state whether any insight "
    "applies to the current observation and why; if none do, ignore them."
)

_SOFT_HEADER = (
    "Below are some memory items that I accumulated from past interactions "
    "in this environment that may be helpful to solve the task. You can use "
    "them when you feel they are relevant.\n"
    f"\n{_PER_STEP_GATE}\n"
)

_STRICT_HEADER = (
    "The following are reasoning insights from prior tasks. "
    "**CRITICAL: MOST insights below will NOT apply to your current task.** "
    "Read them once for context, then check each insight's `When to use` "
    "against the actual issue you are debugging. **Apply an insight ONLY if "
    "its failure mode explicitly matches what you observe in the codebase.** "
    "If the issue is a direct code bug that does not match any insight's "
    "specific trigger condition, IGNORE the insights entirely and solve via "
    "direct file inspection and editing. Do not invent uses for unrelated "
    "techniques.\n"
    f"\n{_PER_STEP_GATE}\n"
)


def build_memory_block(retrieved_items: list, strict: bool = False) -> str:
    if not retrieved_items:
        return ""
    header = _STRICT_HEADER if strict else _SOFT_HEADER
    parts = [header]
    # Frame the retrieved skills as "recovery techniques (consider doing
    # these)" before listing them.
    parts.append("## Recovery techniques (consider doing these):")
    parts.append("")
    for i, it in enumerate(retrieved_items, 1):
        parts.append(f"### Insight {i}: {it.title}")
        parts.append(f"_When to use_: {it.description}")
        parts.append(it.content)
        parts.append("")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Task prompt (mini-SWE-agent style)
# ---------------------------------------------------------------------------

_TASK_PROMPT_BASE = (
    "You are a helpful assistant that can interact with a computer shell to "
    "solve programming tasks.\n\n"
    "The repository is checked out at /testbed inside a Docker container. Use "
    "the `bash` tool to inspect and modify files. Each tool call runs as a "
    "fresh `docker exec` (shell state does NOT persist between calls); chain "
    "related commands with `&&` / `;` inside one call, or use absolute "
    "paths.\n\n"
    "Workflow:\n"
    "  1. Read the issue (provided in the first turn).\n"
    "  2. Locate the relevant file(s) under /testbed (e.g. with `grep -rn`).\n"
    "  3. Edit the file(s) -- `sed -i`, a python -c heredoc, or printf-into-file.\n"
    "  4. Run the relevant tests with `pytest` to verify your fix.\n"
    "  5. When ready, submit by calling bash with a command that does:\n"
    "        echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git -C /testbed diff\n"
    "     The Bridge will detect the sentinel and capture the git diff as "
    "your submitted patch. The official SWE-bench scorer then grades it.\n\n"
    "Do not waste turns on commentary. Issue tool calls."
)


# ---------------------------------------------------------------------------
# Per-episode runner
# ---------------------------------------------------------------------------

def _build_policy_spec(*, model: str, temperature: float, max_steps: int,
                         memory_block: str, gemini_api_key: str | None = None) -> PolicySpec:
    """Construct PolicySpec with the memory block prepended to task_prompt.

    `gemini_api_key` is threaded into client_kwargs so each concurrent episode uses a
    DISTINCT key from the GEMINI_API_KEYS pool (round-robin assigned in main).
    Without it, litellm falls back to the single GEMINI_API_KEY env var and
    every concurrent episode hammers ONE key -> 429 throttling -> mid-episode
    degraded responses -> empty patches. LiteLLMClient forwards client_kwargs
    to litellm.completion via self.defaults, so gemini_api_key here reaches the call."""
    task_prompt = (memory_block + "\n" + _TASK_PROMPT_BASE
                    if memory_block else _TASK_PROMPT_BASE)
    overrides: dict = {}
    # reasoning_effort: DEFAULT "low". A model that does not take the
    # parameter drops it (see envharness.infra.model); on Claude it becomes a
    # thinking budget. Override via RB_REASONING_EFFORT (off/none = omit;
    # minimal/low/medium/high force a level).
    _re = os.environ.get("RB_REASONING_EFFORT", "low")
    if _re.lower() not in ("off", "none", "default", ""):
        overrides["reasoning_effort"] = _re
    _tb = os.environ.get("CLAUDE_THINKING_BUDGET")
    if _tb:
        overrides["thinking_budget"] = int(_tb)
    # The pool key is threaded in so each concurrent episode uses a DISTINCT
    # key (round-robin assigned in main). Without it every episode hammers one
    # key -> 429 -> degraded responses -> empty patches. `api_key` is litellm's
    # name for it; the resolver drops the key on a provider it cannot
    # authenticate, so a Gemini pool does no harm on GPT or Claude.
    if gemini_api_key:
        overrides["api_key"] = gemini_api_key
    client_factory, client_kwargs = client_spec(model, **overrides)
    return PolicySpec(
        client_factory=client_factory,
        client_kwargs=client_kwargs,
        action_format="think_action",
        # max_history bounds MESSAGES (~2/step in think_action), so 200 only
        # truncates past ~step 100; most episodes never reach it. Overridable
        # via RB_MAX_HISTORY for A/B experiments.
        max_history=int(os.environ.get("RB_MAX_HISTORY", "200")),
        temperature=temperature,
        task_prompt=task_prompt,
    )


def _build_episode_spec(*, instance_id: str, max_steps: int,
                          step_timeout: int,
                          policy_spec: PolicySpec) -> EpisodeSpec:
    return EpisodeSpec(
        env=EnvSpec(
            # Bash-only bridge so think_action's single-tool dispatch works.
            import_path=(
                "experiments.swebench.bash_bridge:SWEBenchEnvBashOnly"
            ),
            reset_options={
                "subset": "verified",
                "instance_id": instance_id,
                "step_timeout_seconds": step_timeout,
                "container_timeout_seconds": 7200,
                "eval_timeout_seconds": 1800,
            },
            reset_seed=0,
        ),
        candidate=Candidate(rules_code="", in_env_actions=[],
                             rationale="rb-eval"),
        policy=policy_spec,
        iteration_id=f"rb-{uuid.uuid4().hex[:6]}",
        task_id=f"verified:{instance_id}",
        max_steps=max_steps,
    )


def run_one(*, idx: int, instance_id: str, repo: str,
             problem_statement: str,
             bank: Bank | None, bank2: Bank | None,
             top_k: int, top_k2: int,
             strict_inject: bool,
             max_steps: int, step_timeout: int,
             episode_timeout: float,
             model: str, temperature: float,
             gemini_api_key: str | None = None) -> dict:
    """Run one held-out episode end-to-end. Includes retrieval, prompt
    assembly, episode dispatch, and per-bank inner-retry on transient
    `subprocess exit N` failures (docker daemon race recovery)."""
    t0 = time.time()
    # Retrieve top-k from each bank independently (stacked when bank2 present).
    # Truncate the query to 6000 chars BEFORE embedding. Some SWE-bench
    # problem_statements run 10k+ chars; embedding the full vs the truncated
    # text yields DIFFERENT query vectors -> different retrieved skills, so a
    # fixed truncation keeps conditions comparable.
    query = problem_statement[:6000]
    retrieved: list = []
    if bank is not None and top_k > 0:
        retrieved.extend(bank.retrieve(query, k=top_k))
    if bank2 is not None and top_k2 > 0:
        retrieved.extend(bank2.retrieve(query, k=top_k2))
    memory_block = build_memory_block(retrieved, strict=strict_inject)

    policy_spec = _build_policy_spec(
        model=model, temperature=temperature, max_steps=max_steps,
        memory_block=memory_block, gemini_api_key=gemini_api_key,
    )
    spec = _build_episode_spec(
        instance_id=instance_id, max_steps=max_steps,
        step_timeout=step_timeout, policy_spec=policy_spec,
    )

    # Inner retry on transient infra failures. Each retry uses a fresh
    # SubprocessRunner so any leaked docker container handle from the
    # previous attempt is irrelevant.
    MAX_RETRIES = 3
    # LLM-transient signatures inside trace.error: the in-episode client
    # already retries with backoff (LiteLLMClient, 5 attempts); if a burst
    # outlives that, re-run the EPISODE in place after a longer cool-down
    # instead of recording an API hiccup as a task failure.
    _LLM_TRANSIENT = ("RateLimitError", "ServiceUnavailableError",
                       "InternalServerError", "APIConnectionError",
                       "Timeout", "429", "502", "503")
    trace = None
    for attempt in range(MAX_RETRIES):
        runner = SubprocessRunner(timeout=episode_timeout)
        trace = runner.run(spec)
        e = trace.error or ""
        is_infra = (e.startswith("subprocess exit")
                     or e.startswith("subprocess timeout"))
        is_llm_transient = any(sig in e for sig in _LLM_TRANSIENT)
        if not (is_infra or is_llm_transient):
            break
        if attempt < MAX_RETRIES - 1:
            # docker races recover fast; API bursts need a longer cool-down
            time.sleep((60 + 60 * attempt) if is_llm_transient
                        else (5 + 5 * attempt))
    el = time.time() - t0

    last_info = trace.steps[-1].info if (trace and trace.steps) else {}
    return {
        "dataset_idx": idx,
        "instance_id": instance_id,
        "repo": repo,
        "success": bool(trace.success) if trace else False,
        "duration_steps": int(trace.duration_steps) if trace else 0,
        "duration_ms": int(el * 1000),
        "submitted": bool(last_info.get("submitted")) if last_info else False,
        "submitted_patch_len": int(last_info.get("submitted_patch_len", 0))
            if last_info else 0,
        "final_reward": float(trace.final_reward) if trace else 0.0,
        "error": trace.error if trace else "no_trace",
        "retrieved_titles": [it.title for it in retrieved],
    }


# ---------------------------------------------------------------------------
# CLI / Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default=None,
                    help="Primary Bank JSONL path. Omit for the no-bank "
                    "baseline condition.")
    p.add_argument("--bank2", default=None,
                    help="Optional second Bank for STACKED retrieval. If set, "
                    "top-k2 lessons from --bank2 are retrieved independently "
                    "of --bank and concatenated into the inject block.")
    p.add_argument("--top-k", type=int, default=5,
                    help="Top-k lessons retrieved from --bank.")
    p.add_argument("--top-k2", type=int, default=5,
                    help="Top-k lessons retrieved from --bank2.")
    p.add_argument("--strict-inject", action="store_true",
                    help="Use the STRICT memory-block header instead of "
                    "the default soft header.")
    p.add_argument("--n", type=int, default=407,
                    help="Number of held-out tasks (default: full 407).")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--instance-ids", default="",
                    help="Comma-separated instance_id allowlist; overrides "
                         "--n/--start. For targeted harness/config probes.")
    p.add_argument("--concurrency", type=int, default=4,
                    help="Max parallel SubprocessRunner episodes. Each spawns "
                    "1 docker container for the Policy step loop + 1 for the "
                    "official scorer (sequential per task).")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--model", default="openai/gpt-4.1-mini")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--step-timeout-seconds", type=int, default=60)
    p.add_argument("--episode-timeout-seconds", type=float, default=2400.0)
    p.add_argument("--out", required=True,
                    help="Output JSONL path (one row per episode).")
    p.add_argument("--resume", action="store_true",
                    help="If --out exists, read it and skip instance_ids "
                    "already evaluated.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Key POOL for round-robin across concurrent episodes: pinning them all to
    # one key throttles them against each other (see _build_policy_spec). The
    # pool belongs to args.model's provider, so a Vertex run needs none.
    _missing = missing_key_message(args.model)
    if _missing:
        print(f"ERROR: {_missing}", file=sys.stderr)
        return 2
    key_pool = _key_pool(args.model)
    gemini_api_key = key_pool[0] if key_pool else None
    key_pool = key_pool or [None]
    print(f"[keys] rotating across {len(key_pool)} API key(s) "
          f"(~{args.concurrency / max(len(key_pool), 1):.1f} concurrent/key)",
          flush=True)

    print("[load] resolving Verified minus Lite ...", flush=True)
    held = verified_minus_lite()
    if args.instance_ids:
        wl = {x.strip() for x in args.instance_ids.split(",") if x.strip()}
        held = [t for t in held if t[0] in wl]
    else:
        held = held[args.start:args.start + args.n]
    print(f"[load] {len(held)} held-out tasks", flush=True)

    bank = Bank.load(args.bank) if args.bank else None
    bank2 = Bank.load(args.bank2) if args.bank2 else None
    if bank is not None:
        print(f"[load] bank:  {len(bank)} items  ({args.bank})", flush=True)
    else:
        print("[load] bank:  (none) -- no-bank baseline mode", flush=True)
    if bank2 is not None:
        print(f"[load] bank2: {len(bank2)} items  ({args.bank2})  "
              f"(top-k2={args.top_k2})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip rows that are a REAL result; re-run only DEAD (infra) rows.
    # Dead-sweep criterion (this is NOT an empty-patch policy):
    #   DEAD (re-run) iff:
    #     * duration_steps == 0            -- agent never took a step (container
    #                                         crash / teardown race / died before
    #                                         acting); no real attempt happened.
    #     * error is infra-transient       -- "subprocess exit N" / "subprocess
    #                                         timeout" (docker race) OR an LLM
    #                                         transient (RateLimit/429/5xx/conn/
    #                                         timeout) that outlived Layer-1's
    #                                         in-episode retries. An API failure
    #                                         is not the agent's result.
    #   REAL (done, never re-run) otherwise -- duration_steps > 0 with no infra
    #     error. This INCLUDES an empty patch: the agent ran but produced no
    #     diff == a genuine solve failure (NOT re-run; re-running to fish for a
    #     patch would inflate SR). It also includes a submitted-but-test-failed
    #     patch. Only infra-dead tasks are re-run, never flipping a fail into
    #     a success.
    _INFRA_SIGS = ("subprocess exit", "subprocess timeout", "RateLimitError",
                   "ServiceUnavailableError", "InternalServerError",
                   "APIConnectionError", "Timeout", "429", "502", "503",
                   "no_trace",
                   # env.step reading a non-utf8 byte from a container command's
                   # output is a transient infra error (not the agent's result),
                   # so re-run it too.
                   "UnicodeDecodeError")
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        n_dead_requeued = 0
        last_row: dict = {}
        for ln in out_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            iid = r.get("instance_id")
            if not iid:
                continue
            last_row[iid] = r
        for iid, r in last_row.items():
            err = str(r.get("error") or "")
            is_infra = bool(err) and any(sig in err for sig in _INFRA_SIGS)
            is_dead = is_infra or int(r.get("duration_steps", 0)) == 0
            if is_dead:
                n_dead_requeued += 1         # infra dead: re-run
            else:
                done_ids.add(iid)            # real result (incl. empty patch)
        print(f"[resume] {len(done_ids)} real results skipped; "
              f"{n_dead_requeued} DEAD (infra/0-step) rows re-queued", flush=True)
    # dataset_idx must be absolute (slice offset by args.start), so rows from
    # --start 100 don't collide with --start 0 rows when merged.
    pending = [(args.start + i, iid, repo, ps)
                for i, (iid, repo, ps) in enumerate(held)
                if iid not in done_ids]
    print(f"[run] pending: {len(pending)} episodes  concurrency={args.concurrency}",
          flush=True)

    mode = "a" if args.resume and out_path.exists() else "w"
    out_f = out_path.open(mode, buffering=1)

    t0 = time.time()
    n_done = 0
    n_won = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                run_one,
                idx=idx, instance_id=iid, repo=repo,
                problem_statement=ps,
                bank=bank, bank2=bank2,
                top_k=args.top_k, top_k2=args.top_k2,
                strict_inject=args.strict_inject,
                max_steps=args.max_steps,
                step_timeout=args.step_timeout_seconds,
                episode_timeout=args.episode_timeout_seconds,
                model=args.model, temperature=args.temperature,
                gemini_api_key=key_pool[idx % len(key_pool)],
            )
            for (idx, iid, repo, ps) in pending
        ]
        for fut in cf.as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                # run_one should not normally raise -- it captures errors
                # on the returned dict. Safety net.
                r = {"instance_id": "<unknown>", "success": False,
                      "error": f"{type(e).__name__}: {e}"}
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1
            if r.get("success"):
                n_won += 1
            if n_done % 5 == 0 or n_done == len(pending):
                el_min = (time.time() - t0) / 60
                rate = n_done / max(el_min, 1e-3)
                print(
                    f"[progress] {n_done}/{len(pending)}  "
                    f"SR={n_won}/{n_done}={n_won / max(n_done, 1):.3f}  "
                    f"elapsed={el_min:.1f}m  rate={rate:.1f}/min",
                    flush=True,
                )

    out_f.close()

    # Final summary
    all_alive = []
    for ln in out_path.read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        try: all_alive.append(json.loads(ln))
        except Exception: pass
    n_all = len(all_alive)
    n_won_all = sum(1 for r in all_alive if r.get("success"))
    print()
    bank_str = args.bank or "(none)"
    if args.bank2:
        bank_str += f" + {args.bank2}"
    print(f"=== bank={bank_str} n={n_all}  "
          f"SR={n_won_all}/{n_all} = {n_won_all / max(n_all, 1):.3f} ===",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
