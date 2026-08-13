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

"""Stage 4: OfficeQA + ReasoningBank eval driver.

OfficeQA clone of `experiments/spreadsheetbench/reasoning_bank_eval.py`. Evaluates three
test-time conditions on the held-out TEST split:
  nobank -- no skills (the baseline)
  orig   -- skills induced from ORIGINAL (baseline) corpus trajectories
  ours   -- skills induced from MUTATED (accepted) corpus trajectories

For each task it retrieves top-K skills by the task question, injects them into
the policy system prompt (cli_skill_preloaded protocol), runs a function-calling
grep/read/glob/answer loop against OfficeQAEnv, and grades with the env's
normalized exact-match `evaluate()`.

Self-contained (drives the env directly, not the Orchestrator). Writes
per-episode JSONL + a summary.json. OfficeQA uses SEPARATE splits, so eval seeds
index the TEST split directly (0..N-1) via reset_options.split=test -- there is
NO 100-offset like spreadsheet.

Run from the repo root:
  export GEMINI_API_KEY=...
  python experiments/officeqa/reasoning_bank_eval.py \
      --config experiments/officeqa/reasoning_bank_eval.yaml \
      --out-dir runs/oqa_eval_001 \
      --bank-overrides orig=runs/oqa_corpus_001/banks/orig_subset.jsonl,ours=runs/oqa_corpus_001/banks/ours_subset_matched.jsonl
"""
from __future__ import annotations

from envharness.infra.model import completion_kwargs
from envharness.infra.model import key_env, key_pool, missing_key_message

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import litellm
import yaml

from envharness.bridges.officeqa.bridge import OfficeQAEnv
from experiments.officeqa.prompts import build_agent_system, task_query
from envharness.core.types import Action
from envharness.reasoning_bank import Bank


def _retrieve_block(bank: Bank | None, query: str, top_k: int,
                    mode: str = "cosine", mmr_lambda: float = 0.5) -> tuple[str, list[str]]:
    if not bank or top_k <= 0:
        return "", []
    items = bank.retrieve(query, k=top_k, mode=mode, mmr_lambda=mmr_lambda)
    block = "\n\n".join(it.text for it in items)
    return block, [it.title for it in items]


def _retrieve_balanced(bank: Bank | None, query: str, top_k: int) -> tuple[str, list[str]]:
    """Source-balanced retrieval: split the bank by source['condition'] and take
    the top ceil(k/G) per group, so a merged orig+ours bank always surfaces
    BOTH banks' most-relevant skills instead of letting one crowd out the other
    under a single global top-k."""
    if not bank or top_k <= 0 or not bank.items:
        return "", []
    from envharness.reasoning_bank.embed import cosine, embed_texts
    qemb = embed_texts([query])[0]
    groups: dict[str, list] = {}
    for it in bank.items:
        cond = (it.source or {}).get("condition", "?") if isinstance(it.source, dict) else "?"
        groups.setdefault(cond, []).append(it)
    per = max(1, -(-top_k // max(len(groups), 1)))   # ceil(k/G)
    picked = []
    for _cond, items in sorted(groups.items()):
        scored = sorted(((cosine(it.embedding, qemb), it) for it in items),
                        key=lambda t: t[0], reverse=True)
        picked += [it for _, it in scored[:per]]
    picked = picked[:top_k] if top_k < len(picked) else picked
    block = "\n\n".join(it.text for it in picked)
    return block, [it.title for it in picked]


def _dispatch(env: OfficeQAEnv, name: str, args_json: str):
    try:
        kwargs = json.loads(args_json) if args_json else {}
    except Exception:
        kwargs = {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    return env.step(Action(name=name, kwargs=kwargs))


def run_episode(*, cfg: dict, bank: Bank | None, seed: int,
                gemini_api_key: str | None, skill_doc: str | None = None,
                skill_dir: str | None = None,
                skill_chat: str | None = None) -> dict:
    t0 = time.time()
    rec = {"seed": seed, "success": False, "duration_steps": 0,
           "duration_ms": 0, "error": "", "retrieved_titles": [], "task_id": ""}
    env = OfficeQAEnv()
    try:
        reset = env.reset(seed=seed, options=dict(cfg["env"].get("reset_options") or {}))
        s = env.get_env_state()
        rec["task_id"] = s.task_id
        from experiments.officeqa.prompts import (
            build_agent_system_doc, build_agent_system_skilldir,
            build_agent_system_skillopt, per_step_gate)
        skill_block = ""   # set in the retrieval branch; used for per-step re-injection
        if skill_chat is not None:
            # SkillOpt-faithful direct-chat: bare "## Skill" in system, once,
            # no per-step gate.
            system = build_agent_system_skillopt(skill_chat)
            rec["retrieved_titles"] = ["<skillopt-chat>"]
            skills_active = False
        elif skill_dir is not None:
            # On-disk skill folder: preload SKILL.md, agent reads references/ on
            # demand. No per-step gate (one-shot system prompt protocol).
            import os as _os
            abs_dir = _os.path.abspath(skill_dir)
            skill_md = ""
            md_path = _os.path.join(abs_dir, "SKILL.md")
            if _os.path.isfile(md_path):
                skill_md = open(md_path).read()
            system = build_agent_system_skilldir(skill_md, abs_dir)
            rec["retrieved_titles"] = ["<skill-dir>"]
            skills_active = False
        elif skill_doc is not None:
            # Whole-doc mandatory injection; no per-task retrieval.
            system = build_agent_system_doc(skill_doc)
            rec["retrieved_titles"] = ["<consolidated-skill-doc>"]
            skills_active = bool(skill_doc.strip())
        else:
            query = task_query(s.question)
            top_k = int(cfg["retrieval"]["top_k"])
            if cfg["retrieval"].get("balanced"):
                skill_block, titles = _retrieve_balanced(bank, query, top_k)
            else:
                skill_block, titles = _retrieve_block(
                    bank, query, top_k,
                    mode=cfg["retrieval"].get("mode", "cosine"),
                    mmr_lambda=float(cfg["retrieval"].get("mmr_lambda", 0.5)))
            rec["retrieved_titles"] = titles
            system = build_agent_system(skill_block or None)
            skills_active = bool(skill_block)
        # Per-step SKILL RE-INJECTION (matches the working alfworld/webarena eval
        # loops). In FC mode the system-prompt skill block decays in salience as
        # the transcript grows over a 30-step horizon; re-presenting the retrieved
        # skill block (not just a gate sentence) on every observation keeps the
        # skills at the decision point. Empty for the base condition.
        if skills_active:
            _sb = (skill_block or "").strip()
            gate = "\n\n" + (f"## Relevant skills (consult before acting)\n{_sb}\n\n"
                             if _sb else "") + per_step_gate()
        else:
            gate = ""
        tools = OfficeQAEnv.tool_schemas()

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": reset.observation.text}]
        max_steps = int(cfg["policy"]["max_steps"])
        no_tool_strikes = 0
        for _ in range(max_steps):
            r = None
            last_err = None
            # Retry transient API errors (503 ServiceUnavailable, 429 rate
            # limit, 5xx Internal) with exponential backoff so a momentary
            # provider hiccup under high concurrency does not get scored as a
            # task failure. Non-transient errors break immediately.
            for attempt in range(5):
                try:
                    _extra = {}
                    if cfg["model"].get("reasoning_effort"):
                        _extra["reasoning_effort"] = cfg["model"]["reasoning_effort"]
                    r = litellm.completion(
                        messages=messages, tools=tools, tool_choice="auto",
                        **completion_kwargs(
                            cfg["model"]["name"],
                            temperature=float(cfg["model"]["temperature"]),
                            max_tokens=int(cfg["model"]["max_response_tokens"]),
                            api_key=gemini_api_key, **_extra),
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    name = type(e).__name__.lower()
                    transient = ("serviceunavailable" in name or "ratelimit" in name
                                 or "internalserver" in name or "timeout" in name
                                 or "overloaded" in str(e).lower())
                    if not transient or attempt == 4:
                        break
                    time.sleep(2 ** attempt)
            if r is None:
                rec["error"] = f"{type(last_err).__name__}: {last_err}"
                break
            msg = r.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            # Record the assistant turn. Only the FIRST tool call gets answered
            # below, so record ONLY that one -- recording all of them leaves
            # unanswered tool_call_ids in later turns, which strict providers
            # reject.
            first_tc = tool_calls[0] if tool_calls else None
            messages.append({"role": "assistant",
                             "content": msg.content or "",
                             "tool_calls": [first_tc.model_dump()
                                            if hasattr(first_tc, "model_dump")
                                            else first_tc]
                             if first_tc is not None else None})
            if not tool_calls:
                no_tool_strikes += 1
                if no_tool_strikes >= 2:
                    break
                messages.append({"role": "user", "content": (
                    "Call grep(pattern, path) / read(path, start, limit) / "
                    "glob(pattern) to locate the evidence, or answer(text) once "
                    "you have the value.")})
                continue
            no_tool_strikes = 0
            tc = tool_calls[0]
            name = tc.function.name
            resp = _dispatch(env, name, tc.function.arguments)
            rec["duration_steps"] += 1
            # Answer the tool call (Gemini via litellm maps role=tool correctly).
            # Re-inject the per-step gate on every observation when skills are
            # active (RB-canonical), so the agent re-evaluates skill relevance
            # each step rather than only at turn 0.
            messages.append({"role": "tool", "tool_call_id": getattr(tc, "id", "") or "0",
                             "name": name, "content": resp.observation.text[:6000] + gate})
            if resp.terminated or resp.truncated:
                break
        _ev = env.evaluate()
        rec["success"] = bool(_ev.success)          # metric 1: normalized EM
        _m = _ev.metrics or {}
        rec["f1"] = float(_m.get("f1", 0.0))        # metric 2: token-level F1 (partial credit)
        rec["em"] = float(_m.get("em", 1.0 if _ev.success else 0.0))
        rec["predicted"] = _m.get("predicted", "")
        rec["gold"] = _m.get("gold", "")
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    return rec


def _worker(seed: int, *, cfg: dict, bank: Bank | None, gemini_api_keys: list[str]) -> dict:
    key = gemini_api_keys[os.getpid() % len(gemini_api_keys)] if gemini_api_keys else None
    if key:
        # Bank.retrieve -> embed_texts needs the provider's key in the env.
        os.environ.update(key_env(cfg["model"]["name"], key))
    return run_episode(cfg=cfg, bank=bank, seed=seed, gemini_api_key=key)


def _run_condition(*, cfg, label, bank, seeds, out_path, pool, gemini_api_keys):
    n_won = n_done = n_error = 0
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn = partial(_worker, cfg=cfg, bank=bank, gemini_api_keys=gemini_api_keys)
    with out_path.open("w", buffering=1) as f:
        for r in pool.map(fn, seeds):
            f.write(json.dumps(r) + "\n")
            # An eval_error (transient API failure) is NOT a policy failure --
            # exclude it from the SR denominator and count it separately, rather
            # than depressing SR.
            if (r.get("error") or "").strip():
                n_error += 1
                continue
            n_done += 1
            n_won += int(bool(r.get("success")))
            if n_done % 10 == 0 or n_done == len(seeds):
                rate = (n_done + n_error) / max(time.time() - t0, 1e-3) * 60
                print(f"    [{label}] {n_done}/{len(seeds)}  "
                      f"SR={n_won}/{n_done}={n_won/max(n_done,1):.3f}  "
                      f"err={n_error} rate={rate:.1f}/min", flush=True)
    return n_won, n_done, n_error


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--gemini-api-keys", default=None)
    p.add_argument("--start-idx", type=int, default=None, help="first held-out task index")
    p.add_argument("--n", type=int, default=None, help="number of held-out tasks")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--conditions", default=None)
    p.add_argument("--bank-overrides", default=None,
                   help="comma-separated cond=path pairs")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    # Keys belong to whatever provider the config's model names, so a GPT run
    # needs no Gemini key and a Vertex run needs no key at all.
    _model = cfg["model"]["name"]
    _missing = missing_key_message(_model)
    if _missing:
        raise SystemExit(_missing)
    gemini_api_keys = [k.strip() for k in (args.gemini_api_keys or "").split(",")
                        if k.strip()] or key_pool(_model)
    start_idx = args.start_idx if args.start_idx is not None else int(cfg["eval"]["start_idx"])
    n = args.n if args.n is not None else int(cfg["eval"]["n"])
    concurrency = args.concurrency or int(cfg["eval"]["concurrency"])
    conditions = (args.conditions.split(",") if args.conditions
                  else list(cfg["conditions"]))
    seeds = list(range(start_idx, start_idx + n))

    overrides: dict[str, str] = {}
    if args.bank_overrides:
        for kv in args.bank_overrides.split(","):
            k, _, v = kv.partition("=")
            if k.strip() and v.strip():
                overrides[k.strip()] = v.strip()

    banks: dict[str, Bank | None] = {}
    for cond in conditions:
        bp = overrides.get(cond, (cfg["conditions"][cond] or {}).get("bank"))
        banks[cond] = Bank.load(bp) if bp else None
        print(f"  {cond:8s} bank: {len(banks[cond]) if banks[cond] else 0} items"
              f"  ({bp or 'none'})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[oqa reasoning_bank_eval] model={cfg['model']['name']}  conditions={conditions}")
    print(f"  held-out tasks: {seeds[0]}..{seeds[-1]} (n={n})  concurrency={concurrency}")

    rows = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=concurrency) as pool:
        for cond in conditions:
            out_path = out_dir / f"{cond}.jsonl"
            print(f"\n>> condition={cond}")
            n_won, n_done, n_error = _run_condition(
                cfg=cfg, label=cond, bank=banks[cond], seeds=seeds,
                out_path=out_path, pool=pool, gemini_api_keys=gemini_api_keys)
            rows.append({"condition": cond, "n_won": n_won, "n": n_done,
                         "n_error": n_error, "sr": n_won / max(n_done, 1)})

    print("\n=== SUMMARY (held-out SR; eval_errors excluded from denominator) ===")
    for r in rows:
        err = f"  (+{r['n_error']} eval_error)" if r.get("n_error") else ""
        print(f"  {r['condition']:8s} {r['n_won']}/{r['n']} = {r['sr']*100:.1f}%{err}")
    by = {r["condition"]: r["sr"] for r in rows}
    if "ours" in by and "orig" in by:
        print(f"  delta(ours-orig) = {(by['ours']-by['orig'])*100:+.1f} pp")
    if "ours" in by and "nobank" in by:
        print(f"  delta(ours-nobank) = {(by['ours']-by['nobank'])*100:+.1f} pp")
    print(f"  wall: {(time.time()-t0)/60:.1f} min")

    with (out_dir / "summary.json").open("w") as f:
        json.dump({"rows": rows, "start_idx": start_idx, "n": n,
                   "config_path": str(args.config)}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
