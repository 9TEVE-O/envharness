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

"""ALFWorld + ReasoningBank eval driver.

The protocol (prompts, history length, retrieval, splits) is fully
captured by `experiments/alfworld/reasoning_bank_eval.yaml`. This script is just
the runner: it loads the YAML, instantiates `AlfworldEnv` and `Bank`,
calls the model via litellm, and writes per-episode JSONL.

Runs the nobank / orig / ours comparison across both eval splits,
optionally over multiple rounds (start seeds), and prints per-round
and cross-round (mean ± std) summaries.
"""
from __future__ import annotations

from envharness.infra.model import completion_kwargs
from envharness.infra.model import key_env, key_pool, missing_key_message

import argparse
import concurrent.futures as cf
import json
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import litellm
import yaml

from envharness.bridges.alfworld import AlfworldEnv
from envharness.core.types import Action
from envharness.prompts.alfworld_skill_prompt import (
    ACTION_RE, SKILLOS_TEMPLATE, build_memory_block, extract_task,
    format_admissible, format_history, normalize_to_admissible,
    strip_task_and_admissibles,
)
from envharness.infra.llm import completion_with_retry
from envharness.reasoning_bank import Bank


# ---------------------------------------------------------------------------
# Per-episode eval
# ---------------------------------------------------------------------------

def run_episode(*, cfg: dict, bank: Bank | None, seed: int, split: str,
                  gemini_api_key: str | None) -> dict:
    t0 = time.time()
    rec = {"seed": seed, "split": split, "success": False,
            "duration_steps": 0, "duration_ms": 0,
            "final_reward": 0.0, "error": "", "retrieved_titles": []}

    env = AlfworldEnv()
    try:
        reset_resp = env.reset(seed=seed, options={
            "split": split, **(cfg["env"].get("reset_options") or {}),
        })
        obs = reset_resp.observation
        obs_text = obs.text or ""
        admissibles = list((obs.data or {}).get("admissible_commands") or [])
        task = extract_task(obs_text)
        current_obs = strip_task_and_admissibles(obs_text)

        top_k = int(cfg["retrieval"]["top_k"])
        _rmode = cfg["retrieval"].get("mode", "cosine")
        _rlam = float(cfg["retrieval"].get("mmr_lambda", 0.5))
        retrieved = (bank.retrieve(task, k=top_k, mode=_rmode, mmr_lambda=_rlam)
                     if (bank and top_k > 0) else [])
        memory_block = build_memory_block(
            retrieved, style=cfg["policy"]["inject_style"])
        rec["retrieved_titles"] = [it.title for it in retrieved]

        history: list[dict] = []
        final_reward = 0.0
        won = False
        for step in range(int(cfg["policy"]["max_steps"])):
            adm_str = format_admissible(admissibles)
            action_history_str, valid_hl = format_history(
                history, int(cfg["policy"]["history_length"]))
            prompt = SKILLOS_TEMPLATE.format(
                task_description=task,
                retrieved_skills=memory_block.rstrip("\n") if memory_block else "",
                step_count=len(history),
                history_length=valid_hl,
                action_history=action_history_str,
                current_step=len(history) + 1,
                current_observation=current_obs,
                admissible_actions=adm_str,
            )
            try:
                # In-place transient retry (429/5xx/timeout wait-and-retry);
                # only non-transient errors fall through to rec["error"].
                r = completion_with_retry(
                    messages=[{"role": "user", "content": prompt}],
                    **completion_kwargs(
                        cfg["model"]["name"],
                        temperature=float(cfg["model"]["temperature"]),
                        max_tokens=int(cfg["model"]["max_response_tokens"]),
                        api_key=gemini_api_key),
                )
                txt = (r.choices[0].message.content or "")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                break

            m = list(ACTION_RE.finditer(txt))
            raw_act = (m[-1].group(1).strip() if m
                       else (txt.strip().splitlines()[-1].strip()
                              if txt.strip() else ""))
            normalized = normalize_to_admissible(raw_act, admissibles)
            env_resp = env.step(Action(name="do", kwargs={"text": normalized}))
            new_text = env_resp.observation.text or ""
            admissibles = list((env_resp.observation.data or {})
                                .get("admissible_commands") or [])
            current_obs = strip_task_and_admissibles(new_text)
            final_reward = float(env_resp.reward or 0.0)
            info = env_resp.info or {}
            won = (info.get("success") is True
                   or bool((info.get("result") or {}).get("won", False))
                   or final_reward >= 1.0)
            history.append({"action": normalized, "obs": current_obs[:500]})
            rec["duration_steps"] = step + 1
            if env_resp.terminated or env_resp.truncated:
                break
        rec["success"] = bool(won)
        rec["final_reward"] = final_reward
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            env.close()
        except Exception:
            pass
    rec["duration_ms"] = int((time.time() - t0) * 1000)
    return rec


# ---------------------------------------------------------------------------
# Driver: 3 conditions × 2 splits in parallel
# ---------------------------------------------------------------------------

def _worker(seed: int, *, cfg: dict, bank: Bank | None, split: str,
             gemini_api_keys: list[str]) -> dict:
    """Process-pool entry. Picks an API key by PID so each worker process
    consistently hits one key (helps the provider's per-key rate-limiter
    spread load). Recreates AlfworldEnv per process so TextWorld's
    non-thread-safe engine state stays isolated.

    Also exports the provider's key into this process's env so
    Bank.retrieve -> embed_texts -> litellm.embedding picks it up. Without
    this, every retrieval-augmented condition fails to authenticate before
    the first step."""
    import os
    key = gemini_api_keys[os.getpid() % len(gemini_api_keys)] if gemini_api_keys else None
    if key:
        os.environ.update(key_env(cfg["model"]["name"], key))
    return run_episode(cfg=cfg, bank=bank, seed=seed, split=split,
                        gemini_api_key=key)


def _run_condition(*, cfg: dict, label: str, bank: Bank | None, split: str,
                    seeds: list[int], out_path: Path, pool: cf.ProcessPoolExecutor,
                    gemini_api_keys: list[str]) -> tuple[int, int]:
    """Submit one (condition, split) batch into an EXTERNALLY-managed
    ProcessPool. Pool lifetime is shared across the cells of one condition
    (both splits) so we don't pay the per-cell warmup cost (AlfredTWEnv
    init is ~10s per worker) while a dead worker only takes down its own
    condition."""
    n_won = n_done = 0
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn = partial(_worker, cfg=cfg, bank=bank, split=split, gemini_api_keys=gemini_api_keys)
    with out_path.open("w", buffering=1) as f:
        for r in pool.map(fn, seeds):
            f.write(json.dumps(r) + "\n")
            n_done += 1
            n_won += int(bool(r.get("success")))
            if n_done % 10 == 0 or n_done == len(seeds):
                rate = n_done / max(time.time() - t0, 1e-3) * 60
                print(f"    [{label}/{split[:6]}] {n_done}/{len(seeds)}  "
                      f"SR={n_won}/{n_done}={n_won/n_done:.3f}  "
                      f"rate={rate:.1f}/min", flush=True)
    return n_won, n_done


def _run_one_round(*, cfg: dict, banks: dict, splits_cfg: dict,
                     start_seed: int, round_dir: Path, concurrency: int,
                     gemini_api_keys, round_idx: int,
                     on_condition_done=None) -> list[dict]:
    """One pass over all (condition, split) cells with the given start_seed.

    A fresh ProcessPoolExecutor is created PER CONDITION so that a single
    worker death (BrokenProcessPool) aborts only that condition's cells,
    not the whole multi-round run; `on_condition_done(rows_so_far)` is
    invoked after each condition so summary.json can be refreshed
    incrementally instead of only at the very end."""
    rows = []
    for cond, bank in banks.items():
        try:
            with cf.ProcessPoolExecutor(max_workers=concurrency) as pool:
                for split, n in splits_cfg.items():
                    seeds = list(range(start_seed, start_seed + int(n)))
                    out_path = round_dir / f"{cond}_{split}.jsonl"
                    print(f"\n>> condition={cond}  split={split}  n={n}  "
                          f"seeds {seeds[0]}..{seeds[-1]}", flush=True)
                    n_won, n_done = _run_condition(
                        cfg=cfg, label=cond, bank=bank, split=split, seeds=seeds,
                        out_path=out_path, pool=pool, gemini_api_keys=gemini_api_keys,
                    )
                    rows.append({"condition": cond, "split": split,
                                  "n_won": n_won, "n": n_done,
                                  "sr": n_won / max(n_done, 1),
                                  "start_seed": start_seed,
                                  "round": round_idx})
        except BrokenProcessPool as e:
            print(f"[WARN] round {round_idx} condition={cond} aborted: "
                  f"worker process died ({e}); continuing with next "
                  f"condition on a fresh pool", flush=True)
        if on_condition_done:
            on_condition_done(rows)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--gemini-api-keys", default=None,
                    help="Comma-separated API keys (round-robin by worker "
                    "PID). Defaults to $GEMINI_API_KEYS, then $GEMINI_API_KEY.")
    p.add_argument("--n-indist", type=int, default=None)
    p.add_argument("--n-ood", type=int, default=None)
    p.add_argument("--start-seeds", default=None,
                    help="Comma-separated start_seeds, one per round. "
                    "Defaults to the config's eval.start_seed (single round) "
                    "if present, else '0,1000,2000'.")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None,
                    help="Override retrieval.top_k from the config (K-curve sweeps).")
    p.add_argument("--retrieval-mode", default=None, choices=["cosine","mmr"],
                    help="Override retrieval.mode (mmr = diversity-aware, helps full banks).")
    p.add_argument("--conditions", default=None,
                    help="comma-separated subset of YAML conditions")
    p.add_argument("--bank-overrides", default=None,
                    help="comma-separated 'cond=path' pairs, e.g. "
                    "'orig=runs/foo/orig.jsonl,ours=runs/foo/ours.jsonl'; "
                    "overrides the YAML conditions[].bank entries.")
    args = p.parse_args(argv)

    import os
    cfg = yaml.safe_load(args.config.read_text())
    # Keys belong to whatever provider the config's model names.
    _model = cfg["model"]["name"]
    _missing = missing_key_message(_model)
    if _missing:
        raise SystemExit(_missing)
    gemini_api_keys = [k.strip() for k in (args.gemini_api_keys or "").split(",")
                        if k.strip()] or key_pool(_model)
    if args.top_k is not None:
        cfg["retrieval"]["top_k"] = int(args.top_k)
    if args.retrieval_mode is not None:
        cfg["retrieval"]["mode"] = args.retrieval_mode
    splits_cfg = cfg["eval"]["splits"]
    if args.n_indist is not None:
        splits_cfg["eval_in_distribution"] = args.n_indist
    if args.n_ood is not None:
        splits_cfg["eval_out_of_distribution"] = args.n_ood
    cfg_start_seed = (cfg.get("eval") or {}).get("start_seed")
    if args.start_seeds:
        start_seeds = [int(s) for s in args.start_seeds.split(",")]
    elif cfg_start_seed is not None:
        start_seeds = [int(cfg_start_seed)]   # eval.start_seed from the YAML
    else:
        start_seeds = [0, 1000, 2000]
    concurrency = (args.concurrency if args.concurrency is not None
                    else int(cfg["eval"]["concurrency"]))
    conditions = (args.conditions.split(",") if args.conditions
                   else list(cfg["conditions"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[reasoning_bank_eval]  config={args.config}")
    print(f"  model        = {cfg['model']['name']}")
    print(f"  conditions   = {conditions}")
    print(f"  splits       = {dict(splits_cfg)}")
    print(f"  rounds       = {len(start_seeds)}  start_seeds = {start_seeds}")
    print(f"  gemini_api_keys     = {len(gemini_api_keys)}  concurrency = {concurrency}")

    # CLI bank overrides (cond=path[,cond=path]) win over YAML conditions[].bank
    bank_overrides: dict[str, str] = {}
    if args.bank_overrides:
        for kv in args.bank_overrides.split(","):
            k, _, v = kv.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                bank_overrides[k] = v

    banks: dict[str, Bank | None] = {}
    for cond in conditions:
        # Lazy lookup: dict.get(cond, default) evaluates the default
        # EAGERLY, so cfg["conditions"][cond] would KeyError for a
        # condition supplied only via --conditions + --bank-overrides.
        if cond in bank_overrides:
            bp = bank_overrides[cond]
        else:
            bp = ((cfg.get("conditions") or {}).get(cond) or {}).get("bank")
        if bp:
            banks[cond] = Bank.load(bp)
            print(f"  {cond:8s} bank: {len(banks[cond])} items  ({bp})")
        else:
            banks[cond] = None
            print(f"  {cond:8s} bank: (none)")

    all_rows = []
    t0 = time.time()

    def _write_summary(rows: list) -> None:
        with (out_dir / "summary.json").open("w") as f:
            json.dump({
                "rows": rows,
                "config_path": str(args.config),
                "start_seeds": start_seeds,
                "concurrency": concurrency,
                "n_api_keys": len(gemini_api_keys),
            }, f, indent=2)

    for round_idx, start_seed in enumerate(start_seeds, 1):
        round_dir = out_dir / f"round{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=========================================================")
        print(f"=== Round {round_idx}/{len(start_seeds)}  "
              f"start_seed={start_seed}  "
              f"elapsed={(time.time()-t0)/60:.1f}m")
        print(f"=========================================================")
        round_rows = _run_one_round(
            cfg=cfg, banks=banks, splits_cfg=splits_cfg,
            start_seed=start_seed, round_dir=round_dir,
            concurrency=concurrency, gemini_api_keys=gemini_api_keys,
            round_idx=round_idx,
            # Refresh summary.json after every completed condition so a
            # mid-run crash preserves all finished cells.
            on_condition_done=lambda partial: _write_summary(all_rows + partial),
        )
        all_rows.extend(round_rows)
        _write_summary(all_rows)
        # Per-round mini-summary
        print(f"\n--- Round {round_idx} per-condition combined SR ---")
        by_cond_rd: dict = {}
        for r in round_rows:
            by_cond_rd.setdefault(r["condition"], {})[r["split"]] = r
        for cond, d in by_cond_rd.items():
            if {"eval_in_distribution", "eval_out_of_distribution"} <= set(d):
                nw = d["eval_in_distribution"]["n_won"] + d["eval_out_of_distribution"]["n_won"]
                nt = d["eval_in_distribution"]["n"] + d["eval_out_of_distribution"]["n"]
                print(f"  {cond:<8s} {nw}/{nt} = {nw/max(nt,1)*100:.1f}%")

    # Cross-round aggregation: mean ± std per (condition, split) + combined
    print("\n\n=========================================================")
    print("=== FINAL: 3-round mean ± std per (condition, split)")
    print("=========================================================")
    import statistics
    by_cell: dict = {}
    for r in all_rows:
        by_cell.setdefault((r["condition"], r["split"]), []).append(r)

    print(f"{'condition':<10} {'split':<28} {'rounds':<8} {'mean':>8} {'std':>8}")
    for (cond, split), runs in sorted(by_cell.items()):
        srs = [x["sr"] * 100 for x in runs]
        mean = statistics.mean(srs)
        std = statistics.stdev(srs) if len(srs) > 1 else 0.0
        print(f"{cond:<10} {split:<28} n={len(srs):<5} {mean:>7.1f}% {std:>7.1f}")

    print("\n--- combined view (in-dist + OOD weighted by 274) ---")
    # Per round: compute combined SR per condition; then mean ± std across rounds.
    by_cond_rounds: dict = {}
    for r in all_rows:
        by_cond_rounds.setdefault((r["condition"], r["round"]), {})[r["split"]] = r
    combined_by_cond: dict = {}
    for (cond, rd), d in by_cond_rounds.items():
        if {"eval_in_distribution", "eval_out_of_distribution"} <= set(d):
            nw = d["eval_in_distribution"]["n_won"] + d["eval_out_of_distribution"]["n_won"]
            nt = d["eval_in_distribution"]["n"] + d["eval_out_of_distribution"]["n"]
            combined_by_cond.setdefault(cond, []).append(nw / max(nt, 1) * 100)
    print(f"{'condition':<10} {'rounds':<8} {'combined mean':>15} {'std':>8}")
    for cond, vals in combined_by_cond.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{cond:<10} n={len(vals):<5} {mean:>14.1f}% {std:>7.1f}")
    print("=========================================================")
    print(f"total wall time: {(time.time()-t0)/60:.1f} min")

    _write_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
