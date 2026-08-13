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

"""Single-unit worker for the SpreadsheetBench dispatcher.

One invocation does ONE unit of work and appends its result under an flock, so
many workers can share one output file (the webarena dispatcher pattern). Three
modes:

  --mode corpus  --task-id N --run-root R --config C
       Run one task's corpus generation (run_harness with n-tasks=1,
       task-offset=N) into runs/<R>/corpus_task<N>/. Appends a done marker.

  --mode eval    --task-id N --condition COND --bank PATH|none --config C --out O
       Run one held-out eval episode (reuses reasoning_bank_eval.run_episode) and append
       {task_id, condition, success, ...} to O.

  --mode summary --out-dir D --conditions a,b,c
       Aggregate the per-condition <cond>.jsonl files in D into summary.json
       (base/orig/ours SR + deltas).

The dispatcher (dispatcher.sh) owns the queue, worker pool, key round-robin and
resumability; this script is intentionally a single unit of work.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os

from envharness.infra.model import key_pool
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # repo root
sys.path.insert(0, str(ROOT))


def _append_locked(out_path: str, rec: dict) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def do_corpus(args) -> int:
    run_name = f"{args.run_root}/corpus_task{args.task_id}"
    cmd = [sys.executable, "scripts/run_harness.py",
           "--config", args.config, "--run-name", run_name,
           "--n-tasks", "1", "--task-offset", str(args.task_id)]
    r = subprocess.run(cmd, cwd=str(ROOT))
    _append_locked(args.out, {"task_id": args.task_id, "rc": r.returncode,
                              "run_name": run_name})
    print(json.dumps({"task_id": args.task_id, "rc": r.returncode}))
    return 0 if r.returncode == 0 else 1


def do_eval(args) -> int:
    import yaml
    from experiments.spreadsheetbench.reasoning_bank_eval import run_episode
    from envharness.reasoning_bank import Bank
    cfg = yaml.safe_load(Path(args.config).read_text())
    # The dispatcher rotates a Gemini pool through the environment; on any
    # other provider there is no pool and the key comes from that provider's
    # own variable.
    key = (key_pool(cfg["model"]["name"]) or [None])[0]
    # dir  = Trace2Skill on-disk skill folder (SKILL.md + references/, read on
    #        demand); .md = consolidated skill library (whole-doc injection);
    #        .jsonl = per-task bank (top-k retrieval); "none" = no-bank base.
    bank = None
    skill_doc = None
    skill_dir = None
    skill_chat = None
    if args.bank and args.bank != "none":
        if Path(args.bank).is_dir():
            skill_dir = args.bank
        elif args.bank.endswith(".chat.md"):
            # SkillOpt-faithful direct-chat: bare "## Skill" in system prompt.
            skill_chat = Path(args.bank).read_text()
        elif args.bank.endswith(".md"):
            skill_doc = Path(args.bank).read_text()
        else:
            bank = Bank.load(args.bank)
    rec = run_episode(cfg=cfg, bank=bank, seed=args.task_id, gemini_api_key=key,
                      skill_doc=skill_doc, skill_dir=skill_dir,
                      skill_chat=skill_chat)
    rec["condition"] = args.condition
    rec["task_idx"] = args.task_id
    _append_locked(args.out, rec)
    print(json.dumps({"task_id": rec.get("task_id"), "idx": args.task_id,
                      "condition": args.condition, "success": rec["success"],
                      "error": rec.get("error", "")[:80]}))
    return 0


def do_summary(args) -> int:
    out_dir = Path(args.out_dir)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    rows = []
    for cond in conditions:
        f = out_dir / f"{cond}.jsonl"
        n = won = 0
        if f.exists():
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                won += int(bool(r.get("success")))
        rows.append({"condition": cond, "n_won": won, "n": n,
                     "sr": won / max(n, 1)})
    by = {r["condition"]: r["sr"] for r in rows}
    print("\n=== SUMMARY (held-out SR) ===")
    # map to base/original/mutated labels
    label = {"nobank": "base", "orig": "original", "ours": "mutated"}
    for r in rows:
        print(f"  {label.get(r['condition'], r['condition']):9s} "
              f"({r['condition']:6s}) {r['n_won']}/{r['n']} = {r['sr']*100:.1f}%")
    if "ours" in by and "orig" in by:
        print(f"  delta(mutated-original) = {(by['ours']-by['orig'])*100:+.1f} pp")
    if "ours" in by and "nobank" in by:
        print(f"  delta(mutated-base)     = {(by['ours']-by['nobank'])*100:+.1f} pp")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"  -> {out_dir/'summary.json'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["corpus", "eval", "summary"])
    p.add_argument("--task-id", type=int)
    p.add_argument("--condition", default="")
    p.add_argument("--bank", default="none")
    p.add_argument("--config", default="")
    p.add_argument("--run-root", default="")
    p.add_argument("--out", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--conditions", default="nobank,orig,ours")
    args = p.parse_args(argv)
    if args.mode == "corpus":
        return do_corpus(args)
    if args.mode == "eval":
        return do_eval(args)
    return do_summary(args)


if __name__ == "__main__":
    sys.exit(main())
