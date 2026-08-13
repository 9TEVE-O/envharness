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

"""Stage 3: size-matched per-task subsets.

For each unique `source.task_id` in a full bank, sample 1 item. Restrict
to the intersection of task_ids present in BOTH banks so the two output
subsets are size-matched and share the same source-task set.

Outputs:
  orig_subset.jsonl              (one item per shared task)
  ours_subset_matched.jsonl       (one item per shared task)

Run::

  <your-env>/bin/python scripts/subset.py \\
      --orig runs/_corpus_release/banks/orig_full.jsonl \\
      --ours runs/_corpus_release/banks/ours_full.jsonl \\
      --out-dir runs/_corpus_release/banks
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envharness.reasoning_bank import Bank


def _by_task(bank: Bank) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for it in bank.items:
        tid = (it.source or {}).get("task_id") or ""
        if tid:
            out[tid].append(it)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True, type=Path)
    p.add_argument("--ours", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    orig = Bank.load(args.orig)
    ours = Bank.load(args.ours)
    print(f"orig items: {len(orig)}   ours items: {len(ours)}")

    orig_by = _by_task(orig)
    ours_by = _by_task(ours)
    print(f"orig source tasks: {len(orig_by)}   ours source tasks: {len(ours_by)}")

    # Restrict to tasks present in BOTH for size matching.
    shared = sorted(set(orig_by) & set(ours_by))
    print(f"shared tasks (size-matched): {len(shared)}")

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    orig_sub = Bank()
    ours_sub = Bank()
    for tid in shared:
        orig_sub.add([rng.choice(orig_by[tid])])
        ours_sub.add([rng.choice(ours_by[tid])])

    orig_out = args.out_dir / "orig_subset.jsonl"
    ours_out = args.out_dir / "ours_subset_matched.jsonl"
    orig_sub.save(orig_out)
    ours_sub.save(ours_out)
    print(f"  -> {orig_out}  ({len(orig_sub)} items)")
    print(f"  -> {ours_out}   ({len(ours_sub)} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
