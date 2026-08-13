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

#!/usr/bin/env python
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

"""Build a envharness Rules corpus (+ matching train subset) from a
legacy MutationLayer corpus JSONL.

Default behaviour produces the small bundled example data:
  - experiments/alfworld/data/example_corpus.jsonl  (N mutated records)
  - experiments/alfworld/data/train_subset.jsonl     (the same game_files)

The train subset is what `ALFWORLD_TRAIN_SUBSET_PATH` points at: it restricts
the (~3553-task) TRAIN split to exactly the games we have mutations for, so a
smoke run lands on a mutated game every reset.

Usage:
  PYTHONPATH=envharness:rl \
    python rl/scripts/build_corpus.py \
      --src runs/setting_a_v1_corpus_indexed.jsonl --limit 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from envharness_rl.corpus import convert_file


def main() -> None:
    here = Path(__file__).resolve().parents[1]  # rl/
    data_dir = here / "experiments" / "alfworld" / "data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Legacy MutationLayer corpus JSONL "
                         "(e.g. runs/setting_a_v1_corpus_indexed.jsonl)")
    ap.add_argument("--corpus-out", default=str(data_dir / "example_corpus.jsonl"))
    ap.add_argument("--subset-out", default=str(data_dir / "train_subset.jsonl"))
    ap.add_argument("--limit", type=int, default=5,
                    help="Max records to keep (None for all).")
    ap.add_argument("--all", action="store_true",
                    help="Keep every record (overrides --limit).")
    ap.add_argument("--no-require-mutation", action="store_true",
                    help="Keep pass-through records too (default drops them).")
    args = ap.parse_args()

    limit = None if args.all else args.limit
    res = convert_file(
        args.src, args.corpus_out, limit=limit,
        require_mutation=not args.no_require_mutation, validate=True)

    # Train subset = the game_files we kept (one {"game_file": ...} per line).
    game_files = [json.loads(l)["game_file"]
                  for l in Path(args.corpus_out).open() if l.strip()]
    subset_path = Path(args.subset_out)
    with subset_path.open("w") as f:
        for gf in game_files:
            f.write(json.dumps({"game_file": gf}) + "\n")

    print(f"corpus -> {res['dst']}  ({res['written']} records)")
    print(f"subset -> {subset_path}  ({len(game_files)} game_files)")
    if res["errors"]:
        print(f"dropped {len(res['errors'])} records (code failed to load):")
        for gf, why in res["errors"][:10]:
            print(f"  {gf}: {why}")


if __name__ == "__main__":
    main()
