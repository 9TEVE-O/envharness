#!/bin/bash
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

# ALFWorld reduced-scale reproduce: the same 4 stages as reproduce.py
# (corpus -> induce -> subset -> eval).
#
#   source ~/.config/envharness/gemini_keys.env
#   export ALFWORLD_DATA=~/eh_from_zero/alfworld_data   # wherever alfworld-download put it
#   bash experiments/alfworld/reproduce_smoke.sh
#
# Every knob below is an env var you can override.
set -u
cd "$(dirname "$0")/../.."
export PY=${PY:-$HOME/miniconda3/envs/eh-alfworld/bin/python}

# Keys are checked by reproduce.py against whatever MODEL names, so a GPT-only
# or Vertex-only setup is not blocked here for lacking a Gemini key.
: "${GEMINI_API_KEYS:=${GEMINI_API_KEY:-}}"
[ -n "$GEMINI_API_KEYS" ] && export GEMINI_API_KEYS \
    GEMINI_API_KEY="${GEMINI_API_KEY:-${GEMINI_API_KEYS%%,*}}"

export N_TASKS_TOTAL=${N_TASKS_TOTAL:-4}
export N_SHARDS=${N_SHARDS:-1}
export TASK_BASE_OFFSET=${TASK_BASE_OFFSET:-0}
export EVAL_START_SEEDS=${EVAL_START_SEEDS:-0}
export EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-8}
export CORPUS_YAML=${CORPUS_YAML:-experiments/alfworld/corpus_smoke.yaml}
export EVAL_YAML=${EVAL_YAML:-experiments/alfworld/reasoning_bank_eval_smoke.yaml}
export EVAL_N_INDIST=${EVAL_N_INDIST:-4}
export EVAL_N_OOD=${EVAL_N_OOD:-4}
# Provider is one switch: MODEL reaches every stage as EH_MODEL, which
# envharness.infra.model honours over whatever a config names.
export MODEL=${MODEL:-openai/gpt-4.1-mini}
export ROOT_RUN=${ROOT_RUN:-runs/alfworld_smoke_$(date +%m%d_%H%M%S)_$$}

echo "=== alfworld preflight ==="
$PY scripts/check_env.py alfworld || exit 1

echo "=== alfworld smoke: $ROOT_RUN ==="
$PY experiments/alfworld/reproduce.py || exit 1

echo
echo "=== 3-color table ==="
$PY - "$ROOT_RUN/eval/summary.json" <<'EOF'
import json, sys, collections
rows = json.load(open(sys.argv[1]))["rows"]
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    a = agg[(r["condition"], r["split"])]
    a[0] += r["n_won"]; a[1] += r["n"]
print(f"{'condition':10s} {'split':26s} {'SR':>8s}")
for (cond, split), (won, n) in sorted(agg.items()):
    print(f"{cond:10s} {split:26s} {won:3d}/{n:<3d} {won/n*100 if n else 0:5.1f}%")
EOF
