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

# SWE-bench reduced-scale reproduce: the same 3 stages as reproduce.py
# (corpus -> banks -> 3-condition eval).
#
#   source ~/.config/envharness/gemini_keys.env
#   bash experiments/swebench/reproduce_smoke.sh
#
# Every knob below is an env var you can override.
set -u
cd "$(dirname "$0")/../.."
export PY=${PY:-$HOME/miniconda3/envs/eh-swebench/bin/python}

# Keys are checked by reproduce.py against whatever MODEL names, so a GPT-only
# or Vertex-only setup is not blocked here for lacking a Gemini key.
: "${GEMINI_API_KEYS:=${GEMINI_API_KEY:-}}"
[ -n "$GEMINI_API_KEYS" ] && export GEMINI_API_KEYS \
    GEMINI_API_KEY="${GEMINI_API_KEY:-${GEMINI_API_KEYS%%,*}}"

# Bank distillation needs tasks that produced BOTH successes and failures
# to pair. At four tasks a run can easily get none, and the driver then
# stops on an empty bank -- a scale artifact, not a failure of the code.
export N_TASKS=${N_TASKS:-12}
export CORPUS_YAML=${CORPUS_YAML:-experiments/swebench/corpus_smoke.yaml}
export CORPUS_CONCURRENCY=${CORPUS_CONCURRENCY:-4}
export CORPUS_RETRIES=${CORPUS_RETRIES:-0}
export EVAL_N=${EVAL_N:-4}
export EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
export EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-250}
export MODEL=${MODEL:-openai/gpt-4.1-mini}
export ROOT_RUN=${ROOT_RUN:-runs/swebench_smoke_$(date +%m%d_%H%M%S)_$$}

echo "=== swebench preflight ==="
$PY scripts/check_env.py swebench || exit 1

echo "=== swebench smoke: $ROOT_RUN ==="
$PY experiments/swebench/reproduce.py
rc=$?
echo "reproduce.py rc=$rc"
[ -f "$ROOT_RUN/summary.json" ] && { echo; echo "=== 3-color table ==="; cat "$ROOT_RUN/summary.json"; }
exit $rc
