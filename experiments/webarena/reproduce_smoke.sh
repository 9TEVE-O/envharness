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

# WebArena reduced-scale reproduce: the same 3 stages as reproduce.py
# (corpus -> banks -> nobank/orig/ours eval x 4 sites).
#
#   source ~/.config/envharness/gemini_keys.env
#   bash experiments/webarena/reproduce_smoke.sh
#
# Requires the 12-container WebArena docker stack to be up
# (IMAGE_DIR=~/webarena_images bash experiments/webarena/setup_stack.sh).
#
# Every knob below is an env var you can override.
set -u
cd "$(dirname "$0")/../.."
export WEBARENA_PYTHON=${WEBARENA_PYTHON:-$HOME/miniconda3/envs/eh-webarena/bin/python}

# Keys are checked by reproduce.py against whatever MODEL names, so a GPT-only
# or Vertex-only setup is not blocked here for lacking a Gemini key.
: "${GEMINI_API_KEYS:=${GEMINI_API_KEY:-}}"
[ -n "$GEMINI_API_KEYS" ] && export GEMINI_API_KEYS \
    GEMINI_API_KEY="${GEMINI_API_KEY:-${GEMINI_API_KEYS%%,*}}"

export N_CORPUS_PER_SITE=${N_CORPUS_PER_SITE:-1}
export N_CORPUS_WORKERS=${N_CORPUS_WORKERS:-4}
export N_EVAL_PER_SITE=${N_EVAL_PER_SITE:-1}
export CORPUS_YAML=${CORPUS_YAML:-$PWD/experiments/webarena/corpus_smoke.yaml}
# Provider is one switch: MODEL reaches every stage as EH_MODEL, which
# envharness.infra.model honours over whatever a config names.
export MODEL=${MODEL:-openai/gpt-4.1-mini}
export ROOT_RUN=${ROOT_RUN:-$PWD/experiments/webarena/runs/smoke_$(date +%m%d_%H%M%S)_$$}

echo "=== webarena preflight (import chain + 12 stack ports) ==="
$WEBARENA_PYTHON scripts/check_env.py webarena || exit 1

echo "=== webarena smoke: $ROOT_RUN ==="
$WEBARENA_PYTHON experiments/webarena/reproduce.py
