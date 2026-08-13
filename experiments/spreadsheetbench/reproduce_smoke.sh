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

# SpreadsheetBench reduced-scale reproduce: the same 5 stages as reproduce.py
# (corpus -> induce -> subset -> eval -> report).
#
#   source ~/.config/envharness/gemini_keys.env
#   bash experiments/spreadsheetbench/reproduce_smoke.sh
#
# Every knob below is an env var you can override.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/eh-sb/bin/python}

# Keys are checked by reproduce.py against whatever MODEL names, so a GPT-only
# or Vertex-only setup is not blocked here for lacking a Gemini key.
: "${GEMINI_API_KEYS:=${GEMINI_API_KEY:-}}"
[ -n "$GEMINI_API_KEYS" ] && export GEMINI_API_KEYS \
    GEMINI_API_KEY="${GEMINI_API_KEY:-${GEMINI_API_KEYS%%,*}}"

export N_TRAIN=${N_TRAIN:-6}
export N_HELD_BASES=${N_HELD_BASES:-2}
# Concurrency is bounded by your provider's rate limit, not by CPU: six
# workers on long-context tasks saturate a default OpenAI 200k TPM tier
# and the 429s truncate episodes. Raise it if your quota allows.
export CORPUS_WORKERS=${CORPUS_WORKERS:-3}
export EVAL_WORKERS=${EVAL_WORKERS:-3}
export CORPUS_YAML=${CORPUS_YAML:-experiments/spreadsheetbench/corpus_smoke.yaml}
export EVAL_YAML=${EVAL_YAML:-experiments/spreadsheetbench/reasoning_bank_eval_smoke.yaml}
# Provider is one switch: MODEL reaches every stage as EH_MODEL, which
# envharness.infra.model honours over whatever a config names.
export MODEL=${MODEL:-openai/gpt-4.1-mini}
export ROOT_RUN=${ROOT_RUN:-runs/sb_smoke_$(date +%m%d_%H%M%S)_$$}

echo "=== spreadsheetbench preflight ==="
$PY scripts/check_env.py spreadsheetbench || exit 1

echo "=== spreadsheetbench smoke: $ROOT_RUN ==="
$PY experiments/spreadsheetbench/reproduce.py
