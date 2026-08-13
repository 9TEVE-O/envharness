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

# toy24 reduced-scale reproduce: the three modes of experiments/toy24/README.md.
#
#   source ~/.config/envharness/gemini_keys.env
#   bash experiments/toy24/reproduce_smoke.sh
#
# PY  override the interpreter (default: the eh-toy24 env)
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/eh-toy24/bin/python}

# Provider is one switch. The other benchmarks route MODEL through their
# reproduce.py; toy24 drives run_harness.py directly, so it exports EH_MODEL
# itself. envharness.infra.model honours it over whatever a config names.
export MODEL=${MODEL:-openai/gpt-4.1-mini}
export EH_MODEL=$MODEL

# $$ as well as the clock: two smokes started in the same second would
# otherwise share a run directory and overwrite each other's traces.
TS=$(date +%m%d_%H%M%S)_$$

echo "=== toy24 preflight ==="
$PY scripts/check_env.py toy24 || exit 1

echo
echo "=== mode 1/3: scripted baseline (no API key) ==="
$PY scripts/run_harness.py --scripted --in-process --run-name "toy24-smoke-baseline-$TS" || exit 1

echo
echo "=== mode 2/3: original env (NoopMutator + Gemini policy) ==="
$PY scripts/run_harness.py --config experiments/toy24/original.yaml \
    --run-name "toy24-smoke-original-$TS" --n-tasks 2 || exit 1

echo
echo "=== mode 3/3: mutated env (LLM harness agent + DifficultyZone) ==="
$PY scripts/run_harness.py --config experiments/toy24/mutated_smoke.yaml \
    --run-name "toy24-smoke-mutated-$TS" || exit 1

echo
echo "=== toy24 smoke summary ==="
for m in baseline original mutated; do
  f="runs/toy24-smoke-$m-$TS/traces.jsonl"
  [ -f "$f" ] && $PY - "$f" "$m" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for kind in ("baseline", "exploration", "accepted"):
    sub = [r for r in rows if r["kind"] == kind]
    if sub:
        sr = sum(r["success"] for r in sub) / len(sub)
        print(f"  {sys.argv[2]:9s} {kind:12s} n={len(sub):3d}  SR={sr:.2f}")
EOF
done
