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

# Shared-queue work-stealing dispatcher for OfficeQA (no containers).
# Mirrors experiments/spreadsheetbench/dispatcher.sh: a queue of task ids, N
# workers that flock-pop ids, run a single-unit worker.py with a round-robin API
# key, and append to a shared output file. Resumable: ids already present in the
# output (corpus done-marker / eval result) are skipped.
#
# Usage:
#   # corpus: one run_harness per task, parallel across the key pool
#   GEMINI_API_KEYS=k1,k2,... N_WORKERS=6 ./dispatcher.sh corpus \
#       <config.yaml> <run_root> <ids_csv_or_range> <done_out.jsonl>
#
#   # eval: one episode per task for ONE condition (bank fixed)
#   GEMINI_API_KEYS=k1,k2,... N_WORKERS=18 ./dispatcher.sh eval \
#       <config.yaml> <condition> <bank|none> <ids_csv_or_range> <out.jsonl>
set -uo pipefail

MODE="$1"; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${SB_PYTHON:-$(command -v python3 || command -v python)}"
WORKER="$ROOT/experiments/officeqa/worker.py"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HOME="${HOME:-/home/$USER}"

# --- API keys (comma-separated) ---
# Optional: only Gemini meters per key, so this pool exists to spread that
# quota across workers. Other providers authenticate from their own env var,
# which the worker resolves in Python (envharness.infra.model).
IFS=',' read -ra KEYS <<< "${GEMINI_API_KEYS:-${GEMINI_API_KEY:-}}"
NUM_KEYS=${#KEYS[@]}

# --- expand an ids spec: "a,b,c" or "START:COUNT" -> newline list ---
expand_ids() {
  local spec="$1"
  if [[ "$spec" == *:* ]]; then
    local start="${spec%%:*}" count="${spec##*:}"
    seq "$start" "$((start + count - 1))"
  else
    echo "$spec" | tr ',' '\n'
  fi
}

QUEUE_DIR="$(mktemp -d /tmp/oqa_dispatch_XXXX)"
QUEUE_FILE="$QUEUE_DIR/queue.txt"
LOCK_FILE="$QUEUE_DIR/queue.lock"
trap 'rm -rf "$QUEUE_DIR"' EXIT

grab_task() { ( flock -x 200; head -1 "$QUEUE_FILE" 2>/dev/null; sed -i '1d' "$QUEUE_FILE" 2>/dev/null ) 200>"$LOCK_FILE"; }

# --- build queue, skipping ids already done in OUT ---
build_queue() {  # $1=ids_spec  $2=out_file
  local spec="$1" out="$2"
  expand_ids "$spec" | "$PY" -c "
import sys, json, os
ids=[x.strip() for x in sys.stdin if x.strip()]
done=set()
out='$out'
if os.path.exists(out):
    for line in open(out):
        line=line.strip()
        if not line: continue
        try:
            d=json.loads(line)
            v=d.get('task_idx')              # eval records: the seed (queue id)
            if v is None: v=d.get('task_id') # corpus records: the int task index
            done.add(int(v))
        except Exception: pass
rem=[i for i in ids if int(i) not in done]
sys.stderr.write(f'[dispatcher] {len(done)} done, {len(rem)} remaining\n')
print('\n'.join(rem))
" > "$QUEUE_FILE" 2>&2
}

case "$MODE" in
  corpus)
    CONFIG="$1"; RUN_ROOT="$2"; IDS="$3"; OUT="$4"
    N_WORKERS="${N_WORKERS:-$NUM_KEYS}"
    build_queue "$IDS" "$OUT"
    TOTAL=$(wc -l < "$QUEUE_FILE")
    echo "[dispatcher] corpus: $TOTAL tasks, $N_WORKERS workers, $NUM_KEYS keys"
    worker_loop() {
      local wid=$1
      while true; do
        local tid; tid=$(grab_task); [ -z "$tid" ] && return 0
        [ -n "${KEYS[0]:-}" ] && \
          export GEMINI_API_KEY="${KEYS[$(( (wid + tid) % NUM_KEYS ))]}"
        "$PY" "$WORKER" --mode corpus --task-id "$tid" \
          --run-root "$RUN_ROOT" --config "$CONFIG" --out "$OUT" >/dev/null 2>&1 || true
        echo "[corpus] task $tid done ($(wc -l < "$OUT" 2>/dev/null || echo 0)/$TOTAL)"
      done
    }
    ;;
  eval)
    CONFIG="$1"; COND="$2"; BANK="$3"; IDS="$4"; OUT="$5"
    N_WORKERS="${N_WORKERS:-$((NUM_KEYS*3))}"
    build_queue "$IDS" "$OUT"
    TOTAL=$(wc -l < "$QUEUE_FILE")
    echo "[dispatcher] eval[$COND]: $TOTAL tasks, $N_WORKERS workers, bank=$BANK"
    worker_loop() {
      local wid=$1
      while true; do
        local tid; tid=$(grab_task); [ -z "$tid" ] && return 0
        [ -n "${KEYS[0]:-}" ] && \
          export GEMINI_API_KEY="${KEYS[$(( (wid + tid) % NUM_KEYS ))]}"
        # per-task timeout: a hung episode (litellm retry storm) must not block
        # the whole dispatcher's wait forever.
        timeout "${EVAL_TIMEOUT:-420}" "$PY" "$WORKER" --mode eval \
          --task-id "$tid" --condition "$COND" \
          --bank "$BANK" --config "$CONFIG" --out "$OUT" 2>/dev/null || true
        local done_n won_n
        done_n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
        won_n=$(grep -c '"success": true' "$OUT" 2>/dev/null || echo 0)
        echo "[eval/$COND] $done_n/$TOTAL  SR=$won_n/$done_n"
      done
    }
    ;;
  *) echo "unknown mode: $MODE (corpus|eval)"; exit 1 ;;
esac

mkdir -p "$(dirname "$OUT")"; touch "$OUT"
PIDS=()
for wid in $(seq 0 $((N_WORKERS - 1))); do worker_loop "$wid" & PIDS+=($!); done
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
echo "[dispatcher] $MODE DONE -> $OUT"
