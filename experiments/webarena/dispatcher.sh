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

# Dynamic task dispatcher for WebArena eval.
# 3 containers per site, tasks dispatched from a shared queue.
#
# Usage:
#   ./dispatcher.sh <site> <bank_or_none> <out_jsonl> <tasks_json> <top_k>
set -euo pipefail

SITE="$1"
BANK="$2"        # path to bank JSONL, or "none"
OUT="$3"
TASKS="$4"       # JSON file with list of task IDs
TOPK="${5:-3}"

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${WEBARENA_PYTHON:-$(command -v python3 || command -v python)}
WORKER="$ROOT/experiments/webarena/worker.py"

# API keys (round-robin)
KEYS=(${GEMINI_API_KEYS:-$GEMINI_API_KEY})
IFS=',' read -ra KEYS <<< "${KEYS[0]}"
NUM_KEYS=${#KEYS[@]}

# Container pools
declare -A CONTAINERS URLS
case "$SITE" in
  reddit)
    CONTAINERS=([0]=forum [1]=forum_1 [2]=forum_2)
    URLS=([0]="http://127.0.0.1:19999" [1]="http://127.0.0.1:19998" [2]="http://127.0.0.1:19997") ;;
  shopping)
    CONTAINERS=([0]=shopping [1]=shopping_1 [2]=shopping_2)
    URLS=([0]="http://127.0.0.1:17770" [1]="http://127.0.0.1:17769" [2]="http://127.0.0.1:17768") ;;
  shopping_admin)
    CONTAINERS=([0]=shopping_admin [1]=shopping_admin_1 [2]=shopping_admin_2)
    URLS=([0]="http://127.0.0.1:17780" [1]="http://127.0.0.1:17779" [2]="http://127.0.0.1:17778") ;;
  gitlab)
    CONTAINERS=([0]=gitlab [1]=gitlab_1 [2]=gitlab_2)
    URLS=([0]="http://127.0.0.1:18023" [1]="http://127.0.0.1:18022" [2]="http://127.0.0.1:18021") ;;
  *) echo "Unknown site: $SITE"; exit 1 ;;
esac
NUM_WORKERS=${N_WORKERS_PER_SITE:-${#CONTAINERS[@]}}

# Build queue (skip already done)
QUEUE_DIR=$(mktemp -d /tmp/wa_dispatch_XXXX)
QUEUE_FILE="$QUEUE_DIR/queue.txt"
LOCK_FILE="$QUEUE_DIR/queue.lock"

$PY -c "
import json, sys, os
ids = json.load(open('$TASKS'))
if isinstance(ids, dict): ids = list(ids.values())[0]
done = set()
if os.path.exists('$OUT'):
    for line in open('$OUT'):
        if line.strip():
            done.add(json.loads(line)['task_id'])
remaining = [str(t) for t in ids if int(t) not in done]
print(f'[dispatcher] {len(done)} done, {len(remaining)} remaining', file=sys.stderr)
print('\n'.join(remaining))
" > "$QUEUE_FILE" 2>&2

TOTAL=$(wc -l < "$QUEUE_FILE")
echo "[dispatcher] $SITE: $TOTAL tasks, $NUM_WORKERS workers, top_k=$TOPK"

grab_task() {
  ( flock -x 200; head -1 "$QUEUE_FILE" 2>/dev/null; sed -i '1d' "$QUEUE_FILE" 2>/dev/null ) 200>"$LOCK_FILE"
}

worker_loop() {
  local wid=$1 container=${CONTAINERS[$wid]} url=${URLS[$wid]}
  while true; do
    local tid=$(grab_task)
    [ -z "$tid" ] && return 0
    local kidx=$(( (wid + $(echo "$tid" | cksum | cut -d' ' -f1)) % NUM_KEYS ))
    export GEMINI_API_KEY="${KEYS[$kidx]}"

    local bank_args=()
    [ "$BANK" != "none" ] && bank_args=(--bank "$BANK" --top-k "$TOPK")

    # Run task; if it fails with infra error, retry once. Wrap in a
    # per-task wall-clock timeout (default 600s) so a hung browser episode
    # gets SIGKILLed and the worker moves on instead of occupying its
    # container forever (a single stuck task otherwise blocks the whole
    # site's progress). Overridable via WORKER_TIMEOUT.
    local result
    result=$(timeout -s KILL "${WORKER_TIMEOUT:-600}" $PY "$WORKER" \
      --task-id "$tid" --container "$container" --container-url "$url" \
      "${bank_args[@]}" --out "$OUT" 2>&1 | tail -1) || true

    # Check if this task got an infra error; if so remove and retry
    if python3 -c "
import json
for line in open('$OUT'):
    r = json.loads(line)
    if r['task_id'] == $tid:
        e = r.get('error','')
        if 'CONNECTION' in e or 'Locator.fill' in e or 'Page.goto' in e:
            exit(0)
exit(1)
" 2>/dev/null; then
      echo "[retry] task $tid had infra error, retrying..."
      # Remove the error record. Hold the same flock worker.py takes on
      # $OUT for its appends: this is a read-truncate-rewrite of the
      # shared results file, and without the lock any record appended in
      # between would be lost.
      ( flock -x 201
        python3 -c "
import json
lines = [l for l in open('$OUT') if json.loads(l)['task_id'] != $tid]
open('$OUT','w').writelines(lines)
" 2>/dev/null
      ) 201>>"$OUT"
      sleep 5
      $PY "$WORKER" \
        --task-id "$tid" --container "$container" --container-url "$url" \
        "${bank_args[@]}" --out "$OUT" 2>&1 | tail -1 || true
    fi

    # If this attempt produced NO record (worker SIGKILLed by the timeout,
    # or crashed before writing), bank a synthetic failure so the task is
    # counted as one attempt and not retried forever.
    ( flock -x 201
      python3 -c "
import json, sys
tid = $tid; out = '$OUT'
try:
    have = any(json.loads(l).get('task_id') == tid
               for l in open(out) if l.strip())
except FileNotFoundError:
    have = False
if not have:
    with open(out, 'a') as f:
        f.write(json.dumps({'task_id': tid, 'success': False,
                            'error': 'timeout_or_crash'}) + '\n')
" 2>/dev/null
    ) 201>>"$OUT"

    local done_count=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    # grep -c prints "0" AND exits 1 on no match; `|| echo 0` would emit "0\n0".
    local won_count=$(grep -c '"success": true' "$OUT" 2>/dev/null || true)
    won_count=${won_count:-0}
    echo "[progress] $done_count done, SR=$won_count/$done_count"
  done
}

mkdir -p "$(dirname "$OUT")"
touch "$OUT"
PIDS=()
for wid in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_loop "$wid" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

FINAL_DONE=$(wc -l < "$OUT")
# grep -c prints "0" AND exits 1 on no match; `|| echo 0` would emit "0\n0".
FINAL_WON=$(grep -c '"success": true' "$OUT" || true)
FINAL_WON=${FINAL_WON:-0}
echo "[dispatcher] DONE $SITE: $FINAL_WON/$FINAL_DONE = $(python3 -c "print(f'{$FINAL_WON/max($FINAL_DONE,1)*100:.1f}%')")"
rm -rf "$QUEUE_DIR"
