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

"""Disk-backed cache for per-task unmutated baseline measurements.

Used by the Orchestrator (at the start of each task, to feed the
HarnessAgent a baseline summary) and by post-hoc checkpoint-evaluation
tools (to avoid re-rolling baselines that another run has already paid for).

Cache file layout (one JSON per cache hit):

    runs/_baseline_cache/baseline_<16hex>.json
      {
        "key":     "<16hex>",
        "params":  {bridge, reset_options, task_id, n, max_steps, policy_*},
        "result":  {sr, n_won, n, avg_steps, rollouts: [{success, steps, error}, ...]}
      }

The orchestrator stores the same `result` shape that checkpoint evaluation
writes -- so cache entries are bidirectional: any new orchestrator
run is reusable by checkpoint evaluation, and vice versa, as long as the
hashed params match.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(env_import_path: str, reset_options: dict | None,
                task_id: int, n: int, max_steps: int,
                policy_model: str | None, policy_action_format: str,
                policy_temperature: float, policy_max_history: int,
                policy_task_prompt: str | None = None) -> str:
    """Deterministic 16-hex hash of every parameter that affects the
    baseline measurement. Any change invalidates the cache.

    policy_task_prompt is hashed because the policy's system prompt changes
    the measurement (e.g. RB memory injection rewrites it per condition) --
    without it, a round-2 injected run hits round-1's cache and reads
    pre-injection baselines. Only a sha digest goes into the key to keep it
    prompt-size independent."""
    canon = json.dumps({
        "bridge": env_import_path,
        "reset_options": reset_options or {},
        "task_id": int(task_id),
        "n": int(n),
        "max_steps": int(max_steps),
        "policy_model": policy_model,
        "policy_action_format": policy_action_format,
        "policy_temperature": float(policy_temperature),
        "policy_max_history": int(policy_max_history),
        "policy_task_prompt_sha": (
            hashlib.sha256(policy_task_prompt.encode()).hexdigest()[:16]
            if policy_task_prompt else None),
    }, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def policy_model_id(client_kwargs: dict | None) -> str | None:
    """Effective policy model name for the cache key, seeing through the
    LoggingLLMClient wrap (run_harness's log_policy_calls nests the real
    client kwargs under `inner_kwargs`). Without this, every logged run
    hashed policy_model=None, so runs with DIFFERENT policy models but
    otherwise-equal params silently shared baseline cache entries."""
    kw = client_kwargs or {}
    inner = kw.get("inner_kwargs") or {}
    return kw.get("model") or inner.get("model")


def load(cache_dir: str | Path, key: str) -> dict | None:
    p = Path(cache_dir) / f"baseline_{key}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def save(cache_dir: str | Path, key: str, params: dict[str, Any],
          result: dict[str, Any]) -> None:
    p = Path(cache_dir) / f"baseline_{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"key": key, "params": params, "result": result}, f, indent=2)
