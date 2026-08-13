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

"""Tests for the disk-backed baseline cache.

Used by the Orchestrator at the start of each task to skip recomputing
per-task baseline rollouts. The cache key MUST be deterministic across
processes (it ships through env-var-passed module strings), so the test
fixes the inputs and checks the key shape + the load/save round-trip.
"""
from envharness.orchestration import baseline_cache as bc


COMMON_PARAMS = dict(
    env_import_path="envharness.bridges.toy24:Toy24Bridge",
    reset_options={"target": 24},
    task_id=7,
    n=5,
    max_steps=12,
    policy_model="openai/gpt-4.1-mini",
    policy_action_format="function_calling",
    policy_temperature=0.4,
    policy_max_history=50,
)


def test_cache_key_is_deterministic():
    k1 = bc.cache_key(**COMMON_PARAMS)
    k2 = bc.cache_key(**COMMON_PARAMS)
    assert k1 == k2
    # 16-hex-char prefix of sha256
    assert len(k1) == 16
    assert all(c in "0123456789abcdef" for c in k1)


def test_cache_key_differs_when_any_param_changes():
    base = bc.cache_key(**COMMON_PARAMS)
    altered = {**COMMON_PARAMS, "task_id": 8}
    assert bc.cache_key(**altered) != base
    altered = {**COMMON_PARAMS, "policy_temperature": 0.7}
    assert bc.cache_key(**altered) != base
    altered = {**COMMON_PARAMS, "reset_options": {"target": 36}}
    assert bc.cache_key(**altered) != base


def test_load_missing_returns_none(tmp_path):
    assert bc.load(tmp_path, "nonexistent") is None


def test_save_then_load_round_trips(tmp_path):
    key = bc.cache_key(**COMMON_PARAMS)
    params = {"bridge": "toy24", "n": 5}
    result = {"sr": 0.6, "n_won": 3, "n": 5,
              "avg_steps": 4.0, "rollouts": []}
    bc.save(tmp_path, key, params, result)
    loaded = bc.load(tmp_path, key)
    assert loaded["key"] == key
    assert loaded["params"] == params
    assert loaded["result"] == result


def test_corrupt_cache_file_yields_none(tmp_path):
    p = tmp_path / "baseline_deadbeef00000000.json"
    p.write_text("not-valid-json {")
    # bc.load swallows the error -- crashing the orchestrator on a bad
    # cache file would be worse than just recomputing.
    assert bc.load(tmp_path, "deadbeef00000000") is None
