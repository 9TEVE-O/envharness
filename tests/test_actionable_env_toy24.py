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

"""Toy24Env -- contract walkthrough as ActionableEnv.

Replaces the old test_bridge_toy24.py for the new ActionableEnv API.
Verifies that toy24 implements reset / step / observe / evaluate /
get_env_state / tool_schemas / env_state_schema cleanly, plus the new
env-owned save_state / from_state contract.
"""
from __future__ import annotations

from envharness import Action, ActionableEnv, EvaluationResult, Observation
from envharness.bridges.toy24 import Toy24Env, Toy24State


def test_toy24_is_actionable_env():
    assert issubclass(Toy24Env, ActionableEnv)
    assert Toy24Env.env_type() == "toy24"


def test_reset_returns_observation(fresh_toy24):
    obs = fresh_toy24.observe()
    assert isinstance(obs, Observation)
    assert "target=24" in obs.text
    assert obs.data["target"] == 24
    assert obs.data["numbers"] == [3.0, 3.0, 7.0, 7.0]


def test_step_walks_to_success(fresh_toy24):
    # mul(3, 7) → 21; numbers = [3, 7, 21]
    r1 = fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    assert r1.reward == 0.0 and not r1.terminated
    # add(3, 21) → 24; numbers = [7, 24]
    r2 = fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    assert r2.reward == 0.0 and not r2.terminated
    # stop → success
    r3 = fresh_toy24.step(Action(name="stop"))
    assert r3.terminated and r3.reward == 1.0


def test_evaluate_after_success(fresh_toy24):
    fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    fresh_toy24.step(Action(name="stop"))
    er: EvaluationResult = fresh_toy24.evaluate()
    assert er.success is True
    assert er.score == 1.0
    assert er.metrics["steps"] == 3


def test_get_env_state_returns_typed_dataclass(fresh_toy24):
    s = fresh_toy24.get_env_state()
    assert isinstance(s, Toy24State)
    assert s.target == 24
    assert s.current_numbers == [3.0, 3.0, 7.0, 7.0]


def test_tool_schemas_present_and_named():
    schemas = Toy24Env.tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"combine", "reset", "stop"}


def test_env_state_schema_is_nonempty():
    s = Toy24Env.env_state_schema()
    assert isinstance(s, str)
    assert "current_numbers" in s and "target" in s


# ----- env-owned save/load --------------------------------------------------

def test_save_state_returns_full_snapshot(fresh_toy24):
    fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    snap = fresh_toy24.save_state()
    # All fields the env needs to fully reconstruct itself.
    assert snap["target"] == 24
    assert snap["initial_numbers"] == [3, 3, 7, 7]
    assert snap["current_numbers"] == [3.0, 7.0, 21.0]
    # Toy24 history records "{a} {op} {b} = {v}" with a = current_numbers[i],
    # b = current_numbers[j]; here i=2 (val 7), j=0 (val 3).
    assert snap["history"] == ["7.0 mul 3.0 = 21.0"]
    assert snap["step_count"] == 1
    assert snap["stopped"] is False
    assert snap["success"] is False


def test_from_state_reconstructs_identical_env(fresh_toy24):
    # Walk a few steps so the state is non-trivial.
    fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    snap = fresh_toy24.save_state()
    restored = Toy24Env.from_state(snap)
    # Observation matches.
    assert restored.observe().text == fresh_toy24.observe().text
    # Continuing from the restored state reaches the same outcome.
    restored.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    restored.step(Action(name="stop"))
    assert restored.evaluate().success is True


def test_save_state_is_json_serializable(fresh_toy24):
    import json
    fresh_toy24.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    snap = fresh_toy24.save_state()
    # If save_state ever returns a non-JSON value (numpy float, dataclass,
    # etc.), this raises immediately rather than silently breaking
    # checkpoint writes downstream.
    text = json.dumps(snap)
    assert json.loads(text) == snap
