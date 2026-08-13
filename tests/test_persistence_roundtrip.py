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

"""Round-trip tests: env + harness stack -> save -> load -> behaves the same.

Save/load is a critical surface: a checkpoint that fails to round-trip
silently corrupts every downstream run that reloads it.
Every test here asserts that:

  (1) save_state() returns a JSON-serializable dict
  (2) the dict can be loaded back into an equivalent live env
  (3) the equivalence is OBSERVABLE: after the same sequence of actions,
      the original and the restored stack produce equal observations,
      equal rewards, and equal evaluate() results

We do this for:
  - Toy24Env alone
  - Setup over Toy24Env
  - Rules (compiled from LLM-style source) over Toy24Env
  - the full stack Rules(Setup(Toy24Env()))
  - pass-through cases (empty Setup, empty Rules)
  - error paths (corrupt JSON, unknown tag, missing fields)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from envharness import (
    Action, Checkpoint, Setup, Rules,
    build_stack, dump_stack, load_checkpoint, save_checkpoint,
)
from envharness.bridges.toy24 import Toy24Env


SOLVE_AFTER_MUL = [
    Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}),
    Action(name="stop", kwargs={}),
]


def _run_to_completion(env, actions):
    """Apply a list of Actions and return (observations, rewards, eval_result)."""
    obs_texts: list[str] = []
    rewards: list[float] = []
    for a in actions:
        resp = env.step(a)
        obs_texts.append(resp.observation.text)
        rewards.append(resp.reward)
        if resp.terminated:
            break
    return obs_texts, rewards, env.evaluate()


# ---------------------------------------------------------------------------
# Toy24Env alone
# ---------------------------------------------------------------------------

def test_toy24_dump_and_build_roundtrip():
    env = Toy24Env()
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    env.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))

    cp = dump_stack(env)
    assert cp.env_type == "toy24"
    assert cp.harnesses == []

    restored = build_stack(cp)
    assert isinstance(restored, Toy24Env)
    assert restored.observe().text == env.observe().text


def test_toy24_save_and_load_to_disk(tmp_path: Path):
    env = Toy24Env()
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    env.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))

    path = save_checkpoint(env, tmp_path / "cp.json",
                            metadata={"why": "smoke"})
    loaded = load_checkpoint(path)
    # Restored env behaves identically going forward.
    o_orig, r_orig, e_orig = _run_to_completion(env, SOLVE_AFTER_MUL)
    o_new,  r_new,  e_new  = _run_to_completion(loaded, SOLVE_AFTER_MUL)
    assert o_orig == o_new
    assert r_orig == r_new
    assert e_orig.success == e_new.success
    assert e_orig.score == e_new.score


# ---------------------------------------------------------------------------
# Setup round-trip
# ---------------------------------------------------------------------------

def test_setup_over_toy24_roundtrip(tmp_path: Path):
    setup_actions = [
        Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}),
    ]
    env = Setup(inner=Toy24Env(), actions=setup_actions)
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})

    cp = dump_stack(env, metadata={"who": "test"})
    assert cp.env_type == "toy24"
    assert len(cp.harnesses) == 1
    assert cp.harnesses[0]["type"] == "setup"
    assert cp.harnesses[0]["state"]["actions"] == [
        {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "mul"}},
    ]

    p = save_checkpoint(env, tmp_path / "setup.json", metadata={"who": "test"})
    loaded = load_checkpoint(p)
    # The loaded stack hasn't been reset yet -- agent loop's job. Reset it.
    loaded.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    # After reset, the loaded Setup should replay mul, matching env's state.
    assert loaded.get_env_state().current_numbers == [3.0, 7.0, 21.0]
    o_orig, r_orig, e_orig = _run_to_completion(env, SOLVE_AFTER_MUL)
    o_new,  r_new,  e_new  = _run_to_completion(loaded, SOLVE_AFTER_MUL)
    assert o_orig == o_new
    assert r_orig == r_new
    assert e_orig.success is True and e_new.success is True


def test_empty_setup_roundtrips():
    env = Setup(inner=Toy24Env(), actions=[])
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    cp = dump_stack(env)
    assert cp.harnesses == [{"type": "setup", "state": {"actions": []}}]
    restored = build_stack(cp)
    restored.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    assert restored.observe().text == env.observe().text


# ---------------------------------------------------------------------------
# Rules round-trip (compiled from source)
# ---------------------------------------------------------------------------

_BLOCK_DIV_SOURCE = """
class _Rules(Rules):
    def filter_action(self, action, env_state):
        if action.name == "combine" and action.kwargs.get("op") == "div":
            return Blocked(reason="div disabled (mutator source)")
        return action
"""


def test_rules_over_toy24_roundtrip(tmp_path: Path):
    from envharness import load_rules_instance
    inner = Toy24Env()
    mutator = load_rules_instance(_BLOCK_DIV_SOURCE, inner=inner)
    mutator.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})

    cp = dump_stack(mutator)
    assert cp.env_type == "toy24"
    assert len(cp.harnesses) == 1
    assert cp.harnesses[0]["type"] == "rules"
    assert "filter_action" in cp.harnesses[0]["state"]["rules_code"]

    p = save_checkpoint(mutator, tmp_path / "mut.json")
    loaded = load_checkpoint(p)
    loaded.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})

    # The restored Rules's filter_action hook still blocks division.
    resp = loaded.step(Action(name="combine", kwargs={"i": 0, "j": 1, "op": "div"}))
    assert resp.observation.data.get("blocked") is True
    assert "div disabled (mutator source)" in resp.observation.text


def test_empty_rules_roundtrips():
    env = Rules(inner=Toy24Env())
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    cp = dump_stack(env)
    assert cp.harnesses == [{"type": "rules", "state": {"rules_code": ""}}]
    loaded = build_stack(cp)
    loaded.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    # Pass-through mutator: identical step semantics.
    o1, r1, e1 = _run_to_completion(env,    SOLVE_AFTER_MUL)
    o2, r2, e2 = _run_to_completion(loaded, SOLVE_AFTER_MUL)
    assert o1 == o2 and r1 == r2 and e1.success == e2.success


# ---------------------------------------------------------------------------
# Full stack: Rules over Setup over Toy24Env
# ---------------------------------------------------------------------------

def test_full_stack_roundtrip(tmp_path: Path):
    from envharness import load_rules_instance
    setup_actions = [Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"})]
    inner = Setup(inner=Toy24Env(), actions=setup_actions)
    mutator = load_rules_instance(_BLOCK_DIV_SOURCE, inner=inner)
    mutator.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})

    cp = dump_stack(mutator)
    assert cp.env_type == "toy24"
    # Inner-first ordering: Setup then Rules.
    assert [h["type"] for h in cp.harnesses] == ["setup", "rules"]

    p = save_checkpoint(mutator, tmp_path / "stack.json",
                          metadata={"experiment": "smoke"})
    loaded = load_checkpoint(p)
    loaded.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})

    # Setup replayed mul → [3, 7, 21]; Rules still rejects div.
    assert loaded.get_env_state().current_numbers == [3.0, 7.0, 21.0]
    resp = loaded.step(Action(name="combine", kwargs={"i": 0, "j": 1, "op": "div"}))
    assert resp.observation.data.get("blocked") is True
    # add + stop still solves.
    loaded.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    loaded.step(Action(name="stop"))
    assert loaded.evaluate().success is True


# ---------------------------------------------------------------------------
# JSON / file-format error paths
# ---------------------------------------------------------------------------

def test_checkpoint_save_then_load_preserves_metadata(tmp_path: Path):
    env = Toy24Env()
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    p = save_checkpoint(env, tmp_path / "cp.json",
                          metadata={"created_by": "test", "n": 7})
    cp = Checkpoint.load(p)
    assert cp.metadata == {"created_by": "test", "n": 7}


def test_unknown_env_type_raises_clear_error():
    bad = {
        "schema_version": 1,
        "env": {"type": "this_env_does_not_exist", "state": {}},
        "harnesses": [],
        "metadata": {},
    }
    cp = Checkpoint.from_dict(bad)
    with pytest.raises(KeyError, match="Unknown env_type"):
        build_stack(cp)


def test_unknown_harness_type_raises_clear_error():
    bad = {
        "schema_version": 1,
        "env": {"type": "toy24", "state": Toy24Env().save_state()},
        "harnesses": [{"type": "imaginary", "state": {}}],
        "metadata": {},
    }
    cp = Checkpoint.from_dict(bad)
    with pytest.raises(KeyError, match="Unknown harness_type"):
        build_stack(cp)


def test_wrong_schema_version_raises():
    with pytest.raises(ValueError, match="schema_version"):
        Checkpoint.from_dict({
            "schema_version": 999,
            "env": {"type": "toy24", "state": {}},
            "harnesses": [],
        })


def test_corrupt_top_level_raises():
    # Not a dict at all.
    with pytest.raises(ValueError, match="expected dict"):
        Checkpoint.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_missing_env_block_raises():
    with pytest.raises(ValueError, match="env"):
        Checkpoint.from_dict({"schema_version": 1, "harnesses": []})


def test_malformed_action_in_setup_state_raises_on_load():
    env = Toy24Env()
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    bad = Checkpoint(
        env_type="toy24",
        env_state=env.save_state(),
        harnesses=[{"type": "setup",
                     "state": {"actions": [{"kwargs": {}}]}}],  # no "name"
        metadata={},
    )
    with pytest.raises(ValueError, match="must be a dict with a 'name'"):
        build_stack(bad)


def test_save_refuses_non_json_serializable(tmp_path: Path):
    # Force a non-JSON value into env_state and confirm save raises.
    env = Toy24Env()
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    # The user-side bug we want to catch: stuffing a non-JSON object
    # into a save dict at construction time.
    cp = Checkpoint(env_type="toy24",
                     env_state={"unjsonable": set([1, 2, 3])},  # set is not JSON
                     harnesses=[])
    with pytest.raises(TypeError):
        cp.save(tmp_path / "bad.json")


# ---------------------------------------------------------------------------
# Exhaustive JSON round-trip on the dict produced by save_state
# ---------------------------------------------------------------------------

def test_dump_stack_dict_is_json_roundtrip_clean(tmp_path: Path):
    from envharness import load_rules_instance
    setup_actions = [Action(name="combine", kwargs={"i": 0, "j": 1, "op": "add"})]
    env = load_rules_instance(
        _BLOCK_DIV_SOURCE,
        inner=Setup(inner=Toy24Env(), actions=setup_actions),
    )
    env.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    cp = dump_stack(env, metadata={"k": "v"})
    blob = cp.to_dict()
    text = json.dumps(blob)               # must not raise
    decoded = json.loads(text)
    cp2 = Checkpoint.from_dict(decoded)
    assert cp2.env_type == cp.env_type
    assert cp2.env_state == cp.env_state
    assert cp2.harnesses == cp.harnesses
    assert cp2.metadata == cp.metadata
