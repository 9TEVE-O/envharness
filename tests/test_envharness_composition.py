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

"""EnvHarness composition: Setup and Rules wrap Toy24Env, and stack
together. Confirms the decorator-pattern invariant that an EnvHarness IS-A
ActionableEnv from the caller's perspective.
"""
from __future__ import annotations

from envharness import (
    Action, ActionableEnv, Blocked, EnvHarness, Setup, Rules,
)
from envharness.bridges.toy24 import Toy24Env


# ---------------------------------------------------------------------------
# Setup: replay action sequence on reset
# ---------------------------------------------------------------------------

def test_setup_is_envharness_and_actionable_env():
    e = Setup(inner=Toy24Env(),
              actions=[Action(name="combine", kwargs={"i": 0, "j": 1, "op": "add"})])
    assert isinstance(e, EnvHarness)
    assert isinstance(e, ActionableEnv)
    assert e.harness_type() == "setup"


def test_setup_replays_actions_on_reset():
    actions = [
        Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}),  # 21
        Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}),  # 24
    ]
    e = Setup(inner=Toy24Env(), actions=actions)
    e.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    # After replay, current_numbers should be [7, 24].
    assert e.get_env_state().current_numbers == [7.0, 24.0]
    # The first observation the agent sees reflects the replayed state.
    assert "24" in e.observe().text


def test_setup_step_is_passthrough_to_inner():
    e = Setup(inner=Toy24Env(),
              actions=[Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"})])
    e.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    # Continuing with the same actions the agent would have used reaches success.
    e.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    e.step(Action(name="stop"))
    assert e.evaluate().success is True


# ---------------------------------------------------------------------------
# Rules: hooks fire on the standard step loop
# ---------------------------------------------------------------------------

class _BlockDivision(Rules):
    """Tiny Rules subclass written manually (mirrors what the LLM
    would emit). Blocks division on the A axis."""
    def filter_action(self, action, env_state):
        if action.name == "combine" and action.kwargs.get("op") == "div":
            return Blocked(reason="division disabled in test")
        return action


def test_rules_is_envharness_and_actionable_env():
    m = _BlockDivision(inner=Toy24Env())
    assert isinstance(m, EnvHarness)
    assert isinstance(m, ActionableEnv)
    assert m.harness_type() == "rules"


def test_rules_filter_action_blocks_division():
    m = _BlockDivision(inner=Toy24Env())
    m.reset(seed=0, options={"numbers": [4, 2, 8, 1], "target": 24})
    # A blocked action returns a special EnvResponse and does NOT advance inner.
    resp = m.step(Action(name="combine", kwargs={"i": 0, "j": 1, "op": "div"}))
    assert resp.terminated is False
    assert resp.reward == 0.0
    assert resp.observation.data.get("blocked") is True
    assert "division disabled in test" in resp.observation.text


def test_rules_allows_unblocked_actions():
    m = _BlockDivision(inner=Toy24Env())
    m.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    m.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}))
    m.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    r = m.step(Action(name="stop"))
    assert r.terminated and r.reward == 1.0


# ---------------------------------------------------------------------------
# Stacking: Rules over Setup over Toy24Env
# ---------------------------------------------------------------------------

def test_stack_mutator_over_setup_over_toy24():
    # Setup pre-applies mul, leaving [3, 7, 21] for the agent.
    setup_actions = [Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"})]
    inner = Setup(inner=Toy24Env(), actions=setup_actions)
    outer = _BlockDivision(inner=inner)

    # From the agent's perspective, outer is just an ActionableEnv.
    assert isinstance(outer, ActionableEnv)

    outer.reset(seed=0, options={"numbers": [3, 3, 7, 7], "target": 24})
    # Setup already applied mul; current numbers are [3, 7, 21].
    assert outer.get_env_state().current_numbers == [3.0, 7.0, 21.0]
    # The Rules's filter_action layer still rejects division.
    blocked_resp = outer.step(
        Action(name="combine", kwargs={"i": 0, "j": 1, "op": "div"})
    )
    assert blocked_resp.observation.data.get("blocked") is True
    # And a normal add still works through both layers.
    outer.step(Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}))
    outer.step(Action(name="stop"))
    assert outer.evaluate().success is True


def test_inner_property_raises_when_unbound():
    # An EnvHarness constructed without inner raises a clear error if you
    # try to use it. (Save/load code paths construct with inner=...; this
    # guards against accidental misuse.)
    e = Setup()
    try:
        e.reset()
    except RuntimeError as exc:
        assert "no inner env bound" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unbound inner")
