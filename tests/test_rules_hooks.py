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

"""The A/T/O hooks on Rules.

Rules is the EnvHarness implementation that wraps an inner ActionableEnv
with three pure-function hooks: filter_action (A), modify_transition (T),
filter_observation (O). All defaults are pass-through; subclasses override
the axes they care about. These tests pin those defaults -- if the default
of any hook changes, update these expectations.

For tests that exercise the full step loop (filter_action firing through
Rules.step over an inner ActionableEnv), see test_envharness_composition.py.

Things Rules does NOT support (by design):
  - S₀ (setup_initial_state) -- use the `Setup` harness instead
  - R / step_reward                -- reserved for a future Rewards harness
"""
from envharness import Action, Blocked, EnvResponse, Observation, Rules


def test_default_filter_action_passes_through():
    r = Rules()
    a = Action(name="noop", kwargs={"x": 1})
    assert r.filter_action(a, env_state=None) is a


def test_default_filter_observation_passes_through():
    r = Rules()
    obs = Observation(text="x")
    assert r.filter_observation(obs, env_state=None) is obs


def test_default_modify_transition_passes_through():
    r = Rules()
    resp = EnvResponse(observation=Observation(text="x"),
                       reward=0.5, terminated=False, truncated=False)
    assert r.modify_transition(Action(name="noop"), resp, env_state=None) is resp




def test_rules_has_only_ato_hooks():
    """Rules exposes only A/T/O. S0 belongs to Setup (action list
    replay); R is omitted because eval success is determined by env's
    `info["won"]`, not cumulative reward."""
    r = Rules()
    assert hasattr(r, "filter_action")
    assert hasattr(r, "modify_transition")
    assert hasattr(r, "filter_observation")
    assert not hasattr(r, "setup_initial_state")  # S0 is Setup, not a Rules hook
    assert not hasattr(r, "modify_reward")
    # step_reward inherited from EnvHarness as delegate-to-inner; Rules
    # does not override it (it's not a Rules-axis hook).
    assert "step_reward" not in Rules.__dict__


def test_subclass_can_override_a_single_hook():
    """The point of subclassing Rules: change ONE hook."""

    class BlockEverything(Rules):
        def filter_action(self, action, env_state):
            return Blocked(reason="forbidden in test")

    r = BlockEverything()
    a = Action(name="combine")
    out = r.filter_action(a, env_state=None)
    assert isinstance(out, Blocked)
    assert out.reason == "forbidden in test"

    # Other hooks remain at their defaults.
    obs = Observation(text="x")
    assert r.filter_observation(obs, env_state=None) is obs


def test_subclass_can_override_modify_transition():
    """Verify the T-axis path: subclass overrides modify_transition."""

    class DoubleReward(Rules):
        def modify_transition(self, action, raw_response, env_state):
            return EnvResponse(
                observation=raw_response.observation,
                reward=raw_response.reward * 2.0,
                terminated=raw_response.terminated,
                truncated=raw_response.truncated,
                info=raw_response.info,
            )

    r = DoubleReward()
    resp = EnvResponse(observation=Observation(text="x"),
                       reward=0.5, terminated=False, truncated=False)
    out = r.modify_transition(Action(name="x"), resp, env_state=None)
    assert out.reward == 1.0
