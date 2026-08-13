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

"""Rules -- the EnvHarness with A / T / O transformation hooks.

`Rules` wraps an inner ActionableEnv and applies up to three per-step
pure-function hooks:

  - filter_action(action, env_state)               -- A axis
  - modify_transition(action, response, env_state) -- T axis
  - filter_observation(obs, env_state)             -- O axis

By design, `Rules` does NOT touch:

  - S0 (initial state). That role belongs to the `Setup` harness:
    `Setup.actions: list[Action]` -- the mutation agent outputs an
    action list, Setup.reset replays each via `inner.step(action)`
    before the policy starts. There is no separate setup_initial_state
    hook; Setup IS the S0 mechanism in this design.
  - R (modify_reward / step_reward). Eval success is the env's
    terminal `info["won"]`, not cumulative reward, so R hooks would be
    no-ops for the eval metric.

All hook defaults are pass-through. A useful `Rules` is a SUBCLASS that
overrides one or more hooks -- which is what the HarnessAgent
emits as Python source. The framework loads that source, finds the
`_Rules` subclass, and instantiates it with `inner=<the wrapped env>`.

Example of LLM-emitted code::

    class _Rules(Rules):
        def filter_action(self, action, env_state):
            if action.name == "combine" and action.kwargs.get("op") == "div":
                return Blocked(reason="division disabled")
            return action

Saved state is the Python source string; loading recompiles it.
"""
from __future__ import annotations

from typing import Any

from envharness.core.actionable_env import ActionableEnv
from envharness.core.envharness import EnvHarness
from envharness.core.registry import register_harness
from envharness.core.types import (
    Action, Blocked, EnvResetResponse, EnvResponse, Observation,
)


@register_harness("rules")
class Rules(EnvHarness):
    """A/T/O code-as-mutation EnvHarness. Subclass to override hooks.

    S0 lives entirely in the `Setup` harness (action-list replay); R
    is intentionally absent here -- see the module docstring.

    The instance carries the source code it was compiled from in
    `self.rules_code` so save_state can round-trip.
    """

    rules_code: str = ""

    # ------------------------------------------------------------------
    # Hooks (defaults: pass-through). Override in subclasses.
    # ------------------------------------------------------------------

    def filter_action(self, action: Action, env_state: Any) -> Action | Blocked:
        """A: transform or reject the agent's action before inner.step."""
        return action

    def modify_transition(self, action: Action, raw_response: EnvResponse,
                          env_state: Any) -> EnvResponse:
        """T: transform inner.step's EnvResponse before the O hook runs."""
        return raw_response

    def filter_observation(self, obs: Observation, env_state: Any) -> Observation:
        """O: transform the Observation the agent sees."""
        return obs

    # ------------------------------------------------------------------
    # ActionableEnv overrides -- the A / T / O step loop. No S0 / R.
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        reset_resp = self.inner.reset(seed, options)
        # O hook fires on the initial observation so the agent's first
        # sight of the env is consistent with what it'll see going
        # forward. We use inner.observe() to refresh in case an outer
        # Setup layer changed state between reset and now.
        env_state = self.inner.get_env_state()
        fresh = self.inner.observe()
        return EnvResetResponse(
            observation=self.filter_observation(fresh, env_state),
            info=dict(reset_resp.info),
        )

    def step(self, action: Action) -> EnvResponse:
        env_state = self.inner.get_env_state()
        filtered = self.filter_action(action, env_state)
        if isinstance(filtered, Blocked):
            # A blocked action leaves the env unchanged, so re-observe and
            # give the agent the CURRENT state alongside the block reason.
            # A bare "[blocked]" Observation would strip env-critical
            # obs.data (webarena's browsergym_raw, alfworld's
            # admissible_commands) and strand or crash format-specific
            # policies on their next act(). The O hook applies as usual.
            fresh = self.filter_observation(self.inner.observe(), env_state)
            return EnvResponse(
                observation=Observation(
                    text=f"[blocked] {filtered.reason}\n\n{fresh.text}",
                    data={**(fresh.data or {}),
                          "blocked": True,
                          "blocked_reason": filtered.reason},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"blocked_reason": filtered.reason},
            )
        raw_response = self.inner.step(filtered)
        # Re-fetch env_state for the post-transition hooks: envs that REPLACE
        # their state object on step (webarena's _build_env_state) would
        # otherwise hand T/O hooks the PRE-step snapshot -- url/axtree one
        # step stale, every step. Envs that mutate state in place (toy24,
        # alfworld) return the same object, so this is a no-op for them.
        # filter_action above deliberately keeps the pre-action state.
        post_state = self.inner.get_env_state()
        transformed = self.modify_transition(filtered, raw_response, post_state)
        new_obs = self.filter_observation(transformed.observation, post_state)
        return EnvResponse(
            observation=new_obs,
            reward=transformed.reward,                  # R untouched by Rules
            terminated=transformed.terminated,
            truncated=transformed.truncated,
            info=transformed.info,
        )

    def observe(self) -> Observation:
        env_state = self.inner.get_env_state()
        return self.filter_observation(self.inner.observe(), env_state)

    # evaluate / get_env_state / step_reward -- inherited delegation.

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {"rules_code": self.rules_code or ""}

    @classmethod
    def from_state(cls, state: dict,  # type: ignore[override]
                   inner: ActionableEnv | None = None) -> "Rules":
        code = (state.get("rules_code") or "").strip()
        if not code:
            # Empty code = pass-through Rules (all hook defaults).
            return cls(inner=inner)
        # Late import: code_loader depends on Rules (this module).
        from envharness.core.code_loader import load_rules_subclass
        subclass = load_rules_subclass(code)
        instance = subclass(inner=inner)
        instance.rules_code = code
        return instance
