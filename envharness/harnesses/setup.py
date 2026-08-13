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

"""Setup -- the S₀ EnvHarness: replay actions on inner.reset()."""
from __future__ import annotations

from typing import Any

from envharness.core.actionable_env import ActionableEnv
from envharness.core.envharness import EnvHarness
from envharness.core.registry import register_harness
from envharness.core.types import Action, EnvResetResponse


@register_harness("setup")
class Setup(EnvHarness):
    """Replay a fixed action sequence on every reset to land on a mutated
    starting state.

    Save format::

        {"actions": [{"name": "<tool>", "kwargs": {...}}, ...]}

    For text-game benches (alfworld, webarena) every action's payload is
    just a string, but the canonical dict shape generalizes to structured
    action benches (toy24's `combine(i, j, op)`) without losing
    information.

    Semantics:

    - `reset(seed, options)` calls `inner.reset(seed, options)` and then
      replays every `action` through `inner.step(action)` in order. The
      Observation returned to the agent reflects the POST-replay state
      (obtained via `inner.observe()`).
    - All other ActionableEnv methods (`step / observe / evaluate /
      get_env_state`) pass through to `inner` unchanged.
    - Replay assumes the inner env's `step` is deterministic given the
      action sequence (i.e., the same env can be reconstructed by
      reset+replay). For envs with internal RNG this assumption holds
      iff `seed` is provided.
    """

    def __init__(self, inner: ActionableEnv | None = None,
                 actions: list[Action] | None = None) -> None:
        super().__init__(inner)
        self.actions: list[Action] = list(actions or [])

    # ------------------------------------------------------------------
    # ActionableEnv overrides
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        reset_resp = self.inner.reset(seed, options)
        for a in self.actions:
            # Replay via inner.step. Errors here are surfaced so callers
            # learn about bad save files immediately.
            self.inner.step(a)
        if self.actions:
            # The replay must not be charged against the episode's budgets:
            # let the env rewind per-episode counters (alfworld's repetition
            # guard / step_count) while keeping the replayed world state.
            self.inner.notify_replay_complete()
        # The agent's first Observation must reflect the post-replay state.
        return EnvResetResponse(
            observation=self.inner.observe(),
            info=dict(reset_resp.info),
        )

    # step / observe / evaluate / get_env_state -- inherited delegation.

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        return {
            "actions": [
                {"name": a.name, "kwargs": dict(a.kwargs)}
                for a in self.actions
            ],
        }

    @classmethod
    def from_state(cls, state: dict,  # type: ignore[override]
                   inner: ActionableEnv | None = None) -> "Setup":
        raw_actions = state.get("actions") or []
        actions: list[Action] = []
        for i, a in enumerate(raw_actions):
            if not isinstance(a, dict) or "name" not in a:
                raise ValueError(
                    f"Setup.from_state: actions[{i}] must be a dict with "
                    f"a 'name' field; got {a!r}"
                )
            actions.append(Action(
                name=str(a["name"]),
                kwargs=dict(a.get("kwargs") or {}),
            ))
        return cls(inner=inner, actions=actions)

    # ------------------------------------------------------------------
    # Convenience for the orchestrator: append actions to a live Setup.
    # ------------------------------------------------------------------

    def append(self, action: Action) -> None:
        self.actions.append(action)
