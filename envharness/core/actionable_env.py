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

"""ActionableEnv -- the universal env interface.

Every benchmark (toy24, alfworld, swebench, webarena, ...) implements this
ABC directly. EnvHarness composes WITH an ActionableEnv to produce a NEW
ActionableEnv (decorator pattern -- see envharness.core.envharness). This
means agents/LLMs and EnvHarness code both program against the same
interface; they cannot tell the difference between "raw bench" and
"bench wrapped in N harness layers".

Two things every concrete env must define:

1. **The Gymnasium-like step loop**: `reset / step / observe / evaluate`
   plus `get_env_state` (a Bridge-safe view of internal state that
   harnesses can read in their hooks).

2. **Env-owned save / load**: `save_state` returns a JSON-serializable
   dict capturing everything needed to reconstruct the env at this
   moment; `from_state(dict) -> ActionableEnv` returns a fresh instance
   in that state. The contract is intentionally minimal:

       env.save_state() -> dict
       cls.from_state(dict) -> env

   What "reconstruct" means is the env's call. For pure-Python toy benches
   the dict holds the full live state. For benches whose runtime cannot
   be cloned cheaply (Docker, browsers, sim engines), the dict typically
   holds only `(reset_seed, reset_options)` and `from_state` re-runs
   `reset(seed, options)` -- valid only at episode boundaries. Either way,
   the save-format is OWNED BY THE ENV and independent of any EnvHarness
   that may wrap it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from envharness.core.tool import Tool
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
    StepInfo, TaskSummary,
)


class ActionableEnv(ABC):
    """The universal env interface. All benchmarks implement directly;
    EnvHarness subclasses extend this contract by also wrapping a child
    ActionableEnv (composition)."""

    tool_registry: ClassVar[list[type[Tool]]] = []

    # ------------------------------------------------------------------
    # Identity (for save/load routing through the registry)
    # ------------------------------------------------------------------
    #
    # NOT @abstractmethod -- if it were, the @register_env(tag) decorator
    # (which injects the implementation after class construction) would
    # arrive too late: ABC inspects the class body first and refuses to
    # instantiate anything still carrying an abstract method. Instead the
    # base raises a clear NotImplementedError on call, so a subclass that
    # forgets to register still fails loudly at the first save/load
    # attempt rather than at instantiation time.

    @classmethod
    def env_type(cls) -> str:
        """Stable string tag for this env. Used as the `type` field in
        save dicts. Set automatically by `@register_env(tag)`; if you
        skip the decorator, override this classmethod manually."""
        raise NotImplementedError(
            f"{cls.__name__}.env_type() is not set. Decorate the class "
            "with @register_env('<tag>') or override env_type() manually."
        )

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        ...

    @abstractmethod
    def step(self, action: Action) -> EnvResponse: ...

    @abstractmethod
    def observe(self) -> Observation:
        """Return a fresh Observation reflecting current state.

        Distinct from `reset()` because harnesses (Setup replay, Rules
        in_env_actions via Setup) may change state between `reset()` returning
        and the agent's first call -- `observe()` lets the outer layer
        re-read state without paying for another `reset`.
        """
        ...

    @abstractmethod
    def evaluate(self) -> EvaluationResult: ...

    @abstractmethod
    def get_env_state(self) -> Any:
        """Bridge-safe view of internal state. MUST NOT contain runtime
        handles (Docker container, browser page, sockets) -- only data
        that harness hooks can read or mutate."""
        ...

    # ------------------------------------------------------------------
    # Save / load (the env's own state, independent of any wrapping harness)
    # ------------------------------------------------------------------

    @abstractmethod
    def save_state(self) -> dict:
        """Return a JSON-serializable dict capturing this env's current state.

        Contents are the env's business. Common patterns:
          - in-memory benches: full state snapshot
          - Docker/browser/sim benches: `{"reset_seed": ..., "reset_options": {...}}`
            and accept that load is only valid at episode boundaries

        The returned dict appears as `env.state` in the checkpoint format.
        """
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "ActionableEnv":
        """Return a fresh instance reconstructed from `state`. Should be
        the inverse of `save_state`."""
        ...

    # ------------------------------------------------------------------
    # Helpful defaults
    # ------------------------------------------------------------------

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return [t.get_info() for t in cls.tool_registry]

    @classmethod
    def env_state_schema(cls) -> str:
        """Human-readable description of the `get_env_state` shape, surfaced
        in Rules prompts so an LLM knows what fields its hooks can read."""
        return "(no env_state schema declared)"

    # ------------------------------------------------------------------
    # Default reset args -- consumed by `persistence.load_checkpoint(auto_reset=True)`
    # so a loaded env is ready for the agent loop without the caller
    # needing to dig the (seed, options) tuple out of the save dict.
    # ------------------------------------------------------------------

    def default_reset_args(self) -> tuple[int | None, dict]:
        """Return (seed, options) that `env.reset()` should be called with
        after loading this env from a Checkpoint. Default: (None, {}) --
        callers that want to reproduce a checkpoint exactly should override
        this to return whatever was stashed by the last reset()."""
        return None, {}

    def reset_after_load(self) -> bool:
        """Should `load_checkpoint(auto_reset=True)` call `env.reset()` on
        this env after `from_state` restored it? Default True (the env was
        only given config-only state, so reset is needed to populate live
        state). Override to False when `from_state` restored a FULL live
        snapshot -- calling reset() then would destroy the snapshot."""
        return True

    # ------------------------------------------------------------------
    # Optional hooks the runner may call. Both default to safe no-ops.
    # ------------------------------------------------------------------

    def step_reward(self, step_info: StepInfo) -> float:
        """Dense per-step training signal. Default 0.0 (no extra signal).

        The runner calls this after each step; the value is recorded on
        `Step.process_reward`. NON-FATAL: if a subclass override raises,
        the runner records the exception on `Step.process_reward_error`
        and the episode continues -- the framework's correctness does
        NOT depend on this signal. Rules overrides it for hooked
        process reward; plain envs and Setup inherit the no-op default."""
        return 0.0

    def notify_replay_complete(self) -> None:
        """Called by the `Setup` harness after it finishes replaying its
        action list inside reset(), BEFORE the agent's first step.

        Envs that keep per-episode counters which must not be charged for
        the replay (step budgets, repetition guards) override this to
        rewind them -- the replayed WORLD state itself stays as-is. E.g.
        alfworld resets `step_count` / its repetition guard so a replayed
        trajectory whose actions echoed identical observations does not
        pre-truncate the episode before the policy acts. Default: no-op."""
        return None

    @classmethod
    def list_tasks(cls, reset_options: dict | None = None,
                    limit: int | None = None) -> list[TaskSummary]:
        """Return a list of `TaskSummary` describing candidate tasks
        available under the given `reset_options`. Used by the
        orchestrator when `mutator_selects_tasks=True` to present a
        shortlist to the HarnessAgent.

        Default raises NotImplementedError. Subclasses override to
        enable agent-driven task selection."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement list_tasks(). Set "
            "OrchestratorConfig.mutator_selects_tasks=False or implement "
            "list_tasks() on this env."
        )

    def close(self) -> None:
        """Release runtime resources (Docker container, browser, ...).
        Subprocess death does NOT auto-release child resources, so Bridges
        with external state MUST do their cleanup here."""
        return None
