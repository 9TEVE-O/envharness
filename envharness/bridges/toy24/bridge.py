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

"""Toy24Env -- minimal ActionableEnv for the 24-point game.

This is the reference ActionableEnv implementation. It demonstrates the
contract for new benchmarks:

  - implement reset / step / observe / evaluate / get_env_state
  - implement save_state / from_state (env's own state, NOT harness state)
  - register with `@register_env("<tag>")` so the persistence layer can
    find it by tag

Toy24's state is small and fully in memory, so `save_state` captures the
whole live snapshot. Benches whose runtime can't be cloned cheaply
(Docker, browsers, sim engines) typically save only `(reset_seed,
reset_options)` and accept that `from_state` is valid only at episode
boundaries.

This module deliberately contains ZERO mutation logic -- mutations come
from EnvHarness layers (Setup / Rules). The env exposes:

  - three tools (combine / reset / stop) via `tool_registry`
  - vanilla EnvResponse with raw reward (1.0 on success, 0.0 else)
  - `Toy24State` dataclass for harness hooks to read or mutate
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from envharness.bridges.toy24.solver import generate_puzzle
from envharness.bridges.toy24.tools import Combine, Reset, Stop
from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
)


EPS = 1e-6


@dataclass
class Toy24State:
    """ActionableEnv-exposed view of state. Safe for harness hooks to mutate.

    Contains NO runtime handles -- pure data. Same shape regardless of how
    the env implements its underlying runtime (which is just an in-memory
    Python state here, but the principle generalizes to docker / browser
    / sim Bridges)."""
    target: int = 24
    initial_numbers: list[int] = field(default_factory=list)
    current_numbers: list[float] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    stopped: bool = False
    success: bool = False
    step_count: int = 0
    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for harness hooks to attach per-episode state
    (counters, RNG state, accumulators). Env never reads this."""


@register_env("toy24")
class Toy24Env(ActionableEnv):
    """The 24-point game as an ActionableEnv."""

    tool_registry = [Combine, Reset, Stop]

    def __init__(self) -> None:
        super().__init__()
        self.state: Toy24State = Toy24State()
        # The reset args used last; needed if a downstream benchmarks-
        # without-snapshot pattern wants to reconstruct via reset replay.
        self._last_reset_seed: int | None = None
        self._last_reset_options: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        opts = options or {}
        rng = random.Random(seed)
        if "numbers" in opts:
            nums = list(opts["numbers"])
        else:
            nums = generate_puzzle(opts.get("difficulty", "easy"),
                                    opts.get("target", 24),
                                    allow_unsolvable=False, rng=rng)
        self.state = Toy24State(
            target=int(opts.get("target", 24)),
            initial_numbers=list(nums),
            current_numbers=[float(n) for n in nums],
        )
        self._last_reset_seed = seed
        self._last_reset_options = dict(opts)
        return EnvResetResponse(
            observation=self._observe(),
            info={"task_id": opts.get("task_id", "")},
        )

    def step(self, action: Action) -> EnvResponse:
        self.state.step_count += 1
        tool_cls = next(
            (t for t in self.tool_registry if t.name == action.name), None
        )
        if tool_cls is None:
            return EnvResponse(
                observation=Observation(
                    text=f"[unknown tool: {action.name}]",
                    data={"numbers": list(self.state.current_numbers)},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "unknown_tool"},
            )
        try:
            result = tool_cls.invoke(self.state, **action.kwargs)
        except TypeError as e:
            return EnvResponse(
                observation=Observation(
                    text=f"[bad args: {e}]",
                    data={"numbers": list(self.state.current_numbers)},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "bad_args"},
            )
        terminated = self.state.stopped
        reward = 1.0 if (self.state.stopped and self.state.success) else 0.0
        return EnvResponse(
            observation=self._observe(extra=str(result)),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "success": self.state.success if self.state.stopped else None,
                "result": result,
            },
        )

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            success=self.state.stopped and self.state.success,
            score=1.0 if (self.state.stopped and self.state.success) else 0.0,
            metrics={
                "steps": self.state.step_count,
                "history_len": len(self.state.history),
            },
        )

    def get_env_state(self) -> Toy24State:
        return self.state

    def observe(self) -> Observation:
        return self._observe()

    def default_reset_args(self) -> tuple[int | None, dict]:
        return self._last_reset_seed, dict(self._last_reset_options)

    def reset_after_load(self) -> bool:
        # If `from_state` restored a full live snapshot (initial_numbers
        # populated), do NOT auto-reset: it would regenerate the puzzle and
        # wipe the snapshot. If the snapshot was config-only (synthesized
        # by the orchestrator from reset_seed + reset_options), DO reset
        # so the env lands on the configured puzzle.
        return not bool(self.state.initial_numbers)

    @classmethod
    def env_state_schema(cls) -> str:
        return (
            "env_state is a Toy24State dataclass with these fields:\n"
            "  target: int                  -- the target number to reach (default 24)\n"
            "  initial_numbers: list[int]   -- the puzzle's starting numbers\n"
            "  current_numbers: list[float] -- numbers currently on the board;\n"
            "                                  Combine and Reset tools mutate this\n"
            "  history: list[str]           -- log strings of operations so far\n"
            "  stopped: bool                -- True after Stop tool is invoked\n"
            "  success: bool                -- set by Stop iff target is in current_numbers\n"
            "  step_count: int              -- number of actions taken so far\n"
            "  extras: dict                 -- free dict for harness per-episode state\n"
            "S0 changes go through in_env_actions (Setup layer), not Rules. Rules hooks can read all\n"
            "fields and may write to `extras` for per-episode bookkeeping."
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    #
    # Toy24's state is small and fully in memory, so save_state captures
    # the complete live snapshot. This is the "lucky" case; for
    # docker/browser/sim benches, save_state would typically return only
    # `{"reset_seed": ..., "reset_options": {...}}` and from_state would
    # re-run `reset(seed, options)`.

    def save_state(self) -> dict:
        return {
            "target":          int(self.state.target),
            "initial_numbers": [int(n) for n in self.state.initial_numbers],
            "current_numbers": [float(n) for n in self.state.current_numbers],
            "history":         list(self.state.history),
            "stopped":         bool(self.state.stopped),
            "success":         bool(self.state.success),
            "step_count":      int(self.state.step_count),
            "extras":          dict(self.state.extras),
            # Reset args preserved so callers can replay-reset if they want
            # to roll back to a fresh episode rather than restore mid-state.
            "reset_seed":      self._last_reset_seed,
            "reset_options":   dict(self._last_reset_options),
        }

    @classmethod
    def from_state(cls, state: dict) -> "Toy24Env":
        env = cls()
        env.state = Toy24State(
            target=int(state.get("target", 24)),
            initial_numbers=[int(n) for n in (state.get("initial_numbers") or [])],
            current_numbers=[float(n) for n in (state.get("current_numbers") or [])],
            history=list(state.get("history") or []),
            stopped=bool(state.get("stopped", False)),
            success=bool(state.get("success", False)),
            step_count=int(state.get("step_count", 0)),
            extras=dict(state.get("extras") or {}),
        )
        env._last_reset_seed = state.get("reset_seed")
        env._last_reset_options = dict(state.get("reset_options") or {})
        return env

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _observe(self, extra: str = "") -> Observation:
        nums = ", ".join(_fmt(x) for x in self.state.current_numbers)
        history = " | ".join(self.state.history[-5:])
        text = (
            f"target={self.state.target}, numbers=[{nums}]"
            + (f". recent: {history}" if history else "")
            + (f". last_result: {extra}" if extra else "")
        )
        return Observation(text=text, data={
            "numbers":    list(self.state.current_numbers),
            "target":     self.state.target,
            "history":    list(self.state.history),
            "step_count": self.state.step_count,
        })


def _fmt(x: float) -> str:
    if abs(x - round(x)) < EPS:
        return str(int(round(x)))
    return f"{x:.4g}"
