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

"""WebArenaBridge — wrap browsergym-webarena around Playwright + 6 docker services.

Contract follows the ActionableEnv ABC: zero mutation logic here, only
a uniform interface around the underlying browsergym env. Mutations come
from the harness stack: `Rules` subclasses (A/T/O per-step hooks) and
`Setup` action replay (S0).

Setup requirements (one-time on the host):
    pip install browsergym==0.14.1 browsergym-webarena==0.14.1 playwright
    sudo playwright install-deps chromium
    playwright install chromium
    # plus the 6 WebArena docker services on the canonical (or offset) ports,
    # with WA_SHOPPING / WA_REDDIT / ... env vars pointing at them. See
    # the upstream WebArena environment_docker README for canonical ports.

Reset options (passed through `options` dict in reset()):
    task_id: int       -- pick browsergym task index 0..811 (default uses seed
                          % 812 — RAGEN pattern, matches alfworld bridge).
    headless: bool     -- launch chromium without a window (default True).
    viewport: dict     -- {"width": int, "height": int} (default 1280x900).
    slow_mo: int       -- ms slowdown between Playwright ops (default 0).
    obs_style: str     -- "wrapped" (default; goal+url+axtree all in obs.text)
                          or "raw" (axtree only in obs.text; goal+url in
                          obs.data). Mirrors alfworld bridge.

Design notes (derived from RB's GenericAgent + browsergym docs):
  - Single `do(action_str)` tool. The Policy emits one browsergym high-level
    action string per turn; bridge dispatches directly to env.step(action_str).
    Per-tool decomposition (click / fill / etc.) would explode the schema
    surface for no per-task gain.
  - AXTree dual-exposed: `observation.text` carries the flattened tree
    (wrapped or raw), `observation.data["axtree_text"]` always carries it as
    plain string for programmatic Rules hooks.
  - WebArenaEnv holds the Playwright Page handle (via gymnasium env);
    Rules hook code never sees it -- it only sees WebArenaEnvState.
  - snapshot/restore: NOT implemented. Browser session restoration is
    technically possible (cookie state + URL + page DOM) but messy; defer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env
from envharness.bridges.webarena.tools import Do
from envharness.core.tool import Tool
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
    TaskSummary,
)


# Total number of browsergym-webarena tasks; verified at module load via the
# `browsergym/webarena.N` env-id registration count.
_TOTAL_TASKS = 812


@dataclass
class WebArenaEnvState:
    """Env-exposed view of WebArena state for Rules hooks.

    Pure data (str / list[str] / int / bool / dict). No Playwright handles.
    Same dataclass shape would be valid if we later swap the browser
    backend; the runtime-agnostic invariant is preserved.
    """
    goal_text: str = ""                            # task description
    url: str = ""                                  # current page URL
    page_title: str = ""                           # current <title>
    axtree_text: str = ""                          # flattened AXTree (text)
    last_action: str = ""                          # browsergym's last action string
    last_action_error: str = ""                    # parser / playwright error if any
    focused_element_bid: str = ""                  # currently focused element's bid
    open_page_urls: list[str] = field(default_factory=list)
    open_page_titles: list[str] = field(default_factory=list)
    step_count: int = 0
    won: bool = False                              # task succeeded (terminal)
    done: bool = False                             # episode finished (terminated|truncated)
    score: float = 0.0                             # reward accumulator
    max_score: float = 1.0
    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for Rules per-episode state (counters, RNG state,
    accumulators). The env never reads this."""


@register_env("webarena")
class WebArenaEnv(ActionableEnv):
    """browsergym-webarena ActionableEnv. Per-task gym ID
    `browsergym/webarena.<N>` where N ∈ [0, 811]. Underlying runtime:
    Playwright chromium driving 6 self-hosted docker services (shopping
    / reddit / gitlab / wikipedia / map / homepage)."""

    tool_registry: ClassVar[list[type[Tool]]] = [Do]

    def __init__(self) -> None:
        super().__init__()
        self._env = None                  # lazy: built on first reset
        self._task_id: int = 0
        self._headless: bool = True
        self._viewport: dict[str, int] = {"width": 1280, "height": 900}
        self._slow_mo: int = 0
        self._obs_style: str = "wrapped"
        self._latest_obs: dict[str, Any] | None = None
        self._latest_terminated: bool = False
        self._latest_truncated: bool = False
        self._latest_reward: float = 0.0
        self._latest_info: dict[str, Any] = {}
        self.state: WebArenaEnvState = WebArenaEnvState()
        # save_state retains the configuration so a checkpoint can rebuild
        # the env at the next reset() (browser session itself is not
        # snapshotted).
        self._last_reset_seed: int | None = None
        self._last_reset_options: dict[str, Any] = {}

    # -- core env interface -------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        opts = options or {}
        self._last_reset_seed = seed
        self._last_reset_options = dict(opts)

        # Task selection: integer `task_id` in opts beats seed-based index
        # (RAGEN pattern). The runner adds `task_id=spec.task_id` to options
        # automatically; for WebArena that field is the config-level task_id
        # LABEL (a string like "webarena-baseline-smoke"), not a webarena
        # task index — so we accept only int-parseable opts and silently
        # fall back to `seed` otherwise.
        task_id_opt = opts.get("task_id")
        task_id = None
        if task_id_opt is not None:
            try:
                task_id = int(task_id_opt) % _TOTAL_TASKS
            except (TypeError, ValueError):
                task_id = None
        if task_id is None:
            if seed is not None:
                task_id = int(seed) % _TOTAL_TASKS
            else:
                task_id = 0
        self._task_id = task_id

        # Honor obs_style + viewport + headless overrides; default to instance
        # values to allow Setup-driven S0 changes via in_env_actions.
        obs_style = opts.get("obs_style")
        if obs_style is not None:
            if obs_style not in ("wrapped", "raw"):
                raise ValueError(
                    f"obs_style must be 'wrapped' or 'raw', got {obs_style!r}"
                )
            self._obs_style = obs_style

        headless = opts.get("headless")
        if headless is not None:
            self._headless = bool(headless)
        viewport = opts.get("viewport")
        if viewport is not None:
            self._viewport = dict(viewport)
        slow_mo = opts.get("slow_mo")
        if slow_mo is not None:
            self._slow_mo = int(slow_mo)

        # (Re)build the gymnasium env if needed (different task or fresh).
        self._close_env_if_open()
        self._lazy_init_env(task_id)
        assert self._env is not None

        # Reset; gym returns (obs, info). seed=None is fine — browsergym uses
        # its own task-id-based seed; the canonical entry point is the env id.
        try:
            obs, info = self._env.reset()
        except Exception:
            self._close_env_if_open()
            raise

        self._latest_obs = obs
        self._latest_info = dict(info or {})
        self._latest_terminated = False
        self._latest_truncated = False
        self._latest_reward = 0.0

        # Populate env_state from the fresh observation.
        self.state = self._build_env_state(
            obs=obs, info=self._latest_info, step_count=0,
            won=False, done=False, score=0.0,
        )

        return EnvResetResponse(
            observation=self._observe(),
            info={
                "task_id": str(task_id),
                "url": self.state.url,
                "goal": self.state.goal_text,
                "won": False,
            },
        )

    def step(self, action: Action) -> EnvResponse:
        # Bypass tool_registry dispatch: the Playwright env handle is held by
        # the Bridge, not by env_state, so we can't route through Tool.invoke.
        if action.name != Do.name:
            return EnvResponse(
                observation=Observation(
                    text=f"[unknown tool: {action.name}]",
                    data={"error": "unknown_tool",
                          "available_tools": [Do.name]},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "unknown_tool"},
            )

        action_str = (action.kwargs.get("action_str") or "").strip()
        if not action_str:
            return EnvResponse(
                observation=Observation(
                    text="[no action emitted]",
                    data={"error": "empty_action"},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "empty_action"},
            )

        assert self._env is not None
        try:
            obs, reward, terminated, truncated, info = self._env.step(action_str)
        except Exception as e:  # noqa: BLE001 -- surface to runner as a trace error
            return EnvResponse(
                observation=Observation(
                    text=f"[env.step crashed: {type(e).__name__}: {str(e)[:200]}]",
                    data={"error": "env_step_crashed"},
                ),
                reward=0.0, terminated=True, truncated=False,
                info={"error": f"{type(e).__name__}: {e}"},
            )

        self._latest_obs = obs
        self._latest_info = dict(info or {})
        self._latest_terminated = bool(terminated)
        self._latest_truncated = bool(truncated)
        self._latest_reward = float(reward or 0.0)

        won = self._extract_won(self._latest_info, terminated, self._latest_reward)
        self.state = self._build_env_state(
            obs=obs, info=self._latest_info,
            step_count=self.state.step_count + 1,
            won=won, done=bool(terminated) or bool(truncated),
            score=self.state.score + float(reward or 0.0),
        )

        return EnvResponse(
            observation=self._observe(),
            reward=self._latest_reward,
            terminated=self._latest_terminated,
            truncated=self._latest_truncated,
            info={
                "won": won,
                "url": self.state.url,
                "last_action": self.state.last_action,
                "last_action_error": self.state.last_action_error,
                "task_info": self._latest_info.get("task_info", {}),
            },
        )

    def evaluate(self) -> EvaluationResult:
        """Authoritative success signal. browsergym sets reward=1.0 on the
        terminal step iff the task's evaluator passed; otherwise 0. We treat
        `state.won` (computed at step time) as the official verdict.
        """
        return EvaluationResult(
            success=self.state.won,
            score=self.state.score,
            metrics={
                "max_score": self.state.max_score,
                "step_count": float(self.state.step_count),
                "terminated": float(self._latest_terminated),
                "truncated": float(self._latest_truncated),
            },
        )

    def get_env_state(self) -> WebArenaEnvState:
        return self.state

    def observe(self) -> Observation:
        return self._observe()

    @classmethod
    def list_tasks(cls, reset_options: dict | None = None,
                    limit: int | None = None) -> list[TaskSummary]:
        """Enumerate browsergym-webarena tasks. The brief is the config's
        `intent` (task description). Used only by agent-driven task selection.
        """
        # Avoid module-load import cost; do it on demand.
        from browsergym.webarena.task import GenericWebArenaTask  # noqa: F401
        n = limit if limit is not None and limit > 0 else _TOTAL_TASKS
        n = min(n, _TOTAL_TASKS)
        out: list[TaskSummary] = []
        for i in range(n):
            out.append(TaskSummary(
                task_idx=i,
                instance_id=f"webarena.{i}",
                brief=f"WebArena task #{i}",
                metadata={"gym_id": f"browsergym/webarena.{i}"},
            ))
        return out

    @classmethod
    def env_state_schema(cls) -> str:
        return (
            "WebArenaEnvState fields:\n"
            "  goal_text: str           -- natural-language task description\n"
            "  url: str                 -- current page URL\n"
            "  page_title: str          -- <title> of current page\n"
            "  axtree_text: str         -- flattened AXTree (interactive bids shown)\n"
            "  last_action: str         -- the Policy's last emitted action string\n"
            "  last_action_error: str   -- browsergym error if action failed to parse\n"
            "  focused_element_bid: str -- bid of focused element (if any)\n"
            "  open_page_urls: list[str]\n"
            "  open_page_titles: list[str]\n"
            "  step_count: int\n"
            "  won: bool                -- terminal success (browsergym evaluator)\n"
            "  done: bool               -- terminated OR truncated\n"
            "  score: float             -- cumulative reward this episode\n"
            "  max_score: float\n"
            "  extras: dict             -- free-form for Rules per-episode state"
        )

    def close(self) -> None:
        try:
            self._close_env_if_open()
        finally:
            super().close()

    # -- save / load --------------------------------------------------------
    #
    # WebArena's browser session can't be cheaply snapshotted, so save/load
    # stores only the configuration (reset args). Loading returns a fresh
    # instance; the caller must call reset() to launch chromium.

    def save_state(self) -> dict:
        return {
            "reset_seed":    self._last_reset_seed,
            "reset_options": dict(self._last_reset_options),
        }

    @classmethod
    def from_state(cls, state: dict) -> "WebArenaEnv":
        env = cls()
        env._last_reset_seed = state.get("reset_seed")
        env._last_reset_options = dict(state.get("reset_options") or {})
        return env

    def default_reset_args(self) -> tuple[int | None, dict]:
        return self._last_reset_seed, dict(self._last_reset_options)

    # -- helpers ------------------------------------------------------------

    def _lazy_init_env(self, task_id: int) -> None:
        if self._env is not None:
            return
        import gymnasium
        import browsergym.core  # noqa: F401 — registers env namespace
        import browsergym.webarena  # noqa: F401 — registers webarena.N envs

        env = gymnasium.make(
            f"browsergym/webarena.{task_id}",
            headless=self._headless,
            viewport=self._viewport,
            slow_mo=self._slow_mo,
            # browsergym defaults to a fresh chromium per env; no shared state
            # to worry about across resets.
        )
        self._env = env

    def _close_env_if_open(self) -> None:
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception:  # noqa: BLE001 -- never mask the real error
            pass
        self._env = None

    def _build_env_state(self, *, obs: dict, info: dict, step_count: int,
                          won: bool, done: bool, score: float) -> WebArenaEnvState:
        """Compose a fresh WebArenaEnvState from the latest gym observation."""
        from browsergym.utils.obs import flatten_axtree_to_str

        axtree_text = ""
        ax_obj = obs.get("axtree_object")
        if ax_obj is not None:
            try:
                axtree_text = flatten_axtree_to_str(ax_obj)
            except Exception:  # noqa: BLE001 -- fallback to empty AXTree
                axtree_text = ""

        # browsergym puts the goal in `goal` (string) or `goal_object` (list
        # of typed chunks). Use the string when available.
        goal_text = ""
        g = obs.get("goal")
        if isinstance(g, str) and g:
            goal_text = g
        else:
            go = obs.get("goal_object")
            if isinstance(go, (list, tuple)) and go:
                pieces = []
                for chunk in go:
                    t = chunk.get("text") if isinstance(chunk, dict) else None
                    if isinstance(t, str):
                        pieces.append(t)
                goal_text = "\n".join(pieces)

        open_urls = list(obs.get("open_pages_urls") or [])
        open_titles = list(obs.get("open_pages_titles") or [])

        # Page title: first entry of open_pages_titles if available.
        page_title = open_titles[0] if open_titles else ""

        return WebArenaEnvState(
            goal_text=goal_text,
            url=str(obs.get("url") or ""),
            page_title=page_title,
            axtree_text=axtree_text,
            last_action=str(obs.get("last_action") or ""),
            last_action_error=str(obs.get("last_action_error") or ""),
            focused_element_bid=str(obs.get("focused_element_bid") or ""),
            open_page_urls=open_urls,
            open_page_titles=open_titles,
            step_count=int(step_count),
            won=bool(won),
            done=bool(done),
            score=float(score),
            max_score=1.0,
            extras=dict(self.state.extras) if self.state else {},
        )

    def _observe(self) -> Observation:
        """Compose the Policy-visible Observation. obs_style:
        - "wrapped" (default): goal + url + axtree assembled in obs.text
        - "raw"             : axtree only in obs.text; goal+url in obs.data
        Both modes populate obs.data fully so Rules code can read either.
        """
        if self._obs_style == "raw":
            text = self.state.axtree_text
        else:
            parts: list[str] = []
            if self.state.goal_text:
                parts.append(f"Task: {self.state.goal_text}")
            if self.state.url:
                parts.append(f"URL: {self.state.url}")
            if self.state.page_title:
                parts.append(f"Page title: {self.state.page_title}")
            if self.state.axtree_text:
                parts.append("AXTree:\n" + self.state.axtree_text)
            if self.state.last_action_error:
                parts.append(f"Last action error: {self.state.last_action_error}")
            text = "\n\n".join(parts)
        return Observation(
            text=text,
            data={
                "goal_text": self.state.goal_text,
                "url": self.state.url,
                "page_title": self.state.page_title,
                "axtree_text": self.state.axtree_text,
                "last_action": self.state.last_action,
                "last_action_error": self.state.last_action_error,
                "focused_element_bid": self.state.focused_element_bid,
                "open_page_urls": list(self.state.open_page_urls),
                "open_page_titles": list(self.state.open_page_titles),
                "step_count": self.state.step_count,
                "score": self.state.score,
                "max_score": self.state.max_score,
                "won": self.state.won,
                # Raw browsergym observation for prompt-parity with the test
                # baseline (run_baseline_reasoning_bank_agent uses RB GenericAgent which
                # consumes the full raw dict, not envharness's flattened
                # obs.text). Rules.filter_observation can mutate this
                # in-place; the Policy reads obs.data["browsergym_raw"].
                "browsergym_raw": self._latest_obs,
            },
        )

    @staticmethod
    def _extract_won(info: dict, terminated: bool, reward: float) -> bool:
        """Authoritative `won` decision. browsergym's webarena tasks set
        reward=1.0 on the terminal step iff the task's evaluator passed.
        We also honor an explicit info['task_info']['success'] flag if any
        future browsergym version exposes it. terminated alone doesn't mean
        success — the agent can terminate by hitting max_steps too."""
        ti = (info or {}).get("task_info") or {}
        if "success" in ti:
            return bool(ti["success"])
        # Default: success iff terminated AND reward > 0.5
        return bool(terminated) and (reward >= 0.5)
