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

"""AlfworldBridge -- wrap ALFWorld's TextWorld backend (Allen AI).

Contract follows the ActionableEnv ABC: zero mutation logic here, only
a uniform interface around the underlying TextWorld engine. Mutations
come from the harness stack: Rules subclasses (A/T/O per-step hooks)
and Setup action replay (S0).

Setup requirements (one-time on the host):
    pip install "alfworld[full]"
    alfworld-download              # populates ~/.cache/alfworld/

Reset options (passed through `options` dict in reset()):
    seed: int          -- deterministic task selection: seed(n) shuffles the
                          gamefile order and reset() plays the shuffled head
                          (same seed + split -> same game; NOT an index).
    task_id: str       -- explicit game-file path (or unique suffix). If it
                          matches a known gamefile, that EXACT game is pinned
                          for this reset; takes priority over seed.
    split: str         -- "train" | "eval_in_distribution" | "eval_out_of_distribution"
                          (default: "train").  Matches alfworld's `train_eval` arg.
    config_path: str   -- path to alfworld's base_config.yaml. If omitted, uses
                          $ALFWORLD_CONFIG, then alfworld's bundled config.

Design notes (derived from surveying RAGEN / AgentGym / AgentBench wrappers):
  - Single `do(text)` tool. Per-verb tools and tool-call JSON wrappers cause
    more parse failures than they save.
  - `admissible_commands` is dual-exposed: in `Observation.text` (so LLM
    policies that ignore side-channels still see it) AND in `Observation.data`
    (for programmatic mutations).
  - "Nothing happens" + repetition guard: 3 consecutive identical obs
    truncates the episode (AgentBench-style; prevents Policy collapse loops
    from burning the runner timeout).
  - snapshot/restore: NOT implemented. Jericho/TextWorld state isn't cleanly
    picklable; no surveyed wrapper supports it. The realistic implementation
    is "replay (seed, action-list)" which is a future enhancement, not core.
  - Rules never sees the TextWorld engine handle -- only the
    AlfworldEnvState dataclass and the standard typed I/O.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env
from envharness.bridges.alfworld.tools import Do
from envharness.core.tool import Tool
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
    TaskSummary,
)


# Default consecutive-identical-observations threshold before truncation. Pass
# `repetition_threshold` in reset_options to override; <=0 disables truncation
# entirely (matches pure_eval's "engine_done or max_steps only" loop).
_REPETITION_THRESHOLD = 3


@dataclass
class AlfworldEnvState:
    """Env-exposed view of ALFWorld state for Rules hooks.

    Pure data (str / list[str] / int / bool / dict). No engine handles. The
    same dataclass shape would be valid if we later swap the backend for a
    different text engine; the runtime-agnostic invariant is preserved.
    """
    goal_text: str = ""                            # natural-language goal
    obs_text: str = ""                             # latest observation text
    admissible_commands: list[str] = field(default_factory=list)
    inventory: list[str] = field(default_factory=list)
    won: bool = False
    done: bool = False
    score: float = 0.0
    max_score: float = 1.0
    step_count: int = 0
    last_action_was_effective: bool = True
    repetition_count: int = 1                      # 1 = no repetition yet
    last_obs_for_repeat: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for Rules per-episode state (counters, RNG state,
    accumulators). Bridge never reads this."""


@register_env("alfworld")
class AlfworldEnv(ActionableEnv):
    """ALFWorld text-game ActionableEnv (TextWorld backend; do NOT use
    AlfredThorEnv -- the Thor backend does not expose admissible_commands)."""

    tool_registry: ClassVar[list[type[Tool]]] = [Do]

    def __init__(self) -> None:
        super().__init__()
        self._env = None                  # lazy: built on first reset
        self._game_files: list[str] = []  # populated from env after init
        self._current_split: str = "train"
        self._repetition_threshold: int = _REPETITION_THRESHOLD
        # Save args -- retained so save_state can produce a load-replayable
        # blob. Update on each reset.
        self._last_reset_seed: int | None = None
        self._last_reset_options: dict[str, Any] = {}
        # obs_style controls _observe()'s text composition:
        #   "wrapped" (default, legacy): "Task: <goal>\n\n<obs>\n\nAdmissible commands: ..."
        #   "raw":                       just the raw TextWorld <obs>; goal +
        #                                admissibles are still in observation.data
        # Use "raw" when the caller (PolicyAgent, training-time prompt template)
        # already structures the prompt itself -- avoids duplicate context that
        # measurably hurts non-thinking-mode models.
        self._obs_style: str = "wrapped"
        self.state: AlfworldEnvState = AlfworldEnvState()

    # -- core env interface -------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        opts = options or {}
        self._last_reset_seed = seed
        self._last_reset_options = dict(opts)
        split = opts.get("split", "train")
        config_path = opts.get("config_path") or os.environ.get("ALFWORLD_CONFIG")
        rt = opts.get("repetition_threshold")
        if rt is not None:
            self._repetition_threshold = int(rt)
        obs_style = opts.get("obs_style")
        if obs_style is not None:
            if obs_style not in ("wrapped", "raw"):
                raise ValueError(
                    f"obs_style must be 'wrapped' or 'raw', got {obs_style!r}"
                )
            self._obs_style = obs_style
        # Optional task-subset filter: pass a JSONL where each line has a
        # "game_file" abs path. After lazy_init we filter `self._game_files`
        # AND the underlying alfworld env's game_files to only those entries.
        # Used to shrink the TRAIN split for faster training iteration (e.g.
        # a type-balanced subset). Filter is applied once per bridge
        # lifetime; harmless on subsequent resets.
        train_subset_path = opts.get("train_subset_path")
        # subset_authoritative: when True, the subset list REPLACES the env's
        # gamefiles with exactly the (existence-verified, file-ordered) paths in
        # the subset -- including paths NOT in alfworld's scanned data dir. This
        # is required for corpora whose mutated game COPIES live under a fresh
        # parent dir outside alfworld's scanned data dir and so are absent
        # from the scanned set; the default intersect-filter would silently
        # drop them. Default False preserves the REPLACE / corpus-gen behavior.
        subset_authoritative = bool(opts.get("subset_authoritative", False))

        # (re)build env if first reset or split changed
        if self._env is None or split != self._current_split:
            self._lazy_init_env(config_path, split)
            if train_subset_path:
                self._apply_train_subset(
                    train_subset_path, authoritative=subset_authoritative)

        # Task selection: explicit task_id beats seed-based selection.
        task_id = opts.get("task_id") or ""
        pinned = False
        if task_id:
            wanted = next((gf for gf in self._game_files
                           if gf == task_id or gf.endswith(task_id)), None)
            if wanted is not None:
                # Pin the selection by narrowing the env's gamefile list to
                # exactly [wanted] for this reset (the same mechanism
                # _apply_train_subset uses). Seeding does NOT select by
                # index: TextworldBatchGymEnv.seed(n) SHUFFLES the gamefile
                # order and reset() plays the shuffled head, so the previous
                # `_safe_seed(idx)` approach played a deterministic but
                # UNRELATED game while info["task_id"] reported the
                # requested one.
                if not self._set_env_gamefiles([wanted]):
                    raise RuntimeError(
                        f"AlfworldEnv.reset(task_id={task_id!r}): no known "
                        f"gamefiles attribute on {type(self._env).__name__}; "
                        "cannot pin the requested game."
                    )
                self._safe_seed(0)   # rebuild the iterator from the pinned list
                pinned = True
            elif seed is not None:
                # unknown task_id -> fall back to seed-based selection
                self._safe_seed(seed)
        elif seed is not None:
            self._safe_seed(seed)

        ob, info = self._env.reset()
        if pinned:
            # Restore the full list so later seed-mode resets on this same
            # instance sample from the whole split again.
            self._set_env_gamefiles(list(self._game_files))
        obs_text = ob[0] if isinstance(ob, (list, tuple)) else ob
        info = self._unwrap_info(info)
        admissible = self._extract_admissible(info)
        goal_text = self._extract_goal(obs_text)
        max_score = float(info.get("max_score", 1.0) or 1.0)

        self.state = AlfworldEnvState(
            goal_text=goal_text,
            obs_text=obs_text,
            admissible_commands=admissible,
            score=0.0,
            max_score=max_score,
            step_count=0,
            won=False,
            done=False,
            last_action_was_effective=True,
            repetition_count=1,
            last_obs_for_repeat=obs_text,
        )
        return EnvResetResponse(
            observation=self._observe(),
            info={
                "task_id": task_id,
                "max_score": max_score,
                "won": False,
                "extra.gamefile": info.get("extra.gamefile"),
                "admissible_commands": list(admissible),
                "goal_condition_success_rate":
                    info.get("goal_condition_success_rate", 0.0),
            },
        )

    def step(self, action: Action) -> EnvResponse:
        # Bypass tool_registry dispatch: the TextWorld engine is held by the
        # Bridge, not by env_state, so we can't route through Tool.invoke.
        if action.name != Do.name:
            return EnvResponse(
                observation=Observation(
                    text=f"[unknown tool: {action.name}]",
                    data={"error": "unknown_tool",
                          "admissible_commands": list(self.state.admissible_commands)},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "unknown_tool"},
            )

        text = (action.kwargs.get("text") or "").strip()
        if not text:
            return EnvResponse(
                observation=Observation(
                    text="[empty command]",
                    data={"error": "empty_command",
                          "admissible_commands": list(self.state.admissible_commands)},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "empty_command"},
            )

        self.state.step_count += 1
        ob, scores, dones, infos = self._env.step([text])
        obs_text = ob[0] if isinstance(ob, (list, tuple)) else ob
        score = float(scores[0] if isinstance(scores, (list, tuple)) else scores)
        engine_done = bool(dones[0] if isinstance(dones, (list, tuple)) else dones)
        info = self._unwrap_info(infos)
        admissible = self._extract_admissible(info)

        # "Nothing happens" -> action was a no-op (AgentBench heuristic)
        effective = "nothing happens" not in obs_text.lower()

        # Repetition: identical text 3 times in a row truncates the episode.
        if obs_text == self.state.last_obs_for_repeat:
            self.state.repetition_count += 1
        else:
            self.state.repetition_count = 1
            self.state.last_obs_for_repeat = obs_text
        truncated = (self._repetition_threshold > 0
                     and self.state.repetition_count >= self._repetition_threshold)

        won = bool(info.get("won", False))
        max_score = float(info.get("max_score", self.state.max_score) or self.state.max_score)

        # Update state BEFORE building the obs so _observe sees the new values.
        self.state.obs_text = obs_text
        self.state.admissible_commands = admissible
        self.state.score = score
        self.state.max_score = max_score
        self.state.won = won
        self.state.done = engine_done or truncated
        self.state.last_action_was_effective = effective

        # Reward is binary success by default; Rules can reshape via R hook.
        # Raw score is still in info for objectives that want it.
        reward = 1.0 if won else 0.0

        return EnvResponse(
            observation=self._observe(),
            reward=reward,
            terminated=engine_done,
            truncated=truncated,
            info={
                # Same shape as toy24: `success` (None until terminated),
                # `result` (the per-step payload), plus alfworld extras.
                "success": (won if (engine_done or truncated) else None),
                "result": {
                    "text": obs_text,
                    "won": won,
                    "score": score,
                    "effective": effective,
                    "admissible_commands": list(admissible),
                },
                "score": score,
                "max_score": max_score,
                "effective": effective,
                "admissible_commands": list(admissible),
                "repetition_count": self.state.repetition_count,
                # Pass-throughs that the upstream AlfWorldEnvironmentManager
                # consumes for per-task-type SR breakdown / reward shaping.
                "won": won,
                "extra.gamefile": info.get("extra.gamefile"),
                "goal_condition_success_rate":
                    info.get("goal_condition_success_rate", 0.0),
            },
        )

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(
            success=self.state.won,
            score=self.state.score,
            metrics={
                "steps": self.state.step_count,
                "max_score": self.state.max_score,
                "won": self.state.won,
                "truncated_by_repetition": (
                    self._repetition_threshold > 0
                    and self.state.repetition_count >= self._repetition_threshold
                    and not self.state.won
                ),
            },
        )

    def get_env_state(self) -> AlfworldEnvState:
        return self.state

    def observe(self) -> Observation:
        return self._observe()

    @classmethod
    def env_state_schema(cls) -> str:
        return (
            "env_state is an AlfworldEnvState dataclass with these fields:\n"
            "  goal_text: str                 -- natural-language task goal\n"
            "  obs_text: str                  -- current ALFWorld observation text\n"
            "  admissible_commands: list[str] -- valid commands at this step\n"
            "                                    (TextWorld backend always supplies this)\n"
            "  inventory: list[str]           -- not auto-populated; reserved for\n"
            "                                    Rules-supplied state\n"
            "  won: bool                      -- True iff the goal has been achieved\n"
            "  done: bool                     -- True iff the episode terminated\n"
            "                                    (engine done OR repetition-truncated)\n"
            "  score: float                   -- raw ALFWorld score so far\n"
            "  max_score: float               -- max achievable score this task\n"
            "  step_count: int                -- actions taken so far\n"
            "  last_action_was_effective:bool -- False iff env printed 'Nothing happens'\n"
            "  repetition_count: int          -- consecutive identical observations;\n"
            "                                    Bridge truncates at 3 to prevent loops\n"
            "  last_obs_for_repeat: str       -- internal: previous obs for repeat check\n"
            "  extras: dict                   -- free dict for Rules per-episode state\n"
            "S0 changes go through in_env_actions (Setup layer), not Rules. Rules hooks may read all\n"
            "fields and write to `extras`. Modifying obs_text in filter_observation is\n"
            "done by returning a new Observation, NOT by writing env_state.obs_text."
        )

    # -- save / load --------------------------------------------------------
    #
    # ALFWorld's TextWorld engine can't be cheaply snapshotted, so save/load
    # captures the (seed, options) needed to reconstruct the bench AT EPISODE
    # BOUNDARIES. Loading returns a fresh instance; the caller must call
    # reset(seed, options) to actually boot the engine.
    #
    # Mid-episode action history is the Setup harness's responsibility -- the
    # persistence walker stores it separately in the harnesses[] list.

    def save_state(self) -> dict:
        return {
            "reset_seed":    self._last_reset_seed,
            "reset_options": dict(self._last_reset_options),
        }

    @classmethod
    def from_state(cls, state: dict) -> "AlfworldEnv":
        env = cls()
        env._last_reset_seed = state.get("reset_seed")
        env._last_reset_options = dict(state.get("reset_options") or {})
        return env

    def default_reset_args(self) -> tuple[int | None, dict]:
        return self._last_reset_seed, dict(self._last_reset_options)

    def notify_replay_complete(self) -> None:
        """Setup finished replaying its S0 trajectory: rewind per-episode
        counters so the replay is not charged against the policy.

        Without this, a replayed action list whose steps echoed identical
        observations (e.g. an inadmissible command answered by "Nothing
        happens" 3x) trips the repetition guard BEFORE the policy's first
        action, and `step_count` starts non-zero, skewing step statistics.
        The replayed WORLD state itself is kept -- only the bookkeeping
        resets (same semantics the pre-refactor execute_in_env_mutation
        had). `state.done` is left alone: it is recomputed from fresh
        counters on the next step."""
        self.state.step_count = 0
        self.state.repetition_count = 1
        self.state.last_obs_for_repeat = self.state.obs_text

    # -- agent-driven task selection --------------------------------------

    @classmethod
    def list_tasks(cls, reset_options: dict | None = None,
                    limit: int | None = None) -> list[TaskSummary]:
        """Enumerate the alfworld task pool for the given split. Each
        TaskSummary's brief is the task's goal text (e.g. "put a
        spraybottle in cabinet"), extracted by booting a temporary
        Bridge and reading the game file paths from the env."""
        opts = dict(reset_options or {})
        n = limit if limit is not None else 20
        # Build a one-off bridge instance to consult the game file list.
        b = cls()
        try:
            b._lazy_init_env(opts.get("config_path") or os.environ.get("ALFWORLD_CONFIG"),
                              opts.get("split", "eval_in_distribution"))
            game_files = list(b._game_files or [])
        finally:
            b.close()
        if not game_files:
            return []
        out: list[TaskSummary] = []
        for i in range(min(n, len(game_files))):
            gf = game_files[i]
            # Extract a short identifier: alfworld game files are paths like
            # ".../json_2.1.1/<split>/<scene>/<task_type>/<...>/<game>.tw-pddl".
            # Use the last 3 path components for a readable handle.
            parts = gf.split(os.sep)
            handle = "/".join(parts[-3:]).replace(".tw-pddl", "")
            # Brief: task_type usually appears in the path; the full goal
            # text isn't available without reset()ing each, which is
            # expensive. The path-derived task_type is good enough for the
            # Rules to recognize broad categories (pick_and_place,
            # look_at_obj_in_light, pick_clean_then_place, etc.).
            task_type = next((p for p in parts if "pick_" in p or "look_" in p), "?")
            out.append(TaskSummary(
                task_idx=i,
                instance_id=handle,
                brief=f"task_type={task_type}  file={parts[-1].replace('.tw-pddl','')}",
                metadata={"split": opts.get("split", "eval_in_distribution"),
                          "game_file": gf},
            ))
        return out

    # -- close --------------------------------------------------------------

    def close(self) -> None:
        super().close()
        # ALFWorld leaks memory across many resets; releasing the env between
        # episodes is the cheapest fix (AgentBench docs this in their issues).
        if self._env is not None:
            try:
                close_fn = getattr(self._env, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            self._env = None

    # -- helpers ------------------------------------------------------------

    def _lazy_init_env(self, config_path: str | None, split: str) -> None:
        try:
            import yaml
            import alfworld.agents.environment as env_module
        except ImportError as e:
            raise RuntimeError(
                "ALFWorld is not installed. Run:\n"
                '  pip install "alfworld[full]"\n'
                "  alfworld-download\n"
                f"Underlying ImportError: {e}"
            ) from e

        # Locate config: explicit arg > env var > vendored default (a verbatim
        # copy of alfworld's base_config.yaml, kept under envharness/third_party/ with
        # the rest of the third-party material) > alfworld's bundled config
        # (some installs ship one, some don't).
        if not config_path:
            try:
                import envharness.third_party as _tp
                vendored = os.path.join(os.path.dirname(_tp.__file__),
                                         "alfworld", "base_config.yaml")
            except ImportError:
                vendored = None
            if vendored and os.path.exists(vendored):
                config_path = vendored
        if not config_path:
            try:
                import alfworld
                bundled = os.path.join(
                    os.path.dirname(alfworld.__file__), "..",
                    "configs", "base_config.yaml",
                )
                if os.path.exists(bundled):
                    config_path = bundled
            except Exception:
                pass
        if not config_path or not os.path.exists(config_path):
            raise RuntimeError(
                "Could not locate ALFWorld base_config.yaml. Either set the "
                "ALFWORLD_CONFIG env var or pass reset_options.config_path."
            )

        # Ensure $ALFWORLD_DATA points at the game data directory; the YAML
        # uses '$ALFWORLD_DATA' as a substitution token. Default to alfworld's
        # standard cache location at ~/.cache/alfworld.
        if not os.environ.get("ALFWORLD_DATA"):
            os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Force TextWorld backend (admissible_commands is only available here).
        # alfworld exposes envs via a factory; the class itself lives at
        # alfworld.agents.environment.alfred_tw_env.AlfredTWEnv.
        env_cls = env_module.get_environment("AlfredTWEnv")
        raw_env = env_cls(config, train_eval=split)
        self._env = raw_env.init_env(batch_size=1)
        self._current_split = split
        try:
            self._game_files = list(getattr(self._env, "game_files", None)
                                     or getattr(raw_env, "game_files", []) or [])
        except Exception:
            self._game_files = []

    def _apply_train_subset(self, subset_path: str,
                             authoritative: bool = False) -> None:
        """Filter `self._game_files` AND the underlying env's game_files to
        only those listed in `subset_path` (JSONL with 'game_file' keys).

        ALFWorld's AlfredTWEnv iterates over `game_files`; replacing that list
        in place + replacing our cached copy makes the bridge expose only the
        subset for both seed-based selection and our task_id lookup. Idempotent
        and silent if the file is missing.

        authoritative=True: SET the env's gamefiles to exactly the subset's
        (existence-verified, file-ordered) paths, even paths not present in the
        scanned data dir. Used by the AUGMENT/COMBINE corpus whose mutated
        copies live outside the alfworld cache. authoritative=False (default):
        intersect-filter the scanned list (REPLACE / corpus-gen behavior).
        """
        import json as _json
        import os as _os
        # A bundled subset lists game files relative to the ALFWorld data root,
        # since the absolute location differs per machine. Absolute entries are
        # kept as-is, so subsets generated by a run still work.
        _root = _os.environ.get("ALFWORLD_DATA", "")
        try:
            with open(subset_path) as f:
                ordered = [_json.loads(ln)["game_file"]
                           for ln in f if ln.strip()]
        except FileNotFoundError:
            return
        ordered = [gf if _os.path.isabs(gf) else _os.path.join(_root, gf)
                   for gf in ordered]
        wanted = set(ordered)
        if not wanted:
            return

        if authoritative:
            present = [gf for gf in ordered if _os.path.exists(gf)]
            missing = len(ordered) - len(present)
            if not present:
                raise RuntimeError(
                    f"_apply_train_subset(authoritative): none of the "
                    f"{len(ordered)} subset game_files exist on disk; "
                    f"check {subset_path}."
                )
            env_obj = self._env
            applied = False
            for attr_chain in (("gamefiles",), ("game_files",),
                                ("envs", 0, "gamefiles"),
                                ("envs", 0, "game_files")):
                try:
                    obj = env_obj
                    for part in attr_chain[:-1]:
                        obj = obj[part] if isinstance(part, int) else getattr(obj, part)
                    if getattr(obj, attr_chain[-1], None) is None:
                        continue
                    setattr(obj, attr_chain[-1], list(present))
                    applied = True
                except Exception:
                    pass
            if not applied:
                raise RuntimeError(
                    "_apply_train_subset(authoritative): no known gamefiles "
                    f"attribute on {type(self._env).__name__}."
                )
            try:
                delattr(self._env, "_gamefiles_iterator")
            except Exception:
                pass
            self._game_files = list(present)
            print(f"[AlfworldEnv] authoritative subset: {len(present)} "
                  f"games ({missing} missing skipped)", flush=True)
            return
        # Filter the env's internal list. The actual attribute on
        # TextworldBatchGymEnv is `gamefiles` (no underscore -- this differs
        # from AlfredTWEnv.game_files, and the bridge holds the OUTER
        # batch-gym env). Subsequent env.seed(n) rebuilds the
        # _gamefiles_iterator FROM self.gamefiles, so replacing this attribute
        # is sufficient -- the next seed call picks games from the 432-task
        # filtered set deterministically. We also patch the raw AlfredTWEnv
        # form (`game_files`) and the batched-envs form for safety on
        # alfworld versions that expose them.
        env_obj = self._env
        applied = False
        for attr_chain in (("gamefiles",),                # textworld batch gym
                            ("game_files",),               # raw AlfredTWEnv
                            ("envs", 0, "gamefiles"),      # batched, textworld name
                            ("envs", 0, "game_files")):    # batched, raw name
            try:
                obj = env_obj
                for part in attr_chain[:-1]:
                    obj = obj[part] if isinstance(part, int) else getattr(obj, part)
                cur = list(getattr(obj, attr_chain[-1], None) or [])
                if not cur:
                    continue
                filtered = [gf for gf in cur if gf in wanted]
                if filtered:
                    setattr(obj, attr_chain[-1], filtered)
                    applied = True
            except Exception:
                pass
        if not applied:
            # Surface a clear error rather than silently fall through to
            # full-train sampling. The Rules's per-task baseline would
            # then mutate the wrong task and the corpus would be useless.
            raise RuntimeError(
                f"_apply_train_subset: no known gamefiles attribute on "
                f"{type(self._env).__name__}; cannot filter to the 432-task "
                f"subset. Inspect with `dir(bridge._env)` and add the right "
                f"attr chain to envharness/bridges/alfworld/bridge.py."
            )
        # Best-effort: invalidate the cached _gamefiles_iterator if present so
        # the env can't yield from the old unfiltered list before the next
        # seed() call rebuilds it. seed() is called by the bridge per reset()
        # so this is belt-and-suspenders.
        try:
            delattr(self._env, "_gamefiles_iterator")
        except Exception:
            pass
        self._game_files = [gf for gf in self._game_files if gf in wanted] \
            or list(wanted)

    def _set_env_gamefiles(self, files: list[str]) -> bool:
        """Point the underlying env's gamefile list at `files` and drop its
        cached iterator so the next seed()/reset() rebuilds from the new
        list. Tries the known attribute spellings across alfworld/textworld
        versions (same chains as _apply_train_subset). Returns True if any
        attribute was set."""
        applied = False
        for attr_chain in (("gamefiles",),                # textworld batch gym
                            ("game_files",),               # raw AlfredTWEnv
                            ("envs", 0, "gamefiles"),
                            ("envs", 0, "game_files")):
            try:
                obj = self._env
                for part in attr_chain[:-1]:
                    obj = obj[part] if isinstance(part, int) else getattr(obj, part)
                if getattr(obj, attr_chain[-1], None) is None:
                    continue
                setattr(obj, attr_chain[-1], list(files))
                applied = True
            except Exception:
                pass
        try:
            delattr(self._env, "_gamefiles_iterator")
        except Exception:
            pass
        return applied

    def _safe_seed(self, n: int) -> None:
        """Seed the underlying env's RNG if it supports it; otherwise no-op.

        NOTE: for TextworldBatchGymEnv this is NOT an index into game_files
        -- seed(n) shuffles the gamefile order deterministically and reset()
        plays the shuffled head. Same (seed, split, gamefile list) -> same
        game, which is all seed-mode callers rely on. To select a SPECIFIC
        game, pass reset_options['task_id'] (pinned via _set_env_gamefiles)."""
        try:
            seed_fn = getattr(self._env, "seed", None)
            if callable(seed_fn):
                seed_fn(int(n))
        except Exception:
            pass

    def _observe(self) -> Observation:
        # obs_style switches text composition:
        #   "wrapped" (default, legacy): goal + raw obs + admissibles all in text
        #   "raw": just the raw TextWorld obs (cleanest for callers that build
        #          their own prompt from observation.data)
        # In both modes observation.data carries goal_text + admissible_commands
        # so a structured caller can always read them.
        if self._obs_style == "raw":
            text = self.state.obs_text or ""
        else:
            text_parts: list[str] = []
            if self.state.goal_text:
                text_parts.append(f"Task: {self.state.goal_text}")
            text_parts.append(self.state.obs_text)
            if self.state.admissible_commands:
                text_parts.append(
                    "Admissible commands: "
                    + ", ".join(self.state.admissible_commands)
                )
            text = "\n\n".join(p for p in text_parts if p)
        return Observation(
            text=text,
            data={
                "goal_text": self.state.goal_text,
                "admissible_commands": list(self.state.admissible_commands),
                "step_count": self.state.step_count,
                "score": self.state.score,
                "max_score": self.state.max_score,
                "won": self.state.won,
                "last_action_was_effective": self.state.last_action_was_effective,
            },
        )

    @staticmethod
    def _extract_admissible(info: dict) -> list[str]:
        adm = (info or {}).get("admissible_commands")
        # After _unwrap_info, this is already the inner list (unwrapped from
        # the batch dimension). Guard against legacy shape just in case.
        if isinstance(adm, list) and adm and isinstance(adm[0], list):
            return list(adm[0])
        if isinstance(adm, list):
            return [str(x) for x in adm]
        return []

    @staticmethod
    def _extract_goal(obs_text: str) -> str:
        # ALFWorld prepends the task description to the initial obs as one of
        # the trailing lines starting with "Your task is to ...".
        for line in (obs_text or "").split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("your task is"):
                return stripped
        return ""

    @staticmethod
    def _unwrap_info(info: Any) -> dict:
        """ALFWorld returns info as dict[str, list-of-batch]. Unwrap batch=0."""
        if not info:
            return {}
        if isinstance(info, dict):
            out = {}
            for k, v in info.items():
                if isinstance(v, list) and len(v) > 0:
                    out[k] = v[0]
                else:
                    out[k] = v
            return out
        return {}
