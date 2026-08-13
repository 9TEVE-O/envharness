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

"""Ray-actor parallel ALFWorld envs for verl-agent GRPO, on envharness.

Mirrors the public API verl-agent's `AlfworldEnvs` expects:
  - `reset() -> (text_obs_list, image_obs_list=None, info_list)`
  - `step(actions) -> (text_obs_list, image_obs_list=None, rewards, dones, info_list)`
  - `get_admissible_commands` property
  - `close()`

What changed vs the legacy `envharness.rl.envharness_alfworld` adapter:

  - Base env is `envharness.bridges.alfworld:AlfworldEnv` (an
    ActionableEnv), NOT the old `AlfworldBridge`.
  - Mutations are envharness harnesses, not `MutationLayer`:
      * A/T/O perturbations  -> `Rules` subclass loaded from `rules_code`
        (`envharness.core.code_loader:load_rules_subclass`).
      * S0 (in_env_actions)  -> `Edits`-style action replay. We replay the
        action list IN PLACE on the already-reset base env rather than
        wrapping in an `Edits` layer, because the base `AlfworldEnv` is a
        persistent TextWorld engine whose `reset()` ADVANCES the game
        queue. Wrapping then resetting would skip a game; replaying in
        place keeps the worker landed on the game we looked the mutation
        up for.

  The composed per-step stack is therefore `Rules(inner=AlfworldEnv)` with
  S0 already applied, and `rules.step(action)` runs filter_action ->
  inner.step -> modify_transition -> filter_observation for us (Blocked is
  handled inside `Rules.step`). No manual hook orchestration here.

Each Ray actor owns one `AlfworldEnv`. TextWorld's PDDL grammar is not
thread-safe, so actors keep each engine in its own process. CPU-only -- no
GPU contention with vLLM rollout.
"""
from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path
from typing import Any

import gymnasium as gym
import ray

from envharness.bridges.alfworld import AlfworldEnv
from envharness.core.code_loader import RulesCodeError, load_rules_subclass
from envharness.core.types import Action, EnvResponse, Observation
from envharness.harnesses.rules import Rules


class EnvharnessAlfworldWorker:
    """One Ray actor = one persistent `AlfworldEnv` (the game queue).

    First reset() seeds the underlying TextWorld engine with `seed`;
    subsequent reset()s pass no seed so the engine's internal iterator
    advances through the (seeded-shuffled) game queue -- same pattern as
    verl-agent's stock AlfworldWorker.
    """

    def __init__(self, seed: int, split: str, repetition_threshold: int = 0,
                 obs_style: str = "raw",
                 train_subset_path: str | None = None,
                 mutation_corpus_path: str | None = None,
                 subset_authoritative: bool = False):
        self._seed = int(seed)
        self._split = split
        self._rep_threshold = int(repetition_threshold)
        if obs_style not in ("raw", "wrapped"):
            raise ValueError(
                f"obs_style must be 'raw' or 'wrapped', got {obs_style!r}")
        self._obs_style = obs_style
        self._train_subset_path = train_subset_path
        # mutation_corpus_path: JSONL produced by envharness_rl.corpus. Each
        # line: {"game_file": ..., "rules_code": "<class _Rules(Rules)...>",
        #        "in_env_actions": [{"name": "do", "kwargs": {"text": ...}}]}.
        # Games NOT in the corpus get a pass-through (unmutated) env. No file
        # = pure unmutated env (control baseline).
        self._mutation_corpus_path = mutation_corpus_path
        self._subset_authoritative = bool(subset_authoritative)
        self._corpus: dict[str, dict] = {}
        if mutation_corpus_path:
            self._load_corpus(mutation_corpus_path)

        self._env = AlfworldEnv()
        # Per-episode harness, rebuilt on every reset() from the game we land
        # on. None when the active game has no rules_code -- step() then
        # targets the bare env (pass-through).
        self._rules: Rules | None = None
        self._rules_error: str | None = None
        self._first_reset = True
        self._last_text = ""
        self._last_admissible: list[str] = []
        self._last_gamefile: str | None = None

    # ------------------------------------------------------------------
    def _load_corpus(self, path: str) -> None:
        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"mutation_corpus_path not found: {path}. Generate via "
                f"envharness_rl.corpus (scripts/build_corpus.py) from a "
                f"envharness alfworld corpus run's traces.jsonl, or "
                f"unset to fall back to the unmutated env.")
        n = 0
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            gf = rec.get("game_file")
            if not gf:
                continue
            # Bundled corpora store game files relative to the ALFWorld data
            # root; the env keys off absolute paths.
            if not _os.path.isabs(gf):
                gf = _os.path.join(_os.environ.get("ALFWORLD_DATA", ""), gf)
                rec["game_file"] = gf
            self._corpus[gf] = rec
            n += 1
        print(f"[EnvharnessAlfworldWorker] loaded mutation corpus: "
              f"{n} entries from {path}", flush=True)

    def _reset_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "split": self._split,
            "repetition_threshold": self._rep_threshold,
            "obs_style": self._obs_style,
        }
        if self._train_subset_path:
            opts["train_subset_path"] = self._train_subset_path
            if self._subset_authoritative:
                opts["subset_authoritative"] = True
        return opts

    # ------------------------------------------------------------------
    def reset(self) -> tuple[str, dict]:
        reset_resp = self._env.reset(
            seed=(self._seed if self._first_reset else None),
            options=self._reset_options(),
        )
        self._first_reset = False
        info = dict(reset_resp.info or {})
        obs: Observation = reset_resp.observation

        # Capture the canonical task line from the UNMUTATED initial obs.
        # verl-agent's extract_task hard-requires the "Your task is to: "
        # substring; S0 replay / an O-axis filter can drop it, so we re-inject
        # below if needed.
        task_line = (obs.data or {}).get("goal_text", "") if obs is not None else ""

        active_gamefile = info.get("extra.gamefile")
        self._last_gamefile = active_gamefile
        self._rules = None
        self._rules_error = None
        rec = self._corpus.get(active_gamefile) if active_gamefile else None

        if rec:
            # S0: replay in_env_actions in place on the already-reset base env
            # (the Edits mechanism, applied without a second reset).
            for a in rec.get("in_env_actions") or []:
                try:
                    self._env.step(a if isinstance(a, Action) else Action(**a))
                except Exception as e:  # noqa: BLE001 -- log + continue
                    print(f"[EnvharnessAlfworldWorker] in_env_actions failed "
                          f"for {active_gamefile}: {e}", flush=True)
            # A/T/O: wrap the base env in the Rules subclass (no reset).
            code = (rec.get("rules_code") or "").strip()
            if code:
                try:
                    self._rules = Rules.from_state(
                        {"rules_code": code}, inner=self._env)
                except RulesCodeError as e:
                    self._rules = None
                    self._rules_error = f"RulesCodeError: {e}"
                    print(f"[EnvharnessAlfworldWorker] rules load failed for "
                          f"{active_gamefile}: {e}", flush=True)
            # Initial obs after S0 (+ O-filter when a Rules layer is active).
            if self._rules is not None:
                try:
                    obs = self._rules.observe()
                except Exception as e:  # noqa: BLE001
                    obs = self._env.observe()
                    print(f"[EnvharnessAlfworldWorker] rules.observe at reset "
                          f"failed for {active_gamefile}: {e}", flush=True)
            else:
                obs = self._env.observe()

        text = obs.text or ""
        if task_line and "Your task is to:" not in text:
            text = f"{task_line}\n\n{text}" if text else task_line
        admissible = (obs.data or {}).get("admissible_commands") or info.get(
            "admissible_commands", [])

        info["admissible_commands"] = list(admissible)
        info.setdefault("won", False)
        info.setdefault("extra.gamefile", None)
        info.setdefault("goal_condition_success_rate", 0.0)
        info["mutation_active"] = bool(self._rules is not None)
        if self._rules_error:
            info["mutation_error"] = self._rules_error
        wrapped_info = {k: [v] for k, v in info.items()}
        wrapped_info["observation_text"] = [text]
        self._last_text = text
        self._last_admissible = list(admissible)
        return text, wrapped_info

    # ------------------------------------------------------------------
    def step(self, action: str) -> tuple[str, float, bool, dict]:
        act = Action(name="do", kwargs={"text": str(action or "")})
        # Rules.step internally runs filter_action (incl. Blocked) ->
        # inner.step -> modify_transition -> filter_observation. The bare env
        # is stepped directly when no Rules layer is active.
        target = self._rules if self._rules is not None else self._env
        try:
            env_resp: EnvResponse = target.step(act)
        except Exception as e:  # noqa: BLE001
            print(f"[EnvharnessAlfworldWorker] step crashed: "
                  f"{type(e).__name__}: {e} -- returning 'Nothing happens'",
                  flush=True)
            env_resp = EnvResponse(
                observation=Observation(text="Nothing happens."),
                reward=0.0, terminated=False, truncated=False,
                info={"won": False,
                      "admissible_commands": list(self._last_admissible),
                      "extra.gamefile": self._last_gamefile,
                      "goal_condition_success_rate": 0.0})

        obs = env_resp.observation
        text = obs.text or ""
        info = dict(env_resp.info or {})
        admissible = (obs.data or {}).get("admissible_commands") or info.get(
            "admissible_commands", [])

        # Resolve `won` defensively: AlfworldEnv.step always sets a bool, but a
        # Mutator-emitted modify_transition can strip/corrupt info. Recover
        # from the legacy nested shape, else warn + default False (downstream
        # reward is 10*float(won) -- a silently-wrong won corrupts training).
        if not isinstance(info.get("won"), bool):
            recovered = None
            result = info.get("result")
            if isinstance(result, dict) and isinstance(result.get("won"), bool):
                recovered = result["won"]
            if recovered is not None:
                info["won"] = recovered
            else:
                info["won"] = False
        info.setdefault("extra.gamefile", self._last_gamefile)
        info.setdefault("goal_condition_success_rate", 0.0)
        info["admissible_commands"] = list(admissible)
        info["mutation_active"] = bool(self._rules is not None)

        wrapped_info = {k: [v] for k, v in info.items()}
        wrapped_info["observation_text"] = [text]
        score = float(env_resp.reward or 0.0)
        done = bool(env_resp.terminated) or bool(env_resp.truncated)
        self._last_text = text
        self._last_admissible = list(admissible)
        return text, score, done, wrapped_info

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:  # noqa: BLE001
            pass


class EnvharnessAlfworldEnvs(gym.Env):
    """Parallel wrapper matching verl-agent's `AlfworldEnvs` API."""

    def __init__(self, alf_config_path, seed: int, env_num: int, group_n: int,
                 resources_per_worker, is_train: bool = True,
                 env_kwargs: dict | None = None):
        super().__init__()
        env_kwargs = env_kwargs or {}
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        if is_train:
            split = "train"
        else:
            split = env_kwargs.get("eval_dataset", "eval_in_distribution")
        if alf_config_path:
            import os as _os
            _os.environ.setdefault("ALFWORLD_CONFIG", alf_config_path)

        self.num_processes = env_num * group_n
        self.group_n = group_n
        self._split = split

        obs_style = env_kwargs.get("obs_style", "raw")
        # TRAIN-ONLY knobs (eval uses the vanilla env so SR stays comparable
        # to the unmutated baseline).
        train_subset_path = env_kwargs.get("train_subset_path") if is_train else None
        mutation_corpus_path = (
            env_kwargs.get("mutation_corpus_path") if is_train else None)
        subset_authoritative = (
            bool(env_kwargs.get("subset_authoritative", False)) if is_train else False)

        worker_cls = ray.remote(**(resources_per_worker or {}))(
            EnvharnessAlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            # All members of one group share a seed -> same task on first
            # reset (matches verl-agent / GiGPO recipe).
            base_seed = seed + (i // max(group_n, 1))
            self.workers.append(worker_cls.remote(
                seed=base_seed, split=split, repetition_threshold=0,
                obs_style=obs_style, train_subset_path=train_subset_path,
                mutation_corpus_path=mutation_corpus_path,
                subset_authoritative=subset_authoritative))

        self.prev_admissible_commands = [None] * self.num_processes

    def reset(self):
        results = ray.get([w.reset.remote() for w in self.workers])
        text_obs_list, info_list = [], []
        for i, (text, info) in enumerate(results):
            for k in list(info.keys()):
                info[k] = info[k][0]
            text_obs_list.append(text)
            self.prev_admissible_commands[i] = info["admissible_commands"]
            info_list.append(info)
        return text_obs_list, None, info_list

    def step(self, actions):
        assert len(actions) == self.num_processes, (
            f"actions={len(actions)} != num_processes={self.num_processes}")
        results = ray.get(
            [w.step.remote(a) for w, a in zip(self.workers, actions)])
        text_obs_list, rewards_list, dones_list, info_list = [], [], [], []
        for i, (text, score, done, info) in enumerate(results):
            for k in list(info.keys()):
                info[k] = info[k][0]
            text_obs_list.append(text)
            dones_list.append(done)
            self.prev_admissible_commands[i] = info["admissible_commands"]
            # verl-agent's compute_reward = 10.0 * float(info['won']).
            rewards_list.append(10.0 * float(info.get("won", False)))
            info_list.append(info)
        return text_obs_list, None, rewards_list, dones_list, info_list

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands

    def close(self):
        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:  # noqa: BLE001
                pass


def build_envharness_alfworld_envs(alf_config_path, seed, env_num, group_n,
                                   resources_per_worker, is_train=True,
                                   env_kwargs=None):
    return EnvharnessAlfworldEnvs(alf_config_path, seed, env_num, group_n,
                                  resources_per_worker, is_train, env_kwargs)
