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

"""EpisodeRunner -- run ONE episode given (EnvSpec, Candidate, PolicySpec).

The runner builds an ActionableEnv stack and runs a clean step loop:

    base = ImportedEnvClass()
    if candidate.in_env_actions: env = Setup(inner=base, actions=...)
    if candidate.rules_code:  env = compile_Mutator_subclass(inner=env)
    env.reset(...) -> obs
    for step in range(max_steps):
        a = policy.act(obs)
        resp = env.step(a)               # Rules.step runs the 6 hooks inside
        ... record Step ...
        obs = resp.observation
    eval_result = env.evaluate()

The 5 mutation hooks (S0/A/O/T/R) and the step_reward hook fire INSIDE the
env stack, so the runner never calls them explicitly.

Two runners:
- InProcessRunner: runs in this process. Cheap; no isolation. For tests.
- SubprocessRunner: spawn a fresh Python subprocess per episode. Rules
  code crashes don't take down the loop. Bridges with external runtime
  (Docker, browser, sim) rely on `env.close()` (in the run_episode
  finally) for cleanup since subprocess death does NOT release child
  resources.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envharness.agents.policy import ActionFormat, PolicyAgent
from envharness.core.actionable_env import ActionableEnv
from envharness.core.code_loader import RulesCodeError, load_rules_instance
from envharness.core.envharness import EnvHarness
from envharness.core.types import (
    Candidate, Step, StepInfo, Trace,
)
from envharness.harnesses.setup import Setup
from envharness.infra.llm import LLMClient  # noqa: F401  (kept for downstream typing)
from envharness.infra.utils import import_symbol as _import_symbol


# ---------------------------------------------------------------------------
# Specs -- JSON-serializable so SubprocessRunner can reconstruct them.
# ---------------------------------------------------------------------------

@dataclass
class EnvSpec:
    """Identifies a base ActionableEnv class + reset options.

    `import_path` is "pkg.module:ClassName"; the worker `import_symbol`s it
    to materialize the class. The import has the side effect of running the
    class's `@register_env(tag)` decorator so the registry is populated --
    this is what makes `env.env_type()` work for save/load round-trips on
    the resulting env stack.
    """
    import_path: str
    reset_options: dict[str, Any] = field(default_factory=dict)
    reset_seed: int | None = None


@dataclass
class PolicySpec:
    """How to build the PolicyAgent. The LLMClient factory is addressed by
    import path so subprocesses can rebuild it."""
    client_factory: str                              # "envharness.infra.llm:LiteLLMClient"
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    action_format: str = "function_calling"
    task_prompt: str = ""
    max_history: int = 50
    temperature: float = 0.4
    """K rollouts under the same candidate produce different trajectories
    only via sampling variance (env+actions deterministic given seed).
    Default 0.4 yields meaningful K=5 spread; set 0.0 for greedy single-
    rollout eval matches."""
    prompt_builder_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeSpec:
    env: EnvSpec
    candidate: Candidate
    policy: PolicySpec
    iteration_id: str
    task_id: str = ""
    max_steps: int = 50


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_env_stack(spec: EpisodeSpec) -> ActionableEnv:
    """Build the ActionableEnv from the spec + candidate.

    Order is innermost-first (matches the persistence layout):

        base env  ──>  Setup (if in_env_actions)  ──>  Rules (if rules_code)
    """
    base_cls = _import_symbol(spec.env.import_path)
    env: ActionableEnv = base_cls()
    cand = spec.candidate
    if cand.in_env_actions:
        env = Setup(inner=env, actions=list(cand.in_env_actions))
    if (cand.rules_code or "").strip():
        env = load_rules_instance(cand.rules_code, inner=env)
    return env


def _build_policy(spec: PolicySpec, tools: list[dict]) -> PolicyAgent:
    client_cls = _import_symbol(spec.client_factory)
    client = client_cls(**spec.client_kwargs)
    return PolicyAgent(
        client=client, tools=tools,
        task_prompt=spec.task_prompt,
        action_format=ActionFormat(spec.action_format),
        max_history=spec.max_history,
        temperature=spec.temperature,
        prompt_builder_kwargs=spec.prompt_builder_kwargs or None,
    )


def _base_env(env: ActionableEnv) -> ActionableEnv:
    """Walk down the harness stack and return the innermost non-harness env."""
    cur: ActionableEnv = env
    while isinstance(cur, EnvHarness):
        cur = cur.inner
    return cur


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(spec: EpisodeSpec) -> Trace:
    """In-process episode loop. Used by InProcessRunner and inside the
    subprocess by SubprocessRunner.

    `env.close()` is always called in a `finally`, regardless of which
    early-return path the episode takes. Subprocess death does NOT release
    child runtime resources (Docker containers, browsers, sim engines), so
    ActionableEnv implementations with external state rely on this contract.
    """
    episode_id = uuid.uuid4().hex[:10]

    # Build the env stack. RulesCodeError surfaces a compile error before
    # we boot any bridge runtime, so a malformed rules_code returns a
    # cheap Trace(error=...) without paying the docker/playwright cost.
    try:
        env = build_env_stack(spec)
    except RulesCodeError as e:
        return Trace(
            episode_id=episode_id, iteration_id=spec.iteration_id,
            task_id=spec.task_id, candidate=spec.candidate,
            rollout_seed=spec.env.reset_seed,
            error=f"RulesCodeError: {e}",
        )

    policy = None
    try:
        try:
            reset = env.reset(
                seed=spec.env.reset_seed,
                options={**spec.env.reset_options, "task_id": spec.task_id},
            )
        except Exception as e:
            return Trace(
                episode_id=episode_id, iteration_id=spec.iteration_id,
                task_id=spec.task_id, candidate=spec.candidate,
                rollout_seed=spec.env.reset_seed,
                error=f"env.reset raised: {type(e).__name__}: {e}",
            )

        # Tool schemas live on the base env class (EnvHarness inherits empty).
        base = _base_env(env)
        policy = _build_policy(spec.policy, type(base).tool_schemas())
        policy.reset()

        obs = reset.observation
        steps: list[Step] = []
        cumulative_reward = 0.0

        def _stamp_process_reward(step: Step) -> None:
            """Compute env.step_reward(StepInfo(...)) and stamp on
            `step.process_reward`. NON-FATAL: exceptions land on
            `step.process_reward_error`, the episode continues."""
            try:
                r = env.step_reward(StepInfo(
                    step=step, step_number=len(steps), history=list(steps),
                ))
                step.process_reward = float(r)
            except Exception as e:
                step.process_reward_error = f"{type(e).__name__}: {e}"

        for _ in range(spec.max_steps):
            try:
                raw_action = policy.act(obs)
            except Exception as e:
                # An LLM/client failure mid-episode must yield a Trace that
                # keeps the steps already taken (and hits the finally-close),
                # not escape run_episode -- under InProcessRunner an escape
                # would bubble into the orchestrator's per-task catch and
                # cost the whole task.
                return Trace(
                    episode_id=episode_id, iteration_id=spec.iteration_id,
                    task_id=spec.task_id, candidate=spec.candidate, steps=steps,
                    rollout_seed=spec.env.reset_seed,
                    duration_steps=len(steps),
                    error=f"policy.act raised: {type(e).__name__}: {e}",
                )
            try:
                env_resp = env.step(raw_action)
            except Exception as e:
                return Trace(
                    episode_id=episode_id, iteration_id=spec.iteration_id,
                    task_id=spec.task_id, candidate=spec.candidate, steps=steps,
                    rollout_seed=spec.env.reset_seed,
                    duration_steps=len(steps),
                    error=f"env.step raised: {type(e).__name__}: {e}",
                )

            # Rules-blocked actions surface as observation.data["blocked"]
            # and info["blocked_reason"]; capture these on the Step record.
            blocked = bool((env_resp.observation.data or {}).get("blocked"))
            blocked_reason = (env_resp.info or {}).get("blocked_reason")

            step_record = Step(
                raw_action=raw_action,
                filtered_action=None if blocked else raw_action,
                blocked_reason=blocked_reason,
                raw_observation=env_resp.observation,
                filtered_observation=env_resp.observation,
                raw_reward=env_resp.reward,
                filtered_reward=env_resp.reward,
                terminated=env_resp.terminated,
                truncated=env_resp.truncated,
                info=dict(env_resp.info or {}),
                policy_raw_response=getattr(policy, "last_raw_response", None) or None,
            )
            _stamp_process_reward(step_record)
            steps.append(step_record)
            cumulative_reward += env_resp.reward
            obs = env_resp.observation
            if env_resp.terminated or env_resp.truncated:
                break

        eval_result = env.evaluate()
        return Trace(
            episode_id=episode_id, iteration_id=spec.iteration_id,
            task_id=spec.task_id, candidate=spec.candidate, steps=steps,
            rollout_seed=spec.env.reset_seed,
            final_reward=cumulative_reward,
            success=eval_result.success,
            duration_steps=len(steps),
            policy_model_id=getattr(policy.client, "model_id", "unknown") if policy else "unknown",
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Runner ABC + implementations
# ---------------------------------------------------------------------------

class EpisodeRunner(ABC):
    @abstractmethod
    def run(self, spec: EpisodeSpec) -> Trace: ...


class InProcessRunner(EpisodeRunner):
    """Run in this process. No isolation. Used for tests and offline
    ScriptedClient flows."""
    def run(self, spec: EpisodeSpec) -> Trace:
        return run_episode(spec)


class SubprocessRunner(EpisodeRunner):
    """Spawn a Python subprocess per episode. Rules code crashes don't
    take down the loop. Subprocess death does NOT auto-release child
    runtime state (Docker containers, browsers, sim engines);
    ActionableEnv implementations with external state must do their
    cleanup in `close()`, which `run_episode`'s try/finally always calls.
    """

    def __init__(self, python: str | None = None, timeout: float = 600.0,
                  subprocess_log_dir: str | Path | None = None):
        self.python = python or sys.executable
        self.timeout = timeout
        self.subprocess_log_dir = Path(subprocess_log_dir) if subprocess_log_dir else None
        if self.subprocess_log_dir:
            self.subprocess_log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, spec: EpisodeSpec) -> Trace:
        spec_json = json.dumps({
            "env": {"import_path": spec.env.import_path,
                    "reset_options": spec.env.reset_options,
                    "reset_seed": spec.env.reset_seed},
            "candidate": spec.candidate.model_dump(),
            "policy": {"client_factory": spec.policy.client_factory,
                       "client_kwargs": spec.policy.client_kwargs,
                       "action_format": spec.policy.action_format,
                       "task_prompt": spec.policy.task_prompt,
                       "max_history": spec.policy.max_history,
                       "temperature": spec.policy.temperature,
                       "prompt_builder_kwargs": spec.policy.prompt_builder_kwargs},
            "iteration_id": spec.iteration_id,
            "task_id": spec.task_id,
            "max_steps": spec.max_steps,
        })
        attempt_id = uuid.uuid4().hex[:10]
        # Launch the worker by ABSOLUTE FILE PATH (not `-m envharness...`):
        # with `-m`, the child's sys.path[0] is its CWD, which silently wins
        # over PYTHONPATH -- so a driver launched from a directory containing
        # a different `envharness/` tree (e.g. a repo root with a sibling dev
        # tree) imports the wrong code and every episode dies with
        # `subprocess exit 1`. Script invocation puts the script's own dir at
        # sys.path[0] instead of CWD, and _child_env() pins the package root.
        worker_path = Path(__file__).resolve().parent / "episode_worker.py"
        try:
            proc = subprocess.run(
                [self.python, str(worker_path)],
                input=spec_json, capture_output=True, text=True,
                timeout=self.timeout,
                env=self._child_env(),
            )
        except subprocess.TimeoutExpired as e:
            self._maybe_save_stderr(attempt_id, getattr(e, "stderr", "") or "")
            return Trace(
                episode_id=attempt_id,
                iteration_id=spec.iteration_id, task_id=spec.task_id,
                candidate=spec.candidate, error="subprocess timeout",
                rollout_seed=spec.env.reset_seed,
            )
        self._maybe_save_stderr(attempt_id, proc.stderr or "")
        if proc.returncode != 0:
            return Trace(
                episode_id=attempt_id, iteration_id=spec.iteration_id,
                task_id=spec.task_id, candidate=spec.candidate,
                error=f"subprocess exit {proc.returncode}",
                subprocess_stderr=proc.stderr[-2000:],
                rollout_seed=spec.env.reset_seed,
            )
        last_line = proc.stdout.strip().split("\n")[-1] if proc.stdout.strip() else ""
        if not last_line:
            return Trace(
                episode_id=attempt_id, iteration_id=spec.iteration_id,
                task_id=spec.task_id, candidate=spec.candidate,
                error="subprocess produced no Trace JSON",
                subprocess_stderr=proc.stderr[-2000:],
                rollout_seed=spec.env.reset_seed,
            )
        trace = Trace.model_validate_json(last_line)
        if proc.stderr:
            trace.subprocess_stderr = proc.stderr[-2000:]
        return trace

    @staticmethod
    def _child_env() -> dict[str, str]:
        """Environment for the episode-worker subprocess.

        Guarantees the child imports the SAME `envharness` package as this
        parent process (see the worker_path comment in `run` for why the
        default module search is not trustworthy here).

        Two knobs:
          - PYTHONPATH gets the parent's package root PREPENDED, so the
            worker's `import envharness` (and spec import_paths like
            `experiments.*`) resolve against this tree first;
          - PYTHONSAFEPATH=1 additionally keeps sys.path[0] auto-insertion
            off on Python >= 3.11 (ignored on older interpreters; the
            script-path invocation already avoids the CWD hazard there).
            Respected if the caller already set it explicitly.
        """
        import envharness as _pkg
        pkg_root = str(Path(_pkg.__file__).resolve().parents[1])
        env = dict(os.environ)
        prev = env.get("PYTHONPATH")
        env["PYTHONPATH"] = pkg_root + (os.pathsep + prev if prev else "")
        env.setdefault("PYTHONSAFEPATH", "1")
        return env

    def _maybe_save_stderr(self, attempt_id: str, stderr: str) -> None:
        if not self.subprocess_log_dir or not stderr:
            return
        try:
            with open(self.subprocess_log_dir / f"{attempt_id}.stderr", "w",
                       encoding="utf-8") as f:
                f.write(stderr)
        except Exception:
            pass
