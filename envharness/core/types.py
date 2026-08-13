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

"""Core data types shared across the framework.

All cross-component data contracts live here. Bridges, Mutators, Objectives,
and the Orchestrator only exchange instances of these types.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Action / Observation -- intentionally generic; Bridges define concrete shapes
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """A typed action: name + JSON-serializable kwargs."""
    model_config = ConfigDict(extra="forbid")
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class Blocked(BaseModel):
    """Returned by Rules.filter_action when an action is rejected."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["blocked"] = "blocked"
    reason: str = ""


class Observation(BaseModel):
    """Anything the Policy sees from the env. Bridges may subclass to tighten."""
    model_config = ConfigDict(extra="allow")
    text: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step / EnvResponse  (per-step result)
# ---------------------------------------------------------------------------

class Step(BaseModel):
    """One environment step, including pre/post mutation transformations."""
    model_config = ConfigDict(extra="forbid")
    raw_action: Action
    filtered_action: Action | None = None      # None if Blocked
    blocked_reason: str | None = None
    raw_observation: Observation | None = None
    filtered_observation: Observation | None = None
    raw_reward: float = 0.0
    filtered_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
    # Full LLM response that produced this step's action -- captures any
    # <think>...</think> reasoning the policy emitted. Optional; only filled
    # by action_format=think_action / text_complete / react paths where the
    # policy exposes its raw text. ReasoningBank induction reads this when
    # present so the LLM has access to the same think+action pairs the
    # ReasoningBank induction recipe expects.
    policy_raw_response: str | None = None
    # Per-step dense reward emitted by the env stack's step_reward hook.
    # Default 0.0; set by runner.run_episode after each step. Step-reward
    # exceptions are caught and recorded as process_reward_error rather than
    # terminating the episode -- the framework's correctness does not depend
    # on this signal being valid, so a buggy Rules-emitted reward function
    # leaves the existing trace untouched.
    process_reward: float = 0.0
    process_reward_error: str | None = None


class EnvResponse(BaseModel):
    """The Pydantic-wrapped Gymnasium 5-tuple returned by Bridge.step()."""
    model_config = ConfigDict(extra="forbid")
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EnvResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: Observation
    info: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool
    score: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Candidate -- the HarnessAgent's only output type
# ---------------------------------------------------------------------------
#
# A Candidate bundles two things:
#   - rules_code: Python source defining `class _Rules(Rules)`
#                    Empty string == pass-through (no mutation).
#   - in_env_actions: Actions the Setup harness replays via inner.step()
#                     on reset, before the Policy starts (the S0 mechanism).
#
# This is intentionally simple: one Rules subclass per Candidate. Composing
# multiple effects = the HarnessAgent writes multiple hook overrides in one class.

class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules_code: str = ""
    in_env_actions: list[Action] = Field(default_factory=list)
    rationale: str = ""


# ---------------------------------------------------------------------------
# Rules decision result
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    REJECT = "reject"


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_axis: Optional[Literal["S0", "A", "O", "T", "R", "task_understanding", "none"]] = None
    label: str = ""           # short, free-form, e.g. "lost_localization"
    description: str = ""     # 1-2 sentence rationale


class DecideResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    failure_analysis: FailureAnalysis | None = None
    rationale: str = ""


# ---------------------------------------------------------------------------
# Trace -- stored verbatim including the mutation source code
# ---------------------------------------------------------------------------

class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    iteration_id: str
    task_id: str

    candidate: Candidate                       # mutation code + in-env actions
    candidate_id: str = ""                     # links K rollouts of one candidate
    rollout_idx: int = 0                       # 0..K-1 within a candidate
    rollout_seed: int | None = None            # seed used by Bridge.reset

    steps: list[Step] = Field(default_factory=list)

    final_reward: float = 0.0
    success: bool = False
    failure_analysis: FailureAnalysis | None = None
    duration_steps: int = 0

    kind: Literal["accepted", "exploration", "baseline"] = "accepted"
    policy_model_id: str = ""
    policy_seed: int | None = None

    error: str | None = None                   # set if subprocess crashed
    subprocess_stderr: str | None = None       # captured for debugging


# ---------------------------------------------------------------------------
# TaskSummary -- one entry in the shortlist the orchestrator presents to
# the HarnessAgent when `OrchestratorConfig.mutator_selects_tasks = True`.
# Each summary describes ONE candidate task the HarnessAgent may pick from.
# Bridges implement `list_tasks(reset_options, limit)` to produce these.
# ---------------------------------------------------------------------------

class TaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_idx: int                                  # what to pass back as the picked id
    instance_id: str                               # human-readable handle (e.g. alfworld task name, swebench instance_id)
    brief: str                                     # ~200-char one-line description shown to the HarnessAgent
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Opt-in extras: repo name, problem-statement excerpt, test counts,
    etc. BASELINE_SR is intentionally NOT exposed by default to avoid
    eval contamination."""


# ---------------------------------------------------------------------------
# StepInfo -- input to ActionableEnv.step_reward
# ---------------------------------------------------------------------------

class StepInfo(BaseModel):
    """Per-step context handed to `ActionableEnv.step_reward`. Carries the
    just-completed Step (which already has raw_observation = env-side,
    filtered_observation = Policy-side, blocked_reason, action, etc.) plus
    the position-in-episode and the history of previous steps.

    Rules-emitted reward functions can hold cross-step state by writing
    to plain instance attributes on `self` (the harness stack is built
    fresh per episode, so attributes don't leak across rollouts).
    """
    model_config = ConfigDict(extra="forbid")
    step: Step
    step_number: int                                # 0-indexed within this episode
    history: list[Step] = Field(default_factory=list)
    """Previous steps in this episode (excluding `step`). Empty on step_number=0."""


# ---------------------------------------------------------------------------
# BaselineSnapshot -- per-task unmutated Policy SR, fed to the HarnessAgent before
# its first propose() so it can calibrate perturbation magnitude. Computed
# once per (task_id, K, policy_spec) and cached on disk.
# ---------------------------------------------------------------------------

class BaselineRolloutSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool
    steps: int


class BaselineSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int                                          # rollouts that produced this snapshot
    n_success: int
    sr: float                                       # n_success / n
    avg_success_steps: float | None = None          # None if no successes
    per_rollout: list[BaselineRolloutSummary] = Field(default_factory=list)
    sample_success_actions: list[Action] = Field(default_factory=list)
    """One full action sequence from a successful baseline rollout (the
    SHORTEST winning trajectory, to minimize prompt bloat). Empty if no
    baseline rollout succeeded."""


# ---------------------------------------------------------------------------
# ObjectiveSignal
# ---------------------------------------------------------------------------

class ObjectiveSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float
    diagnostic: str
    suggestion_prompt: str
    weights: dict[str, float] | None = None
