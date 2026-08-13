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

"""Run an episode through `run_episode` (InProcessRunner path) with the
new ActionableEnv + EnvHarness wiring.

Verifies that:
  - the runner builds the env stack from (EnvSpec, Candidate) correctly
  - a Rules-mutated env applies its `filter_action` hook during the loop
  - blocked actions show up on Step.blocked_reason
  - the resulting Trace records what the agent saw
"""
from __future__ import annotations

from envharness.bridges.toy24 import Toy24Env  # noqa: F401 registers env
from envharness.core.types import Action, Candidate
from envharness.orchestration.runner import (
    EnvSpec, EpisodeSpec, InProcessRunner, PolicySpec, build_env_stack,
)


SOLVING_SCRIPT = [
    {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "mul"}},
    {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "add"}},
    {"name": "stop", "kwargs": {}},
]


def _base_spec(candidate: Candidate, *,
                max_steps: int = 8, seed: int | None = 0) -> EpisodeSpec:
    return EpisodeSpec(
        env=EnvSpec(
            import_path="envharness.bridges.toy24:Toy24Env",
            reset_options={"numbers": [3, 3, 7, 7], "target": 24},
            reset_seed=seed,
        ),
        candidate=candidate,
        policy=PolicySpec(
            client_factory="envharness.infra.llm:ScriptedClient",
            client_kwargs={"model_id": "scripted/seq", "script": SOLVING_SCRIPT},
            task_prompt="Combine to make 24.",
            action_format="function_calling",
        ),
        iteration_id="it",
        task_id="t",
        max_steps=max_steps,
    )


def test_runner_passthrough_candidate_solves():
    """A pass-through Candidate (empty rules_code, no actions) just lets
    the scripted Policy solve the puzzle."""
    spec = _base_spec(Candidate())
    trace = InProcessRunner().run(spec)
    assert trace.error is None
    assert trace.success is True
    assert trace.final_reward == 1.0
    assert trace.duration_steps == 3


def test_runner_setup_replays_actions_then_solves():
    """Candidate with one in_env_action: Setup replays it before policy."""
    cand = Candidate(
        in_env_actions=[Action(name="combine",
                                kwargs={"i": 2, "j": 0, "op": "mul"})],
    )
    # Policy script must continue from the post-replay state -- add+stop only.
    spec = _base_spec(cand)
    spec.policy.client_kwargs["script"] = SOLVING_SCRIPT[1:]
    trace = InProcessRunner().run(spec)
    assert trace.error is None
    assert trace.success is True
    assert trace.duration_steps == 2


def test_runner_rules_blocks_action():
    """A Candidate with rules_code that blocks `mul` makes the scripted
    Policy fail (its first action gets blocked)."""
    rules_code = '''
class _Rules(Rules):
    def filter_action(self, action, env_state):
        if action.name == "combine" and action.kwargs.get("op") == "mul":
            return Blocked(reason="mul disabled in test")
        return action
'''
    cand = Candidate(rules_code=rules_code)
    spec = _base_spec(cand, max_steps=4)
    trace = InProcessRunner().run(spec)
    # First step blocked; subsequent attempts (also mul/add/stop) get evaluated.
    assert trace.error is None
    # The blocked step is recorded.
    blocked_steps = [s for s in trace.steps if s.blocked_reason]
    assert len(blocked_steps) >= 1
    assert any("mul disabled" in (s.blocked_reason or "") for s in blocked_steps)


def test_runner_mutation_code_error_returns_trace():
    """Malformed rules_code surfaces as Trace.error, not a crash."""
    cand = Candidate(rules_code="class _Rules(Rules):\n    def f(")
    trace = InProcessRunner().run(_base_spec(cand))
    assert trace.error is not None
    assert "MutationCodeError" in trace.error or "RulesCodeError" in trace.error


def test_build_env_stack_returns_actionable_env():
    """Direct check of the builder helper: returned object is an
    ActionableEnv composed of Setup + Rules over the base toy24."""
    from envharness.core.actionable_env import ActionableEnv
    from envharness.core.envharness import EnvHarness
    cand = Candidate(
        in_env_actions=[Action(name="combine",
                                kwargs={"i": 0, "j": 1, "op": "add"})],
        rules_code="class _Rules(Rules):\n    pass",
    )
    env = build_env_stack(_base_spec(cand))
    assert isinstance(env, ActionableEnv)
    assert isinstance(env, EnvHarness)        # outermost is Rules
    assert isinstance(env.inner, EnvHarness)  # next is Setup
    base = env.inner.inner
    assert type(base).__name__ == "Toy24Env"
