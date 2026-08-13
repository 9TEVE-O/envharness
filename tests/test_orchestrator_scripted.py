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

"""End-to-end Orchestrator smoke with a scripted HarnessAgent + scripted Policy.

This is the new-architecture equivalent of the deleted old
test_orchestrator_scripted.py. It exercises the full propose -> K-rollout
-> decide -> save-checkpoint loop without any LLM calls.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from envharness import load_checkpoint
from envharness.agents.harness_agent import ScriptedHarnessAgent
from envharness.bridges.toy24 import Toy24Env  # noqa: F401 -- registers "toy24"
from envharness.core.types import (
    Candidate, DecideResult, Decision, FailureAnalysis,
)
from envharness.orchestration.budget import CappedAdaptive
from envharness.orchestration.objectives import DifficultyZone
from envharness.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from envharness.orchestration.runner import (
    EnvSpec, InProcessRunner, PolicySpec,
)
from envharness.orchestration.storage import TraceStore


PASSTHROUGH_RULES = """
class _Rules(Rules):
    pass
""".strip()


def _scripted_agent_fn(call_kind, args):
    if call_kind == "propose":
        return Candidate(rules_code=PASSTHROUGH_RULES, rationale="smoke")
    if call_kind == "decide":
        traces = args["traces"]
        any_success = any(t.success for t in traces)
        return DecideResult(
            decision=Decision.ACCEPT,
            rationale=f"K={len(traces)} any_success={any_success}",
            failure_analysis=None,
        )
    if call_kind == "refine":
        return args["candidate"]
    raise AssertionError(f"unexpected call_kind={call_kind}")


def _make_orchestrator(log_dir: Path, n_tasks: int = 1) -> Orchestrator:
    env_spec = EnvSpec(
        import_path="envharness.bridges.toy24:Toy24Env",
        reset_options={"numbers": [3, 3, 7, 7], "target": 24},
    )
    policy_spec = PolicySpec(
        client_factory="envharness.infra.llm:ScriptedClient",
        client_kwargs={"model_id": "scripted/seq", "script": [
            {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "mul"}},
            {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "add"}},
            {"name": "stop", "kwargs": {}},
        ]},
        task_prompt="Combine to make 24.",
        action_format="function_calling",
    )
    return Orchestrator(
        env_spec=env_spec,
        policy_spec=policy_spec,
        harness_agent=ScriptedHarnessAgent(_scripted_agent_fn),
        objective=DifficultyZone(),
        budget=CappedAdaptive(max_k=2),
        runner=InProcessRunner(),
        trace_store=TraceStore(log_dir / "traces.jsonl"),
        tool_schemas=Toy24Env.tool_schemas(),
        env_state_schema=Toy24Env.env_state_schema(),
        config=OrchestratorConfig(
            task_id="toy24-orch-test",
            task_description="Combine to make 24.",
            n_tasks=n_tasks,
            max_episode_steps=8,
            k_per_candidate=2,
            base_seed=0,
            compute_baseline=False,    # skip baseline for the scripted plumbing
        ),
        log_dir=log_dir,
    )


def test_orchestrator_run_produces_accepted_traces(tmp_path: Path):
    orch = _make_orchestrator(tmp_path)
    accepted = orch.run()
    # 1 task × K=2 rollouts = 2 accepted traces.
    assert len(accepted) == 2
    assert all(t.success for t in accepted)
    assert all(t.kind == "accepted" for t in accepted)


def test_orchestrator_writes_checkpoint_per_task(tmp_path: Path):
    orch = _make_orchestrator(tmp_path, n_tasks=2)
    orch.run()
    cps = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert len(cps) == 2
    assert any("task_0000_" in p.name for p in cps)
    assert any("task_0001_" in p.name for p in cps)


def test_orchestrator_checkpoint_round_trips(tmp_path: Path):
    """The most important integration: a checkpoint the orchestrator
    saves must be load_checkpoint-able and the loaded env behaves."""
    orch = _make_orchestrator(tmp_path)
    orch.run()
    cps = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert cps, "orchestrator must produce at least one checkpoint"
    env = load_checkpoint(cps[0])    # auto_reset=True by default
    # The loaded env is reset to the configured puzzle.
    obs = env.observe()
    assert "target=24" in obs.text
    assert obs.data["numbers"] == [3.0, 3.0, 7.0, 7.0]


def test_orchestrator_passthrough_skip_option(tmp_path: Path):
    """skip_passthrough_candidates: empty Candidate => no rollouts."""
    def empty_agent_fn(call_kind, args):
        if call_kind == "propose":
            return Candidate(rationale="empty")
        return DecideResult(decision=Decision.ACCEPT, rationale="ok")

    orch = Orchestrator(
        env_spec=EnvSpec(import_path="envharness.bridges.toy24:Toy24Env",
                          reset_options={"numbers": [3, 3, 7, 7], "target": 24}),
        policy_spec=PolicySpec(
            client_factory="envharness.infra.llm:ScriptedClient",
            client_kwargs={"model_id": "scripted/seq", "script": []},
            task_prompt="x", action_format="function_calling",
        ),
        harness_agent=ScriptedHarnessAgent(empty_agent_fn),
        objective=DifficultyZone(),
        budget=CappedAdaptive(max_k=2),
        runner=InProcessRunner(),
        trace_store=TraceStore(tmp_path / "traces.jsonl"),
        tool_schemas=Toy24Env.tool_schemas(),
        env_state_schema=Toy24Env.env_state_schema(),
        config=OrchestratorConfig(
            task_id="empty-test",
            task_description="x", n_tasks=1,
            max_episode_steps=8, k_per_candidate=2, base_seed=0,
            compute_baseline=False,
            skip_passthrough_candidates=True,
        ),
        log_dir=tmp_path,
    )
    accepted = orch.run()
    assert accepted == []        # rollouts skipped, no traces accepted
