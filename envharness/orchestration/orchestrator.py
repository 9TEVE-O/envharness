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

"""Orchestrator -- the feedback loop driver.

For each outer iteration:
  1. HarnessAgent.propose() -> Candidate (rules_code + in_env_actions)
  2. Inner search loop:
       a. K-rollout: run K episodes on the SAME candidate (different seeds)
       b. HarnessAgent.decide(candidate, K traces) -> Decision + FailureAnalysis
       c. ACCEPT -> stop. REFINE/REJECT -> next attempt.
       d. BudgetPolicy.should_stop() may force exit.
  3. ACCEPT'd candidate's K traces are kind="accepted"; others "exploration".
  4. All traces are persisted to TraceStore.

Logging: an optional `OrchestratorLogger` writes every meaningful event
(propose / decide / candidate eval) to a JSONL file plus a human log.
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from envharness.orchestration import baseline_cache as _baseline_cache
from envharness.orchestration.budget import BudgetPolicy
from envharness.agents.harness_agent import HarnessAgent, HarnessAgentContext
from envharness.core.types import (
    BaselineRolloutSummary, BaselineSnapshot,
    Candidate, Decision, ObjectiveSignal, Trace,
)
from envharness.infra.utils import import_symbol
from envharness.orchestration.objectives import MutationObjective
from envharness.orchestration.runner import EnvSpec, EpisodeRunner, EpisodeSpec, PolicySpec
from envharness.orchestration.storage import TraceStore
from envharness.persistence.checkpoint import Checkpoint


def _checkpoint_from_candidate(env_spec: EnvSpec, candidate: Candidate,
                                task_id: int,
                                metadata: dict | None = None) -> Checkpoint:
    """Build a portable Checkpoint from (env_spec, candidate) WITHOUT booting
    the env stack.

    DECOUPLING NOTE: a Checkpoint can also be produced from any source
    (hand-written, loaded from another run, induced by a different agent
    architecture). The persistence format is agent-agnostic; this function
    is one path among several.
    """
    env_cls = import_symbol(env_spec.import_path)
    env_type = env_cls.env_type()
    harnesses: list[dict] = []
    if candidate.in_env_actions:
        harnesses.append({
            "type": "setup",
            "state": {"actions": [{"name": a.name, "kwargs": dict(a.kwargs)}
                                  for a in candidate.in_env_actions]},
        })
    if (candidate.rules_code or "").strip():
        harnesses.append({
            "type": "rules",
            "state": {"rules_code": candidate.rules_code},
        })
    return Checkpoint(
        env_type=env_type,
        env_state={"reset_seed": int(task_id),
                   "reset_options": dict(env_spec.reset_options or {})},
        harnesses=harnesses,
        metadata=dict(metadata or {}),
    )


@dataclass
class OrchestratorConfig:
    task_id: str
    task_description: str
    # Number of tasks to sweep (one task per outer pass). Each outer pass
    # targets a different env instance: task_id = base_seed + idx*stride
    # + offset. `n_iterations` is accepted as an alias (see __post_init__).
    n_tasks: int = -1
    n_iterations: int = -1     # legacy alias for n_tasks
    max_episode_steps: int = 50
    k_per_candidate: int = 5
    base_seed: int = 0
    rollout_concurrency: int = 1
    """Max parallel K-rollouts within one candidate. 1 = sequential (default,
    backwards-compat). Higher trades a slight scheduling overhead and more
    memory (one Bridge per active subprocess) for K/rollout_concurrency wall
    speedup. When >1, sets ENVHARNESS_LOG_PER_PID=1 so subprocess
    LoggingLLMClient writes to `<name>.<pid>.<ext>` -- avoids POSIX append
    races on log lines larger than PIPE_BUF (4096 bytes)."""
    task_id_stride: int = -1               # preferred name
    task_id_outer_stride: int = -1          # legacy alias for task_id_stride
    task_id_base_offset: int = 100
    """Per-task identifier is `base_seed + task_idx * task_id_stride
    + task_id_base_offset`. Defaults (stride=1000, offset=100) match legacy.
    For a full-dataset sweep on an N-game benchmark, set stride=1 + offset=0
    so n_tasks=N hits each unique game once (otherwise `% N` collisions map
    many outer passes onto the same game)."""
    discard_rollout_steps_after_decide: bool = False
    """After the HarnessAgent's decide+refine loop for a task completes, clear
    `Trace.steps` (the bulky per-step record of Policy actions/obs) on every
    trace in that pass. Keeps the cheap fields -- success, duration_steps,
    reward, error -- so SR statistics still work. The perturbation itself is
    preserved in the saved Checkpoint."""

    skip_passthrough_candidates: bool = False
    """If True, when the HarnessAgent returns a Candidate with EMPTY rules_code AND
    EMPTY in_env_actions (pass-through == no actual mutation), the orchestrator
    SKIPS the K-rollout phase for that task and ends the task immediately.
    Useful with Mutators that emit `Candidate(rules_code="", rationale="skip-...")`
    as a "this task isn't worth mutating" signal (e.g. a fail-targeted agent's
    skip-easy-task path). Saves K rollouts × pass-through == effective baseline
    re-rolls that would just reproduce the cached baseline. Default False to
    preserve legacy behavior."""

    compute_baseline: bool = True
    """If True (default), at the start of EACH task run K unmutated rollouts of
    the Policy on that task_id and expose the result to the HarnessAgent via
    `HarnessAgentContext.baseline`. Lets the HarnessAgent calibrate perturbation
    magnitude per task instead of pushing every task toward the SR band
    uniformly. Disable for plain baseline-only runs (NoopMutator) where
    re-rolling the same env at the start is wasteful."""
    baseline_cache_dir: str | None = "runs/_baseline_cache"
    """Disk cache for per-task baseline measurements. Cache key includes
    bridge, reset_options, task_id, K, max_steps, and the Policy spec. Set
    to None to always re-roll."""
    mutator_baseline_mode: str = "summary"
    """How much of the baseline rollouts the HarnessAgent sees in its prompt.
    "summary" (DEFAULT, legacy behavior):
        - per-rollout (success, step_count) summary
        - one sample successful trajectory as action-only list (kwargs only)
        - failure trajectories not shown
        Use this to reproduce summary-mode runs byte-for-byte.
    "full":
        - all of summary, PLUS
        - one full successful trajectory (think + action + obs) AND
        - one full failure trajectory (think + action + obs)
        - both truncated to last ~12 steps; per-step content also truncated
        Enables the HarnessAgent to diagnose WHY the Policy succeeded/failed,
        not just THAT it did. Recommended for new SWE-bench band-targeted
        experiments. Only takes effect on FRESH baseline compute (cache hit
        falls back to summary because cache stores summary stats only)."""

    explicit_task_ids: list[int] | None = None
    """If set, OVERRIDES the deterministic stride formula:
    `task_id = base_seed + task_idx * task_id_stride + task_id_base_offset`.
    Lets the launcher pass an arbitrary (e.g. randomly-sampled) list of
    task_ids; the orchestrator iterates through them in order, using
    `explicit_task_ids[task_idx]` as each outer pass's task_id. n_tasks
    is implicitly the length of this list (CLI / yaml --n-tasks ignored
    when this is set)."""

    mutator_selects_tasks: bool = False
    """If True, the orchestrator builds a shortlist of un-tried
    candidate task_idxs each iteration via `bridge.list_tasks()` and asks
    `harness_agent.select_task(ctx)` to pick one (vs the default deterministic
    stride formula). Requires the Bridge to implement list_tasks()."""
    shortlist_size: int = 10
    """How many candidate tasks to present to the HarnessAgent per iteration
    when `mutator_selects_tasks=True`. Larger = more selection power +
    bigger prompt; smaller = cheaper + less interesting choice."""

    def __post_init__(self):
        # n_tasks <-> n_iterations alias resolution.
        if self.n_tasks >= 0 and self.n_iterations >= 0 and self.n_tasks != self.n_iterations:
            raise ValueError(
                f"OrchestratorConfig: conflicting n_tasks={self.n_tasks} vs "
                f"n_iterations={self.n_iterations}. Use one, not both."
            )
        if self.n_tasks < 0 and self.n_iterations < 0:
            raise ValueError("OrchestratorConfig requires n_tasks (or legacy n_iterations).")
        if self.n_tasks < 0: self.n_tasks = self.n_iterations
        if self.n_iterations < 0: self.n_iterations = self.n_tasks
        # task_id_stride <-> task_id_outer_stride alias.
        if self.task_id_stride >= 0 and self.task_id_outer_stride >= 0 and self.task_id_stride != self.task_id_outer_stride:
            raise ValueError(
                "OrchestratorConfig: conflicting task_id_stride and task_id_outer_stride."
            )
        if self.task_id_stride < 0 and self.task_id_outer_stride < 0:
            # Legacy default
            self.task_id_stride = 1000
            self.task_id_outer_stride = 1000
        elif self.task_id_stride < 0:
            self.task_id_stride = self.task_id_outer_stride
        elif self.task_id_outer_stride < 0:
            self.task_id_outer_stride = self.task_id_stride


class _Logger:
    """Optional helper: writes structured events to <log_dir>/orchestrator.jsonl
    and a human-readable line to <log_dir>/orchestrator.log."""

    def __init__(self, log_dir: Path | None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.jsonl_path = (self.log_dir / "orchestrator.jsonl") if self.log_dir else None
        self.text_path = (self.log_dir / "orchestrator.log") if self.log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._py = logging.getLogger("envharness.orchestration.orchestrator")
        if not self._py.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s"))
            self._py.addHandler(h)
        self._py.setLevel(logging.INFO)

    def event(self, kind: str, **fields):
        rec = {"ts": time.time(),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kind": kind, **fields}
        line = json.dumps(rec, default=str, ensure_ascii=False)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        msg = " ".join(f"{k}={v}" for k, v in fields.items())
        text = f"{kind}  {msg}"
        if self.text_path:
            with open(self.text_path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime())
                         + text + "\n")
        self._py.info(text)


class Orchestrator:
    def __init__(self,
                 env_spec: EnvSpec,
                 policy_spec: PolicySpec,
                 harness_agent: HarnessAgent,
                 objective: MutationObjective,
                 budget: BudgetPolicy,
                 runner: EpisodeRunner,
                 trace_store: TraceStore,
                 tool_schemas: list[dict],
                 env_state_schema: str,
                 config: OrchestratorConfig,
                 log_dir: str | Path | None = None):
        # NOTE: a HarnessAgent is ONE way to produce candidate harnesses;
        # it's not the only way. Loading a Checkpoint, hand-coding a
        # Candidate, or any other procedure that yields a Candidate works
        # equally well with the runner. The Orchestrator is the
        # agent-search driver specifically -- use the runner directly for
        # other workflows.
        self.env_spec = env_spec
        self.policy_spec = policy_spec
        self.harness_agent = harness_agent
        self.objective = objective
        self.budget = budget
        self.runner = runner
        self.trace_store = trace_store
        self.tool_schemas = tool_schemas
        self.env_state_schema = env_state_schema
        self.config = config
        self.log = _Logger(Path(log_dir) if log_dir else None)
        # Tell subprocess LoggingLLMClient instances to split logs per-pid when
        # we're running K rollouts concurrently. POSIX O_APPEND atomicity only
        # holds up to PIPE_BUF (4096B); typical policy_calls lines exceed that.
        if self.config.rollout_concurrency > 1:
            os.environ["ENVHARNESS_LOG_PER_PID"] = "1"
        # Work-stealing endpoint dispatch for K-rollouts: build an EndpointPool
        # from the policy's api_base_pool (or singleton api_base). Each K
        # rollout leases an endpoint, runs, releases -- so K subprocesses
        # spread across the N endpoints without per-PID clustering, and a
        # single-endpoint deployment degenerates to all K hitting one
        # endpoint (vLLM batches them). None if the policy spec has neither
        # field; then no per-rollout endpoint pinning is applied.
        from envharness.infra.endpoint_pool import pool_from_client_kwargs
        self._endpoint_pool = pool_from_client_kwargs(policy_spec.client_kwargs)

    def run(self) -> list[Trace]:
        accepted: list[Trace] = []
        self._preflight_check()
        # explicit_task_ids overrides n_tasks: iterate exactly the supplied list.
        if self.config.explicit_task_ids:
            n_tasks = len(self.config.explicit_task_ids)
        else:
            n_tasks = self.config.n_tasks
        self.log.event("run_start",
                        task_id=self.config.task_id,
                        n_tasks=n_tasks,
                        k_per_candidate=self.config.k_per_candidate,
                        base_seed=self.config.base_seed,
                        explicit_task_ids=("yes" if self.config.explicit_task_ids
                                             else "no"))
        n_failed = 0
        for task_idx in range(n_tasks):
            # `task_start` fires per outer pass (one task instance);
            # `iteration_start` is the alias event under the older name.
            self.log.event("task_start", task_idx=task_idx)
            self.log.event("iteration_start", outer=task_idx)
            try:
                for tr in self._iterate(task_idx):
                    self.trace_store.add(tr)
                    if tr.kind == "accepted":
                        accepted.append(tr)
            except Exception as e:
                # Per-task crash isolation: an unexpected failure must not
                # kill the remaining tasks in the run. HarnessAgent LLM /
                # parse / rollout-collection errors are already salvaged
                # INSIDE _iterate (see the `aborted` guard: the task ends
                # gracefully and its completed rollout traces are returned
                # and stored). This catch handles what's left -- baseline
                # compute or the first propose() -- where no candidate
                # rollouts exist yet, so nothing is lost. The failed task
                # gets a `task_failed` event the driver can grep for.
                n_failed += 1
                self.log.event("task_failed", task_idx=task_idx,
                                error_type=type(e).__name__,
                                error=str(e)[:500])
            self.log.event("task_end", task_idx=task_idx,
                            total_traces_so_far=len(self.trace_store))
            self.log.event("iteration_end", outer=task_idx,
                            total_traces_so_far=len(self.trace_store))
        self.log.event("run_end", n_accepted=len(accepted),
                        n_total=len(self.trace_store),
                        n_failed=n_failed)
        return accepted

    def _preflight_check(self) -> None:
        """Fail fast on config combinations that silently degenerate.

        Currently checks: a band objective (DifficultyZone) + compute_baseline
        + K>1 + Policy temperature == 0 would collapse the K rollouts to one
        deterministic trajectory (env+actions deterministic given seed), so
        per-task SR can only be {0, 1} and the band [lo, hi] is unreachable.
        The K rollouts all use the same task seed, so this holds
        unconditionally."""
        if not self.config.compute_baseline:
            return
        if self.config.k_per_candidate <= 1:
            return
        if self._target_band_or_none() is None:
            return
        temp = float(getattr(self.policy_spec, "temperature", 0.0) or 0.0)
        if temp > 0.0:
            return
        raise ValueError(
            "Orchestrator preflight: band objective + compute_baseline "
            f"+ k_per_candidate={self.config.k_per_candidate} "
            f"+ policy.temperature={temp} would collapse K rollouts to "
            "one deterministic trajectory and make the SR band unreachable. "
            "Set policy.temperature > 0 (default 0.4) or k_per_candidate=1."
        )

    def _target_band_or_none(self) -> tuple[float, float] | None:
        """Pull (lo, hi) from the active objective (DifficultyZone uses .lo
        and .hi). Returns None if no band is configured -- in that case,
        no best-effort checkpoint is saved and the existing ACCEPT-only
        save path is the sole way to land a checkpoint."""
        obj = self.objective
        if obj is None:
            return None
        lo = getattr(obj, "lo", None)
        hi = getattr(obj, "hi", None)
        if lo is None or hi is None:
            return None
        return float(lo), float(hi)

    def _iterate(self, task_idx: int) -> list[Trace]:
        # `iteration_id` retained as the name of the per-pass UUID for log
        # back-compat (it's a unique pass identifier; not the same as task_id,
        # which is the alfworld game index).
        iteration_id = uuid.uuid4().hex[:8]
        history = self.trace_store.all()

        # If mutator_selects_tasks is True, let the HarnessAgent pick
        # task_idx from a shortlist (instead of the deterministic stride).
        # Default `candidate_tasks=[]` keeps backwards-compat -- existing
        # configs see the same task_idx flow as before.
        candidate_tasks = []
        if self.config.mutator_selects_tasks:
            candidate_tasks, picked_idx = self._build_shortlist_and_pick(
                task_idx, history, iteration_id)
            task_idx = picked_idx    # override

        # Deterministic task identifier for this pass. When the HarnessAgent returns
        # a Candidate with in_env_actions, all K rollouts of this pass share
        # this task_id for fully deterministic replay (see _rollout_k).
        if self.config.explicit_task_ids:
            task_id = int(self.config.explicit_task_ids[task_idx])
        else:
            task_id = (self.config.base_seed
                        + task_idx * self.config.task_id_stride
                        + self.config.task_id_base_offset)

        # Per-task baseline phase. Runs K unmutated rollouts on `task_id` before
        # the HarnessAgent's first propose(), so it can see how the Policy
        # behaves on the untouched env and calibrate accordingly. Cached on
        # disk under the configured baseline_cache_dir.
        baseline: BaselineSnapshot | None = None
        baseline_traces_for_ctx: list[Trace] = []
        if self.config.compute_baseline:
            baseline = self._compute_or_load_baseline(task_idx, task_id)
            # Only thread full trajectories into ctx when explicitly opted in;
            # default "summary" preserves backward-compat.
            if self.config.mutator_baseline_mode == "full":
                baseline_traces_for_ctx = list(getattr(self, '_last_baseline_traces', []))

        ctx = HarnessAgentContext(
            history_traces=history,
            tool_schemas=self.tool_schemas,
            env_state_schema=self.env_state_schema,
            objective=self.objective,
            iteration_id=iteration_id,
            task_description=self.config.task_description,
            task_id=task_id,
            baseline=baseline,
            baseline_traces=baseline_traces_for_ctx,
            candidate_tasks=candidate_tasks,
        )

        candidate = self.harness_agent.propose(ctx)
        self.log.event("candidate_proposed", iteration_id=iteration_id,
                        attempt=0, rules_code_len=len(candidate.rules_code),
                        n_in_env_actions=len(candidate.in_env_actions),
                        rationale=candidate.rationale[:120])

        # PASSTHROUGH-SKIP optimization: if Rules returns empty rules_code
        # AND empty in_env_actions (pure pass-through, would just reproduce
        # baseline), skip the K-rollout phase and end the task immediately.
        # Saves K episodes worth of compute per skipped task. Opt-in via
        # `orchestrator.skip_passthrough_candidates: true`.
        if (self.config.skip_passthrough_candidates
                and not candidate.rules_code.strip()
                and not candidate.in_env_actions):
            self.log.event("task_skipped_passthrough",
                            iteration_id=iteration_id, task_idx=task_idx,
                            task_id=task_id,
                            rationale=candidate.rationale[:120])
            return []

        attempts = 0
        last_decision = Decision.REFINE
        last_signal: ObjectiveSignal = self.objective.evaluate(history)
        all_traces: list[Trace] = []
        candidate_traces_groups: list[list[Trace]] = []
        # Best-effort tracker. If the HarnessAgent never ACCEPTs an attempt before
        # budget_stop, we save the closest-to-band attempt under
        # iter_<N>_best_effort.json -- so every outer iter produces a
        # checkpoint, never zero. Tagged with metadata.best_effort=True.
        band = self._target_band_or_none()
        accepted_saved = False
        best_effort: dict | None = None

        # Salvage guard: a HarnessAgent LLM / parse failure (or a rollout
        # collection error) must END the task gracefully, not raise out of
        # _iterate -- an escape lands in run()'s per-task catch, which can
        # only log, and every already-paid rollout trace of this task would
        # be silently dropped (traces reach the TraceStore via our return).
        aborted: str | None = None
        while True:
            attempts += 1
            candidate_id = uuid.uuid4().hex[:8]
            try:
                traces_for_candidate = self._rollout_k(
                    candidate, candidate_id, iteration_id, task_idx, attempts,
                    ctx_task_id=ctx.task_id,
                )
            except Exception as e:
                aborted = f"rollout_k: {type(e).__name__}: {e}"
                break
            sr = sum(1 for t in traces_for_candidate if t.success) / max(len(traces_for_candidate), 1)
            self.log.event("candidate_evaluated",
                            iteration_id=iteration_id, attempt=attempts,
                            candidate_id=candidate_id,
                            k=len(traces_for_candidate),
                            success_rate=round(sr, 3),
                            n_errors=sum(1 for t in traces_for_candidate if t.error))

            try:
                decide_result = self.harness_agent.decide(candidate, traces_for_candidate, ctx)
            except Exception as e:
                candidate_traces_groups.append(traces_for_candidate)
                all_traces.extend(traces_for_candidate)
                aborted = f"decide: {type(e).__name__}: {e}"
                break
            for t in traces_for_candidate:
                t.failure_analysis = decide_result.failure_analysis
            last_decision = decide_result.decision
            self.log.event("mutator_decision",
                            iteration_id=iteration_id, attempt=attempts,
                            decision=last_decision.value,
                            failure_axis=(decide_result.failure_analysis.primary_axis
                                          if decide_result.failure_analysis else None),
                            failure_label=(decide_result.failure_analysis.label
                                           if decide_result.failure_analysis else None),
                            rationale=decide_result.rationale[:200])

            # Track best-effort: the attempt whose SR is closest to the
            # target band center. Used as a fallback if budget_stop without
            # ACCEPT, so every iter still produces a checkpoint.
            non_trivial = (bool(candidate.in_env_actions)
                            or bool((candidate.rules_code or "").strip()))
            if band is not None and non_trivial:
                lo, hi = band
                center = (lo + hi) / 2
                dist = abs(sr - center)
                if best_effort is None or dist < best_effort["dist"]:
                    best_effort = {
                        "sr": sr, "dist": dist,
                        "candidate": candidate,
                        "candidate_id": candidate_id,
                        "attempts": attempts,
                        "k": len(traces_for_candidate),
                    }

            # Persist the env-state checkpoint when the HarnessAgent ACCEPTs a
            # non-trivial mutation. NoopMutator's empty Candidates produce
            # nothing here, so plain baseline runs are unaffected on disk.
            if (last_decision == Decision.ACCEPT
                    and self.log.log_dir is not None
                    and non_trivial):
                try:
                    from envharness.persistence.checkpoint import Checkpoint
                    cp = _checkpoint_from_candidate(
                        candidate=candidate,
                        env_spec=self.env_spec,
                        task_id=ctx.task_id,
                        metadata={
                            "iteration_id": iteration_id,
                            "task_idx": task_idx,
                            "outer": task_idx,           # legacy alias
                            "attempt": attempts,
                            "candidate_id": candidate_id,
                            "rollout_success_rate": round(sr, 3),
                            "k": len(traces_for_candidate),
                            "best_effort": False,
                        },
                    )
                    cp_path = (self.log.log_dir / "checkpoints"
                                / f"task_{task_idx:04d}_attempt_{attempts:02d}.json")
                    cp.save(cp_path)
                    accepted_saved = True
                    self.log.event("checkpoint_saved",
                                    iteration_id=iteration_id,
                                    path=str(cp_path.relative_to(self.log.log_dir)),
                                    n_actions=len(candidate.in_env_actions),
                                    rules_code_len=len(candidate.rules_code or ""),
                                    best_effort=False)
                except Exception as e:
                    self.log.event("checkpoint_save_failed",
                                    iteration_id=iteration_id, error=str(e))

            candidate_traces_groups.append(traces_for_candidate)
            all_traces.extend(traces_for_candidate)

            # Refresh the objective signal so BudgetPolicy.should_stop sees
            # this task's latest state rather than the pre-task snapshot of
            # history (matters for ObjectiveDriven's score_threshold gate).
            last_signal = self.objective.evaluate(history + all_traces)

            if self.budget.should_stop(attempts, last_decision, last_signal):
                self.log.event("budget_stop", iteration_id=iteration_id,
                                attempts=attempts)
                break
            if last_decision == Decision.ACCEPT:
                break
            if last_decision == Decision.REFINE:
                try:
                    candidate = self.harness_agent.refine(candidate, traces_for_candidate, ctx)
                except Exception as e:
                    aborted = f"refine: {type(e).__name__}: {e}"
                    break
                self.log.event("candidate_refined", iteration_id=iteration_id,
                                attempt=attempts,
                                rules_code_len=len(candidate.rules_code))
            elif last_decision == Decision.REJECT:
                ctx.history_traces = history + all_traces
                try:
                    candidate = self.harness_agent.propose(ctx)
                except Exception as e:
                    aborted = f"propose: {type(e).__name__}: {e}"
                    break
                self.log.event("candidate_reproposed", iteration_id=iteration_id,
                                attempt=attempts,
                                rules_code_len=len(candidate.rules_code))

        # Save best-effort checkpoint if no ACCEPTed save happened. This
        # guarantees every outer iter produces exactly one checkpoint --
        # never zero -- so downstream evaluation always has something to
        # compare against the baseline.
        if (not accepted_saved
                and best_effort is not None
                and self.log.log_dir is not None):
            try:
                from envharness.persistence.checkpoint import Checkpoint
                be = best_effort
                cp = _checkpoint_from_candidate(
                    candidate=be["candidate"],
                    env_spec=self.env_spec,
                    task_id=ctx.task_id,
                    metadata={
                        "iteration_id": iteration_id,
                        "task_idx": task_idx,
                        "outer": task_idx,         # legacy alias
                        "attempt": be["attempts"],
                        "candidate_id": be["candidate_id"],
                        "rollout_success_rate": round(be["sr"], 3),
                        "k": be["k"],
                        "best_effort": True,
                        "distance_to_band_center": round(be["dist"], 3),
                        "reason": "budget_stop_without_accept",
                    },
                )
                cp_path = (self.log.log_dir / "checkpoints"
                            / f"task_{task_idx:04d}_best_effort.json")
                cp.save(cp_path)
                self.log.event("checkpoint_saved",
                                iteration_id=iteration_id,
                                path=str(cp_path.relative_to(self.log.log_dir)),
                                n_actions=len(be["candidate"].in_env_actions),
                                rules_code_len=len(be["candidate"].rules_code or ""),
                                best_effort=True,
                                rollout_success_rate=round(be["sr"], 3))
            except Exception as e:
                self.log.event("checkpoint_save_failed",
                                iteration_id=iteration_id, error=str(e))

        if aborted is not None:
            self.log.event("task_aborted", iteration_id=iteration_id,
                            attempts=attempts, error=aborted[:500],
                            n_salvaged_traces=len(all_traces))

        # final candidate's K traces -> kind="accepted" ONLY when the loop
        # actually ended on an ACCEPT. A budget exhaustion after REFINE /
        # REJECT (or a salvaged abort) means no candidate was ever committed
        # -- every group stays "exploration", since "accepted" means the
        # HarnessAgent committed the candidate. This matters downstream: the
        # `ours` skill banks are distilled from kind="accepted" traces, so
        # mislabeling an exhausted task would feed a REJECTED mutation's
        # rollouts into the bank.
        final_accepted = (aborted is None and last_decision == Decision.ACCEPT)
        for i, group in enumerate(candidate_traces_groups):
            is_final = i == len(candidate_traces_groups) - 1
            kind = "accepted" if (is_final and final_accepted) else "exploration"
            for t in group:
                t.kind = kind

        # The Rules has fully consumed these traces (decide + any refine).
        # If configured, drop the bulky step data before the TraceStore writes.
        # The checkpoint (saved above) is the durable artifact; SR aggregates
        # still work because Trace.success/duration_steps/reward are intact.
        if self.config.discard_rollout_steps_after_decide:
            for t in all_traces:
                t.steps = []

        return all_traces

    # ------------------------------------------------------------------
    # HarnessAgent-driven task selection.
    # ------------------------------------------------------------------

    def _build_shortlist_and_pick(self, default_task_idx: int,
                                    history: list[Trace],
                                    iteration_id: str) -> tuple[list, int]:
        """When `mutator_selects_tasks=True`: query the Bridge for
        candidate tasks, sample a shortlist of un-tried task_idxs, ask
        the HarnessAgent to pick one, and return (shortlist, picked_task_idx).

        Robust fallbacks:
          - Bridge doesn't implement list_tasks -> log + fall back to
            default_task_idx (default stride formula).
          - Empty shortlist -> log + fall back to default_task_idx.
          - HarnessAgent returns a task_idx not in shortlist -> LLMHarnessAgent's
            select_task handles via fallback to shortlist[0].
        """
        from envharness.infra.utils import import_symbol
        try:
            env_cls = import_symbol(self.env_spec.import_path)
        except Exception as e:
            self.log.event("shortlist_build_failed", reason=f"import env class: {e}")
            return [], default_task_idx
        try:
            all_summaries = env_cls.list_tasks(
                reset_options=self.env_spec.reset_options,
                limit=None,
            )
        except NotImplementedError:
            self.log.event("shortlist_build_failed",
                            reason=f"{env_cls.__name__} does not implement list_tasks")
            return [], default_task_idx
        except Exception as e:
            self.log.event("shortlist_build_failed", reason=f"list_tasks raised: {e}")
            return [], default_task_idx
        if not all_summaries:
            self.log.event("shortlist_empty", default_task_idx=default_task_idx)
            return [], default_task_idx

        # Filter out already-tried task_idxs from prior iterations in this run.
        # We track tried task_idxs by examining checkpoint files; if logging
        # log_dir is set, this is reliable across crashes too.
        tried_idxs: set[int] = set()
        if self.log.log_dir is not None:
            ckpt_dir = self.log.log_dir / "checkpoints"
            if ckpt_dir.exists():
                import re
                for f in ckpt_dir.glob("task_*.json"):
                    m = re.match(r"task_(\d+)_", f.name)
                    if m:
                        tried_idxs.add(int(m.group(1)))
        eligible = [s for s in all_summaries if s.task_idx not in tried_idxs]
        if not eligible:
            # All tasks tried; reset to full pool (means user n_tasks > pool size).
            eligible = all_summaries

        # Sample up to shortlist_size deterministically: take a sliding
        # window starting at default_task_idx (mod pool size). This keeps
        # the shortlist stable across iterations while still presenting
        # different tasks.
        n = self.config.shortlist_size
        pool = eligible
        start = default_task_idx % len(pool) if pool else 0
        shortlist = pool[start:start + n] if start + n <= len(pool) else pool[start:] + pool[: n - (len(pool) - start)]
        shortlist = shortlist[:n]

        # Ask the HarnessAgent to pick
        from envharness.agents.harness_agent import HarnessAgentContext
        sel_ctx = HarnessAgentContext(
            history_traces=history,
            tool_schemas=self.tool_schemas,
            env_state_schema=self.env_state_schema,
            objective=self.objective,
            iteration_id=iteration_id,
            task_description=self.config.task_description,
            task_id=default_task_idx,
            candidate_tasks=shortlist,
        )
        try:
            picked = self.harness_agent.select_task(sel_ctx)
        except Exception as e:
            self.log.event("select_task_failed",
                            reason=f"{type(e).__name__}: {e}",
                            fallback_task_idx=shortlist[0].task_idx)
            picked = shortlist[0].task_idx
        # Validate that picked is in the shortlist.
        if picked not in {s.task_idx for s in shortlist}:
            self.log.event("select_task_invalid_pick",
                            picked=picked,
                            shortlist_idxs=[s.task_idx for s in shortlist],
                            fallback_task_idx=shortlist[0].task_idx)
            picked = shortlist[0].task_idx
        self.log.event("task_selected",
                        presented=[s.task_idx for s in shortlist],
                        picked=picked,
                        default_was=default_task_idx)
        return shortlist, picked

    # ------------------------------------------------------------------
    # Baseline phase: K unmutated rollouts per task, cached on disk and
    # surfaced to the HarnessAgent via HarnessAgentContext.baseline.
    # ------------------------------------------------------------------

    def _compute_or_load_baseline(self, task_idx: int,
                                     task_id: int) -> BaselineSnapshot:
        # Side effect: populates self._last_baseline_traces with the K full
        # Trace objects when freshly computed (empty list on cache hit). The
        # caller can read this to thread full trajectories into HarnessAgentContext.
        self._last_baseline_traces = []
        K = self.config.k_per_candidate
        policy_model = _baseline_cache.policy_model_id(self.policy_spec.client_kwargs)
        key = _baseline_cache.cache_key(
            env_import_path=self.env_spec.import_path,
            reset_options=self.env_spec.reset_options,
            task_id=task_id,
            n=K,
            max_steps=self.config.max_episode_steps,
            policy_model=policy_model,
            policy_action_format=self.policy_spec.action_format,
            policy_temperature=self.policy_spec.temperature,
            policy_max_history=self.policy_spec.max_history,
            policy_task_prompt=self.policy_spec.task_prompt,
        )
        cache_dir = self.config.baseline_cache_dir
        if cache_dir is not None:
            cached = _baseline_cache.load(cache_dir, key)
            if cached is not None:
                snap = self._baseline_snapshot_from_cached(cached.get("result", {}))
                self.log.event("baseline_cache_hit", task_idx=task_idx,
                                task_id=task_id, key=key, n=snap.n,
                                sr=round(snap.sr, 3),
                                avg_success_steps=snap.avg_success_steps)
                return snap

        self.log.event("baseline_compute_start", task_idx=task_idx,
                        task_id=task_id, n=K, key=key)
        traces = self._rollout_baseline_k(task_idx, task_id)
        for t in traces:
            t.kind = "baseline"
            self.trace_store.add(t)
        self._last_baseline_traces = traces
        snap = self._baseline_snapshot_from_traces(traces)

        # Only cache if every baseline rollout completed cleanly. A subprocess
        # error (missing API key, runtime crash, etc.) yields a fake "0% SR"
        # that would mislead all future runs reading this cache key.
        n_with_error = sum(1 for t in traces if t.error)
        if cache_dir is not None and n_with_error == 0:
            # Write in the same shape evaluate_checkpoints.py already reads, so
            # the two tools share a cache symmetrically. `sample_success_actions`
            # is an addition the orchestrator uses; older readers ignore it.
            result = {
                "sr": snap.sr,
                "n_won": snap.n_success,
                "n": snap.n,
                "avg_steps": round(
                    sum(r.steps for r in snap.per_rollout) / max(snap.n, 1), 2),
                "rollouts": [{"success": r.success, "steps": r.steps, "error": None}
                              for r in snap.per_rollout],
                "sample_success_actions": [a.model_dump()
                                            for a in snap.sample_success_actions],
            }
            params = {
                "bridge": self.env_spec.import_path,
                "reset_options": self.env_spec.reset_options or {},
                "task_id": int(task_id),
                "n": int(K),
                "max_steps": int(self.config.max_episode_steps),
                "policy_model": policy_model,
                "policy_action_format": self.policy_spec.action_format,
                "policy_temperature": float(self.policy_spec.temperature),
                "policy_max_history": int(self.policy_spec.max_history),
            }
            try:
                _baseline_cache.save(cache_dir, key, params, result)
            except Exception as e:
                self.log.event("baseline_cache_save_failed",
                                task_idx=task_idx, key=key, error=str(e))
        elif n_with_error > 0:
            self.log.event("baseline_cache_skip",
                            task_idx=task_idx, key=key,
                            n_with_error=n_with_error,
                            reason="rollouts had subprocess errors")

        self.log.event("baseline_compute_done", task_idx=task_idx,
                        task_id=task_id, n=snap.n, sr=round(snap.sr, 3),
                        avg_success_steps=snap.avg_success_steps,
                        sample_action_seq_len=len(snap.sample_success_actions))
        return snap

    def _rollout_baseline_k(self, task_idx: int,
                              task_id: int) -> list[Trace]:
        """K unmutated rollouts of the Policy on the same task_id.

        All K rollouts hit `reset_seed=task_id` (the deterministic per-task
        identifier from `_iterate`'s formula). `_rollout_k` mirrors this
        choice so the baseline-vs-mutated comparison is apples-to-apples.
        """
        K = self.config.k_per_candidate
        conc = max(1, int(self.config.rollout_concurrency))
        baseline_iter_id = f"baseline-{task_idx:04d}"
        pass_through = Candidate(rules_code="", in_env_actions=[])

        specs_and_seeds: list[tuple[int, EpisodeSpec]] = []
        for k in range(K):
            specs_and_seeds.append((k, EpisodeSpec(
                env=EnvSpec(
                    import_path=self.env_spec.import_path,
                    reset_options=self.env_spec.reset_options,
                    reset_seed=task_id,
                ),
                candidate=pass_through,
                policy=self.policy_spec,
                iteration_id=baseline_iter_id,
                task_id=self.config.task_id,
                max_steps=self.config.max_episode_steps,
            )))

        def _run_one(item: tuple[int, EpisodeSpec]) -> tuple[int, int, Trace, float]:
            k, spec = item
            t0 = time.time()
            trace = self._run_spec_with_endpoint(spec)
            return k, spec.env.reset_seed, trace, time.time() - t0

        results: dict[int, Trace] = {}
        if conc == 1:
            for item in specs_and_seeds:
                k, seed, trace, dur = _run_one(item)
                trace.candidate_id = "baseline"
                trace.rollout_idx = k
                results[k] = trace
                self._log_rollout(baseline_iter_id, 0, "baseline", k,
                                   seed, dur, trace)
        else:
            with _cf.ThreadPoolExecutor(max_workers=conc) as pool:
                futures = [pool.submit(_run_one, item) for item in specs_and_seeds]
                for fut in _cf.as_completed(futures):
                    k, seed, trace, dur = fut.result()
                    trace.candidate_id = "baseline"
                    trace.rollout_idx = k
                    results[k] = trace
                    self._log_rollout(baseline_iter_id, 0, "baseline", k,
                                       seed, dur, trace)
        return [results[k] for k in range(K)]

    def _run_spec_with_endpoint(self, spec) -> Trace:
        """Lease an endpoint from the EndpointPool (if configured) and run
        the spec pinned to that endpoint. The lease releases on exit so the
        endpoint is back in the pool for the next waiting rollout. Falls
        back to plain runner.run(spec) when no pool is configured."""
        if self._endpoint_pool is None:
            return self.runner.run(spec)
        from envharness.infra.endpoint_pool import pin_endpoint
        with self._endpoint_pool.lease() as endpoint:
            pinned = pin_endpoint(spec, endpoint)
            return self.runner.run(pinned)

    @staticmethod
    def _baseline_snapshot_from_traces(traces: list[Trace]) -> BaselineSnapshot:
        n = len(traces)
        successes = [t for t in traces if t.success]
        n_success = len(successes)
        sr = n_success / max(n, 1)
        avg_success_steps = (
            sum(t.duration_steps for t in successes) / n_success
            if n_success else None
        )
        # Pick the SHORTEST winning trajectory for the sample (less prompt bloat;
        # also tends to be the cleanest demonstration of the Policy's "fast
        # path" when one exists).
        sample_actions = []
        if successes:
            shortest = min(successes, key=lambda t: t.duration_steps)
            sample_actions = [step.raw_action for step in shortest.steps]
        return BaselineSnapshot(
            n=n, n_success=n_success, sr=sr,
            avg_success_steps=avg_success_steps,
            per_rollout=[BaselineRolloutSummary(success=t.success,
                                                  steps=t.duration_steps)
                          for t in traces],
            sample_success_actions=sample_actions,
        )

    @staticmethod
    def _baseline_snapshot_from_cached(result: dict) -> BaselineSnapshot:
        from envharness.core.types import Action
        n = int(result.get("n", 0))
        rollouts = result.get("rollouts", []) or []
        n_success = sum(1 for r in rollouts if r.get("success"))
        success_steps = [int(r["steps"]) for r in rollouts if r.get("success")]
        avg_success_steps = (sum(success_steps) / len(success_steps)
                              if success_steps else None)
        sample_actions = [Action(**a) for a in
                            (result.get("sample_success_actions") or [])]
        return BaselineSnapshot(
            n=n,
            n_success=n_success,
            sr=float(result.get("sr", 0.0)),
            avg_success_steps=avg_success_steps,
            per_rollout=[BaselineRolloutSummary(
                              success=bool(r.get("success")),
                              steps=int(r.get("steps", 0)),
                          ) for r in rollouts],
            sample_success_actions=sample_actions,
        )

    def _rollout_k(self, candidate: Candidate, candidate_id: str,
                    iteration_id: str, task_idx: int, attempt: int,
                    ctx_task_id: int) -> list[Trace]:
        K = self.config.k_per_candidate
        conc = max(1, int(self.config.rollout_concurrency))

        # Seed selection: ALL K rollouts pin to ctx.task_id -- the same task
        # _compute_or_load_baseline already targeted -- so the candidate is
        # graded against its own baseline on the same task. Variance across
        # the K rollouts therefore comes from Policy temperature alone;
        # cross-task variety comes from the outer n_tasks sweep.
        specs_and_seeds: list[tuple[int, EpisodeSpec]] = []
        for k in range(K):
            seed = ctx_task_id
            specs_and_seeds.append((k, EpisodeSpec(
                env=EnvSpec(
                    import_path=self.env_spec.import_path,
                    reset_options=self.env_spec.reset_options,
                    reset_seed=seed,
                ),
                candidate=candidate,
                policy=self.policy_spec,
                iteration_id=iteration_id,
                task_id=self.config.task_id,
                max_steps=self.config.max_episode_steps,
            )))

        def _run_one(item: tuple[int, EpisodeSpec]) -> tuple[int, int, Trace, float]:
            k, spec = item
            t0 = time.time()
            trace = self._run_spec_with_endpoint(spec)
            return k, spec.env.reset_seed, trace, time.time() - t0

        results: dict[int, Trace] = {}
        if conc == 1:
            for item in specs_and_seeds:
                k, seed, trace, dur = _run_one(item)
                trace.candidate_id = candidate_id
                trace.rollout_idx = k
                # Truthful default: nothing is accepted yet. The end of
                # _iterate promotes the final group to kind="accepted".
                # Without this, the mid-loop objective refresh
                # (`objective.evaluate(history + all_traces)`) would count
                # every in-flight attempt as accepted (Trace.kind's model
                # default is "accepted").
                trace.kind = "exploration"
                results[k] = trace
                self._log_rollout(iteration_id, attempt, candidate_id, k,
                                   seed, dur, trace)
        else:
            with _cf.ThreadPoolExecutor(max_workers=conc) as pool:
                futures = [pool.submit(_run_one, item) for item in specs_and_seeds]
                for fut in _cf.as_completed(futures):
                    k, seed, trace, dur = fut.result()
                    trace.candidate_id = candidate_id
                    trace.rollout_idx = k
                    trace.kind = "exploration"   # see sequential branch above
                    results[k] = trace
                    self._log_rollout(iteration_id, attempt, candidate_id, k,
                                       seed, dur, trace)
        return [results[k] for k in range(K)]

    def _log_rollout(self, iteration_id: str, attempt: int,
                       candidate_id: str, k: int, seed: int, dur: float,
                       trace: Trace) -> None:
        self.log.event("rollout_done",
                        iteration_id=iteration_id, attempt=attempt,
                        candidate_id=candidate_id, rollout=k,
                        seed=seed,
                        duration_ms=int(dur * 1000),
                        success=trace.success,
                        reward=round(trace.final_reward, 3),
                        steps=trace.duration_steps,
                        error=trace.error)
