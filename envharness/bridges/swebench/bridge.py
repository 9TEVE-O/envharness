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

"""SWEBenchBridge -- wrap SWE-bench's per-instance Docker containers.

Contract follows the ActionableEnv ABC: zero mutation logic here; the
underlying runtime (Docker container per task instance) lives ONLY in
this env and is invisible from upstream. Mutations come from the
harness stack: Rules subclasses (A/T/O per-step hooks) and Setup
action replay (S0).

Setup requirements (one-time on the host):
    pip install swebench datasets
    # Docker daemon must be running. Per-instance images are pulled
    # lazily on first use (`docker.io/swebench/sweb.eval.x86_64.<id>`).

Reset options (passed through `options` dict in reset()):
    instance_id: str   -- e.g. "django__django-11099". If omitted, seed
                          mod len(dataset) indexes into the dataset
                          (deterministic alfworld-style fallback).
    subset: str        -- "lite" (default) | "verified" | "full" |
                          "multimodal" | "multilingual". Maps to the
                          princeton-nlp/SWE-bench_* HuggingFace dataset.
    split: str         -- "test" (default) | "dev". Same conventions as
                          SWE-bench's official harness.

Action contract:
    Single tool: `bash(command: str)`. Each step runs
    `docker exec -w /testbed <cid> bash -c <command>`. Stateless between
    calls -- no persistent shell session -- so `cd X && ...` patterns must
    be used inside one action.

    Termination is detected by a literal first-line sentinel in stdout:
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Everything after the sentinel
    is captured as the submitted patch. Pattern copied from mini-SWE-agent.

Evaluation:
    `evaluate()` shells out to `swebench.harness.run_evaluation` with the
    captured patch. The official scorer spins up a fresh container, applies
    the patch, runs the FAIL_TO_PASS + PASS_TO_PASS test sets, and writes a
    report JSON. We read that report and return EvaluationResult.

Snapshot/restore: NOT implemented. `--rm` containers can't be cheaply
forked. S0 mutation = the Setup harness replaying bash actions on reset.

Design lineage:
  - Mini-SWE-agent (github.com/SWE-agent/mini-swe-agent): single bash tool,
    stateless docker exec per step, --rm container per episode.
  - Official scorer: `swebench.harness.run_evaluation.main(...)`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env
from envharness.bridges.swebench.tools import Bash, StrReplaceEditor
from envharness.core.tool import Tool
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
    TaskSummary,
)


# Sentinel mini-SWE-agent uses to mark a submission. We keep the same string
# so prompts / trajectories transfer between the two systems.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Default per-step timeout (seconds) for `docker exec`. Mini-SWE-agent uses 60s.
DEFAULT_STEP_TIMEOUT = 60

# Default container lifespan upper bound (seconds). Container `sleep N` then exits.
DEFAULT_CONTAINER_TIMEOUT = 7200      # 2h, matches mini-SWE-agent

# Max observation length (chars). mini-SWE-agent uses 10_000. Rules's O hook
# can override per-bridge, but this is the Bridge-level default cap.
DEFAULT_OBS_TRUNCATE = 10_000

# Subset short name -> (HF dataset id, default split). Same names mini-SWE-agent
# accepts; users override via `reset_options.subset`. SWE-Gym entries added for
# training (SWE-Gym is the canonical training-set counterpart to SWE-bench;
# Pan et al. 2024). For local-parquet-based training (no HF download) pass
# `options["instance_row"]` directly -- the dataset lookup is bypassed.
_SUBSET_MAP: dict[str, tuple[str, str]] = {
    "lite":         ("princeton-nlp/SWE-bench_Lite",       "test"),
    "verified":     ("princeton-nlp/SWE-bench_Verified",   "test"),
    "full":         ("princeton-nlp/SWE-bench",            "test"),
    "multimodal":   ("princeton-nlp/SWE-bench_Multimodal", "test"),
    "multilingual": ("princeton-nlp/SWE-bench_Multilingual", "test"),
    "swe_gym":      ("SWE-Gym/SWE-Gym",                    "train"),
    "swe_gym_lite": ("SWE-Gym/SWE-Gym-Lite",               "train"),
}

# Cached HF datasets are large (~1GB metadata each). Load each subset once
# per process and share. Indexed by (subset, split).
_DATASET_CACHE: dict[tuple[str, str], Any] = {}
_DATASET_CACHE_LOCK = threading.Lock()


@dataclass
class SWEBenchEnvState:
    """Env-exposed view of SWE-bench state for Rules hooks.

    Pure data (str / int / bool / dict). No docker container handles. The
    same dataclass shape would be valid if we later swap docker for podman,
    modal, or remote-execution; the runtime-agnostic invariant is preserved.
    """
    instance_id: str = ""              # e.g. "django__django-11099"
    repo: str = ""                     # e.g. "django/django"
    base_commit: str = ""              # pre-fix git ref (instance is reset to this)
    problem_statement: str = ""        # the GitHub-issue text the Policy must address
    hints_text: str = ""               # optional extra hints from the dataset row
    last_command: str = ""             # most recent bash command issued
    last_output: str = ""              # most recent stdout+stderr (post-truncation)
    last_returncode: int = 0           # most recent docker-exec returncode
    submitted: bool = False            # True once the sentinel was emitted
    submitted_patch: str = ""          # lines after the sentinel, joined
    step_count: int = 0
    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form bag for Rules per-episode state (counters, RNG state,
    accumulators). Bridge never reads this."""


@register_env("swebench")
class SWEBenchEnv(ActionableEnv):
    """SWE-bench ActionableEnv. Per-instance --rm docker container; per-step
    `docker exec`. Submission via the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
    sentinel; scoring via the official `swebench.harness.run_evaluation`.
    """

    tool_registry: ClassVar[list[type[Tool]]] = [Bash, StrReplaceEditor]

    def __init__(self) -> None:
        super().__init__()
        self._container_id: str | None = None
        self._container_name: str | None = None
        self._subset: str = "lite"
        self._split: str = "test"
        self._step_timeout: int = DEFAULT_STEP_TIMEOUT
        self._container_timeout: int = DEFAULT_CONTAINER_TIMEOUT
        self._obs_truncate: int = DEFAULT_OBS_TRUNCATE
        self._max_workers: int = 1
        self._eval_timeout: int = 1800     # 30 min cap on official run_evaluation
        self.state: SWEBenchEnvState = SWEBenchEnvState()
        # save_state retains the configuration so a checkpoint can rebuild
        # the env (modulo container rebirth) at the next reset() call.
        self._last_reset_seed: int | None = None
        self._last_reset_options: dict[str, Any] = {}

    # -- core env interface -------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        opts = options or {}
        self._last_reset_seed = seed
        self._last_reset_options = dict(opts)
        self._subset = (opts.get("subset") or "lite").lower()
        if self._subset not in _SUBSET_MAP:
            raise ValueError(
                f"Unknown SWE-bench subset {self._subset!r}; choose one of "
                f"{sorted(_SUBSET_MAP)}"
            )
        default_split = _SUBSET_MAP[self._subset][1]
        self._split = opts.get("split") or default_split
        self._step_timeout = int(opts.get("step_timeout_seconds")
                                  or DEFAULT_STEP_TIMEOUT)
        self._container_timeout = int(opts.get("container_timeout_seconds")
                                       or DEFAULT_CONTAINER_TIMEOUT)
        self._obs_truncate = int(opts.get("obs_truncate_chars")
                                  or DEFAULT_OBS_TRUNCATE)
        self._eval_timeout = int(opts.get("eval_timeout_seconds") or 1800)

        # Pick the instance: three paths, in priority order.
        #   1. options["instance_row"] -- caller-supplied dict (skip HF entirely).
        #      Used by the RL adapter to drive the Bridge from a local parquet
        #      without depending on HF download at training time.
        #   2. options["instance_id"] -- look up by id in the HF dataset.
        #   3. seed mod len(ds) -- deterministic fallback.
        instance_row = opts.get("instance_row")
        instance_id = opts.get("instance_id") or ""
        if instance_row is not None:
            row = dict(instance_row)
            for required in ("instance_id", "repo", "base_commit", "problem_statement"):
                if required not in row:
                    raise ValueError(
                        f"reset_options['instance_row'] missing required field "
                        f"{required!r}; got keys {sorted(row)}"
                    )
        else:
            ds = _load_dataset(self._subset, self._split)
            if instance_id:
                try:
                    idx = next(i for i, r in enumerate(ds)
                                if r["instance_id"] == instance_id)
                except StopIteration:
                    raise ValueError(
                        f"instance_id={instance_id!r} not found in "
                        f"{self._subset}/{self._split}"
                    )
                row = ds[idx]
            else:
                idx = (seed or 0) % len(ds)
                row = ds[idx]

        # Tear down any prior container before starting the new one. close()
        # is also called by the runner's try/finally, but a manual reset()
        # mid-episode (test/dev path) should also recycle the container.
        self._teardown_container()

        # Start the per-instance container. The image MUST already be pulled,
        # OR docker will auto-pull on first `docker run` (slow first time;
        # ~3GB / instance).
        image = _instance_image_name(row["instance_id"], self._subset)
        # PRE-PULL: ensure the image is present locally before `docker run`.
        # Avoids the daemon's lazy-pull-then-run race that otherwise manifests
        # as `docker run` returning exit 1 ("image not found" / partial pull).
        # Failures of `docker pull` are silently tolerated -- the next
        # `docker run` will lazy-pull if needed.
        try:
            subprocess.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=300,
            )
        except Exception:
            pass
        # `--rm` so container auto-removes on stop; `sleep N` keeps it alive
        # for at most N seconds. `-w /testbed` matches mini-swe-agent + the
        # pre-built image layout (repo is checked out at /testbed inside).
        #
        # Retry on docker daemon race: when the host is under heavy
        # concurrent docker load (load average >> #cpus, 100+ containers),
        # `docker run` occasionally returns non-zero or "Conflict" or
        # "address already in use" even though the image+name are valid.
        # Retry up to 5 times with exponential backoff (3s, 6s, 12s, 24s, 48s),
        # regenerating the container name each attempt so a stale entry from
        # a half-aborted previous attempt does not block the rename.
        MAX_DOCKER_RUN_ATTEMPTS = 5
        proc = None
        last_err = ""
        for attempt in range(MAX_DOCKER_RUN_ATTEMPTS):
            cname = f"envharness-swebench-{uuid.uuid4().hex[:8]}"
            try:
                proc = subprocess.run(
                    ["docker", "run", "-d", "--rm",
                     "--name", cname,
                     "-w", "/testbed",
                     image, "sleep", str(self._container_timeout)],
                    capture_output=True, text=True, timeout=120,
                )
            except subprocess.TimeoutExpired as e:
                last_err = f"timeout: {e}"
                proc = None
            if proc is not None and proc.returncode == 0:
                break
            last_err = (
                f"exit={proc.returncode if proc else 'TIMEOUT'} "
                f"stderr={(proc.stderr[-500:] if proc else '')!r}"
            )
            if attempt < MAX_DOCKER_RUN_ATTEMPTS - 1:
                import time as _time
                _time.sleep(3 * (2 ** attempt))
        if proc is None or proc.returncode != 0:
            raise RuntimeError(
                f"`docker run` failed for image {image} after "
                f"{MAX_DOCKER_RUN_ATTEMPTS} attempts (daemon race?): "
                f"{last_err}"
            )
        self._container_id = proc.stdout.strip()
        self._container_name = cname

        # Populate env_state. Note: `base_commit` is the pre-fix commit;
        # the pre-built image already has /testbed checked out at this commit,
        # so we don't need to git-reset on every reset (mini-swe-agent skips
        # this too). If the HarnessAgent's perturbation leaves /testbed dirty, a
        # later step's `git status` will surface it; we don't force a clean.
        self.state = SWEBenchEnvState(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            hints_text=row.get("hints_text") or "",
            last_command="",
            last_output="",
            last_returncode=0,
            submitted=False,
            submitted_patch="",
            step_count=0,
        )
        return EnvResetResponse(
            observation=self._observe(),
            info={
                "instance_id": self.state.instance_id,
                "repo": self.state.repo,
                "base_commit": self.state.base_commit,
                "container_id": self._container_id,
            },
        )

    def step(self, action: Action) -> EnvResponse:
        # Bypass tool_registry dispatch: the docker container handle lives on
        # the Bridge, not on env_state, so we can't route through Tool.invoke.
        # We support two tools:
        #   bash               -- arbitrary shell (existing path)
        #   str_replace_editor -- structured file edit ops (OpenHands-style)
        if action.name == StrReplaceEditor.name:
            return self._step_str_replace_editor(action)
        if action.name != Bash.name:
            return EnvResponse(
                observation=Observation(
                    text=f"[unknown tool: {action.name}]",
                    data={"error": "unknown_tool"},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "unknown_tool"},
            )
        command = (action.kwargs.get("command") or "").strip()
        if not command:
            return EnvResponse(
                observation=Observation(
                    text="[empty command]",
                    data={"error": "empty_command"},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "empty_command"},
            )
        if self._container_id is None:
            return EnvResponse(
                observation=Observation(
                    text="[container not running -- call reset() first]",
                    data={"error": "no_container"},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "no_container"},
            )

        self.state.step_count += 1
        self.state.last_command = command

        # Run the command. We merge stderr into stdout (mini-swe-agent
        # convention) so the model sees the full picture in one stream.
        # `bash -c` (NOT -lc) because mini-SWE-agent uses non-login shells
        # and login shells re-source .bashrc / .profile which sometimes
        # print banners that pollute output. Quality-of-life env vars
        # (PAGER, MANPAGER, LESS, PIP_PROGRESS_BAR, TQDM_DISABLE) mirror
        # mini-SWE-agent's swebench.yaml `environment.env` block so
        # pagers / progress bars don't burn observation budget.
        # ALIGNMENT: plain `bash -c command`, NO conda-activate wrapper. A
        # conda wrapper plus stdout-only submission detection can miss the
        # agent's `echo SENTINEL && git diff` (submitted=False on solved
        # tasks). mini-SWE-agent uses a non-login `bash -c` and merges
        # stderr into stdout so the model sees one stream.
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/testbed",
                 "-e", "PAGER=cat",
                 "-e", "MANPAGER=cat",
                 "-e", "LESS=-R",
                 "-e", "PIP_PROGRESS_BAR=off",
                 "-e", "TQDM_DISABLE=1",
                 self._container_id,
                 "bash", "-c", command],
                capture_output=True, text=True, timeout=self._step_timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as e:
            output = (
                f"[docker exec timed out after {self._step_timeout}s]\n"
                + ((e.stdout or "") if isinstance(e.stdout, str) else "")
                + ((e.stderr or "") if isinstance(e.stderr, str) else "")
            )
            returncode = 124       # canonical timeout code
            timed_out = True

        # Submission detection: first non-empty line of output == sentinel.
        # Everything after the sentinel line is the submitted patch (mini-
        # swe-agent's convention -- agent does `echo SENTINEL && cat patch.txt`).
        # Detection + patch capture must run on the MERGED stdout+stderr
        # stream: a stdout-only variant drops submissions whose git-diff
        # prologue lands after a stderr line.
        submitted = False
        submitted_patch = ""
        first_line = ""
        for ln in (output or "").splitlines():
            if ln.strip():
                first_line = ln.strip()
                break
        if first_line == SUBMIT_SENTINEL and returncode == 0:
            submitted = True
            # Drop everything up to and including the sentinel line, keep rest.
            lines = (output or "").splitlines()
            after: list[str] = []
            seen = False
            for ln in lines:
                if not seen and ln.strip() == SUBMIT_SENTINEL:
                    seen = True
                    continue
                if seen:
                    after.append(ln)
            submitted_patch = "\n".join(after).strip()

        truncated_output = _truncate_text(output, self._obs_truncate)

        self.state.last_output = truncated_output
        self.state.last_returncode = returncode
        if submitted:
            self.state.submitted = True
            self.state.submitted_patch = submitted_patch

        reward = 1.0 if submitted else 0.0     # raw signal; R-axis hook can reshape
        terminated = submitted
        # A per-command timeout truncates the episode (truncated = timed_out).
        truncated = timed_out

        return EnvResponse(
            observation=self._observe(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                # Same shape as alfworld + toy24: `success` (None until terminated),
                # `result` (per-step payload), plus swebench extras.
                "success": (submitted if terminated else None),
                "result": {
                    "output": truncated_output,
                    "returncode": returncode,
                    "submitted": submitted,
                    "timed_out": timed_out,
                },
                "returncode": returncode,
                "submitted": submitted,
                "submitted_patch_len": len(submitted_patch),
                "timed_out": timed_out,
                "step_count": self.state.step_count,
            },
        )

    def _extract_patch(self) -> str:
        """Canonical patch extraction (OpenHands `complete_runtime`): stage
        EVERYTHING (so new files are included) and diff the staged tree
        against the pre-fix commit, with NO truncation. This is authoritative
        -- a parse-agent-stdout approach misses new/staged files (the
        prompt's plain `git diff` only shows unstaged tracked changes) and
        understates the solve rate. Best-effort: "" on any failure."""
        if self._container_id is None:
            return ""
        base = self.state.base_commit
        cmd = ('cd /testbed && '
               'git config --global core.pager "" 2>/dev/null; '
               'git add -A 2>/dev/null; '
               f'git diff --no-color --cached {base}')
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/testbed", self._container_id,
                 "bash", "-c", cmd],
                capture_output=True, text=True, timeout=600,
            )
        except Exception:
            return ""
        return (proc.stdout or "").strip() if proc.returncode == 0 else ""

    def evaluate(self) -> EvaluationResult:
        # Require an EXPLICIT submit. Do NOT add a live `_extract_patch`
        # fallback at episode end: (a) it can overwrite a good
        # sentinel-captured patch with an empty `git diff --cached` on
        # containers where staging silently no-ops, and (b) it lets
        # non-submitting rollouts look scorable, inflating the empty-patch
        # rate and depressing SR. The Policy must emit the submit sentinel
        # to win.
        if not self.state.submitted or not self.state.submitted_patch:
            return EvaluationResult(
                success=False, score=0.0,
                metrics={"reason": "no_submission",
                          "steps": self.state.step_count},
            )

        # Hand the patch to the official harness. This is the slow path
        # (~1-5 min/instance because it spins up a *fresh* container of the
        # SAME image, applies the patch, runs the FAIL_TO_PASS + PASS_TO_PASS
        # test sets, parses results).
        try:
            resolved, report = _run_official_evaluation(
                instance_id=self.state.instance_id,
                model_patch=self.state.submitted_patch,
                subset=self._subset,
                split=self._split,
                timeout=self._eval_timeout,
            )
        except Exception as e:
            # Don't mask a scoring error as success. Report and treat as failure.
            # ALSO surface on stderr: Trace only persists eval SUCCESS (not
            # metrics), so without this line a broken scorer (e.g. missing
            # swebench package) silently zeroes every SR.
            print(f"[SWEBenchEnv.evaluate] SCORER ERROR for "
                  f"{self.state.instance_id}: {type(e).__name__}: {e}",
                  file=__import__("sys").stderr, flush=True)
            return EvaluationResult(
                success=False, score=0.0,
                metrics={
                    "reason": "scorer_error",
                    "error": f"{type(e).__name__}: {e}",
                    "steps": self.state.step_count,
                    "submitted_patch_len": len(self.state.submitted_patch),
                },
            )
        return EvaluationResult(
            success=bool(resolved),
            score=1.0 if resolved else 0.0,
            metrics={
                "resolved": bool(resolved),
                "steps": self.state.step_count,
                "submitted_patch_len": len(self.state.submitted_patch),
                **(report or {}),
            },
        )

    def get_env_state(self) -> SWEBenchEnvState:
        return self.state

    def observe(self) -> Observation:
        return self._observe()

    @classmethod
    def env_state_schema(cls) -> str:
        return (
            "env_state is a SWEBenchEnvState dataclass with these fields:\n"
            "  instance_id: str            -- SWE-bench task id, e.g. 'django__django-11099'\n"
            "  repo: str                   -- GitHub slug, e.g. 'django/django'\n"
            "  base_commit: str            -- pre-fix git ref the container starts at\n"
            "  problem_statement: str      -- the GitHub-issue text the Policy must fix\n"
            "  hints_text: str             -- optional extra hints (often empty)\n"
            "  last_command: str           -- most recent bash command issued\n"
            "  last_output: str            -- most recent stdout+stderr (post-truncation)\n"
            "  last_returncode: int        -- most recent docker-exec exit code\n"
            "  submitted: bool             -- True once the Policy emitted the sentinel\n"
            "  submitted_patch: str        -- the captured git diff (post-sentinel lines)\n"
            "  step_count: int             -- actions taken so far\n"
            "  extras: dict                -- free dict for Rules per-episode state\n"
            "S0 changes go through in_env_actions (Setup layer), not Rules. Rules hooks may read all\n"
            "fields and write to `extras`. Modifying last_output in filter_observation\n"
            "is done by returning a new Observation, NOT by writing env_state.last_output."
        )

    # -- save / load --------------------------------------------------------
    #
    # SWE-bench containers are --rm; we cannot snapshot a live container
    # cheaply. save/load stores only the configuration (reset args) so the
    # episode-boundary contract is preserved. Loading returns a fresh
    # instance; the caller must call reset() to pull the image + boot the
    # container.

    def save_state(self) -> dict:
        return {
            "reset_seed":    self._last_reset_seed,
            "reset_options": dict(self._last_reset_options),
        }

    @classmethod
    def from_state(cls, state: dict) -> "SWEBenchEnv":
        env = cls()
        env._last_reset_seed = state.get("reset_seed")
        env._last_reset_options = dict(state.get("reset_options") or {})
        return env

    def default_reset_args(self) -> tuple[int | None, dict]:
        return self._last_reset_seed, dict(self._last_reset_options)

    # -- agent-driven task selection ---------------------------------------

    @classmethod
    def list_tasks(cls, reset_options: dict | None = None,
                    limit: int | None = None) -> list[TaskSummary]:
        """Enumerate SWE-bench instances for the given subset/split.

        Disk-cached under runs/_task_summary_cache/<subset>__<split>.json
        because the HuggingFace dataset is slow to enumerate (first load
        ~30s + per-row metadata access). Subsequent calls are ~instant.
        """
        opts = dict(reset_options or {})
        subset = (opts.get("subset") or "lite").lower()
        if subset not in _SUBSET_MAP:
            raise ValueError(f"Unknown SWE-bench subset {subset!r}; "
                             f"choose one of {sorted(_SUBSET_MAP)}")
        hf_id, default_split = _SUBSET_MAP[subset]
        split = opts.get("split") or default_split
        n = limit if limit is not None else 50

        # Disk cache
        cache_dir = Path("runs/_task_summary_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"swebench__{subset}__{split}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                return [TaskSummary(**r) for r in cached[:n]]
            except Exception:
                pass    # corrupt cache; rebuild

        from datasets import load_dataset    # heavy import; lazy
        ds = load_dataset(hf_id, split=split)
        all_summaries = []
        for i, row in enumerate(ds):
            ps = (row.get("problem_statement") or "")[:200]
            ps_one_line = " ".join(ps.split())  # collapse whitespace
            ftp = row.get("FAIL_TO_PASS") or "[]"
            n_f2p = len(json.loads(ftp)) if isinstance(ftp, str) else len(ftp)
            all_summaries.append(TaskSummary(
                task_idx=i,
                instance_id=row.get("instance_id", f"unknown-{i}"),
                brief=ps_one_line,
                metadata={"repo": row.get("repo", ""),
                          "n_fail_to_pass_tests": n_f2p,
                          "subset": subset, "split": split},
            ))
        # Cache all (not just first n) so future limit values can be served
        try:
            cache_path.write_text(json.dumps([s.model_dump() for s in all_summaries]))
        except Exception:
            pass
        return all_summaries[:n]

    # -- close --------------------------------------------------------------

    def close(self) -> None:
        super().close()
        self._teardown_container()

    # -- str_replace_editor --------------------------------------------------
    #
    # Mirrors OpenHands' file-edit tool surface. The model emits a single
    # structured action ({operation, path, ...}) instead of building a fragile
    # `sed -i` invocation. This is the single biggest scaffolding lever for
    # 7-8B models: bash-only baselines fare poorly on Lite precisely because
    # small models botch `sed` escapes; this tool removes that failure mode.
    #
    # Runs `python3 -c "..."` inside the docker container so the editor
    # behavior is identical across docker / podman / future runtimes.

    def _step_str_replace_editor(self, action: Action) -> EnvResponse:
        if self._container_id is None:
            return EnvResponse(
                observation=Observation(
                    text="[container not running -- call reset() first]",
                    data={"error": "no_container"},
                ),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "no_container"},
            )
        self.state.step_count += 1

        op = (action.kwargs.get("operation") or "").strip()
        path = (action.kwargs.get("path") or "").strip()
        # Path safety: must be absolute and live under /testbed. (We don't
        # block edits to system files at the container level -- the container
        # is ephemeral and we'd rather give the model a clear error so it
        # can correct course.)
        if not path or not path.startswith("/testbed"):
            obs = (
                f"[str_replace_editor error: `path` must be an absolute path "
                f"under /testbed. Got {path!r}.]"
            )
            self.state.last_command = f"str_replace_editor({op}, {path})"
            self.state.last_output = obs
            self.state.last_returncode = 2
            return EnvResponse(
                observation=Observation(text=obs, data={"error": "bad_path"}),
                reward=0.0, terminated=False, truncated=False,
                info={"error": "bad_path"},
            )

        # Embed the editor program. Runs inside the container; reads/writes
        # `path`. Communicates back via stdout (success message or error).
        editor_py = r"""
import json, os, sys
op = os.environ['EH_OP']
path = os.environ['EH_PATH']
def _read(p):
    try:
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"[error] could not read {p}: {type(e).__name__}: {e}")
        sys.exit(2)
def _write(p, content):
    try:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"[error] could not write {p}: {type(e).__name__}: {e}")
        sys.exit(2)

if op == "view":
    if os.path.isdir(path):
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            print(f"[error] could not list dir {path}: {e}"); sys.exit(2)
        print(f"[directory listing: {path}]")
        for e in entries:
            full = os.path.join(path, e)
            tag = "/" if os.path.isdir(full) else ""
            print(f"  {e}{tag}")
        sys.exit(0)
    if not os.path.isfile(path):
        print(f"[error] no such file or directory: {path}"); sys.exit(2)
    text = _read(path)
    rng = os.environ.get("EH_VIEW_RANGE", "")
    lines = text.splitlines()
    if rng:
        try:
            a, b = rng.split("-", 1); a, b = int(a), int(b)
        except Exception:
            print(f"[error] bad view_range {rng!r}; want \"start-end\"")
            sys.exit(2)
        if b < 0:
            b = len(lines)   # schema promises "[start, -1] reads to end of file"
        a = max(1, a); b = min(len(lines), b)
        print(f"[file: {path}, lines {a}-{b} of {len(lines)}]")
        for i in range(a-1, b):
            print(f"{i+1:6d}\t{lines[i]}")
    else:
        print(f"[file: {path}, {len(lines)} lines]")
        for i, ln in enumerate(lines):
            print(f"{i+1:6d}\t{ln}")
    sys.exit(0)

if op == "create":
    if os.path.exists(path):
        print(f"[error] create failed: {path} already exists. Use str_replace to edit it.")
        sys.exit(2)
    body = os.environ.get("EH_FILE_TEXT", "")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _write(path, body)
    print(f"[created: {path} ({len(body)} chars)]")
    sys.exit(0)

if op == "str_replace":
    if not os.path.isfile(path):
        print(f"[error] str_replace failed: no such file: {path}"); sys.exit(2)
    text = _read(path)
    old_str = os.environ.get("EH_OLD_STR", "")
    new_str = os.environ.get("EH_NEW_STR", "")
    if not old_str:
        print("[error] str_replace requires non-empty old_str"); sys.exit(2)
    n = text.count(old_str)
    if n == 0:
        # Show a tiny diagnostic so the model can re-locate.
        snippet = old_str.splitlines()[0][:80] if old_str else ""
        print(f"[error] str_replace failed: old_str not found in {path}. "
              f"First line of old_str (truncated): {snippet!r}")
        sys.exit(2)
    if n > 1:
        print(f"[error] str_replace failed: old_str appears {n} times in "
              f"{path}. Make old_str more specific so it matches exactly once.")
        sys.exit(2)
    new_text = text.replace(old_str, new_str, 1)
    _write(path, new_text)
    # Print a small confirmation showing the replaced region in context.
    before_lines = text[:text.index(old_str)].count("\n")
    n_replaced = old_str.count("\n") + 1
    n_new = new_str.count("\n") + 1
    print(f"[str_replace: {path}: replaced {n_replaced} line(s) with "
          f"{n_new} line(s) starting at line {before_lines + 1}]")
    sys.exit(0)

if op == "insert":
    if not os.path.isfile(path):
        print(f"[error] insert failed: no such file: {path}"); sys.exit(2)
    text = _read(path)
    try:
        insert_line = int(os.environ.get("EH_INSERT_LINE", "0"))
    except Exception:
        print(f"[error] bad insert_line"); sys.exit(2)
    new_str = os.environ.get("EH_NEW_STR", "")
    lines = text.splitlines(keepends=True)
    if insert_line < 0 or insert_line > len(lines):
        print(f"[error] insert_line {insert_line} out of range (file has {len(lines)} lines)")
        sys.exit(2)
    insert_block = new_str if new_str.endswith("\n") else new_str + "\n"
    new_lines = lines[:insert_line] + [insert_block] + lines[insert_line:]
    _write(path, "".join(new_lines))
    print(f"[insert: {path}: inserted {len(insert_block.splitlines())} line(s) after line {insert_line}]")
    sys.exit(0)

print(f"[error] unknown operation: {op}. "
      f"Valid: view | create | str_replace | insert.")
sys.exit(2)
"""

        env_extra = [
            "-e", f"EH_OP={op}",
            "-e", f"EH_PATH={path}",
            "-e", f"EH_FILE_TEXT={action.kwargs.get('file_text', '')}",
            "-e", f"EH_OLD_STR={action.kwargs.get('old_str', '')}",
            "-e", f"EH_NEW_STR={action.kwargs.get('new_str', '')}",
            "-e", f"EH_INSERT_LINE={action.kwargs.get('insert_line', 0)}",
            "-e", f"EH_VIEW_RANGE={action.kwargs.get('view_range', '')}",
            # Quality-of-life vars to match the bash path
            "-e", "PAGER=cat", "-e", "MANPAGER=cat", "-e", "LESS=-R",
            "-e", "PIP_PROGRESS_BAR=off", "-e", "TQDM_DISABLE=1",
        ]
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/testbed",
                 *env_extra, self._container_id,
                 "python3", "-c", editor_py],
                capture_output=True, text=True, timeout=self._step_timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as e:
            output = f"[docker exec timed out after {self._step_timeout}s]"
            returncode = 124
            timed_out = True

        truncated = _truncate_text(output, self._obs_truncate)
        self.state.last_command = f"str_replace_editor({op}, {path})"
        self.state.last_output = truncated
        self.state.last_returncode = returncode
        return EnvResponse(
            observation=Observation(text=truncated, data={
                "tool": "str_replace_editor",
                "operation": op,
                "path": path,
                "returncode": returncode,
            }),
            reward=0.0,
            terminated=False,
            # Editor timeouts are recoverable observations too (same policy
            # as the bash path above); the timeout text is in the obs.
            truncated=False,
            info={
                "success": None,    # editor never terminates the episode
                "tool": "str_replace_editor",
                "operation": op,
                "path": path,
                "returncode": returncode,
                "timed_out": timed_out,
            },
        )

    # -- helpers ------------------------------------------------------------

    def _teardown_container(self) -> None:
        if self._container_id is None:
            return
        # `docker stop` is graceful; the container was started with `--rm`
        # so it gets removed automatically once stopped. Force-kill as a
        # fallback if stop hangs OR returns nonzero (busy daemon) --
        # eval-container cleanup must not leak (containers otherwise live
        # up to the 2h sleep timeout).
        stopped = False
        try:
            proc = subprocess.run(["docker", "stop", "-t", "1", self._container_id],
                                   capture_output=True, text=True, timeout=15)
            stopped = proc.returncode == 0
        except Exception:
            stopped = False
        if not stopped:
            try:
                subprocess.run(["docker", "rm", "-f", self._container_id],
                                capture_output=True, text=True, timeout=15)
            except Exception:
                pass
        self._container_id = None
        self._container_name = None

    def _observe(self) -> Observation:
        # First-turn obs: problem statement + admissible-action hint. Later
        # turns: append the most recent command's output. Dual exposure in
        # text + data (alfworld convention).
        parts: list[str] = []
        if self.state.step_count == 0:
            parts.append(f"Repository: {self.state.repo}")
            parts.append(f"Instance: {self.state.instance_id}")
            parts.append(f"Base commit: {self.state.base_commit}")
            parts.append("")
            parts.append("Task (from the GitHub issue):")
            parts.append(self.state.problem_statement.strip())
            if self.state.hints_text.strip():
                parts.append("")
                parts.append("Hints:")
                parts.append(self.state.hints_text.strip())
            parts.append("")
            parts.append(
                "You can issue bash commands inside the task container. Working "
                f"directory: /testbed. Each command is a fresh `docker exec`, "
                "so shell state does not persist (use `&&` to chain). To submit, "
                f"echo the sentinel `{SUBMIT_SENTINEL}` followed by your git "
                "diff -- everything after the sentinel becomes the submitted patch."
            )
        else:
            parts.append(f"$ {self.state.last_command}")
            parts.append(self.state.last_output or "(no output)")
            parts.append(f"[returncode={self.state.last_returncode}]")
        text = "\n".join(parts)
        return Observation(
            text=text,
            data={
                "instance_id": self.state.instance_id,
                "repo": self.state.repo,
                "base_commit": self.state.base_commit,
                "step_count": self.state.step_count,
                "submitted": self.state.submitted,
                "last_returncode": self.state.last_returncode,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _instance_image_name(instance_id: str, subset: str = "lite") -> str:
    """Compute the per-instance pre-built image name SWE-bench publishes.

    Docker disallows `__` in repo names, so SWE-bench replaces it with
    `_1776_` (their "magic token"). The same scheme is what mini-SWE-agent
    and SWE-agent use, so the same image set works across all three.

    Example: "django__django-11099" -> "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"

    SWE-Gym instances: the `swebench/` Docker Hub org does NOT host SWE-Gym
    images. SWE-Gym (Pan et al. 2024) publishes its pre-built per-instance
    images under the `xingyaoww/` org (https://hub.docker.com/u/xingyaoww,
    per the SWE-Gym repo's README).
    """
    safe = instance_id.replace("__", "_1776_")
    org = "xingyaoww" if subset.startswith("swe_gym") else "swebench"
    return f"{org}/sweb.eval.x86_64.{safe}:latest"


def _load_dataset(subset: str, split: str):
    """Load (and memoize) the SWE-bench HF dataset for this subset/split."""
    key = (subset, split)
    with _DATASET_CACHE_LOCK:
        cached = _DATASET_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "`datasets` is required for SWEBenchBridge. Install with:\n"
                "  pip install datasets\n"
                f"Underlying ImportError: {e}"
            ) from e
        hf_id, _default_split = _SUBSET_MAP[subset]
        ds = load_dataset(hf_id, split=split)
        _DATASET_CACHE[key] = ds
        return ds


def _truncate_text(text: str, cap: int) -> str:
    """Head+tail truncation, matching mini-swe-agent's `<output_head>` /
    `<output_tail>` convention so long outputs (test runs, big diffs)
    don't flood the Policy's context. Total length stays under `cap`."""
    if not text or len(text) <= cap:
        return text or ""
    half = max(cap // 2 - 50, 200)
    head = text[:half]
    tail = text[-half:]
    elided = len(text) - len(head) - len(tail)
    return (head
            + f"\n\n[... {elided} characters elided ...]\n\n"
            + tail)


def _run_official_evaluation(*, instance_id: str, model_patch: str,
                                subset: str, split: str,
                                timeout: int) -> tuple[bool, dict[str, Any]]:
    """Invoke `swebench.harness.run_evaluation.main(...)` against one
    (instance_id, model_patch) and return (resolved, report_dict).

    Why subprocess and not direct import? `run_evaluation` writes a report
    JSON to cwd; setting up + isolating it is cleaner via a tempdir + a
    subprocess. Direct import would also pull SWE-bench's heavy CLI parser
    into the parent process. The fact that scoring already takes minutes
    dwarfs the ~1s subprocess startup.
    """
    hf_id = _SUBSET_MAP[subset][0]
    run_id = f"envharness-{uuid.uuid4().hex[:8]}"
    tmpdir = Path(tempfile.mkdtemp(prefix="envharness-swebench-eval-"))
    try:
        preds_path = tmpdir / "preds.jsonl"
        # SWE-bench expects one prediction per line keyed by instance_id.
        # `model_name_or_path` is the predictor identifier (free-form).
        with open(preds_path, "w") as f:
            json.dump({
                "instance_id": instance_id,
                "model_name_or_path": "envharness-policy",
                "model_patch": model_patch,
            }, f)
            f.write("\n")

        # Run the official harness. cwd=tmpdir so the report JSON lands there.
        # Use sys.executable, NOT bare "python": after a host reboot / conda-
        # base auto-activate, bare "python" can resolve to an interpreter
        # WITHOUT swebench (e.g. /opt/conda/bin/python), making this exit
        # non-zero and grading EVERY task as success=False.
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", hf_id,
            "--split", split,
            "--predictions_path", str(preds_path),
            "--instance_ids", instance_id,
            "--max_workers", "1",
            "--run_id", run_id,
            "--cache_level", "instance",      # keep the instance image cached
            "--report_dir", str(tmpdir),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(tmpdir),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"swebench.harness.run_evaluation exit {proc.returncode}; "
                f"stderr tail: {proc.stderr[-1500:]}"
            )
        # Report file naming: <model_name>.<run_id>.json (this changed across
        # swebench versions). Glob the report_dir for any *.json that
        # contains the instance_id.
        report = _find_and_parse_report(tmpdir, instance_id)
        resolved = bool(report.get("resolved_instances", 0) >= 1
                         if isinstance(report.get("resolved_instances"), int)
                         else report.get("resolved"))
        # `report` shape: {total_instances, submitted_instances,
        #                  completed_instances, resolved_instances,
        #                  unresolved_instances, empty_patch_instances,
        #                  error_instances, ...}
        return resolved, report
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def _find_and_parse_report(report_dir: Path, instance_id: str) -> dict[str, Any]:
    """Locate the per-run report JSON the harness wrote and parse it.

    Different swebench versions use slightly different naming -- we glob and
    pick the most recently modified report-shaped JSON in the dir.
    """
    candidates = sorted(
        (p for p in report_dir.glob("*.json") if p.name != "preds.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        try:
            with open(p) as f:
                obj = json.load(f)
        except Exception:
            continue
        # Top-level shape: {total_instances, resolved_instances, ...}.
        if isinstance(obj, dict) and (
            "resolved_instances" in obj or "resolved_ids" in obj
        ):
            # Also include the per-instance verdict if present.
            obj["instance_id"] = instance_id
            if "resolved_ids" in obj and instance_id in (obj.get("resolved_ids") or []):
                obj["resolved"] = True
            elif "resolved_instances" in obj:
                obj.setdefault("resolved",
                                int(obj.get("resolved_instances") or 0) >= 1)
            return obj
    raise RuntimeError(
        f"could not find a SWE-bench report JSON in {report_dir}; "
        f"directory contents: {sorted(p.name for p in report_dir.iterdir())}"
    )
