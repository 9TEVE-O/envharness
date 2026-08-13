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

"""Link -- serially compose TWO ActionableEnvs into ONE long-horizon episode.

`Link(env_a, env_b)` is an EnvHarness whose `step()` runs `env_a` until A's
task is finished, then transitions to `env_b` for the rest of the episode.
The composed env's `evaluate()` returns
`a.evaluate().success AND b.evaluate().success` -- both sub-tasks must pass.

The composed episode runs under a longer horizon than either sub-env alone
provides: the Policy manages one step budget across both segments, carries
a longer chat history, and is handed a different task mid-episode.

Env-agnostic invariant
----------------------
Link only ever calls the ActionableEnv ABC:
`reset / step / observe / evaluate / get_env_state / close`. It never
imports any concrete Bridge class, never inspects `Action.kwargs` or
`Observation.data` keys it doesn't recognize, and never type-checks env_a
or env_b against any specific class. Therefore Link works for ANY pair of
ActionableEnvs:

    Link(Toy24Env(), Toy24Env())
    Link(SWEBenchEnv(), SWEBenchEnv())
    Link(SWEBenchEnv(), WebArenaEnv())
    Link(AlfworldEnv(), Toy24Env())


Knobs (ctor params, all optionally overridable per-episode via `reset(
options={"link": {...}})`)
-----------------------------
- `a_done_via`:
    In BOTH modes, `step.terminated or step.truncated` always ends stage A
    (an ended sub-env cannot be stepped further).
    - `"submitted"`  -- ADDITIONALLY end stage A when
                       `step.info["submitted"] is True`. Use for benches
                       that have an explicit submit action (SWE-bench's
                       sentinel echo). Default.
    - `"terminated"` -- end stage A only on terminated/truncated. Use for
                       benches whose natural episode end == task finish
                       (ALFWorld, Toy24).
- `carry_context`:
    - `True`  (default): when transitioning to B, prepend a
                          ``[switched to new task]`` banner + a tail of A's
                          last observation to B's first observation.text.
                          Lets the Policy realize the env changed without
                          relying on the chat history alone.
    - `False`: no splice. B's first observation is exactly what
                B.reset() returned.

Per-leg reset options (the env-agnostic plumbing)
-------------------------------------------------
Two equivalent ways to feed per-leg reset options into the children:

  1. `link.reset(options={"a": {...env_a opts...},
                          "b": {...env_b opts...},
                          "link": {"carry_context": False, ...}})`
     -- routed: `options["a"]` -> `env_a.reset(seed, opts_a)`,
                `options["b"]` -> stashed for `env_b.reset(None, opts_b)`
                                    at handoff,
                `options["link"]` -> overrides ctor knobs for this episode.
     This is the canonical form. Generic across any pair of envs because
     Link never inspects WHAT's inside opts_a / opts_b -- it only routes.

  2. Legacy / single-env form: `link.reset(options={...env_a opts...})`
     -- if `options` has none of the reserved keys ("a", "b", "link"),
        the whole dict is treated as env_a's reset options (back-compat
        with naive composition). env_b.reset(None, None) at handoff.

Checkpointing is NOT supported in this schema version (a Link is a TREE,
not a single chain; schema v1 only walks `self._inner`). Both `save_state`
and `from_state` raise NotImplementedError.
"""
from __future__ import annotations

from typing import Any

from envharness.core.actionable_env import ActionableEnv
from envharness.core.envharness import EnvHarness
from envharness.core.registry import register_harness
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
    StepInfo,
)


_DONE_VIA_CHOICES = {"submitted", "terminated"}


@register_harness("link")
class Link(EnvHarness):
    """Serial composition of two ActionableEnvs into one long-horizon episode.

    `self.inner` (the EnvHarness ABC slot) holds env_a; env_b lives on
    `self._env_b`. This means a `dump_stack(Link(...))` walker would only
    see env_a -- which is one of the reasons Link checkpointing is
    explicitly disabled until schema v2.
    """

    def __init__(self, env_a: ActionableEnv | None = None,
                 env_b: ActionableEnv | None = None,
                 *,
                 carry_context: bool = True,
                 a_done_via: str = "submitted",
                 carry_chars: int = 1500) -> None:
        if a_done_via not in _DONE_VIA_CHOICES:
            raise ValueError(
                f"Link.a_done_via must be one of {_DONE_VIA_CHOICES}; "
                f"got {a_done_via!r}"
            )
        super().__init__(env_a)
        self._env_b: ActionableEnv | None = env_b
        self._carry_context = bool(carry_context)
        self._a_done_via = a_done_via
        self._carry_chars = int(carry_chars)

        # Per-episode mutable state (reset on every Link.reset())
        self._stage: str = "A"
        self._a_success: bool = False
        self._a_evaluated: bool = False
        self._a_eval_error: str | None = None
        self._b_success: bool = False
        self._b_evaluated: bool = False
        self._b_eval_error: str | None = None
        self._last_obs_a: Observation | None = None
        # Stashed env_b reset options + seed, set by Link.reset based on
        # options["b"]. Used at handoff time. Default (None, None) preserves
        # the legacy "no options for env_b" behavior.
        self._b_reset_seed: int | None = None
        self._b_reset_opts: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Children accessors -- read-only views for tests / introspection.
    # ------------------------------------------------------------------

    @property
    def env_a(self) -> ActionableEnv:
        return self.inner

    @property
    def env_b(self) -> ActionableEnv:
        if self._env_b is None:
            raise RuntimeError("Link has no env_b bound.")
        return self._env_b

    @property
    def stage(self) -> str:
        return self._stage

    # ------------------------------------------------------------------
    # ActionableEnv overrides
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> EnvResetResponse:
        # Always start in stage A. env_b is NOT reset here -- we lazy-reset
        # it at handoff so its runtime resources (Docker, browser, ...) only
        # spin up when actually needed. Saves wall+memory on Link episodes
        # that crash early in stage A.
        self._stage = "A"
        self._a_success = False
        self._a_evaluated = False
        self._a_eval_error = None
        self._b_success = False
        self._b_evaluated = False
        self._b_eval_error = None
        self._last_obs_a = None

        # Route `options` into the three buckets without inspecting any
        # bench-specific keys. Two accepted shapes:
        #   structured: {"a": <a_opts>, "b": <b_opts>, "link": <link_opts>}
        #   legacy:     <a_opts>  (back-compat single-env composition)
        opts = options or {}
        a_opts, b_opts, link_opts = self._split_options(opts)

        # Per-episode override of link-level knobs (carry_context, a_done_via,
        # carry_chars). Keeps Link generic across one-off test composition
        # AND runner-driven structured composition.
        if link_opts is not None:
            if "carry_context" in link_opts:
                self._carry_context = bool(link_opts["carry_context"])
            if "a_done_via" in link_opts:
                v = link_opts["a_done_via"]
                if v not in _DONE_VIA_CHOICES:
                    raise ValueError(
                        f"options.link.a_done_via must be one of "
                        f"{_DONE_VIA_CHOICES}; got {v!r}"
                    )
                self._a_done_via = v
            if "carry_chars" in link_opts:
                self._carry_chars = int(link_opts["carry_chars"])
            # Optional separate seed for env_b (otherwise None -- handoff
            # uses None which is interpreted as "deterministic" by most
            # benches).
            if "b_seed" in link_opts:
                self._b_reset_seed = link_opts["b_seed"]

        # Stash env_b's reset opts so _step_a can use them at handoff
        # without touching any bench-specific knowledge here.
        self._b_reset_opts = b_opts

        r = self.env_a.reset(seed, a_opts)
        self._last_obs_a = r.observation
        return EnvResetResponse(
            observation=r.observation,
            info={**dict(r.info), "link_stage": "A"},
        )

    @staticmethod
    def _split_options(opts: dict[str, Any]) -> tuple[
        dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None
    ]:
        """Route a generic options dict into (a_opts, b_opts, link_opts).

        Recognizes the structured form `{"a": {...}, "b": {...}, "link": {...}}`
        and falls back to legacy "treat the whole dict as a_opts" when none
        of the reserved keys are present. Returns None for any bucket that
        was not supplied (so children get their default reset behavior).
        """
        reserved = {"a", "b", "link"}
        if any(k in opts for k in reserved):
            return (opts.get("a"), opts.get("b"), opts.get("link"))
        # Legacy form: dict is env_a's options, nothing for env_b.
        return (opts or None), None, None

    def step(self, action: Action) -> EnvResponse:
        if self._stage == "A":
            return self._step_a(action)
        return self._step_b(action)

    def _step_a(self, action: Action) -> EnvResponse:
        resp = self.env_a.step(action)
        self._last_obs_a = resp.observation

        if not self._a_is_finished(resp):
            # Stage A continues -- pass through. We patch info so the
            # caller (runner) can always tell which stage produced the step.
            return EnvResponse(
                observation=resp.observation,
                reward=resp.reward,
                terminated=False,           # mask sub-env termination; only
                truncated=False,            # the Link decides when WE end
                info={**dict(resp.info), "link_stage": "A"},
            )

        # Stage A is done. Score it, then transition.
        try:
            self._a_success = bool(self.env_a.evaluate().success)
        except Exception as e:
            # Some Bridges raise if evaluate() is called before a submit;
            # if A_done_via is 'terminated' and the env never actually
            # accepted a submission, treat A as failed and keep going.
            self._a_success = False
            a_eval_error: str | None = f"{type(e).__name__}: {e}"
        else:
            a_eval_error = None
        self._a_eval_error = a_eval_error
        self._a_evaluated = True

        # Hand off to env_b. We DO NOT close env_a yet -- evaluate()
        # may have lazy state we want to keep alive until Link.close().
        # b options were stashed by Link.reset (see _split_options); pass
        # them through verbatim -- Link itself never inspects what's inside.
        try:
            r_b = self.env_b.reset(seed=self._b_reset_seed,
                                    options=self._b_reset_opts)
        except Exception as e:
            # If env_b can't even reset, the episode is over.
            return EnvResponse(
                observation=Observation(
                    text=f"[Link: env_b.reset failed: {type(e).__name__}: {e}]",
                    data={"link_stage": "failed_at_handoff",
                          "a_success": self._a_success},
                ),
                reward=0.0,
                terminated=True, truncated=False,
                info={"link_stage": "failed_at_handoff",
                      "a_success": self._a_success,
                      "a_eval_error": a_eval_error,
                      "handoff_error": f"{type(e).__name__}: {e}"},
            )

        first_obs_b = r_b.observation
        if self._carry_context:
            first_obs_b = self._splice_handoff(self._last_obs_a, first_obs_b)

        self._stage = "B"
        return EnvResponse(
            observation=first_obs_b,
            reward=0.0,                    # handoff itself isn't graded
            terminated=False, truncated=False,
            info={
                "link_stage": "switched_to_B",
                "a_success": self._a_success,
                "a_eval_error": a_eval_error,
                **dict(r_b.info),
            },
        )

    def _step_b(self, action: Action) -> EnvResponse:
        resp = self.env_b.step(action)
        if not (resp.terminated or resp.truncated):
            return EnvResponse(
                observation=resp.observation,
                reward=resp.reward,
                terminated=False,
                truncated=False,
                info={**dict(resp.info), "link_stage": "B"},
            )

        # Stage B finished -- compute combined verdict. Cache it so
        # Link.evaluate() can reuse it instead of re-running the evaluator
        # (for SWE-bench a re-run spins another official-scorer container).
        try:
            b_success = bool(self.env_b.evaluate().success)
        except Exception as e:
            b_success = False
            b_eval_error: str | None = f"{type(e).__name__}: {e}"
        else:
            b_eval_error = None
        self._b_success = b_success
        self._b_eval_error = b_eval_error
        self._b_evaluated = True

        combined = self._a_success and b_success
        return EnvResponse(
            observation=resp.observation,
            reward=float(combined),
            terminated=True,
            truncated=resp.truncated,
            info={
                **dict(resp.info),
                "link_stage": "done",
                "a_success": self._a_success,
                "b_success": b_success,
                "b_eval_error": b_eval_error,
                "combined_success": combined,
            },
        )

    def observe(self) -> Observation:
        active = self.env_a if self._stage == "A" else self.env_b
        obs = active.observe()
        if self._stage == "A":
            return obs
        if self._carry_context and self._last_obs_a is not None:
            # On re-observe in stage B we don't re-splice -- the splice
            # only happens on the handoff message. observe() returns the
            # live env_b observation as-is.
            return obs
        return obs

    def evaluate(self) -> EvaluationResult:
        # Prefer the verdicts cached at handoff (_step_a) / termination
        # (_step_b): re-invoking the sub-env evaluators is expensive
        # (SWE-bench spins an official-scorer container per call) and
        # env_a may raise if evaluate() precedes a submission.
        if self._a_evaluated:
            a_success = self._a_success
            a_score = float(self._a_success)
            a_metrics: dict[str, Any] = {"cached_at_handoff": True}
            if self._a_eval_error:
                a_metrics["eval_error"] = self._a_eval_error
        else:
            try:
                a = self.env_a.evaluate()
                a_success, a_score = a.success, a.score
                a_metrics = a.metrics
            except Exception as e:
                # Some envs raise if evaluate() is called before a
                # submission (same degrade shape as env_b below).
                a_success, a_score = False, 0.0
                a_metrics = {"not_evaluated": True,
                             "reason": f"{type(e).__name__}: {e}"}
        if self._b_evaluated:
            b_success = self._b_success
            b_score = float(self._b_success)
            b_metrics: dict[str, Any] = {"cached_at_termination": True}
            if self._b_eval_error:
                b_metrics["eval_error"] = self._b_eval_error
        else:
            try:
                b = self.env_b.evaluate()
                b_metrics = b.metrics
                b_success = b.success
                b_score = b.score
            except Exception as e:
                # If env_b was never reset (Link ended before handoff) this is
                # the natural state. Report b as not-yet-scored.
                b_metrics = {"not_evaluated": True,
                             "reason": f"{type(e).__name__}: {e}"}
                b_success = False
                b_score = 0.0
        return EvaluationResult(
            success=a_success and b_success,
            score=(a_score + b_score) / 2.0,
            metrics={
                "a_success": a_success,
                "a_score": a_score,
                "a_metrics": a_metrics,
                "b_success": b_success,
                "b_score": b_score,
                "b_metrics": b_metrics,
                "a_verdict_cached": self._a_evaluated,
                "b_verdict_cached": self._b_evaluated,
                "link_stage": self._stage,
                "a_evaluated_at_handoff": self._a_evaluated,
            },
        )

    def get_env_state(self) -> Any:
        return {
            "stage": self._stage,
            "a_success": self._a_success,
            "a_state": self.env_a.get_env_state(),
            "b_state": (self.env_b.get_env_state()
                        if self._env_b is not None else None),
            "carry_context": self._carry_context,
            "a_done_via": self._a_done_via,
        }

    def step_reward(self, step_info: StepInfo) -> float:
        # Delegate to whichever sub-env is currently active. The runner
        # produces one StepInfo per step regardless of stage; the active
        # env is the one whose state actually changed.
        active = self.env_a if self._stage == "A" else self.env_b
        return active.step_reward(step_info)

    def close(self) -> None:
        # Close both -- subprocess death doesn't auto-release docker etc.
        # We catch errors on each so a broken env_a close doesn't skip
        # env_b cleanup (matters: SWE-bench leaks containers otherwise).
        errors = []
        try:
            self.env_a.close()
        except Exception as e:
            errors.append(("env_a", e))
        if self._env_b is not None:
            try:
                self.env_b.close()
            except Exception as e:
                errors.append(("env_b", e))
        # Detach both so the EnvHarness ABC.close() contract is honored.
        self._inner = None
        self._env_b = None
        if errors:
            # Surface the first error so callers can log -- but only after
            # both close() attempts ran. Subprocess wrapping in the runner
            # will catch this regardless.
            kind, e = errors[0]
            raise RuntimeError(
                f"Link.close: {kind} raised {type(e).__name__}: {e}"
            )

    def default_reset_args(self) -> tuple[int | None, dict]:
        # Link does not own reset args -- env_a's are the canonical ones.
        return self.env_a.default_reset_args()

    def reset_after_load(self) -> bool:
        return self.env_a.reset_after_load()

    @classmethod
    def env_state_schema(cls) -> str:
        return (
            "Link.env_state = {\n"
            "  stage: 'A' | 'B' | 'done',\n"
            "  a_success: bool,                 # filled at handoff\n"
            "  a_state: <env_a's env_state schema>,\n"
            "  b_state: <env_b's env_state schema>,\n"
            "  carry_context: bool,\n"
            "  a_done_via: 'submitted' | 'terminated',\n"
            "}"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _a_is_finished(self, resp: EnvResponse) -> bool:
        # An ended sub-env cannot be stepped further, so termination /
        # truncation ALWAYS finishes stage A regardless of mode. Otherwise
        # a "submitted"-mode env that terminates without ever submitting
        # would be masked (terminated=False) and Link would keep stepping
        # a dead env until the runner's max_steps.
        if resp.terminated or resp.truncated:
            return True
        if self._a_done_via == "submitted":
            return bool((resp.info or {}).get("submitted"))
        # "terminated": only the terminated/truncated check above applies.
        return False

    def _splice_handoff(self, last_obs_a: Observation | None,
                        first_obs_b: Observation) -> Observation:
        """Splice a [switched to new task] banner + a tail of A's last
        observation onto the front of B's first observation. Carries via
        observation.text (string-only, so it works for any env regardless
        of observation.data shape)."""
        tail = ""
        if last_obs_a is not None and last_obs_a.text:
            t = last_obs_a.text
            if len(t) > self._carry_chars:
                t = "..." + t[-self._carry_chars:]
            tail = (f"\n\n[end of previous task A -- excerpt of last observation]\n"
                    f"{t}\n")
        banner = (
            "[Link: SWITCHED TO A NEW TASK. The previous task is over; "
            "your work on it is preserved in the chat history above but "
            "the environment below is a DIFFERENT task in a different "
            "repository. Re-orient before acting.]"
            f"{tail}\n"
            "[Begin new task B]\n"
        )
        new_data = dict(first_obs_b.data or {})
        new_data["linked_from_a"] = True
        return Observation(
            text=banner + (first_obs_b.text or ""),
            data=new_data,
        )

    # ------------------------------------------------------------------
    # Save / load  -- NOT supported in schema v1 (tree shape).
    # ------------------------------------------------------------------

    def save_state(self) -> dict:
        raise NotImplementedError(
            "Link checkpointing is not supported in checkpoint schema v1. "
            "A Link is a TREE (env_a + env_b), but schema v1's dump_stack "
            "walks a single chain via `self.inner`. Re-instantiate Link "
            "from a higher-level run config instead."
        )

    @classmethod
    def from_state(cls, state: dict,  # type: ignore[override]
                   inner: ActionableEnv | None = None) -> "Link":
        raise NotImplementedError(
            "Link.from_state is not supported in checkpoint schema v1; "
            "see Link.save_state for context."
        )
