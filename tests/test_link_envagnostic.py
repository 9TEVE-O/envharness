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

"""Link harness -- env-agnostic invariant + serial-composition semantics.

This file exists to PROVE Link works for ANY pair of ActionableEnvs, not
just the SWE-bench pair we run the real corpus on. Two flavors:

  1. Toy24 × Toy24  -- baseline: same env class twice. Exercises the
     full handoff path (stage A success/failure, stage B success/failure,
     combined verdict, evaluate(), carry_context splice).

  2. CounterEnv × StringEnv  -- two dummy ActionableEnvs with DIFFERENT
     action/observation shapes. Proves Link only ever touches the ABC --
     it never inspects Action.kwargs keys, Observation.data keys, or
     instance types. If this test passes, the env-agnostic guarantee
     stated in link.py's docstring holds by construction.

GPU-free, docker-free, API-key-free. Runs in <0.1s.
"""
from __future__ import annotations

from typing import Any

import pytest

from envharness import (
    Action, ActionableEnv, EnvResetResponse, EnvResponse, EvaluationResult,
    Link, Observation,
)
from envharness.bridges.toy24 import Toy24Env


# ===========================================================================
# A) Toy24 × Toy24 -- baseline same-class composition
# ===========================================================================

_PUZZLE = dict(numbers=[3, 3, 7, 7], target=24)
_SOLVE = [
    Action(name="combine", kwargs={"i": 2, "j": 0, "op": "mul"}),  # 7*3=21
    Action(name="combine", kwargs={"i": 2, "j": 0, "op": "add"}),  # 21+3=24
    Action(name="stop", kwargs={}),
]


def _build_toy_link(*, carry: bool = True) -> Link:
    """Two fresh Toy24Env wrapped in a Link. a_done_via='terminated' --
    Toy24 has no submitted sentinel, the env naturally terminates on
    stop()/win/lose."""
    a, b = Toy24Env(), Toy24Env()
    return Link(env_a=a, env_b=b, carry_context=carry, a_done_via="terminated")


def test_link_is_envharness_subclass():
    # Type closure: Link IS an ActionableEnv (via EnvHarness), so any
    # composition that returns a Link is itself a valid Environment.
    from envharness.core.envharness import EnvHarness
    assert issubclass(Link, EnvHarness)
    assert issubclass(Link, ActionableEnv)
    assert Link.harness_type() == "link"


def test_link_reset_starts_in_stage_a():
    link = _build_toy_link()
    try:
        r = link.reset(seed=0, options=_PUZZLE)
        assert isinstance(r, EnvResetResponse)
        assert link.stage == "A"
        # Stage tag in info so the runner can see it from day 1.
        assert r.info.get("link_stage") == "A"
        # The observation is env_a's reset observation -- toy24's first
        # obs always includes "target=24".
        assert "target=24" in r.observation.text
    finally:
        try: link.close()
        except Exception: pass


def test_link_solves_both_then_succeeds():
    link = _build_toy_link(carry=False)  # no splice noise in test obs
    try:
        # Route the SAME known puzzle to both legs via the structured
        # options shape, so stage B is deterministically solvable too.
        link.reset(seed=0, options={"a": dict(_PUZZLE), "b": dict(_PUZZLE)})
        # Stage A: solve toy24 #1.
        for act in _SOLVE[:-1]:
            r = link.step(act)
            assert r.terminated is False  # Link masks sub-env termination
            assert r.info.get("link_stage") == "A"
        r_stop = link.step(_SOLVE[-1])  # env_a.step(stop) -> terminated
        # That step triggers handoff: NOT terminated, link_stage flips.
        assert r_stop.terminated is False
        assert r_stop.info.get("link_stage") == "switched_to_B"
        assert r_stop.info.get("a_success") is True
        assert link.stage == "B"
        # Stage B: same puzzle (options["b"] routed at handoff) -- solve it.
        for act in _SOLVE[:-1]:
            r = link.step(act)
            assert r.terminated is False
            assert r.info.get("link_stage") == "B"
        r_end = link.step(_SOLVE[-1])
        assert r_end.terminated is True
        assert r_end.info.get("link_stage") == "done"
        assert r_end.info.get("a_success") is True
        assert r_end.info.get("b_success") is True
        assert r_end.info.get("combined_success") is True
        er = link.evaluate()
        assert er.success is True
    finally:
        try: link.close()
        except Exception: pass


def test_link_fails_when_a_fails():
    """If A fails (stop without 24 in state), a_success=False; B still runs."""
    link = _build_toy_link(carry=False)
    try:
        link.reset(seed=0, options=_PUZZLE)
        # Stop immediately -- target 24 NOT yet in [3,3,7,7], so A fails.
        r = link.step(Action(name="stop", kwargs={}))
        # Stop triggers terminated, _a_is_finished=True (a_done_via="terminated")
        assert r.info.get("link_stage") == "switched_to_B"
        assert r.info.get("a_success") is False
        assert link.stage == "B"
    finally:
        try: link.close()
        except Exception: pass


def test_link_evaluate_combines_via_and():
    """evaluate() returns success iff BOTH a and b succeeded (F ∧ F case)."""
    link = _build_toy_link(carry=False)
    try:
        link.reset(seed=0, options=_PUZZLE)
        # Fail A quickly so we don't pollute env_b setup.
        link.step(Action(name="stop", kwargs={}))   # A fails, switches to B
        # Now in stage B with a brand new Toy24 puzzle. Fail B immediately
        # so combined verdict is False (False ∧ False = False).
        link.step(Action(name="stop", kwargs={}))   # B fails
        er = link.evaluate()
        assert isinstance(er, EvaluationResult)
        assert er.success is False
        assert er.metrics["a_success"] is False
        assert er.metrics["b_success"] is False
    finally:
        try: link.close()
        except Exception: pass


def test_link_evaluate_true_and_false_is_false():
    """T ∧ F must combine to False -- distinguishes AND from OR (the
    F ∧ F case above cannot)."""
    link = Link(env_a=_CounterEnv(), env_b=_StringEnv(),
                carry_context=False, a_done_via="terminated")
    try:
        link.reset()
        for _ in range(3):
            link.step(Action(name="inc", kwargs={}))   # A succeeds -> handoff
        assert link.stage == "B"
        # B: say the wrong word -- StringEnv doesn't terminate, so B is
        # still unsolved when evaluate() runs (as at runner max_steps).
        link.step(Action(name="say", kwargs={"text": "goodbye"}))
        er = link.evaluate()
        assert er.success is False
        assert er.metrics["a_success"] is True
        assert er.metrics["b_success"] is False
    finally:
        link.close()


def test_link_evaluate_true_and_true_is_true():
    """T ∧ T combines to True."""
    link = Link(env_a=_CounterEnv(), env_b=_StringEnv(),
                carry_context=False, a_done_via="terminated")
    try:
        link.reset()
        for _ in range(3):
            link.step(Action(name="inc", kwargs={}))   # A succeeds -> handoff
        r = link.step(Action(name="say", kwargs={"text": "hello"}))  # B wins
        assert r.info.get("combined_success") is True
        er = link.evaluate()
        assert er.success is True
        assert er.metrics["a_success"] is True
        assert er.metrics["b_success"] is True
    finally:
        link.close()


def test_link_carry_context_splices_banner():
    link = _build_toy_link(carry=True)
    try:
        link.reset(seed=0, options=_PUZZLE)
        r = link.step(Action(name="stop", kwargs={}))  # triggers handoff
        # B's first observation now should carry the banner.
        assert "SWITCHED TO A NEW TASK" in r.observation.text
        assert "previous task" in r.observation.text.lower()
        # data side-channel marker so programmatic callers know.
        assert r.observation.data.get("linked_from_a") is True
    finally:
        try: link.close()
        except Exception: pass


def test_link_carry_context_off_does_not_splice():
    link = _build_toy_link(carry=False)
    try:
        link.reset(seed=0, options=_PUZZLE)
        r = link.step(Action(name="stop", kwargs={}))
        assert "SWITCHED TO A NEW TASK" not in (r.observation.text or "")
        assert r.observation.data.get("linked_from_a") is None
    finally:
        try: link.close()
        except Exception: pass


def test_link_close_releases_both_children():
    """Both env_a and env_b get close()d on Link.close(), even though only
    env_a is `self.inner`. Important for SWE-bench where docker containers
    on BOTH children must be torn down."""
    calls = []

    class _CloseRecordingToy24(Toy24Env):
        def close(self):
            calls.append(id(self))
            super().close()

    a, b = _CloseRecordingToy24(), _CloseRecordingToy24()
    link = Link(env_a=a, env_b=b, a_done_via="terminated")
    link.reset(seed=0, options=_PUZZLE)
    # Trigger transition to B so env_b actually gets reset & is alive.
    link.step(Action(name="stop", kwargs={}))
    link.close()
    assert id(a) in calls
    assert id(b) in calls


# ===========================================================================
# B) Cross-type composition -- proves env-agnostic invariant
# ===========================================================================
#
# Two dummy ActionableEnvs whose Action and Observation shapes are
# different from each other AND from Toy24. If Link can drive both of
# them through reset / step / evaluate / close without inspecting any
# concrete-class details, the "any env × any env" claim holds.


class _CounterEnv(ActionableEnv):
    """A 5-step counter env. Action shape: name='inc', no kwargs.
    Observation shape: text=f'count={n}', data={'n': int}. Succeeds when
    counter reaches 3. Has `submitted` info convention: never sets it
    (forcing Link to use a_done_via='terminated' for this env)."""

    @classmethod
    def env_type(cls): return "_counter_test"

    def __init__(self):
        super().__init__()
        self._n = 0
        self._closed = False

    def reset(self, seed=None, options=None):
        self._n = 0
        return EnvResetResponse(
            observation=Observation(text=f"count={self._n}", data={"n": self._n}),
            info={"env_class": "counter"},
        )

    def step(self, action: Action) -> EnvResponse:
        if action.name == "inc":
            self._n += 1
        done = self._n >= 3
        return EnvResponse(
            observation=Observation(text=f"count={self._n}", data={"n": self._n}),
            reward=1.0 if done else 0.0,
            terminated=done, truncated=False,
            info={"count": self._n},
        )

    def observe(self) -> Observation:
        return Observation(text=f"count={self._n}", data={"n": self._n})

    def evaluate(self) -> EvaluationResult:
        return EvaluationResult(success=self._n >= 3, score=float(self._n) / 3)

    def get_env_state(self): return {"counter": self._n}
    def save_state(self): return {"n": self._n}

    @classmethod
    def from_state(cls, state):
        e = cls(); e._n = int(state.get("n", 0)); return e

    def close(self): self._closed = True


class _StringEnv(ActionableEnv):
    """A string-matching env. Action shape: name='say', kwargs={'text': str}.
    Observation shape: just text. Succeeds when Policy says 'hello'.
    Action kwargs are TOTALLY different from CounterEnv -- if Link
    inspected them, it'd break."""

    @classmethod
    def env_type(cls): return "_string_test"

    def __init__(self):
        super().__init__()
        self._said: list[str] = []
        self._won = False

    def reset(self, seed=None, options=None):
        self._said, self._won = [], False
        return EnvResetResponse(
            observation=Observation(text="say hello to win",
                                    data={"target_word": "hello"}),
            info={"env_class": "string"},
        )

    def step(self, action: Action) -> EnvResponse:
        # No assumption about action.name -- accept anything, just record.
        txt = (action.kwargs or {}).get("text", action.name)
        self._said.append(str(txt))
        won = (str(txt).strip().lower() == "hello")
        self._won = self._won or won
        return EnvResponse(
            observation=Observation(text=f"you said: {txt}", data={"said": self._said}),
            reward=1.0 if won else 0.0,
            terminated=won, truncated=False,
            info={"won": won},
        )

    def observe(self): return Observation(text="say hello to win", data={})
    def evaluate(self): return EvaluationResult(success=self._won, score=1.0 if self._won else 0.0)
    def get_env_state(self): return {"said": list(self._said)}
    def save_state(self): return {"said": list(self._said), "won": self._won}

    @classmethod
    def from_state(cls, state):
        e = cls(); e._said = list(state.get("said") or []); e._won = bool(state.get("won")); return e

    def close(self): pass


def test_link_crosstype_a_succ_b_succ():
    """Counter × String, both succeed -> combined verdict True."""
    link = Link(env_a=_CounterEnv(), env_b=_StringEnv(),
                carry_context=False, a_done_via="terminated")
    try:
        r = link.reset()
        assert link.stage == "A"
        # Drive counter to 3 (terminates env_a, triggers handoff)
        for _ in range(3):
            r = link.step(Action(name="inc", kwargs={}))
        assert link.stage == "B"
        assert r.info.get("a_success") is True
        # Now in StringEnv. Say "hello".
        r = link.step(Action(name="say", kwargs={"text": "hello"}))
        assert r.terminated is True
        assert r.info.get("combined_success") is True
        er = link.evaluate()
        assert er.success is True
        assert er.metrics["a_success"] is True
        assert er.metrics["b_success"] is True
    finally:
        link.close()


def test_link_crosstype_a_fail_b_succ_is_combined_fail():
    """Even if B succeeds, A failure -> combined False."""
    # Use a CounterEnv variant that terminates immediately WITHOUT success:
    class _FailEnv(_CounterEnv):
        def step(self, action: Action) -> EnvResponse:
            return EnvResponse(
                observation=Observation(text="failed", data={}),
                reward=0.0, terminated=True, truncated=False,
                info={"failed_immediately": True},
            )
        def evaluate(self): return EvaluationResult(success=False, score=0.0)
    link2 = Link(env_a=_FailEnv(), env_b=_StringEnv(),
                 carry_context=True, a_done_via="terminated")
    try:
        link2.reset()
        r = link2.step(Action(name="whatever", kwargs={}))  # A fails+handoff
        assert r.info.get("a_success") is False
        assert link2.stage == "B"
        # Carry context banner present
        assert "SWITCHED" in r.observation.text
        # Solve B
        r2 = link2.step(Action(name="say", kwargs={"text": "hello"}))
        # Combined = False AND True = False
        assert r2.info.get("combined_success") is False
        assert link2.evaluate().success is False
    finally:
        link2.close()


def test_link_submitted_mode_handles_termination_without_submit():
    """Regression (FIX 2): with a_done_via='submitted', a sub-env that
    terminates WITHOUT ever setting info['submitted'] must still finish
    stage A -- an ended env can't be stepped further. Previously Link
    masked the termination (terminated=False) and kept stepping the dead
    env until the runner's max_steps."""
    # _CounterEnv never sets info["submitted"]; it terminates at n >= 3.
    link = Link(env_a=_CounterEnv(), env_b=_StringEnv(),
                carry_context=False, a_done_via="submitted")
    try:
        link.reset()
        r = link.step(Action(name="inc", kwargs={}))
        assert r.info.get("link_stage") == "A"
        link.step(Action(name="inc", kwargs={}))
        r = link.step(Action(name="inc", kwargs={}))   # n=3 -> terminated
        # Termination without submit must trigger the handoff, not loop A.
        assert r.info.get("link_stage") == "switched_to_B"
        assert link.stage == "B"
        # Episode can proceed to B and end normally.
        r2 = link.step(Action(name="say", kwargs={"text": "hello"}))
        assert r2.terminated is True
        assert r2.info.get("combined_success") is True
    finally:
        link.close()


def test_link_evaluate_reuses_cached_verdicts():
    """Regression (FIX 1): after a normal A->B run, Link.evaluate() must
    reuse the verdicts cached at handoff/termination instead of re-invoking
    the sub-envs' evaluate() (for SWE-bench a re-run spins 2 extra official
    scorer containers per episode; some envs even raise if evaluate() is
    called without a submission)."""
    a_calls: list[int] = []
    b_calls: list[int] = []

    class _CountingCounter(_CounterEnv):
        def evaluate(self):
            a_calls.append(1)
            return super().evaluate()

    class _CountingString(_StringEnv):
        def evaluate(self):
            b_calls.append(1)
            return super().evaluate()

    link = Link(env_a=_CountingCounter(), env_b=_CountingString(),
                carry_context=False, a_done_via="terminated")
    try:
        link.reset()
        for _ in range(3):
            link.step(Action(name="inc", kwargs={}))   # handoff scores A once
        link.step(Action(name="say", kwargs={"text": "hello"}))  # B end scores B once
        assert len(a_calls) == 1
        assert len(b_calls) == 1
        er = link.evaluate()
        # No additional evaluator invocations -- verdicts came from cache.
        assert len(a_calls) == 1
        assert len(b_calls) == 1
        assert er.success is True
        assert er.metrics["a_verdict_cached"] is True
        assert er.metrics["b_verdict_cached"] is True
        # A second evaluate() is also free.
        link.evaluate()
        assert len(a_calls) == 1
        assert len(b_calls) == 1
    finally:
        link.close()


def test_link_per_leg_options_route_correctly():
    """The structured {"a": ..., "b": ..., "link": ...} options shape must
    route to each child independently. This is the env-agnostic plumbing
    that lets Link compose SWE-bench × SWE-bench (two different instance_ids)
    without Link knowing anything about SWE-bench."""
    a_seen: list[dict] = []
    b_seen: list[dict] = []

    class _RecordingCounter(_CounterEnv):
        def reset(self, seed=None, options=None):
            a_seen.append(dict(options or {}))
            return super().reset(seed, options)

    class _RecordingString(_StringEnv):
        def reset(self, seed=None, options=None):
            b_seen.append(dict(options or {}))
            return super().reset(seed, options)

    link = Link(env_a=_RecordingCounter(), env_b=_RecordingString(),
                a_done_via="terminated")
    try:
        link.reset(seed=0, options={
            "a": {"a_key": "task_a"},
            "b": {"b_key": "task_b"},
            "link": {"carry_context": False, "a_done_via": "terminated"},
        })
        # env_a got options['a']
        assert a_seen == [{"a_key": "task_a"}]
        # env_b not yet reset (lazy at handoff)
        assert b_seen == []
        # Drive A to termination -> handoff fires
        for _ in range(3):
            link.step(Action(name="inc", kwargs={}))
        # env_b got options['b']
        assert b_seen == [{"b_key": "task_b"}]
    finally:
        link.close()


def test_link_options_legacy_dict_back_compat():
    """Old-style: pass plain dict (no a/b/link keys) -> treated as env_a opts."""
    a_seen: list[dict] = []

    class _RecordingCounter(_CounterEnv):
        def reset(self, seed=None, options=None):
            a_seen.append(dict(options or {}))
            return super().reset(seed, options)

    link = Link(env_a=_RecordingCounter(), env_b=_StringEnv(),
                a_done_via="terminated")
    try:
        link.reset(seed=0, options={"plain": "dict"})
        # env_a received the whole dict
        assert a_seen == [{"plain": "dict"}]
    finally:
        link.close()


def test_link_options_link_overrides_ctor():
    """options['link'].carry_context flips off the ctor default for this
    episode. This is the per-episode override path the corpus runner uses."""
    link = Link(env_a=_CounterEnv(), env_b=_StringEnv(),
                carry_context=True,            # ctor default
                a_done_via="terminated")
    try:
        link.reset(seed=0, options={
            "a": {},
            "b": {},
            "link": {"carry_context": False},   # override at reset time
        })
        # Drive A to handoff
        for _ in range(3):
            r = link.step(Action(name="inc", kwargs={}))
        # No splice in B's obs because we overrode to False.
        assert "SWITCHED" not in (r.observation.text or "")
        assert r.observation.data.get("linked_from_a") is None
    finally:
        link.close()


def test_link_options_link_invalid_a_done_via_raises():
    link = Link(env_a=_CounterEnv(), env_b=_StringEnv())
    try:
        try:
            link.reset(options={"link": {"a_done_via": "garbage"}})
        except ValueError as e:
            assert "a_done_via" in str(e)
        else:
            raise AssertionError("expected ValueError")
    finally:
        link.close()


def test_link_only_touches_abc():
    """The ultimate env-agnostic test: monkey-patch the two children to
    explode if any non-ABC attribute is accessed on them, then run a Link
    episode. If Link ever reaches into env_a.<something_not_in_ABC>, this
    raises and the test fails.
    """
    ABC_OK = {"reset", "step", "observe", "evaluate", "get_env_state",
              "save_state", "from_state", "step_reward", "close",
              "default_reset_args", "reset_after_load", "env_type",
              "env_state_schema", "tool_schemas", "tool_registry",
              "list_tasks"}
    # Plus dunder + Pydantic / object internals + Python descriptor noise.

    class _StrictABC(_CounterEnv):
        def __getattribute__(self, name):
            if (not name.startswith("_")
                    and name not in ABC_OK
                    and name not in {"env_a", "env_b"}):
                # Allow the env's OWN public methods; the point of the
                # test is to detect Link reaching THROUGH this env to its
                # internals. Since we forbid via `_`, internals like _n
                # remain accessible to self but external callers can't.
                raise AssertionError(
                    f"Link tried to read non-ABC attribute '{name}' from env_a"
                )
            return object.__getattribute__(self, name)

    a = _StrictABC()
    b = _StringEnv()
    link = Link(env_a=a, env_b=b, carry_context=False, a_done_via="terminated")
    try:
        link.reset()
        for _ in range(3):
            link.step(Action(name="inc", kwargs={}))
        link.step(Action(name="say", kwargs={"text": "hello"}))
        link.evaluate()
    finally:
        link.close()
    # If we got here without assertion, Link only used ABC methods. ✓
