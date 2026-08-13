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

"""Tests for the core data contracts in envharness.core.types.

When you add a new field to one of these models, add a test below so that
breaking changes to consumers (Bridges, Mutators, Objectives) get caught
at the type layer instead of inside a 10-min Docker rollout.
"""
import pytest
from pydantic import ValidationError

from envharness.core.types import (
    Action, Blocked, Observation, EnvResponse, Candidate, Decision,
    DecideResult, FailureAnalysis, Trace, EvaluationResult,
)


def test_action_extra_fields_forbidden():
    """Action uses extra='forbid' — typos in kwargs surface as ValidationError."""
    Action(name="combine", kwargs={"i": 0})
    with pytest.raises(ValidationError):
        Action(name="combine", kwargs={"i": 0}, oops="typo")


def test_blocked_default_kind():
    b = Blocked(reason="too risky")
    assert b.kind == "blocked"
    assert b.reason == "too risky"


def test_observation_text_and_extras():
    obs = Observation(text="see room", data={"key": "value"})
    assert obs.text == "see room"
    assert obs.data["key"] == "value"


def test_env_response_round_trip_json():
    """EnvResponse must be JSON-serializable (subprocess workers exchange JSON)."""
    r = EnvResponse(
        observation=Observation(text="hello"),
        reward=1.5, terminated=True, truncated=False,
        info={"won": True},
    )
    s = r.model_dump_json()
    r2 = EnvResponse.model_validate_json(s)
    assert r2.reward == 1.5
    assert r2.observation.text == "hello"
    assert r2.info == {"won": True}


def test_candidate_defaults():
    c = Candidate()
    assert c.rules_code == ""
    assert c.in_env_actions == []
    assert c.rationale == ""


def test_decision_enum_strings():
    """Decision is a str-Enum: code can compare to either the enum or the string."""
    assert Decision.ACCEPT == "accept"
    assert Decision.REJECT.value == "reject"
    assert Decision("refine") is Decision.REFINE


def test_decide_result_with_failure_analysis():
    dr = DecideResult(
        decision=Decision.REFINE,
        failure_analysis=FailureAnalysis(primary_axis="A", label="too_strict"),
        rationale="A axis too narrow",
    )
    assert dr.decision is Decision.REFINE
    assert dr.failure_analysis.primary_axis == "A"


def test_failure_analysis_primary_axis_enum_constraint():
    """primary_axis must be one of the six known axes (or 'task_understanding' / 'none')."""
    FailureAnalysis(primary_axis="S0")
    FailureAnalysis(primary_axis="none")
    with pytest.raises(ValidationError):
        FailureAnalysis(primary_axis="Z")


def test_trace_minimal_shape():
    """A Trace must carry a Candidate; everything else has defaults."""
    t = Trace(episode_id="ep1", iteration_id="it1", task_id="t1",
              candidate=Candidate())
    assert t.kind == "accepted"           # default
    assert t.success is False
    assert t.steps == []


def test_evaluation_result_default_score():
    r = EvaluationResult(success=True)
    assert r.score == 0.0
    assert r.metrics == {}
