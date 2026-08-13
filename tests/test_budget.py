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

"""Tests for BudgetPolicy implementations (when does the inner loop stop?).

Each Budget is a small state machine called once per attempt:
  should_stop(attempts, last_decision, objective_signal) -> bool
"""
import pytest

from envharness.orchestration.budget import (
    FixedBudget, CappedAdaptive, ObjectiveDriven,
)
from envharness.core.types import Decision, ObjectiveSignal


def _sig(score: float = 0.5) -> ObjectiveSignal:
    return ObjectiveSignal(score=score, diagnostic="", suggestion_prompt="")


# -- FixedBudget ------------------------------------------------------------

def test_fixed_budget_rejects_invalid_k():
    with pytest.raises(ValueError):
        FixedBudget(k=0)


def test_fixed_budget_stops_at_k():
    b = FixedBudget(k=3)
    assert b.should_stop(0, Decision.REFINE, _sig()) is False
    assert b.should_stop(2, Decision.REFINE, _sig()) is False
    assert b.should_stop(3, Decision.REFINE, _sig()) is True


# -- CappedAdaptive ---------------------------------------------------------

def test_capped_adaptive_rejects_invalid_max_k():
    with pytest.raises(ValueError):
        CappedAdaptive(max_k=0)


def test_capped_adaptive_stops_on_accept_regardless_of_attempts():
    """ACCEPT means the Rules is done — Budget yields immediately."""
    b = CappedAdaptive(max_k=10)
    assert b.should_stop(1, Decision.ACCEPT, _sig()) is True


def test_capped_adaptive_runs_to_cap_on_refine():
    b = CappedAdaptive(max_k=3)
    assert b.should_stop(0, Decision.REFINE, _sig()) is False
    assert b.should_stop(2, Decision.REFINE, _sig()) is False
    assert b.should_stop(3, Decision.REFINE, _sig()) is True


# -- ObjectiveDriven -------------------------------------------------------

def test_objective_driven_stops_at_cap():
    b = ObjectiveDriven(score_threshold=0.95, max_k=4)
    assert b.should_stop(4, Decision.REFINE, _sig(0.0)) is True


def test_objective_driven_stops_on_score_above_threshold():
    """ObjectiveDriven stops as soon as score crosses threshold,
    regardless of the Rules's decision."""
    b = ObjectiveDriven(score_threshold=0.5, max_k=10)
    assert b.should_stop(1, Decision.REFINE, _sig(0.4)) is False
    assert b.should_stop(1, Decision.REFINE, _sig(0.9)) is True


def test_objective_driven_stops_on_accept():
    """An ACCEPT short-circuits the score check."""
    b = ObjectiveDriven(score_threshold=0.99, max_k=10)
    assert b.should_stop(1, Decision.ACCEPT, _sig(0.0)) is True
