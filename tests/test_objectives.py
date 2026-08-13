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

"""Tests for MutationObjective implementations.

Objectives consume a list of recent Traces and emit an ObjectiveSignal
that's fed into the Rules's decide() prompt. When you add a new
objective, model its tests on this file.
"""
import pytest

from envharness.orchestration.objectives import DifficultyZone
from envharness.core.types import (
    Candidate, ObjectiveSignal, Trace,
)


def _trace(success: bool, kind: str = "accepted") -> Trace:
    return Trace(
        episode_id="ep", iteration_id="it", task_id="t",
        candidate=Candidate(), success=success, kind=kind,
    )


def test_difficulty_zone_rejects_bad_band():
    with pytest.raises(ValueError):
        DifficultyZone(target_band=(0.7, 0.3))      # lo > hi
    with pytest.raises(ValueError):
        DifficultyZone(target_band=(-0.1, 0.5))      # below 0
    with pytest.raises(ValueError):
        DifficultyZone(target_band=(0.3, 1.5))       # above 1


def test_difficulty_zone_empty_history_bootstrap():
    obj = DifficultyZone(target_band=(0.4, 0.6))
    sig = obj.evaluate([])
    assert isinstance(sig, ObjectiveSignal)
    assert sig.score == 0.0
    assert "bootstrap" in sig.diagnostic.lower()


def test_difficulty_zone_in_band_high_score():
    obj = DifficultyZone(target_band=(0.4, 0.6))
    traces = [_trace(True), _trace(False)]   # SR = 0.5, exact center
    sig = obj.evaluate(traces)
    assert sig.score == 1.0
    assert "in target band" in sig.diagnostic.lower()


def test_difficulty_zone_too_easy_flagged():
    obj = DifficultyZone(target_band=(0.3, 0.7))
    traces = [_trace(True)] * 10              # SR = 1.0
    sig = obj.evaluate(traces)
    assert sig.score < 1.0
    assert "too easy" in sig.diagnostic.lower()
    assert "increase difficulty" in sig.suggestion_prompt.lower()


def test_difficulty_zone_too_hard_flagged():
    obj = DifficultyZone(target_band=(0.3, 0.7))
    traces = [_trace(False)] * 10
    sig = obj.evaluate(traces)
    assert sig.score < 1.0
    assert "too hard" in sig.diagnostic.lower()
    assert "decrease difficulty" in sig.suggestion_prompt.lower()


def test_difficulty_zone_ignores_exploration_traces():
    """Only kind='accepted' traces feed the SR signal."""
    obj = DifficultyZone(target_band=(0.4, 0.6))
    traces = [_trace(True, kind="exploration")] * 5    # ignored
    sig = obj.evaluate(traces)
    assert "bootstrap" in sig.diagnostic.lower()


def test_difficulty_zone_axis_weights_emitted():
    """weights dict covers the 5 mutation axes."""
    obj = DifficultyZone(target_band=(0.4, 0.6))
    traces = [_trace(False), _trace(True)]
    sig = obj.evaluate(traces)
    assert sig.weights is not None
    assert set(sig.weights.keys()) == {"S0", "A", "O", "T", "R"}
    # Weights are normalized to sum ~1.
    assert sum(sig.weights.values()) == pytest.approx(1.0, abs=0.01)
