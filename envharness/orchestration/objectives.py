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

"""MutationObjective ABC + built-in implementations.

Each Objective consumes recent traces and outputs an ObjectiveSignal
(score + diagnostic + suggestion_prompt + optional weights). Rules's
operation mechanism is fixed; what it optimizes is whatever Objective is
plugged in.

Objective only specifies DIRECTION, not concrete numerical mutations. The
HarnessAgent LLM reads the suggestion_prompt and decides the actual values.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from envharness.core.types import ObjectiveSignal, Trace


class MutationObjective(ABC):
    name: str = ""

    @abstractmethod
    def evaluate(self, recent_traces: list[Trace]) -> ObjectiveSignal: ...


# ---------------------------------------------------------------------------
# DifficultyZone -- maintain Policy success rate inside a target band
# ---------------------------------------------------------------------------

class DifficultyZone(MutationObjective):
    name = "difficulty_zone"

    def __init__(self, target_band: tuple[float, float] = (0.3, 0.7),
                 window: int = 10):
        lo, hi = target_band
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"invalid target_band {target_band}")
        self.lo, self.hi = lo, hi
        self.window = window

    def evaluate(self, recent_traces):
        accepted = [t for t in recent_traces if t.kind == "accepted"]
        recent = accepted[-self.window:]
        if not recent:
            return ObjectiveSignal(
                score=0.0,
                diagnostic="No accepted traces yet -- bootstrap phase.",
                suggestion_prompt="Start with mild mutations across all axes.",
            )
        sr = sum(1 for t in recent if t.success) / len(recent)
        center = (self.lo + self.hi) / 2
        half_width = (self.hi - self.lo) / 2
        score = max(0.0, 1.0 - abs(sr - center) / max(half_width, 1e-6))

        if sr > self.hi:
            verdict = f"TOO EASY by {sr - self.hi:.2f}"
            direction = "Increase difficulty."
        elif sr < self.lo:
            verdict = f"TOO HARD by {self.lo - sr:.2f}"
            direction = "Decrease difficulty."
        else:
            verdict = "in target band"
            direction = "Maintain. Prefer diversity over change."

        # axis distribution from failure analyses
        axes = Counter(
            t.failure_analysis.primary_axis for t in recent
            if t.failure_analysis and t.failure_analysis.primary_axis
        )
        weights = _weights_from_failure_axes(axes, n_recent=len(recent))

        return ObjectiveSignal(
            score=score,
            diagnostic=(
                f"Recent {len(recent)} accepted episodes: success_rate={sr:.2f}, "
                f"target_band=[{self.lo:.2f}, {self.hi:.2f}]. {verdict}. "
                f"Failure-axis distribution: {dict(axes)}."
            ),
            suggestion_prompt=(
                f"{direction} Target SR is in [{self.lo:.2f}, {self.hi:.2f}]; "
                f"current is {sr:.2f}. Prefer mutations on axes that are "
                f"under-represented in recent failures."
            ),
            weights=weights,
        )


def _weights_from_failure_axes(axes: Counter, n_recent: int) -> dict[str, float]:
    # idle axes (no recent failures) get higher weight when increasing difficulty
    inv = {a: 1.0 / (axes.get(a, 0) + 1.0) for a in ("S0", "A", "O", "T", "R")}
    s = sum(inv.values()) or 1.0
    return {a: round(v / s, 3) for a, v in inv.items()}


# ---------------------------------------------------------------------------
# RedTeam -- search for configurations that cause Policy to fail
# ---------------------------------------------------------------------------

class RedTeam(MutationObjective):
    name = "red_team"

    def __init__(self, window: int = 10):
        self.window = window

    def evaluate(self, recent_traces):
        recent = recent_traces[-self.window:]
        if not recent:
            return ObjectiveSignal(
                score=0.0,
                diagnostic="No traces yet.",
                suggestion_prompt="Probe for failure-inducing mutations across all axes.",
            )
        attacks = sum(1 for t in recent if not t.success)
        score = attacks / len(recent)
        failure_axes = Counter(
            t.failure_analysis.primary_axis for t in recent
            if t.failure_analysis and t.failure_analysis.primary_axis
        )
        labels = Counter(
            t.failure_analysis.label for t in recent
            if t.failure_analysis and t.failure_analysis.label
        )
        return ObjectiveSignal(
            score=score,
            diagnostic=(
                f"Adversarial attempts last {len(recent)}: {attacks} caused Policy "
                f"failure. Failure axes: {dict(failure_axes)}. "
                f"Top labels: {dict(labels.most_common(5))}."
            ),
            suggestion_prompt=(
                "Continue searching for failure-inducing mutations. Build on "
                "axes/labels already proven effective; explore untested combinations. "
                "Report a minimal repro for any new failure mode."
            ),
            weights=None,
        )
