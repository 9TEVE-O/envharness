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

"""BudgetPolicy -- decides when the inner search loop terminates.

Three built-in implementations:
  FixedBudget(k)              -- always run K episodes per outer iteration
  CappedAdaptive(max_k)       -- Rules self-decides ACCEPT, capped at max_k
  ObjectiveDriven(threshold)  -- stop when ObjectiveSignal.score >= threshold
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from envharness.core.types import Decision, ObjectiveSignal


class BudgetPolicy(ABC):
    @abstractmethod
    def should_stop(self, attempts: int, last_decision: Decision,
                    objective_signal: ObjectiveSignal | None) -> bool: ...


class FixedBudget(BudgetPolicy):
    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def should_stop(self, attempts, last_decision, objective_signal):
        return attempts >= self.k


class CappedAdaptive(BudgetPolicy):
    """Rules decides; cap is a safety net."""
    def __init__(self, max_k: int = 8):
        if max_k < 1:
            raise ValueError("max_k must be >= 1")
        self.max_k = max_k

    def should_stop(self, attempts, last_decision, objective_signal):
        if last_decision == Decision.ACCEPT:
            return True
        return attempts >= self.max_k


class ObjectiveDriven(BudgetPolicy):
    """Stop when objective score crosses threshold OR cap is hit."""
    def __init__(self, score_threshold: float = 0.8, max_k: int = 20):
        self.score_threshold = score_threshold
        self.max_k = max_k

    def should_stop(self, attempts, last_decision, objective_signal):
        if attempts >= self.max_k:
            return True
        if last_decision == Decision.ACCEPT:
            return True
        if objective_signal and objective_signal.score >= self.score_threshold:
            return True
        return False
