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

"""Shared pytest fixtures.

Design principle: NO GPU, NO docker, NO API key. Tests run against the
toy24 ActionableEnv (pure stdlib) and ScriptedClient (canned LLM responses)
so they finish in seconds on a laptop and on CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the release root importable without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from envharness.bridges.toy24 import Toy24Env


# Known-solvable toy24 puzzle. Solution sequence:
#   combine(i=2, j=0, op=mul) → 7 * 3 = 21,  state=[3, 7, 21]
#   combine(i=2, j=0, op=add) → 21 + 3 = 24, state=[7, 24]
#   stop                        → 24 in state → success
PUZZLE_NUMBERS = [3, 3, 7, 7]
PUZZLE_TARGET = 24
SOLVING_SCRIPT = [
    {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "mul"}},
    {"name": "combine", "kwargs": {"i": 2, "j": 0, "op": "add"}},
    {"name": "stop", "kwargs": {}},
]


@pytest.fixture
def fresh_toy24() -> Toy24Env:
    """A reset Toy24Env pinned to the known-solvable puzzle."""
    e = Toy24Env()
    e.reset(seed=0, options={"numbers": PUZZLE_NUMBERS, "target": PUZZLE_TARGET})
    return e
