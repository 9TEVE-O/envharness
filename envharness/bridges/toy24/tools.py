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

"""Tools for the toy24 game.

Three tools, no runtime knobs (no disabled-op checks, no integer-only checks
-- those are Rules-hook concerns now). Tools just compute the math.
"""
from __future__ import annotations

from typing import Any

from envharness.core.tool import Tool


def _apply(op: str, a: float, b: float) -> float | None:
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div":
        if abs(b) < 1e-6: return None
        return a / b
    return None


class Combine(Tool):
    name = "combine"
    description = ("Combine the numbers at positions i and j using one of "
                   "{add, sub, mul, div}. Result replaces them; positions "
                   "shift accordingly.")

    @classmethod
    def invoke(cls, env_state: Any, i: int, j: int, op: str) -> Any:
        if env_state.stopped:
            return {"error": "already stopped"}
        n = len(env_state.current_numbers)
        if not (0 <= i < n and 0 <= j < n) or i == j:
            return {"error": f"invalid indices i={i} j={j}, n={n}"}
        a, b = env_state.current_numbers[i], env_state.current_numbers[j]
        v = _apply(op, a, b)
        if v is None:
            return {"error": f"operation {op} failed on {a}, {b}"}
        rest = [x for k, x in enumerate(env_state.current_numbers) if k not in (i, j)]
        env_state.current_numbers = rest + [v]
        env_state.history.append(f"{a} {op} {b} = {v}")
        return {"ok": True, "result": v, "numbers": list(env_state.current_numbers)}


class Reset(Tool):
    name = "reset"
    description = "Reset numbers back to the puzzle's initial values."

    @classmethod
    def invoke(cls, env_state: Any) -> Any:
        env_state.current_numbers = [float(n) for n in env_state.initial_numbers]
        env_state.history.append("reset")
        return {"ok": True, "numbers": list(env_state.current_numbers)}


class Stop(Tool):
    name = "stop"
    description = ("Declare the puzzle solved. Success iff a remaining number "
                   "equals the target (within 1e-6).")

    @classmethod
    def invoke(cls, env_state: Any) -> Any:
        env_state.stopped = True
        env_state.success = any(abs(x - env_state.target) < 1e-6
                                  for x in env_state.current_numbers)
        return {"ok": True, "success": env_state.success}
